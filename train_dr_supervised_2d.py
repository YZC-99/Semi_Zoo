import random
from tqdm import tqdm
from dataloader.fundus import SemiDataset
from dataloader.samplers import TwoStreamBatchSampler, LabeledBatchSampler,UnlabeledBatchSampler
import torch
import glob
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR,LambdaLR
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss,MSELoss
from torch.cuda.amp import autocast,GradScaler
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from model.netwotks.unet import UNet, MCNet2d_compete_v1,UNet_DTC2d
from model.netwotks.unet_two_decoder import UNet_two_Decoder,UNet_MiT,UNet_ResNet
from utils import ramps,losses
from utils.losses import OhemCrossEntropy
from utils.test_utils import DR_metrics
import random
from utils.util import get_optimizer,PolyLRwithWarmup, compute_sdf,compute_sdf_luoxd,compute_sdf_multi_class
import time
import logging
import os
import shutil
import logging
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--num_works',type=int,default=0)
parser.add_argument('--model',type=str,default='unet')
parser.add_argument('--backbone',type=str,default='b2')
parser.add_argument('--lr_decouple',action='store_true')

parser.add_argument('--exp',type=str,default='IDRID')
parser.add_argument('--save_period',type=int,default=5000)
parser.add_argument('--val_period',type=int,default=100)

parser.add_argument('--dataset_name',type=str,default='IDRID')
parser.add_argument('--unlabeled_txt',type=str,default='unlabeled_addDDR.txt')

parser.add_argument('--optim',type=str,default='AdamW')
parser.add_argument('--amp',type=bool,default=True)
parser.add_argument('--num_classes',type=int,default=5)
parser.add_argument('--base_lr',type=float,default=0.0001)

parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--labeled_bs',type=int,default=16)

parser.add_argument('--od_rim',type=bool,default=True)
parser.add_argument('--oc_label',type=int,default=2)
parser.add_argument('--image_size',type=int,default=1024)

parser.add_argument('--labeled_num',type=int,default=100,help="5%:100,")
parser.add_argument('--total_num',type=int,default=11249,help="SEG:2859;SEG_add_DDR:11249")


parser.add_argument('--scale_num',type=int,default=2)
parser.add_argument('--max_iterations',type=int,default=10000)

parser.add_argument('--ohem',type=float,default=-1.0)
parser.add_argument('--with_dice',type=bool,default=False)

parser.add_argument('--beta', type=float,  default=0.3,
                    help='balance factor to control regional and sdm loss')
parser.add_argument('--consistency', type=float,
                    default=1.0, help='consistency')
parser.add_argument('--consistency_type', type=str,
                    default="kl", help='consistency_type')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')

parser.add_argument('--cps_la_weight_final', type=float,  default=0.1)
parser.add_argument('--cps_la_rampup_scheme', type=str,  default='None', help='cps_la_rampup_scheme')
parser.add_argument('--cps_la_rampup',type=float,  default=40.0)
parser.add_argument('--cps_la_with_dice',type=bool,default=False)

parser.add_argument('--cps_un_weight_final', type=float,  default=0.1, help='consistency')
parser.add_argument('--cps_un_rampup_scheme', type=str,  default='None', help='cps_rampup_scheme')
parser.add_argument('--cps_un_rampup', type=float,  default=40.0, help='cps_rampup')
parser.add_argument('--cps_un_with_dice', type=bool,  default=True, help='cps_un_with_dice')

def build_model(model,backbone,in_chns,class_num1,class_num2,fuse_type):
    if model == "UNet":
        return UNet(in_chns=in_chns,class_num=class_num1)
    elif model == 'UNet_ResNet':
        return UNet_ResNet(in_chns=in_chns, class_num=class_num1, phi=backbone, pretrained=True)
    elif model == 'UNet_MiT':
        return UNet_MiT( in_chns=in_chns, class_num=class_num1,phi=backbone,pretrained=True)
    elif model == 'UNet_two_Decoder':
        return UNet_two_Decoder( in_chns=in_chns, class_num1=class_num1,class_num2=class_num2,fuse_type=fuse_type)




def create_version_folder(snapshot_path):
    # 检查是否存在版本号文件夹
    version_folders = [name for name in os.listdir(snapshot_path) if name.startswith('version')]

    if not version_folders:
        # 如果不存在版本号文件夹，则创建version0
        new_folder = os.path.join(snapshot_path, 'version0')
    else:
        # 如果存在版本号文件夹，则创建下一个版本号文件夹
        last_version = max(version_folders)
        last_version_number = int(last_version.replace('version', ''))
        next_version = 'version{:02d}'.format(last_version_number + 1)
        new_folder = os.path.join(snapshot_path, next_version)

    os.makedirs(new_folder)
    return new_folder


args = parser.parse_args()
snapshot_path = "./exp_2d_dr/" + args.exp + "/"
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs


if __name__ == '__main__':
    """
    1、搜寻snapshot_path下面的含有version的文件夹，如果没有就创建version0，即：
    snapshot_path = snapshot_path + "/" + "version0"
    2、如果有version的文件夹，如果当前文件夹下有version0和version01，则创建version02，以此类推
    """

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    snapshot_path = create_version_folder(snapshot_path)

    device = "cuda:{}".format(args.device)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # init model
    model = build_model(model=args.model,backbone=args.backbone,in_chns=3,class_num1=args.num_classes,class_num2=2,fuse_type=None)
    model.to(device)

    # init dataset
    labeled_dataset = SemiDataset(name='./dataset/{}'.format(args.dataset_name),
                                  root="/home/gu721/yzc/data/dr/{}".format(args.dataset_name),
                                  mode='semi_train',
                                  size=args.image_size,
                                  id_path='train.txt')


    labeledtrainloader = DataLoader(labeled_dataset,batch_size=args.batch_size, num_workers=args.num_works, pin_memory=True,)

    model.train()
    # init optimizer
    optimizer = get_optimizer(model=model,name=args.optim,base_lr=args.base_lr,lr_decouple=args.lr_decouple)


    # scheduler = StepLR(optimizer,step_size=100,gamma=0.999)
    scheduler = PolyLRwithWarmup(optimizer,total_steps=args.max_iterations,warmup_steps=args.max_iterations * 0.01)
    # init summarywriter
    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_iterations // len(labeledtrainloader) + 1
    lr_ = args.base_lr
    model.train()

    # ce_loss = BCEWithLogitsLoss()
    if args.ohem > 0:
        ce_loss = OhemCrossEntropy(thres=args.ohem,weight=torch.tensor([1.0,2.8,3.0],device=device))
    else:
        ce_loss = CrossEntropyLoss(ignore_index=255)
    # mse_loss = MSELoss()


    # 验证集
    # init dataset
    val_dataset = SemiDataset(name='./dataset/{}'.format(args.dataset_name),
                                    # root="D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/REFUGE",
                                  root="/home/gu721/yzc/data/odoc/{}".format(args.dataset_name),
                                  mode='val',
                                  size=args.image_size)

    val_labeledtrainloader = DataLoader(val_dataset,batch_size=1,num_workers=args.num_works)
    val_iteriter = tqdm(val_labeledtrainloader)


    # 开始训练
    iterator = tqdm(range(max_epoch), ncols=70)

    DR_val_metrics = DR_metrics(device)
    AUC = {}
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch,labeled_sampled_batch in enumerate(labeledtrainloader):
            time2 = time.time()

            labeled_batch, label_label_batch = labeled_sampled_batch['image'].to(device), labeled_sampled_batch['label'].to(device)

            all_batch = labeled_batch
            all_label_batch = label_label_batch
            all_label_batch[all_label_batch > 2] = 0

            outputs = model(all_batch)

            # calculate the loss
            # 这里需要注意，如果是分割三个类别以上，则需要分开计算dist和分开计算mse
            outputs_soft = torch.argmax(outputs,dim=1)
            loss_seg_ce = ce_loss(outputs,all_label_batch)
            # loss_seg_dice = losses.dice_loss(outputs_soft,all_label_batch)
            # loss = loss_seg_ce + loss_seg_dice
            loss = loss_seg_ce
            # loss =  loss_seg_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg_ce, iter_num)
            # writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)

            logging.info(
                'iteration %d : loss : %f' %
                (iter_num, loss.item()))
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            logging.info('iteration %d : loss_seg : %f' % (iter_num, loss_seg_ce.item()))
            # logging.info('iteration %d : loss_dice : %f' % (iter_num, loss_seg_dice.item()))

            if iter_num % 1 == 0:
                image = all_batch[0]
                writer.add_image('train/Image', image, iter_num)

                image = torch.argmax(outputs,dim=1)
                # image = image[0] / (args.num_classes - 1)
                image = image[0]
                # 将单通道张量转换为 RGB 通道
                w,h = image.size()
                image_rgb = torch.zeros(3, w,h)
                for i in range(3):
                    image_rgb[i, :, :] = (image == i).float()
                writer.add_image('train/Predicted_label', image_rgb, iter_num)


                image = all_label_batch[0]
                w,h = image.size()
                image_rgb = torch.zeros(3, w,h)
                for i in range(3):
                    image_rgb[i, :, :] = (image == i).float()
                # image = image / (args.num_classes - 1)
                writer.add_image('train/Groundtruth_label',
                                 image_rgb, iter_num)


            # eval
            if iter_num % args.val_period == 0:
                model.eval()
                show_id = random.randint(0,len(val_iteriter))
                for id,data in enumerate(val_iteriter):
                    img,label = data['image'].to(device),data['label'].to(device)
                    outputs = model(img)

                    DR_val_metrics.add(outputs,label)

                    if id == show_id:
                        image = img[0]
                        writer.add_image('val/image', image, iter_num)
                        image = torch.argmax(outputs,dim=1)
                        image = image[0] / (args.num_classes - 1)
                        writer.add_image('val/pred', image.unsqueeze(0), iter_num)
                        image = label
                        image = image / (args.num_classes - 1)
                        writer.add_image('val/Groundtruth_label',
                                         image, iter_num)

                val_metrics = DR_val_metrics.get_metrics()
                AUC = val_metrics[0]
                MA_auc, HE_auc, EX_auc, SE_auc = AUC['MA_auc'],AUC['HE_auc'],AUC['EX_auc'],AUC['SE_auc']



                logging.info("MA_auc:{}--HE_auc:--{}--EX_auc:{}--SE_auc:--{}".format(
                                                                                        MA_auc,
                                                                                        HE_auc,
                                                                                        EX_auc,
                                                                                       SE_auc,
                                                                                       ))
                writer.add_scalar('val/MA_auc',MA_auc, iter_num)
                writer.add_scalar('val/HE_auc',HE_auc, iter_num)
                writer.add_scalar('val/EX_auc',EX_auc, iter_num)
                writer.add_scalar('val/SE_auc',SE_auc, iter_num)
                model.train()

            if iter_num % args.save_period == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
