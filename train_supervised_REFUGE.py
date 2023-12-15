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
from model.netwotks.segformer import SegFormer
from model.netwotks.deeplabv3plus import DeepLabV3Plus
from model.netwotks.unet_two_decoder import UNet_two_Decoder,UNet_MiT,UNet_ResNet,SR_UNet_ResNet
from utils import ramps,losses
from utils.losses import OhemCrossEntropy
from utils.test_utils import ODOC_metrics
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
parser.add_argument('--model',type=str,default='UNet_ResNet')
parser.add_argument('--backbone',type=str,default='resnet34')
parser.add_argument('--lr_decouple',action='store_true')

parser.add_argument('--exp',type=str,default='supervised/REFUGE')
parser.add_argument('--save_period',type=int,default=5000)
parser.add_argument('--val_period',type=int,default=100)

parser.add_argument('--dataset_name',type=str,default='REFUGE')
parser.add_argument('--unlabeled_txt',type=str,default='unlabeled_addDDR.txt')

parser.add_argument('--optim',type=str,default='AdamW')
parser.add_argument('--amp',type=bool,default=True)
parser.add_argument('--num_classes',type=int,default=3)
parser.add_argument('--base_lr',type=float,default=0.005)

parser.add_argument('--batch_size',type=int,default=16)
parser.add_argument('--labeled_bs',type=int,default=16)

parser.add_argument('--od_rim',type=bool,default=True)
parser.add_argument('--oc_label',type=int,default=2)
parser.add_argument('--image_size',type=int,default=256)

parser.add_argument('--labeled_num',type=int,default=100,help="5%:100,")
parser.add_argument('--total_num',type=int,default=11249,help="SEG:2859;SEG_add_DDR:11249")


parser.add_argument('--scale_num',type=int,default=2)
parser.add_argument('--max_iterations',type=int,default=10000)

parser.add_argument('--ohem',type=float,default=-1.0)
parser.add_argument('--with_ce',action='store_false')
parser.add_argument('--with_dice',action='store_true')

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
    elif model == 'SR_UNet_ResNet':
        return SR_UNet_ResNet(in_chns=in_chns, class_num=class_num1, phi=backbone, pretrained=True)
    elif model == 'UNet_MiT':
        return UNet_MiT( in_chns=in_chns, class_num=class_num1,phi=backbone,pretrained=True)
    elif model == 'Segformer':
        return SegFormer(num_classes=class_num1,phi=backbone,pretrained=True)
    elif model == 'Deeplabv3+':
        return DeepLabV3Plus(backbone=backbone,nclass=class_num1)
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
# snapshot_path = "./exp_2d/" + args.exp + "/"
snapshot_path = "./exp_2d/" + args.exp + "/"
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs


if __name__ == '__main__':
    assert args.with_ce  or args.with_dice,"ce 和 dice至少有一个！！！！！"


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
                                  root="/home/gu721/yzc/data/odoc/{}".format("REFUGE"),
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
        ce_loss = OhemCrossEntropy(thres=args.ohem,weight=torch.tensor([1.0,4.0,12.0],device=device))
    else:
        ce_loss = CrossEntropyLoss(ignore_index=255)

    # mse_loss = MSELoss()


    # 验证集
    # init dataset
    val_dataset = SemiDataset(name='./dataset/{}'.format(args.dataset_name),
                                    # root="D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/REFUGE",
                                  root="/home/gu721/yzc/data/odoc/{}".format("REFUGE"),
                                  mode='val',
                                  size=args.image_size)

    val_labeledtrainloader = DataLoader(val_dataset,batch_size=1,num_workers=args.num_works)
    val_iteriter = tqdm(val_labeledtrainloader)


    # 开始训练
    iterator = tqdm(range(max_epoch), ncols=70)

    ODOC_val_metrics = ODOC_metrics(device)
    best_OD_DICE,best_OC_DICE = 0,0
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

            if args.with_dice:
                loss_seg_dice = losses.dice_loss(outputs_soft,all_label_batch)
                loss = loss_seg_ce + loss_seg_dice
            else:
                loss = loss_seg_ce
            if not args.with_ce:
                loss_seg_ce = torch.zeros(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg_ce, iter_num)
            if args.with_dice:
                writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)

            logging.info(
                'iteration %d : loss : %f' %
                (iter_num, loss.item()))
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            logging.info('iteration %d : loss_seg : %f' % (iter_num, loss_seg_ce.item()))
            if args.with_dice:
                logging.info('iteration %d : loss_dice : %f' % (iter_num, loss_seg_dice.item()))

            if iter_num % 50 == 0:
                image = all_batch[0]
                writer.add_image('train/Image', image, iter_num)

                image = torch.argmax(outputs,dim=1)
                image = image[0] / (args.num_classes - 1)
                writer.add_image('train/Predicted_label', image.unsqueeze(0), iter_num)


                image = all_label_batch[0].unsqueeze(0)
                image = image / (args.num_classes - 1)
                writer.add_image('train/Groundtruth_label',
                                 image, iter_num)


            # eval
            if iter_num % args.val_period == 0:
                model.eval()
                show_id = random.randint(0,len(val_iteriter))
                for id,data in enumerate(val_iteriter):
                    img,label = data['image'].to(device),data['label'].to(device)
                    outputs = model(img)

                    ODOC_val_metrics.add_multi_class(outputs,label)

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

                Dice_IoU = ODOC_val_metrics.get_metrics()
                OD_DICE,OD_IOU,OC_DICE,OC_IOU =  Dice_IoU['od_dice'],Dice_IoU['od_iou'],Dice_IoU['oc_dice'],Dice_IoU['oc_iou']
                OD_BIOU,OC_BIOU =  Dice_IoU['od_biou'],Dice_IoU['oc_biou']
                logging.info("OD_Dice:{}--OD_IoU:--{}--OC_Dice:{}--OC_IoU:--{}".format(
                                                                                        OD_DICE,
                                                                                        OD_IOU,
                                                                                        OC_DICE,
                                                                                       OC_IOU,
                                                                                       ))
                writer.add_scalar('val/OD_Dice',OD_DICE, iter_num)
                writer.add_scalar('val/OD_IOU',OD_IOU, iter_num)
                writer.add_scalar('val/OD_BIOU',OD_BIOU, iter_num)
                writer.add_scalar('val/OC_Dice',OC_DICE, iter_num)
                writer.add_scalar('val/OC_IOU',OC_IOU, iter_num)
                writer.add_scalar('val/OC_BIOU',OC_BIOU, iter_num)

                if OD_DICE > best_OD_DICE:
                    best_OD_DICE = OD_DICE
                    name = "OD_DICE" + str(round(best_OD_DICE.item(), 4)) +'_iter_' + str(iter_num)  + '.pth'
                    save_mode_path = os.path.join(
                        snapshot_path, name)

                    previous_files = glob.glob(os.path.join(snapshot_path, '*OD_DICE*.pth'))
                    for file_path in previous_files:
                        os.remove(file_path)

                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

                if OC_DICE > best_OC_DICE:
                    best_OC_DICE = OC_DICE
                    previous_files = glob.glob(os.path.join(snapshot_path, '*OC_DICE*.pth'))
                    for file_path in previous_files:
                        os.remove(file_path)
                    name = "OC_DICE" + str(round(best_OC_DICE.item(), 4)) +'_iter_' + str(iter_num)  + '.pth'
                    save_mode_path = os.path.join(
                        snapshot_path, name)
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))



                model.train()
            # change lr
            # if iter_num % 2500 == 0:
            #     lr_ = base_lr * 0.1 ** (iter_num // 2500)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_
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
