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
from model.netwotks.unet_two_decoder import UNet_two_Decoder,UNet_MiT,SR_UNet_two_Decoder,UNet_two_Out
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
parser.add_argument('--model',type=str,default='unet')
parser.add_argument('--backbone',type=str,default='b2')
parser.add_argument('--fuse_type',type=str,default=None)
parser.add_argument('--lr_decouple',action='store_true')

# parser.add_argument('--exp',type=str,default='supervised/RIM-ONE_Vessel')
parser.add_argument('--exp',type=str,default='demo')


parser.add_argument('--save_period',type=int,default=5000)
parser.add_argument('--val_period',type=int,default=100)

parser.add_argument('--dataset_name',type=str,default='RIM-ONE')
parser.add_argument('--unlabeled_txt',type=str,default='unlabeled_addDDR.txt')

parser.add_argument('--optim',type=str,default='AdamW')
parser.add_argument('--amp',type=bool,default=True)
parser.add_argument('--num_classes',type=int,default=3)
parser.add_argument('--base_lr',type=float,default=0.005)
parser.add_argument('--vessel_loss_weight',type=float,default=0.1)
parser.add_argument('--no_vessel_weight_decay',action='store_false')
parser.add_argument('--CLAHE',action='store_true')
parser.add_argument('--preprocess',action='store_true')

parser.add_argument('--batch_size',type=int,default=16)
parser.add_argument('--labeled_bs',type=int,default=8)

parser.add_argument('--oc_label',type=int,default=2)
parser.add_argument('--image_size',type=int,default=256)

parser.add_argument('--labeled_num',type=int,default=99,help="RIM-ONE:99")
parser.add_argument('--total_num',type=int,default=144,help="HRF:45")


parser.add_argument('--scale_num',type=int,default=2)
parser.add_argument('--max_iterations',type=int,default=10000)

parser.add_argument('--ohem',type=float,default=-1.0)
parser.add_argument('--ce_weight',action='store_true')
parser.add_argument('--with_ce',action='store_false')
parser.add_argument('--with_softfocal',action='store_true')
parser.add_argument('--with_dice',action='store_true')



def build_model(model,backbone,in_chns,class_num1,class_num2,fuse_type):
    if model == "UNet":
        return UNet(in_chns=in_chns,class_num=class_num1)
    elif model == 'UNet_MiT':
        return UNet_MiT(in_chns=in_chns, class_num=class_num1,phi=backbone,pretrained=True)
    elif model == 'UNet_two_Decoder':
        return UNet_two_Decoder(in_chns=in_chns, class_num1=class_num1,class_num2=class_num2,phi=backbone,fuse_type=fuse_type)
    elif model == 'SR_UNet_two_Decoder':
        return SR_UNet_two_Decoder(in_chns=in_chns, class_num1=class_num1,class_num2=class_num2,phi=backbone)
    elif model == 'UNet_two_Out':
        return UNet_two_Out(in_chns=in_chns, class_num1=class_num1,class_num2=class_num2,phi=backbone,fuse_type=fuse_type)

def get_vessel_loss_weight(iter):
    # 发现训练容易塌陷，所以考虑对vessel的权重进行退火衰减
    weight = args.vessel_loss_weight * ( 1 - iter / args.max_iterations)
    return weight


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
snapshot_path = "./exp_2d/" + args.exp + "/"
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs


if __name__ == '__main__':
    assert args.with_ce  or args.with_dice or args.with_softfocal,"ce , dice , focal至少有一个！！！！！"


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
    in_chns = 3
    model = build_model(model=args.model,backbone=args.backbone,in_chns=in_chns,class_num1=args.num_classes,class_num2=2,fuse_type=args.fuse_type)
    model.to(device)

    # init dataset
    odoc_dataset = SemiDataset(name='./dataset/{}'.format(args.dataset_name),
                                  root="/home/gu721/yzc/data/odoc/{}".format(args.dataset_name),
                                  mode='semi_train',
                                  size=args.image_size,
                                  id_path='train_addHRF.txt',CLAHE=args.CLAHE,preprocess=args.preprocess)

    vessel_dataset = SemiDataset(name='./dataset/{}'.format(args.dataset_name),
                                  root="/home/gu721/yzc/data/vessel/{}".format(''),
                                  mode='semi_train',
                                  size=args.image_size,
                                  id_path='train_addHRF.txt',CLAHE=args.CLAHE,preprocess=args.preprocess)

    odoc_idxs = list(range(args.labeled_num))
    vessel_idxs = list(range(args.labeled_num,args.total_num))

    odoc_batch_sampler = LabeledBatchSampler(odoc_idxs,labeled_bs)
    vessel_batch_sampler = UnlabeledBatchSampler(vessel_idxs, args.batch_size - args.labeled_bs)

    # init dataloader
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    odoc_trainloader = DataLoader(odoc_dataset,batch_sampler = odoc_batch_sampler, num_workers=args.num_works, pin_memory=True,worker_init_fn=worker_init_fn)
    vessel_trainloader = DataLoader(vessel_dataset,batch_sampler = vessel_batch_sampler, num_workers=args.num_works, pin_memory=True,worker_init_fn=worker_init_fn)



    model.train()
    # init optimizer
    # init optimizer
    optimizer = get_optimizer(model=model,name=args.optim,base_lr=args.base_lr,lr_decouple=args.lr_decouple)

    # scheduler = StepLR(optimizer,step_size=100,gamma=0.999)
    scheduler = PolyLRwithWarmup(optimizer,total_steps=args.max_iterations,warmup_steps=args.max_iterations * 0.01)
    # init summarywriter
    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_iterations // len(odoc_trainloader) + 1
    lr_ = args.base_lr
    model.train()

    ce_loss_vessel = BCEWithLogitsLoss()
    if args.ohem > 0:
        ce_loss_odoc = OhemCrossEntropy(thres=args.ohem,weight=torch.tensor([1.0,4.0,8.0],device=device))
    if args.ce_weight:
        ce_loss_odoc = CrossEntropyLoss(weight=torch.tensor([1.0,4.0,8.0],device=device))
    else:
        ce_loss_odoc = CrossEntropyLoss()
    # mse_loss = MSELoss()


    # 验证集
    # init dataset
    val_dataset = SemiDataset(name='./dataset/{}'.format(args.dataset_name),
                                    # root="D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/REFUGE",
                                  root="/home/gu721/yzc/data/odoc/{}".format(args.dataset_name),
                                  mode='val',
                                  size=args.image_size,CLAHE=args.CLAHE,preprocess=args.preprocess)

    val_labeledtrainloader = DataLoader(val_dataset,batch_size=1,num_workers=args.num_works)
    val_iteriter = tqdm(val_labeledtrainloader)


    # 开始训练
    iterator = tqdm(range(max_epoch), ncols=70)

    ODOC_val_metrics = ODOC_metrics(device)
    best_OD_DICE,best_OC_DICE = 0,0
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch,(odoc_sampled_batch,vessel_sampled_batch) in enumerate(zip(odoc_trainloader,vessel_trainloader)):
            time2 = time.time()

            odoc_labeled_batch, odoc_label_batch = odoc_sampled_batch['image'].to(device), odoc_sampled_batch['label'].to(device)
            vessel_labeled_batch, vessel_label_batch = vessel_sampled_batch['image'].to(device), vessel_sampled_batch['label'].to(device)
            if args.preprocess:
                odoc_labeled_batch_edges_info = odoc_sampled_batch['image_edges_info'].to(device)
                vessel_labeled_batch_edges_info = vessel_sampled_batch['image_edges_info'].to(device)




            all_batch = torch.cat([odoc_labeled_batch,vessel_labeled_batch],dim=0)

            outputs_odoc,outputs_vessel = model(all_batch)

            # calculate the loss of od and oc
            odoc_label_batch[odoc_label_batch > 2] = 0

            loss_seg_softfocal_odoc = torch.zeros(1,device=device)
            loss_seg_dice_odoc = torch.zeros(1,device=device)
            loss_seg_ce_odoc = ce_loss_odoc(outputs_odoc[:labeled_bs,...],odoc_label_batch)
            loss_seg_ce_vessel = ce_loss_vessel(outputs_vessel[labeled_bs:,0, ...], vessel_label_batch.float())

            if args.with_dice:
                outputs_odoc_soft = torch.argmax(outputs_odoc[:labeled_bs,...], dim=1)
                loss_seg_dice_odoc = losses.dice_loss(outputs_odoc_soft, odoc_label_batch)
            if not args.with_ce:
                loss_seg_ce_odoc = torch.zeros(1,device=device)
            if args.with_softfocal:
                loss_seg_softfocal_odoc = losses.softmax_focalloss(outputs_odoc[:labeled_bs,...],odoc_label_batch)

            if not args.no_vessel_weight_decay:
                vessel_loss_weight = args.vessel_loss_weight
            else:
                vessel_loss_weight = get_vessel_loss_weight(iter_num)
            loss = loss_seg_ce_odoc + \
                   loss_seg_dice_odoc + \
                   vessel_loss_weight * loss_seg_ce_vessel + \
                   loss_seg_softfocal_odoc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/vessel_loss_weight', vessel_loss_weight, iter_num)
            writer.add_scalar('loss/loss_seg_ce_odoc', loss_seg_ce_odoc, iter_num)
            writer.add_scalar('loss/loss_seg_softfocal_odoc', loss_seg_softfocal_odoc, iter_num)
            writer.add_scalar('loss/loss_seg_dice_odoc', loss_seg_dice_odoc, iter_num)
            writer.add_scalar('loss/loss_seg_ce_vessel', loss_seg_ce_vessel, iter_num)

            logging.info(
                'iteration %d : loss : %f' %
                (iter_num, loss.item()))
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            logging.info('iteration %d : loss_seg_ce_odoc : %f' % (iter_num, loss_seg_ce_odoc.item()))
            logging.info('iteration %d : loss_seg_softfocal_odoc : %f' % (iter_num, loss_seg_softfocal_odoc.item()))
            logging.info('iteration %d : loss_seg_dice_odoc : %f' % (iter_num, loss_seg_dice_odoc.item()))
            logging.info('iteration %d : loss_seg_ce_vessel : %f' % (iter_num, loss_seg_ce_vessel.item()))

            if iter_num % 50 == 0:
                with torch.no_grad():
                    image = all_batch[0]
                    writer.add_image('train/Image', image, iter_num)


                    image = torch.argmax(outputs_vessel,dim=1)
                    image = image[0]
                    writer.add_image('train/Predicted_label_vessel', image.unsqueeze(0), iter_num)

                    image = vessel_label_batch[0].unsqueeze(0)
                    image = image / 2
                    writer.add_image('train/Groundtruth_label_vessel',
                                     image, iter_num)

                    image = torch.argmax(outputs_odoc,dim=1)
                    image = image[0] / (args.num_classes - 1)
                    writer.add_image('train/Predicted_label', image.unsqueeze(0), iter_num)


                    image = odoc_label_batch[0].unsqueeze(0)
                    image = image / (args.num_classes - 1)
                    writer.add_image('train/Groundtruth_label',
                                     image, iter_num)


            # eval
            if iter_num % args.val_period == 0:
                with torch.no_grad():
                    model.eval()
                    show_id = random.randint(0,len(val_iteriter))
                    for id,data in enumerate(val_iteriter):
                        img,label = data['image'].to(device),data['label'].to(device)
                        outputs_odoc,_ = model(img)

                        ODOC_val_metrics.add_multi_class(outputs_odoc,label)

                        if id == show_id:
                            image = img[0]
                            writer.add_image('val/image', image, iter_num)
                            image = torch.argmax(outputs_odoc,dim=1)
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
