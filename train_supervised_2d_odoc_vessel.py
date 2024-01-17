import random
from tqdm import tqdm
from dataloader.fundus import ODOC_Vessel_Dataset,SemiDataset
import torch
import glob
import argparse
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss,MSELoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.vessel_prior_loss import KinkLoss
from tensorboardX import SummaryWriter
import segmentation_models_pytorch as smp
from utils.losses import OhemCrossEntropy,annealing_softmax_focalloss,softmax_focalloss,weight_softmax_focalloss
from utils.test_utils import ODOC_metrics
from utils.util import color_map,gray_to_color
import random
from utils.util import get_optimizer,PolyLRwithWarmup
from utils.boundary_utils import gt2boundary_tensor
from utils.bulid_model import build_model
import time
import os
import logging
import math
from copy import copy
import sys
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


parser = argparse.ArgumentParser()

# ==============model===================
parser.add_argument('--model',type=str,default='unet')
parser.add_argument('--backbone',type=str,default='b2')
parser.add_argument('--fpn_out_c',type=int,default=256,help='the out-channels of the FPN module')
parser.add_argument('--fpn_pretrained',action='store_true')
parser.add_argument('--sr_out_c',type=int,default=256,help='the out-channels of the SR module')
parser.add_argument('--sr_pretrained',action='store_true')
parser.add_argument('--decoder_attention_type',type=str,default=None,choices=['scse'])
parser.add_argument('--ckpt_weight',type=str,default=None)
parser.add_argument('--exclude_keys',type=str,default=None)
# ==============model===================

# ==============loss===================
parser.add_argument('--ce_weight', type=float, nargs='+', default=[1.0,1.0,1.0], help='List of floating-point values')
parser.add_argument('--ohem',type=float,default=-1.0)
parser.add_argument('--annealing_softmax_focalloss',action='store_true')
parser.add_argument('--softmax_focalloss',action='store_true')
parser.add_argument('--weight_softmax_focalloss',action='store_true')
parser.add_argument('--with_dice',action='store_true')
# ==============loss===================


# ==============aux params===================
parser.add_argument('--KinkLoss',type=float,default=-1)

parser.add_argument('--vessel_loss_weight',type=float,default=0.1)
parser.add_argument('--vessel_type',type=str,default='oc-rim5')

# ==============lr===================
parser.add_argument('--base_lr',type=float,default=0.00025)
parser.add_argument('--lr_decouple',action='store_true')
parser.add_argument('--warmup',type=float,default=0.01)
parser.add_argument('--scheduler',type=str,default='poly-v2')
# ==============lr===================

# ==============training params===================
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--num_works',type=int,default=0)
parser.add_argument('--num_classes',type=int,default=3)
parser.add_argument('--exp',type=str,default='RIM-ONE')
parser.add_argument('--save_period',type=int,default=5000)
parser.add_argument('--val_period',type=int,default=100)
parser.add_argument('--dataset_name',type=str,default='RIM-ONE')
parser.add_argument('--CLAHE',type=int,default=2)
parser.add_argument('--optim',type=str,default='AdamW')
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--image_size',type=int,default=256)
parser.add_argument('--max_iterations',type=int,default=10000)
parser.add_argument('--autodl',action='store_true')




def step_decay(current_epoch,total_epochs=60,base_lr=0.0001):
    initial_lrate = base_lr
    epochs_drop = total_epochs
    lrate = initial_lrate * math.pow(1-(1+current_epoch)/epochs_drop,0.9)
    return lrate




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
snapshot_path = "./exp_2d_odoc/" + args.exp + "/"
max_iterations = args.max_iterations
base_lr = args.base_lr


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
    model = build_model(args,model=args.model,backbone=args.backbone,in_chns=3,class_num1=args.num_classes,class_num2=2,fuse_type=None,ckpt_weight=args.ckpt_weight)
    model.to(device)

    # init dataset
    root_base = '/home/gu721/yzc/data/odoc/'
    if args.autodl:
        root_base = '/root/autodl-tmp/'

    labeled_dataset = ODOC_Vessel_Dataset(name='./dataset/{}'.format(args.dataset_name),
                                  root="{}{}".format(root_base,args.dataset_name),
                                  mode='semi_train',
                                  size=args.image_size,
                                  id_path='train.txt',
                                  pseudo_vessel=True
                                          )



    labeledtrainloader = DataLoader(labeled_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_works,
                                    pin_memory=True,
                                    shuffle=True
                                    )

    # 验证集
    # init dataset
    val_dataset = SemiDataset(name='./dataset/{}'.format(args.dataset_name),
                                  root="{}{}".format(root_base,args.dataset_name),
                                  mode='val',
                                  size=args.image_size)

    val_labeledtrainloader = DataLoader(val_dataset,batch_size=1,num_workers=1)
    val_iteriter = tqdm(val_labeledtrainloader)

    model.train()
    # init optimizer
    optimizer = get_optimizer(model=model,name=args.optim,base_lr=args.base_lr,lr_decouple=args.lr_decouple)


    # scheduler = StepLR(optimizer,step_size=100,gamma=0.999)
    scheduler = PolyLRwithWarmup(optimizer,total_steps=args.max_iterations,warmup_steps=args.max_iterations * args.warmup)

    # init summarywriter
    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_iterations // len(labeledtrainloader) + 1
    lr_ = args.base_lr
    model.train()

    vessel_bce_loss = BCEWithLogitsLoss()
    # class_weights = [0.001,1.0,0.1,0.01,0.1]
    class_weights = args.ce_weight
    if args.ohem > 0:
        # online hard example mining
        ce_loss = OhemCrossEntropy(thres=args.ohem,weight=torch.tensor(class_weights,device=device))
    else:
        ce_loss = CrossEntropyLoss(ignore_index=255,weight=torch.tensor(class_weights,device=device))

    dice_loss = smp.losses.DiceLoss(mode='multiclass',from_logits=True)
    kink_loss = KinkLoss()
    # mse_loss = MSELoss()




    print("=================共计训练epoch: {}====================".format(max_epoch))

    # 开始训练
    iterator = tqdm(range(max_epoch), ncols=70)
    ODOC_val_metrics = ODOC_metrics(device)
    best_OD_DICE,best_OC_DICE = 0,0
    for epoch_num in iterator:
        torch.cuda.empty_cache()
        time1 = time.time()
        for i_batch,labeled_sampled_batch in enumerate(labeledtrainloader):
            time2 = time.time()

            odoc_labeled_batch = labeled_sampled_batch['image'].to(device)
            odoc_label_batch = labeled_sampled_batch['odoc_label'].to(device)
            pseudo_vessel_label_batch = labeled_sampled_batch['vessel_mask'].to(device)

            if 'oc-rim' in args.vessel_type :
                # get the boundary of the oc
                oc_label_batch = torch.zeros_like(odoc_label_batch)
                oc_label_batch[odoc_label_batch == 2] = 1
                boundary_width = int(args.vessel_type.split('oc-rim')[-1])
                oc_boundary_label_batch = gt2boundary_tensor(oc_label_batch,boundary_width=boundary_width)
                # get the area of the pseudo vessel cover the oc-boundary
                vessel_mask = copy(oc_boundary_label_batch)
                vessel_mask[vessel_mask != pseudo_vessel_label_batch] = 0

            elif 'all-oc-rim' in args.vessel_type:
                # get the boundary of the oc
                oc_label_batch = torch.zeros_like(odoc_label_batch)
                oc_label_batch[odoc_label_batch == 2] = 1
                boundary_width = int(args.vessel_type.split('all-oc-rim')[-1])
                oc_boundary_label_batch = gt2boundary_tensor(oc_label_batch, boundary_width=boundary_width)
                # get the area of the pseudo vessel cover the oc-boundary
                vessel_mask = oc_boundary_label_batch

            elif 'minus-od-rim' in args.vessel_type :
                # get the boundary of the od
                od_label_batch = torch.zeros_like(odoc_label_batch)
                od_label_batch[odoc_label_batch == 1] = 1
                boundary_width = int(args.vessel_type.split('minus-od-rim')[-1])
                od_boundary_label_batch = gt2boundary_tensor(od_label_batch,boundary_width=boundary_width)
                # minus the od-rim
                od_label_batch[od_boundary_label_batch > 0] = 0
                # get the area of the pseudo vessel cover the od-boundary
                vessel_mask = copy(od_label_batch)
                vessel_mask[vessel_mask != pseudo_vessel_label_batch] = 0
            elif args.vessel_type == 'all-vessel':
                vessel_mask = pseudo_vessel_label_batch
            elif args.vessel_type == 'oc':
                pseudo_vessel_label_batch[odoc_label_batch !=2] = 0
                vessel_mask = pseudo_vessel_label_batch
            elif args.vessel_type == 'od':
                pseudo_vessel_label_batch[odoc_label_batch !=1] = 0
                vessel_mask = pseudo_vessel_label_batch

            all_batch = odoc_labeled_batch
            all_label_batch = odoc_label_batch

            loss_seg_ce = torch.zeros(1,device=device)
            loss_seg_dice = torch.zeros(1,device=device)
            loss_seg_vessel = torch.zeros(1,device=device)
            loss_kink = torch.zeros(1,device=device)

            features = 0
            if 'Dual' in args.model:
                if args.KinkLoss > 0:
                    features,features2 = model.forward_logits(all_batch)
                    odoc_outputs,vessel_outputs = model.forward_seg(features,features2)
                    one_hot_vessel_mask = torch.nn.functional.one_hot(vessel_mask.to(torch.int64), num_classes=2).permute(0,3,1,2).float()
                    loss_seg_vessel = vessel_bce_loss(vessel_outputs,one_hot_vessel_mask)
                else:
                    odoc_outputs,vessel_outputs = model(all_batch)
                    one_hot_vessel_mask = torch.nn.functional.one_hot(vessel_mask.to(torch.int64), num_classes=2).permute(0,3,1,2).float()
                    loss_seg_vessel = vessel_bce_loss(vessel_outputs,one_hot_vessel_mask)
            else:
                if args.KinkLoss > 0:
                    features = model.forward_logits(all_batch)
                    odoc_outputs = model.forward_seg(features)
                else:
                    odoc_outputs = model(all_batch)

            loss_seg_ce = ce_loss(odoc_outputs, all_label_batch)

            if args.KinkLoss > 0:
                loss_kink = args.KinkLoss * kink_loss(features, odoc_label_batch, vessel_mask)
            if args.vessel_loss_weight > 0:
                loss_seg_vessel =  args.vessel_loss_weight * loss_seg_vessel

            loss = loss_seg_ce + loss_kink + loss_seg_vessel

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.scheduler == 'poly':
                scheduler.step()
            elif args.scheduler == 'poly-v2':
                current_lr = step_decay(epoch_num,max_epoch,args.base_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            iter_num = iter_num + 1
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg_ce, iter_num)
            writer.add_scalar('loss/loss_kink', loss_kink, iter_num)
            writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)


            with torch.no_grad():
                if iter_num % 50 == 0:
                    image = all_batch[0]
                    writer.add_image('train/Image', image, iter_num)

                    image = torch.argmax(outputs,dim=1)
                    image = image[0]
                    colored_image = gray_to_color(image, color_map)
                    writer.add_image('train/Predicted_label', colored_image, iter_num,dataformats='CHW')


                    image = all_label_batch[0]
                    colored_image = gray_to_color(image, color_map)
                    writer.add_image('train/Groundtruth_label',
                                     colored_image, iter_num,dataformats='CHW')


            # eval
            with torch.no_grad():
                if iter_num % (54 / args.batch_size) == 0:
                    model.eval()
                    show_id = random.randint(0,len(val_iteriter))
                    for id,data in enumerate(val_iteriter):
                        img,label = data['image'].to(device),data['label'].to(device)
                        if 'Dual' in args.model:
                            outputs,_ = model(img)
                        else:
                            outputs = model(img)

                        ODOC_val_metrics.add_multi_class(outputs.detach(),label)

                        if id == show_id:
                            image = img[0]
                            writer.add_image('val/image', image, iter_num)

                            image = torch.argmax(outputs,dim=1)
                            image = gray_to_color(image,color_map)
                            writer.add_image('val/pred', image, iter_num,dataformats='CHW')
                            image = label
                            image = gray_to_color(image,color_map)
                            writer.add_image('val/Groundtruth_label',
                                             image, iter_num,dataformats='CHW')
                    torch.cuda.empty_cache()
                    val_metrics = ODOC_val_metrics.get_metrics()
                    OD_DICE, OD_IOU, OC_DICE, OC_IOU = val_metrics['od_dice'], val_metrics['od_iou'], val_metrics['oc_dice'], \
                                                       val_metrics['oc_iou']
                    OD_rim_DICE, OD_rim_IOU = val_metrics['od_rim_dice'],val_metrics['od_rim_iou']
                    OD_BIOU, OC_BIOU = val_metrics['od_biou'], val_metrics['oc_biou']
                    writer.add_scalar('val/OD_Dice', OD_DICE, iter_num)
                    writer.add_scalar('val/OD_IOU', OD_IOU, iter_num)
                    writer.add_scalar('val/OD_BIOU', OD_BIOU, iter_num)

                    writer.add_scalar('val/OD_rim_DICE', OD_rim_DICE, iter_num)
                    writer.add_scalar('val/OD_rim_IOU', OD_rim_IOU, iter_num)

                    writer.add_scalar('val/OC_Dice', OC_DICE, iter_num)
                    writer.add_scalar('val/OC_IOU', OC_IOU, iter_num)
                    writer.add_scalar('val/OC_BIOU', OC_BIOU, iter_num)
                    model.train()

                    if OD_DICE > best_OD_DICE:
                        best_OD_DICE = OD_DICE
                        name = "OD_DICE" + str(round(best_OD_DICE.item(), 4)) + '_iter_' + str(iter_num) + '.pth'
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
                        name = "OC_DICE" + str(round(best_OC_DICE.item(), 4)) + '_iter_' + str(iter_num) + '.pth'
                        save_mode_path = os.path.join(
                            snapshot_path, name)
                        torch.save(model.state_dict(), save_mode_path)
                        logging.info("save model to {}".format(save_mode_path))

                if iter_num % args.save_period == 0:
                    save_mode_path = os.path.join(
                        snapshot_path, 'iter_' + str(iter_num) + '.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    # logging.info("save model to {}".format(save_mode_path))

                if iter_num >= max_iterations:
                    break
                time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
