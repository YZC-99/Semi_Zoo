import random
from tqdm import tqdm
from dataloader.fundus import SemiDataset,IDRIDDataset,BinaryIDRIDDataset
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
import segmentation_models_pytorch as smp
from utils import ramps,losses
from utils.losses import OhemCrossEntropy,annealing_softmax_focalloss,softmax_focalloss,weight_softmax_focalloss
from utils.test_utils import DR_metrics,Binary_DR_metrics
from utils.util import color_map,gray_to_color
import random
from utils.util import get_optimizer,PolyLRwithWarmup, compute_sdf,compute_sdf_luoxd,compute_sdf_multi_class
import time
import logging
import os
import shutil
import logging
import sys
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



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
parser.add_argument('--CLAHE',type=int,default=2)

parser.add_argument('--optim',type=str,default='AdamW')
parser.add_argument('--amp',type=bool,default=True)
parser.add_argument('--num_classes',type=int,default=5)
parser.add_argument('--base_lr',type=float,default=0.00025)

parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--labeled_bs',type=int,default=16)

parser.add_argument('--od_rim',type=bool,default=True)
parser.add_argument('--oc_label',type=int,default=2)
parser.add_argument('--image_size',type=int,default=512)


parser.add_argument('--scale_num',type=int,default=2)
parser.add_argument('--max_iterations',type=int,default=10000)

parser.add_argument('--ce_weight', type=float, nargs='+', default=[0.001,1.0,0.1,0.1,0.1], help='List of floating-point values')
parser.add_argument('--ohem',type=float,default=-1.0)
parser.add_argument('--annealing_softmax_focalloss',action='store_true')
parser.add_argument('--softmax_focalloss',action='store_true')
parser.add_argument('--weight_softmax_focalloss',action='store_true')
parser.add_argument('--with_dice',type=bool,default=False)

parser.add_argument('--autodl',action='store_true')





def build_model(model,backbone,in_chns,class_num1,class_num2,fuse_type):
    if model == "UNet":
        net =  smp.Unet(
            encoder_name = backbone,
            encoder_weights = 'imagenet',
            in_channels = in_chns,
            classes= class_num1
        )
    elif model == 'DeepLabV3p':
        net =  smp.DeepLabV3Plus(
            encoder_name = backbone,
            encoder_weights = 'imagenet',
            in_channels = in_chns,
            classes= class_num1
        )
    return net


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
    root_base = '/home/gu721/yzc/data/dr/'
    if args.autodl:
        root_base = '/root/autodl-tmp/'

    labeled_dataset = BinaryIDRIDDataset(name='./dataset/{}'.format(args.dataset_name),
                                  root="{}{}".format(root_base,args.dataset_name),
                                  mode='semi_train',
                                  size=args.image_size,
                                  id_path='train.txt',
                                  CLAHE=args.CLAHE)


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

    # MA_ce_loss = BCEWithLogitsLoss(pos_weight=torch.tensor([2274],device=device))
    # HE_ce_loss = BCEWithLogitsLoss(pos_weight=torch.tensor([684],device=device))
    # EX_ce_loss = BCEWithLogitsLoss(pos_weight=torch.tensor([789],device=device))
    # SE_ce_loss = BCEWithLogitsLoss(pos_weight=torch.tensor([607],device=device))

    MA_ce_loss = BCEWithLogitsLoss(pos_weight=torch.tensor([100],device=device))
    HE_ce_loss = BCEWithLogitsLoss(pos_weight=torch.tensor([100],device=device))
    EX_ce_loss = BCEWithLogitsLoss(pos_weight=torch.tensor([100],device=device))
    SE_ce_loss = BCEWithLogitsLoss(pos_weight=torch.tensor([100],device=device))
    Background_ce_loss = BCEWithLogitsLoss()



    # 验证集
    # init dataset
    val_dataset = BinaryIDRIDDataset(name='./dataset/{}'.format(args.dataset_name),
                                    # root="D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/REFUGE",
                                  root="{}{}".format(root_base,args.dataset_name),
                                  mode='val',
                                  size=args.image_size,
                                  CLAHE=args.CLAHE)

    val_labeledtrainloader = DataLoader(val_dataset,batch_size=1,num_workers=args.num_works)
    val_iteriter = tqdm(val_labeledtrainloader)


    # 开始训练
    iterator = tqdm(range(max_epoch), ncols=70)

    DR_val_metrics = Binary_DR_metrics(device)
    best_AUC_PR_EX = 0
    for epoch_num in iterator:
        torch.cuda.empty_cache()
        time1 = time.time()
        for i_batch,labeled_sampled_batch in enumerate(labeledtrainloader):
            time2 = time.time()

            target_part_name = 'segmentation_head'
            for name, param in model.named_parameters():
                if target_part_name in name and param.grad is not None:
                    # print(param.grad[0].norm())
                    writer.add_scalar(f'gradients/{name}', param.grad.norm(), global_step=iter_num)

                    for class_index in range(args.num_classes):
                        class_gradient = param.grad[class_index]
                        writer.add_scalar(f'gradients/{target_part_name}_class_{class_index}', class_gradient.norm(),
                                          global_step=iter_num)

            # target_part_name = 'segmentation_head'
            # for name, param in model.named_parameters():
            #     if target_part_name in name and param.grad is not None:
            #         # Assuming the gradient tensor has shape (batch_size, num_classes, width, height)
            #         for class_index in range(args.num_classes):
            #             class_gradient = param.grad[:, class_index, :, :]
            #             writer.add_scalar(f'gradients/{target_part_name}_class_{class_index}', class_gradient.norm(),
            #                               global_step=iter_num)

            labeled_batch = labeled_sampled_batch['image'].to(device)

            MA_label_batch = labeled_sampled_batch['MA_mask'].to(device)
            HE_label_batch = labeled_sampled_batch['HE_mask'].to(device)
            EX_label_batch = labeled_sampled_batch['EX_mask'].to(device)
            SE_label_batch = labeled_sampled_batch['SE_mask'].to(device)

            outputs = model(labeled_batch)
            # outputs_prob = torch.sigmoid(outputs)
            loss_seg_ce = 0
            # MA_loss = 0
            # HE_loss = 0
            # EX_loss = 0
            # SE_loss = 0

            background_label_batch = torch.ones_like(MA_label_batch,device=device)

            if len(torch.unique(MA_label_batch)) > 1:
                background_label_batch[MA_label_batch > 0] = 0
                MA_loss = MA_ce_loss(outputs[:,1,...],MA_label_batch.float())
                loss_seg_ce += MA_loss
            if len(torch.unique(HE_label_batch)) > 1:
                background_label_batch[HE_label_batch > 0] = 0
                HE_loss = HE_ce_loss(outputs[:,2,...],HE_label_batch.float())
                loss_seg_ce += HE_loss
            if len(torch.unique(EX_label_batch)) > 1:
                background_label_batch[EX_label_batch > 0] = 0
                EX_loss = EX_ce_loss(outputs[:,3,...],EX_label_batch.float())
                loss_seg_ce += EX_loss
            if len(torch.unique(SE_label_batch)) > 1:
                background_label_batch[SE_label_batch > 0] = 0
                SE_loss = SE_ce_loss(outputs[:,4,...],SE_label_batch.float())
                loss_seg_ce += SE_loss

            background_loss = Background_ce_loss(outputs[:,0,...],background_label_batch.float())
            loss_seg_ce += background_loss
            # loss_seg_ce = args.ce_weight[0] * background_loss + \
            #               args.ce_weight[1] * MA_loss + \
            #               args.ce_weight[2] * HE_loss + \
            #               args.ce_weight[3] * EX_loss +\
            #               args.ce_weight[4] * SE_loss

            # loss_seg_ce = background_loss + \
            #               MA_loss + \
            #               HE_loss + \
            #               EX_loss + \
            #               SE_loss
            # loss_seg_ce = 0.25 * MA_loss + \
            #               0.25 * HE_loss + \
            #               0.25 * EX_loss + \
            #               0.25 * SE_loss
            # loss_seg_ce = MA_loss + \
            #               HE_loss + \
            #               EX_loss + \
            #               SE_loss

            loss = loss_seg_ce

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

            with torch.no_grad():
                if iter_num % 50 == 0:
                    image = labeled_batch[0]
                    writer.add_image('train/Image', image, iter_num)

                    outputs = torch.sigmoid(outputs[0])

                    MA_img = (outputs[1,...] > 0.5).to(torch.int8)
                    HE_img = (outputs[2,...] > 0.5).to(torch.int8) * 2
                    EX_img = (outputs[3,...] > 0.5).to(torch.int8) * 3
                    SE_img = (outputs[4,...] > 0.5).to(torch.int8) * 4


                    MA_colored_image = gray_to_color(MA_img, color_map)
                    writer.add_image('train/MA_Predicted_label', MA_colored_image, iter_num,dataformats='CHW')
                    HE_colored_image = gray_to_color(HE_img, color_map)
                    writer.add_image('train/HE_Predicted_label', HE_colored_image, iter_num,dataformats='CHW')
                    EX_colored_image = gray_to_color(EX_img, color_map)
                    writer.add_image('train/EX_Predicted_label', EX_colored_image, iter_num,dataformats='CHW')
                    SE_colored_image = gray_to_color(SE_img, color_map)
                    writer.add_image('train/SE_Predicted_label', SE_colored_image, iter_num,dataformats='CHW')


                    image = MA_label_batch[0]+ 2* HE_label_batch[0]+ 3 * EX_label_batch[0]+ 4 *SE_label_batch[0]
                    colored_image = gray_to_color(image, color_map)
                    writer.add_image('train/Groundtruth_label',
                                     colored_image, iter_num,dataformats='CHW')


            # eval
            with torch.no_grad():
                if iter_num % args.val_period == 0:
                    model.eval()
                    show_id = random.randint(0,len(val_iteriter))
                    for id,data in enumerate(val_iteriter):
                        img = data['image'].to(device)
                        MA_mask = data['MA_mask'].to(device)
                        HE_mask = data['HE_mask'].to(device)
                        EX_mask = data['EX_mask'].to(device)
                        SE_mask = data['SE_mask'].to(device)

                        label = MA_mask + 2 * HE_mask + 3 * EX_mask + 4 * SE_mask

                        # one_hot_label = torch.nn.functional.one_hot(torch.cat([MA_mask,HE_mask,EX_mask,SE_mask],dim=1)).permute(3,0,1,2)

                        outputs = model(img)
                        outputs = torch.sigmoid(outputs)
                        DR_val_metrics.add(outputs.detach(),MA_mask , HE_mask , EX_mask , SE_mask)

                        if id == show_id:
                            image = img[0]
                            writer.add_image('val/image', image, iter_num)

                            outputs = outputs[0]
                            MA_img = (outputs[1, ...] > 0.5).to(torch.int8)
                            HE_img = (outputs[2, ...] > 0.5).to(torch.int8) * 2
                            EX_img = (outputs[3, ...] > 0.5).to(torch.int8) * 3
                            SE_img = (outputs[4, ...] > 0.5).to(torch.int8) * 4

                            MA_colored_image = gray_to_color(MA_img, color_map)
                            writer.add_image('val/MA_Predicted_label', MA_colored_image, iter_num, dataformats='CHW')
                            HE_colored_image = gray_to_color(HE_img, color_map)
                            writer.add_image('val/HE_Predicted_label', HE_colored_image, iter_num, dataformats='CHW')
                            EX_colored_image = gray_to_color(EX_img, color_map)
                            writer.add_image('val/EX_Predicted_label', EX_colored_image, iter_num, dataformats='CHW')
                            SE_colored_image = gray_to_color(SE_img, color_map)
                            writer.add_image('val/SE_Predicted_label', SE_colored_image, iter_num, dataformats='CHW')

                            image = label
                            image = gray_to_color(image,color_map)
                            writer.add_image('val/Groundtruth_label',
                                             image, iter_num,dataformats='CHW')
                    torch.cuda.empty_cache()
                    val_metrics = DR_val_metrics.get_metrics()

                    AUC_PR = val_metrics[0]
                    AUC_ROC = val_metrics[1]
                    Dice,IoU = val_metrics[2]['Dice'],val_metrics[2]['IoU']
                    MA_AUC_PR, HE_AUC_PR, EX_AUC_PR, SE_AUC_PR = AUC_PR['MA_AUC_PR'],AUC_PR['HE_AUC_PR'],AUC_PR['EX_AUC_PR'],AUC_PR['SE_AUC_PR']
                    MA_AUC_ROC, HE_AUC_ROC, EX_AUC_ROC, SE_AUC_ROC = AUC_ROC['MA_AUC_ROC'],AUC_ROC['HE_AUC_ROC'],AUC_ROC['EX_AUC_ROC'],AUC_ROC['SE_AUC_ROC']


                    logging.info("MA_AUC_PR:{}--HE_AUC_PR:{}--EX_AUC_PR:{}--SE_AUC_PR:{}".format(
                                                                                            MA_AUC_PR,
                                                                                            HE_AUC_PR,
                                                                                            EX_AUC_PR,
                                                                                            SE_AUC_PR,
                                                                                           ))
                    logging.info("MA_AUC_ROC:{}--HE_AUC_ROC:{}--EX_AUC_ROC:{}--SE_AUC_ROC:{}".format(
                                                                                            MA_AUC_ROC,
                                                                                            HE_AUC_ROC,
                                                                                            EX_AUC_ROC,
                                                                                            SE_AUC_ROC,
                                                                                           ))

                    writer.add_scalar('val_AUC_PR/MA',MA_AUC_PR, iter_num)
                    writer.add_scalar('val_AUC_PR/HE',HE_AUC_PR, iter_num)
                    writer.add_scalar('val_AUC_PR/EX',EX_AUC_PR, iter_num)
                    writer.add_scalar('val_AUC_PR/SE',SE_AUC_PR, iter_num)

                    writer.add_scalar('val_AUC_ROC/MA',MA_AUC_ROC, iter_num)
                    writer.add_scalar('val_AUC_ROC/HE',HE_AUC_ROC, iter_num)
                    writer.add_scalar('val_AUC_ROC/EX',EX_AUC_ROC, iter_num)
                    writer.add_scalar('val_AUC_ROC/SE',SE_AUC_ROC, iter_num)

                    writer.add_scalar('val_Region/Dice',Dice, iter_num)
                    writer.add_scalar('val_Region/IoU',IoU, iter_num)

                    model.train()

                    if EX_AUC_PR > best_AUC_PR_EX:
                        best_AUC_PR_EX = EX_AUC_PR
                        name = "best_AUC_PR_EX" + str(round(best_AUC_PR_EX.item(), 4)) +'_iter_' + str(iter_num)  + '.pth'
                        save_mode_path = os.path.join(
                            snapshot_path, name)

                        previous_files = glob.glob(os.path.join(snapshot_path, '*best_AUC_PR_EX*.pth'))
                        for file_path in previous_files:
                            os.remove(file_path)

                        torch.save(model.state_dict(), save_mode_path)
                        logging.info("save model to {}".format(save_mode_path))


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
