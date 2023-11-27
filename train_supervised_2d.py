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
from torch.utils.tensorboard import SummaryWriter
from model.mcnet.unet import UNet, MCNet2d_compete_v1,UNet_DTC2d
from utils import ramps,losses
from utils.test_utils import ODOC_metrics
import random
from utils.util import PolyLRwithWarmup, compute_sdf,compute_sdf_luoxd,compute_sdf_multi_class
import time
import logging
import os
import shutil
import logging
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '5'

parser = argparse.ArgumentParser()
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--num_works',type=int,default=4)
parser.add_argument('--backbone',type=str,default='resnet34')

parser.add_argument('--exp',type=str,default='supervised/SEG')

parser.add_argument('--dataset_name',type=str,default='SEG')
parser.add_argument('--unlabeled_txt',type=str,default='unlabeled_addDDR.txt')

parser.add_argument('--amp',type=bool,default=True)
parser.add_argument('--num_classes',type=int,default=3)
parser.add_argument('--base_lr',type=float,default=0.005)

parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--labeled_bs',type=int,default=16)

parser.add_argument('--od_rim',type=bool,default=True)
parser.add_argument('--oc_label',type=int,default=2)
parser.add_argument('--image_size',type=int,default=256)

parser.add_argument('--labeled_num',type=int,default=100,help="5%:100,")
parser.add_argument('--total_num',type=int,default=11249,help="SEG:2859;SEG_add_DDR:11249")


parser.add_argument('--scale_num',type=int,default=2)
parser.add_argument('--max_iterations',type=int,default=10000)

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



def get_unsup_cont_weight(epoch, weight, scheme, ramp_up_or_down ):
    if  scheme == 'sigmoid_rampup':
        return weight * ramps.sigmoid_rampup(epoch, ramp_up_or_down)
    elif scheme == 'linear_rampup':
        return weight * ramps.linear_rampup(epoch, ramp_up_or_down)
    elif scheme == 'log_rampup':
        return weight * ramps.log_rampup(epoch, ramp_up_or_down)
    elif scheme == 'exp_rampup':
        return weight * ramps.exp_rampup(epoch, ramp_up_or_down)
    elif scheme == 'quadratic_rampdown':
        return weight * ramps.quadratic_rampdown(epoch, ramp_up_or_down)
    elif scheme == 'cosine_rampdown':
        return weight * ramps.cosine_rampdown(epoch, ramp_up_or_down)
    else:
        return weight


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def get_supervised_loss(outputs, label_batch,  with_dice=True):
    loss_seg = F.cross_entropy(outputs, label_batch)
    outputs_soft = F.softmax(outputs, dim=1)
    if with_dice:
        loss_seg_dice = losses.dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
        supervised_loss = 0.5 * (loss_seg + loss_seg_dice)
    else:
        loss_seg_dice = torch.zeros([1]).to(device)
        supervised_loss = loss_seg + loss_seg_dice
    return supervised_loss, loss_seg, loss_seg_dice

def pseudo_labeling_from_most_confident_prediction(pred_1, pred_2, max_1, max_2):
        # stack很有意思，因为是堆叠，所以会产生一个新的维度，dim就是指定这个维度应该放在哪里
        prob_all_ex_3 = torch.stack([pred_1, pred_2], dim=2)  # bs, n_c, n_branch - 1, h, w
        max_all_ex_3 = torch.stack([max_1, max_2], dim=1)  # bs, n_branch - 1, h, w
        # 获取堆叠后的最大分数,注意，这里返回后的维度，现在分数不再是softmax的，而是来自不同branch的最大分数
        max_conf_each_branch_ex_3, _ = torch.max(prob_all_ex_3, dim=1)  # bs, n_branch - 1, h, w
        # 再计算伪标签的值和索引，branch_id_max_conf_ex_3代表的是第几个branch
        max_conf_ex_3, branch_id_max_conf_ex_3 = torch.max(max_conf_each_branch_ex_3, dim=1,
                                                              keepdim=True)  # bs, h, w
        # branch_id_max_conf_ex_3是索引，因此可以通过索引从堆叠的标签中获取最终通过竞争机制得到的伪标签
        pseudo_12 = torch.gather(max_all_ex_3, dim=1, index=branch_id_max_conf_ex_3)[:, 0]
        # 返回的是一个布尔值，因此从max_conf_ex_3的伪标签
        max_conf_fg_ex_3 = max_conf_ex_3[:, 0][pseudo_12 == 1]
        try:
            mean_max_conf_fg_ex_3, var_max_conf_fg_ex_3, min_max_conf_fg_ex_3, max_max_conf_fg_ex_3 = torch.mean(
                max_conf_fg_ex_3).detach().cpu(), torch.var(max_conf_fg_ex_3).detach().cpu(), torch.min(
                max_conf_fg_ex_3).detach().cpu(), torch.max(max_conf_fg_ex_3).detach().cpu()
        except:
            mean_max_conf_fg_ex_3, var_max_conf_fg_ex_3, min_max_conf_fg_ex_3, max_max_conf_fg_ex_3 = 0, 0, 0, 0
        return pseudo_12, branch_id_max_conf_ex_3, [mean_max_conf_fg_ex_3, var_max_conf_fg_ex_3, min_max_conf_fg_ex_3, max_max_conf_fg_ex_3]


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
snapshot_path = "./exp_dtc_2d/" + args.exp + "/"
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
    model = UNet(in_chns=3,class_num=args.num_classes)
    model.to(device)

    # init dataset
    labeled_dataset = SemiDataset(name='./dataset/{}'.format(args.dataset_name),
                                  root="/home/gu721/yzc/data/odoc",
                                  mode='semi_train',
                                  size=args.image_size,
                                  id_path='train.txt')


    labeledtrainloader = DataLoader(labeled_dataset,batch_size=args.batch_size, num_workers=args.num_works, pin_memory=True,)

    model.train()
    # init optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    # scheduler = StepLR(optimizer,step_size=100,gamma=0.999)
    scheduler = PolyLRwithWarmup(optimizer,total_steps=args.max_iterations,warmup_steps=args.max_iterations * 0.01)
    # init summarywriter
    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_iterations // len(labeledtrainloader) + 1
    lr_ = args.base_lr
    model.train()

    # ce_loss = BCEWithLogitsLoss()
    ce_loss = CrossEntropyLoss()
    # mse_loss = MSELoss()


    # 验证集
    # init dataset
    val_dataset = SemiDataset(name='./dataset/{}'.format(args.dataset_name),
                                    # root="D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/REFUGE",
                                  root="/home/gu721/yzc/data/odoc",
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

            outputs = model(all_batch)


            # calculate the loss
            # 这里需要注意，如果是分割三个类别以上，则需要分开计算dist和分开计算mse

            loss = ce_loss(outputs,all_label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)

            logging.info(
                'iteration %d : loss : %f' %
                (iter_num, loss.item()))
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

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
            if iter_num % 100 == 0:
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
            if iter_num % 1000 == 0:
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
