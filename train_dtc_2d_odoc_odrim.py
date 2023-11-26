import random
from tqdm import tqdm
from dataloader.fundus import SemiDataset
from dataloader.samplers import LabeledBatchSampler,UnlabeledBatchSampler
import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss,MSELoss
from torch.cuda.amp import autocast,GradScaler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model.mcnet.unet import MCNet2d_compete_v1,UNet_DTC2d
from utils import ramps,losses
from utils.test_utils import ODOC_metrics
import random
from utils.util import compute_sdf,compute_sdf_luoxd,compute_sdf_multi_class
import time
import logging
import os
import shutil
import logging
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--seed',type=int,default=42)

parser.add_argument('--backbone',type=str,default='resnet34')

parser.add_argument('--amp',type=bool,default=True)
parser.add_argument('--num_classes',type=int,default=3)
parser.add_argument('--base_lr',type=float,default=0.001)

parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--labeled_bs',type=int,default=2)

parser.add_argument('--image_size',type=int,default=256)

parser.add_argument('--labeled_num',type=int,default=100)
parser.add_argument('--total_num',type=int,default=360)
parser.add_argument('--scale_num',type=int,default=2)
parser.add_argument('--max_iterations',type=int,default=60000)

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

parser.add_argument('--exp',type=str,default='refuge400_odrim')


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
    # outputs_soft = F.softmax(outputs, dim=1)
    # if with_dice:
    #     loss_seg_dice = losses.dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
    #     supervised_loss = 0.5 * (loss_seg + loss_seg_dice)
    # else:
    loss_seg_dice = torch.zeros([1]).cuda()
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

    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # init model
    scale_num = 2
    model = UNet_DTC2d(in_chns=3,class_num=args.num_classes,outchannel_minus1 = False)
    model.cuda()

    # init dataset
    labeled_dataset = SemiDataset(name='./dataset/semi_refuge400',
                                  # root="D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/REFUGE",
                                  root="/home/gu721/yzc/data/odoc/REFUGE/",
                                  mode='semi_train',
                                  size=args.image_size,
                                  id_path='labeled.txt')

    unlabeled_dataset = SemiDataset(name='./dataset/semi_refuge400',
                                    # root="D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/REFUGE",
                                    root="/home/gu721/yzc/data/odoc/REFUGE/",
                                    mode='semi_train',
                                    size=args.image_size,
                                    id_path='unlabeled.txt')

    labeled_idxs = list(range(args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num,args.total_num))
    labeled_batch_sampler = LabeledBatchSampler(labeled_idxs,labeled_bs)
    unlabeled_batch_sampler = UnlabeledBatchSampler(unlabeled_idxs, args.batch_size - args.labeled_bs)

    # init dataloader
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    labeledtrainloader = DataLoader(labeled_dataset, batch_sampler=labeled_batch_sampler, num_workers=0, pin_memory=True,
                                    worker_init_fn=worker_init_fn)
    unlabeledtrainloader = DataLoader(unlabeled_dataset, batch_sampler=unlabeled_batch_sampler, num_workers=0,
                                      pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    # init optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    # init summarywriter
    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_iterations // len(labeledtrainloader) + 1
    lr_ = args.base_lr
    model.train()

    # ce_loss = BCEWithLogitsLoss()
    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()


    # 验证集
    # init dataset
    val_dataset = SemiDataset(name='./dataset/semi_refuge400',
                                    # root="D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/REFUGE",
                                  root="/home/gu721/yzc/data/odoc/REFUGE/",
                                  mode='val',
                                  size=args.image_size)
    val_labeledtrainloader = DataLoader(val_dataset,batch_size=1)
    val_iteriter = tqdm(val_labeledtrainloader)


    # 开始训练
    iterator = tqdm(range(max_epoch), ncols=70)

    ODOC_val_metrics = ODOC_metrics('cuda')

    for epoch_num in iterator:
        time1 = time.time()
        for i_batch,(labeled_sampled_batch, unlabeled_sampled_batch) in enumerate(zip(labeledtrainloader,unlabeledtrainloader)):
            time2 = time.time()

            unlabeled_batch, unlabel_label_batch = unlabeled_sampled_batch['image'].cuda(), unlabeled_sampled_batch['label'].cuda()
            labeled_batch, label_label_batch = labeled_sampled_batch['image'].cuda(), labeled_sampled_batch['label'].cuda()

            all_batch = torch.cat([labeled_batch,unlabeled_batch],dim=0)
            all_label_batch = torch.cat([label_label_batch,unlabel_label_batch],dim=0)

            out_dict = model(all_batch)
            outputs_tanh, outputs = out_dict["output_tanh_1"],out_dict["output1"]

            outputs_tanh_od = outputs_tanh[:,0,...].unsqueeze(1)
            outputs_tanh_oc = outputs_tanh[:,1,...].unsqueeze(1)

            outputs_od = outputs[:,0,...].unsqueeze(1)
            outputs_oc = outputs[:,1,...].unsqueeze(1)

            outputs_soft = torch.sigmoid(outputs)

            outputs_soft_od = outputs_soft[:,0,...].unsqueeze(1)
            outputs_soft_oc = outputs_soft[:,1,...].unsqueeze(1)

            # calculate the loss
            # 这里需要注意，如果是分割三个类别以上，则需要分开计算dist和分开计算mse

            # 先转OD的
            with torch.no_grad():
                od_all_label_batch = torch.zeros_like(all_label_batch)
                oc_all_label_batch = torch.zeros_like(all_label_batch)

                # 这里的od选择可能回影响后续的任务
                # od_all_label_batch[all_label_batch > 0] = 1
                od_all_label_batch[all_label_batch == 1] = 1
                oc_all_label_batch[all_label_batch > 1] = 2


                #luoxd的方法
                od_gt_dis = compute_sdf_luoxd(od_all_label_batch[:].cpu().numpy(),outputs[:labeled_bs,0,...].shape)
                od_gt_dis = torch.from_numpy(od_gt_dis).float().unsqueeze(1).cuda()
                oc_gt_dis = compute_sdf_luoxd(oc_all_label_batch[:].cpu().numpy(),outputs[:labeled_bs,0,...].shape)
                oc_gt_dis = torch.from_numpy(oc_gt_dis).float().unsqueeze(1).cuda()


            # 计算gt_dis的mse损失对应原论文公式(6)
            loss_sdf = mse_loss(outputs_tanh_od[:labeled_bs, ...], od_gt_dis) + \
                       mse_loss(outputs_tanh_oc[:labeled_bs, ...], oc_gt_dis)

            loss_seg = ce_loss(outputs_od[:labeled_bs,0, ...], od_all_label_batch[:labeled_bs].float()) + \
                       ce_loss(outputs_oc[:labeled_bs,0, ...], oc_all_label_batch[:labeled_bs].float())

            loss_seg_dice = losses.dice_loss(outputs_soft_od[:labeled_bs,...], od_all_label_batch[:labeled_bs].unsqueeze(1)) + \
                            losses.dice_loss(outputs_soft_oc[:labeled_bs,...], oc_all_label_batch[:labeled_bs].unsqueeze(1))

            # 统一将水平集函数转化为mask
            dis_to_mask_od = torch.sigmoid(-1500*outputs_tanh_od)
            dis_to_mask_oc= torch.sigmoid(-1500*outputs_tanh_oc)
            # 原论文公式(4)
            # 计算包括了标记与未标记的部分
            # 这么做的目的是强制两个推理的一致性，一个专注像素级推理，另一个专注几何结构
            # dis_to_mask：(B,num_lcasses - 1,h,w)
            # outputs_soft：(B,num_lcasses ,h,w)
            #所以需要分开算
            consistency_loss = torch.mean((dis_to_mask_od - outputs_soft_od) ** 2) + \
                               torch.mean((dis_to_mask_oc - outputs_soft_oc) ** 2)
            supervised_loss = loss_seg_dice + args.beta * loss_sdf
            consistency_weight = get_current_consistency_weight(iter_num//150)

            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss_hausdorff', loss_sdf, iter_num)
            writer.add_scalar('loss/consistency_weight',
                              consistency_weight, iter_num)
            writer.add_scalar('loss/consistency_loss',
                              consistency_loss, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_consis: %f, loss_haus: %f, loss_seg: %f, loss_dice: %f' %
                (iter_num, loss.item(), consistency_loss.item(), loss_sdf.item(),
                 loss_seg.item(), loss_seg_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            if iter_num % 50 == 0:
                image = all_batch[0]
                writer.add_image('train/Image', image, iter_num)

                image_od = (outputs_od > 0.5).to(torch.int8)
                image_oc = (outputs_oc > 0.5).to(torch.int8)
                image = image_od[0] + image_oc[0] * 2
                image = image / (args.num_classes - 1)
                writer.add_image('train/Predicted_label', image, iter_num)


                image = dis_to_mask_od[0] + dis_to_mask_oc[0] * 2
                image  = image / (args.num_classes - 1)
                writer.add_image('train/Dis2Mask', image, iter_num)

                image = outputs_tanh[0,0,...]
                writer.add_image('train/OD_DistMap', image.unsqueeze(0), iter_num)

                image = outputs_tanh[0,1,...]
                writer.add_image('train/OC_DistMap', image.unsqueeze(0), iter_num)

                image = all_label_batch[0].unsqueeze(0)
                image = image / (args.num_classes - 1)
                writer.add_image('train/Groundtruth_label',
                                 image, iter_num)

                image = od_gt_dis[0]
                writer.add_image('train/Groundtruth_OD_DistMap',
                                 image, iter_num)

                image = oc_gt_dis[0]
                writer.add_image('train/Groundtruth_OC_DistMap',
                                 image, iter_num)


            # eval
            if iter_num % 100 == 0:
                model.eval()
                show_id = random.randint(0,len(val_iteriter))
                for id,data in enumerate(val_iteriter):
                    img,label = data['image'].cuda(),data['label'].cuda()
                    out_dict = model(img)
                    outputs_tanh, outputs = out_dict["output_tanh_1"], out_dict["output1"]
                    outputs_od,outputs_oc = outputs[:,0,...].unsqueeze(1),outputs[:,1,...].unsqueeze(1)
                    ODOC_val_metrics.add(outputs,label)

                    if id == show_id:
                        image = img[0]
                        writer.add_image('val/image', image, iter_num)
                        image_od = (outputs_od > 0.5).to(torch.int8)
                        image_oc = (outputs_oc > 0.5).to(torch.int8)
                        image = image_od[0] + image_oc[0] * 2
                        image = image / (args.num_classes - 1)
                        writer.add_image('val/pred', image, iter_num)
                        image = label
                        image = image / (args.num_classes - 1)
                        writer.add_image('val/Groundtruth_label',
                                         image, iter_num)

                Dice_IoU = ODOC_val_metrics.get_metrics()
                logging.info("OD_Dice:{}--OD_IoU:--{}--OC_Dice:{}--OC_IoU:--{}".format(
                                                                                        Dice_IoU['od_dice'],
                                                                                        Dice_IoU['od_iou'],
                                                                                        Dice_IoU['oc_dice'],
                                                                                        Dice_IoU['oc_iou'],
                                                                                       ))
                writer.add_scalar('val/OD_Dice',Dice_IoU['od_dice'], iter_num)
                writer.add_scalar('val/OD_IOU',Dice_IoU['od_iou'], iter_num)
                writer.add_scalar('val/OC_Dice',Dice_IoU['oc_dice'], iter_num)
                writer.add_scalar('val/OC_IOU',Dice_IoU['oc_iou'], iter_num)
                model.train()
            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
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
