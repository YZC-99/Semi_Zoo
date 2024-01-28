import torch
import segmentation_models_pytorch as smp
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss,MSELoss
from utils.losses import OhemCrossEntropy,annealing_softmax_focalloss,softmax_focalloss,weight_softmax_focalloss
from utils.blv_loss import BlvLoss,Softmaxfocal_BlvLoss

def ce_dice_criteria(outputs,all_label_batch):
    dice_criteria = smp.losses.DiceLoss(mode='multiclass', from_logits=True,log_loss=True)
    ce_criteria = CrossEntropyLoss(ignore_index=255)

    loss_dice = dice_criteria(outputs, all_label_batch)
    loss_ce = ce_criteria(outputs, all_label_batch)
    return loss_ce + loss_dice

def criteria(args,outputs,all_label_batch,iter_num):
    dice_criteria = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
    ce_criteria = CrossEntropyLoss(ignore_index=255)
    weight_ce_criteria = CrossEntropyLoss(ignore_index=255,weight=torch.tensor([1.0,2.0,2.0,2.0,2.0],device=outputs.device))
    blv_criteria = BlvLoss(cls_num_list=args.cls_num_list)
    softmax_focal_blv_criteria = Softmaxfocal_BlvLoss(cls_num_list=args.cls_num_list)


    seg_criteria = {'ce':ce_criteria,
                    'weight-ce':weight_ce_criteria,
                    'dice':dice_criteria,
                    'ce-dice': ce_dice_criteria,
                    'blv': blv_criteria,
                    'softmax_focal_blv': softmax_focal_blv_criteria,
                    'softmax_focal':softmax_focalloss}

    if args.main_criteria == 'annealing_softmax_focal':
        loss_seg_main = annealing_softmax_focalloss(outputs,all_label_batch,
                                                         t=iter_num,t_max=args.max_iterations * 0.6)
    else:
        loss_seg_main = seg_criteria[args.main_criteria](outputs,all_label_batch)

    return loss_seg_main

