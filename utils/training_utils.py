import torch
import segmentation_models_pytorch as smp
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss,MSELoss
from utils.losses import OhemCrossEntropy,annealing_softmax_focalloss,softmax_focalloss,weight_softmax_focalloss

def ce_dice_criteria(outputs,all_label_batch):
    dice_criteria = smp.losses.DiceLoss(mode='multiclass', from_logits=True,log_loss=True)
    ce_criteria = CrossEntropyLoss(ignore_index=255)

    loss_dice = dice_criteria(outputs, all_label_batch)
    loss_ce = ce_criteria(outputs, all_label_batch)
    return loss_ce + loss_dice

def criteria(args,outputs,all_label_batch,iter_num):
    dice_criteria = smp.losses.DiceLoss(mode='multiclass', from_logits=True,log_loss=True)
    ce_criteria = CrossEntropyLoss(ignore_index=255)


    seg_criteria = {'ce':ce_criteria,
                    'dice':dice_criteria,
                    'ce-dice': ce_dice_criteria,
                    'softmax_focal':softmax_focalloss}

    if args.main_criteria == 'annealing_softmax_focal':
        loss_seg_main = annealing_softmax_focalloss(outputs,all_label_batch,
                                                         t=iter_num,t_max=args.max_iterations * 0.6)
    else:
        loss_seg_main = seg_criteria[args.main_criteria](outputs,all_label_batch)

    return loss_seg_main
