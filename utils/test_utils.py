from torchmetrics import JaccardIndex,Dice
from utils.my_metrics import BoundaryIoU
import torch
import copy

class ODOC_metrics(object):
    def __init__(self,device):
        self.od_dice_score = Dice(num_classes=1, multiclass=False,average='samples').to(device)
        self.od_binary_jaccard = JaccardIndex(num_classes=2, task='binary', average='micro').to(device)
        self.od_binary_boundary_jaccard = BoundaryIoU(num_classes=2, task='binary').to(device)

        self.oc_dice_score = Dice(num_classes=1, multiclass=False,average='samples').to(device)
        self.oc_binary_jaccard = JaccardIndex(num_classes=2, task='binary', average='micro').to(device)
        self.oc_binary_boundary_jaccard = BoundaryIoU(num_classes=2, task='binary').to(device)

    def add_multi_class(self,outputs,label):
        pred = torch.argmax(outputs,dim=1)
        #包含oc的od
        od_pred = copy.deepcopy(pred)
        od_label = copy.deepcopy(label)
        od_pred[od_pred > 0] = 1
        od_label[od_label > 0] = 1
        # self.od_dice_score.add_state(od_pred,od_label)
        # self.od_binary_jaccard.add_state(od_pred,od_label)
        self.od_dice_score.update(od_pred,od_label)
        self.od_binary_jaccard.update(od_pred,od_label)
        self.od_binary_boundary_jaccard.update(od_pred,od_label)
        #oc
        oc_pred = copy.deepcopy(pred)
        oc_label = copy.deepcopy(label)
        oc_pred[oc_pred == 2] = 1
        oc_label[oc_label == 2] = 1
        # self.od_dice_score.add_state(oc_pred,oc_label)
        # self.od_binary_jaccard.add_state(oc_pred,oc_label)
        self.oc_dice_score.update(oc_pred,oc_label)
        self.oc_binary_jaccard.update(oc_pred,oc_label)
        self.oc_binary_boundary_jaccard.update(oc_pred,oc_label)

    def add(self,outputs,label):
        outputs_od, outputs_oc = outputs[:, 0, ...].unsqueeze(1), outputs[:, 1, ...].unsqueeze(1)
        od_pred = (outputs_od > 0.5).to(torch.int8)
        oc_pred = (outputs_oc > 0.5).to(torch.int8)


        od_pred = od_pred + oc_pred
        od_pred[od_pred > 0] = 1
        #包含oc的od
        od_label = copy.deepcopy(label)
        od_label[od_label > 0] = 1

        self.od_dice_score.update(od_pred[0,...],od_label)
        self.od_binary_jaccard.update(od_pred[0,...],od_label)
        self.od_binary_boundary_jaccard.update(od_pred[0,...],od_label)
        #oc
        oc_label = copy.deepcopy(label)
        oc_label[oc_label != 2] = 0
        oc_label[oc_label != 0] = 1

        self.oc_dice_score.update(oc_pred[0,...],oc_label)
        self.oc_binary_jaccard.update(oc_pred[0,...],oc_label)
        self.oc_binary_boundary_jaccard.update(oc_pred[0,...],oc_label)

    def get_metrics(self):
        od_dice = self.od_dice_score.compute()
        od_iou = self.od_binary_jaccard.compute()
        od_biou = self.od_binary_boundary_jaccard.compute()
        oc_dice = self.oc_dice_score.compute()
        oc_iou = self.oc_binary_jaccard.compute()
        oc_biou = self.oc_binary_boundary_jaccard.compute()
        self.od_dice_score.reset()
        self.od_binary_jaccard.reset()
        self.od_binary_boundary_jaccard.reset()
        self.oc_dice_score.reset()
        self.oc_binary_jaccard.reset()
        self.oc_binary_boundary_jaccard.reset()
        return {
                   "od_dice":od_dice,
                   "od_iou":od_iou,
                   "od_biou":od_biou,
                   "oc_dice":oc_dice,
                   "oc_iou":oc_iou,
                   "oc_biou":oc_biou,
        }
