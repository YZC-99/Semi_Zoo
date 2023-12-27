from torchmetrics import JaccardIndex,Dice,PrecisionRecallCurve
from sklearn.metrics import auc,roc_auc_score,precision_recall_curve,roc_curve
from torcheval.metrics import MulticlassAUROC,MulticlassAUPRC,MultilabelAUPRC
from torcheval.metrics import BinaryAUPRC,BinaryAUROC
from utils.my_metrics import BoundaryIoU
import torch
from copy import copy
import numpy as np

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
        od_pred = torch.zeros_like(pred)
        od_label = torch.zeros_like(label)
        od_pred[pred > 0] = 1
        od_label[label > 0] = 1
        self.od_dice_score.update(od_pred,od_label)
        self.od_binary_jaccard.update(od_pred,od_label)
        self.od_binary_boundary_jaccard.update(od_pred,od_label)
        #oc
        oc_pred = torch.zeros_like(pred)
        oc_label = torch.zeros_like(label)
        oc_pred[pred == 2] = 1
        oc_label[label == 2] = 1
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
        od_label = torch.zeros_like(label)
        od_label[label > 0] = 1

        self.od_dice_score.update(od_pred[0,...],od_label)
        self.od_binary_jaccard.update(od_pred[0,...],od_label)
        self.od_binary_boundary_jaccard.update(od_pred[0,...],od_label)
        #oc
        oc_label = torch.zeros_like(label)
        oc_label[label == 2] = 1

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



class DR_metrics(object):
    def __init__(self,device):
        self.AUC_PR = MulticlassAUPRC(num_classes=5,average=None).to(device)
        self.AUC_ROC = MulticlassAUROC(num_classes=5,average=None).to(device)
        self.Dice = Dice(num_classes=1, multiclass=False,average='samples').to(device)
        self.IoU = JaccardIndex(num_classes=2, task='binary', average='micro').to(device)

        self.background_count = 0
        self.MA_count = 0
        self.HE_count = 0
        self.EX_count = 0
        self.SE_count = 0
        self.auc_pr = torch.zeros(5,device=device)
        self.auc_roc = torch.zeros(5,device=device)
        self.dice = 0
        self.iou = 0

    def add(self,preds,labels):
        preds = torch.softmax(preds,dim=1)

        region_preds = copy(preds)
        region_preds = torch.argmax(region_preds,dim=1)
        region_preds[region_preds > 0] = 1
        region_labels = torch.zeros_like(labels)
        region_labels[labels > 0] =1
        self.Dice.update(region_preds.long(), region_labels)
        self.IoU.update(region_preds.long(), region_labels)
        current_dice = self.Dice.compute()
        self.Dice.reset()
        current_iou = self.IoU.compute()
        self.IoU.reset()
        self.dice += current_dice
        self.iou += current_iou


        current_labels = torch.unique(labels)

        preds, labels = preds.squeeze(),labels.squeeze()
        preds = preds.permute(1,2,0)
        preds = preds.view(-1,5)
        labels = labels.view(-1)
        self.AUC_PR.update(preds,labels)
        self.AUC_ROC.update(preds,labels)

        current_auc_pr = self.AUC_PR.compute()
        self.AUC_PR.reset()
        current_auc_roc = self.AUC_ROC.compute()
        self.AUC_ROC.reset()


        for i in range(5):
            if i in labels:
                self.auc_pr[i] += current_auc_pr[i]
                self.auc_roc[i] += current_auc_roc[i]

        self.background_count += 1
        if 1 in current_labels:
            self.MA_count += 1
        if 2 in current_labels:
            self.HE_count += 1
        if 3 in current_labels:
            self.EX_count += 1
        if 4 in current_labels:
            self.SE_count += 1

    def my_reset(self):
        self.background_count = 0
        self.MA_count = 0
        self.HE_count = 0
        self.EX_count = 0
        self.SE_count = 0
        self.auc_pr = torch.zeros_like(self.auc_pr)
        self.auc_roc = torch.zeros_like(self.auc_roc)
        self.dice = 0
        self.iou = 0

    def get_metrics(self):
        auc_pr = self.auc_pr
        auc_roc = self.auc_roc
        Dice = self.dice /self.background_count
        IoU = self.iou /self.background_count

        results = [{
            'MA_AUC_PR': auc_pr[1] / self.MA_count,
            'HE_AUC_PR': auc_pr[2] / self.HE_count,
            'EX_AUC_PR': auc_pr[3] / self.EX_count,
            'SE_AUC_PR': auc_pr[4] / self.SE_count,
        },
        {   'MA_AUC_ROC': auc_roc[1] / self.MA_count,
            'HE_AUC_ROC': auc_roc[2] / self.HE_count,
            'EX_AUC_ROC': auc_roc[3] / self.EX_count,
            'SE_AUC_ROC': auc_roc[4] / self.SE_count,
        },
            {'Dice': Dice,
             'IoU': IoU,

        }
        ]


        self.my_reset()

        return results


class Sklearn_DR_metrics(object):
    def __init__(self):
        self.background_count = 0
        self.MA_count = 0
        self.HE_count = 0
        self.EX_count = 0
        self.SE_count = 0
        self.MA = 0
        self.HE = 0
        self.EX = 0
        self.SE = 0

        self.dice = 0
        self.iou = 0

    def get_scores(self,pred,label):
        y_pred = pred.flatten()
        y_true = label.flatten()
        return y_pred, y_true

    def PR_AUC(self,pred,label):
        # y_pred,y_true = self.get_scores(pred.cpu().numpy(),label.cpu().numpy())
        # precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        precision, recall, thresholds = PrecisionRecallCurve(task="binary")(label, pred)
        precision, recall, thresholds = precision.cpu().numpy(), recall.cpu().numpy(), thresholds.cpu().numpy()


        sorted_indices = np.argsort(recall)
        recall = recall[sorted_indices]
        precision = precision[sorted_indices]

        pr_auc = auc(recall, precision)
        return pr_auc

    def add(self,probs,labels):
        probs = torch.softmax(probs,dim=1)

        assert torch.all((labels == 0) | (labels == 1) | (labels == 2) | (labels == 3) | (
                    labels == 4)), "labels should only contain values 1, 2, 3, or 4,now unique:==>{}<==".format(torch.unique(labels))

        current_labels = torch.unique(labels)

        MA_labels = torch.zeros_like(labels)
        MA_labels[labels == 1] = 1
        HE_labels = torch.zeros_like(labels)
        HE_labels[labels == 2] = 1
        EX_labels = torch.zeros_like(labels)
        EX_labels[labels == 3] = 1
        SE_labels = torch.zeros_like(labels)
        SE_labels[labels == 4] = 1

        if 1 in current_labels:
            self.MA += self.PR_AUC(probs[:,1,...],MA_labels)
            self.MA_count += 1
        if 2 in current_labels:
            self.HE += self.PR_AUC(probs[:,2,...],HE_labels)
            self.HE_count += 1
        if 3 in current_labels:
            self.EX += self.PR_AUC(probs[:,3,...],EX_labels)
            self.EX_count += 1
        if 4 in current_labels:
            self.SE += self.PR_AUC(probs[:,4,...],SE_labels)
            self.SE_count += 1

    def my_reset(self):
        self.background_count = 0
        self.MA_count = 0
        self.HE_count = 0
        self.EX_count = 0
        self.SE_count = 0
        self.MA = 0
        self.HE = 0
        self.EX = 0
        self.SE = 0
        self.dice = 0
        self.iou = 0

    def get_metrics(self):

        results = [{
            'MA_AUC_PR': self.MA / self.MA_count,
            'HE_AUC_PR': self.HE / self.HE_count,
            'EX_AUC_PR': self.EX / self.EX_count,
            'SE_AUC_PR': self.SE / self.SE_count,
        },
        {   'MA_AUC_ROC': 0,
            'HE_AUC_ROC': 0,
            'EX_AUC_ROC': 0,
            'SE_AUC_ROC': 0,
        },
            {'Dice': 0,
             'IoU': 0,

        }
        ]


        self.my_reset()

        return results



class Binary_DR_metrics(object):
    def __init__(self,device):
        self.AUC_PR = BinaryAUPRC().to(device)
        self.AUC_ROC = BinaryAUROC().to(device)
        self.Dice = Dice(num_classes=1, multiclass=False,average='samples').to(device)
        self.IoU = JaccardIndex(num_classes=2, task='binary', average='micro').to(device)

        self.background_count = 0
        self.MA_count = 0
        self.HE_count = 0
        self.EX_count = 0
        self.SE_count = 0
        self.auc_pr = torch.zeros(4,device=device)
        self.auc_roc = torch.zeros(4,device=device)
        self.dice = 0
        self.iou = 0

    def add(self,preds,MA_mask , HE_mask , EX_mask , SE_mask):
        # preds[preds > 0.5] = 1
        _,num,_,_ = preds.size()
        preds = preds.squeeze()
        preds = preds.view(num,-1)

        MA_mask = MA_mask.squeeze().view(-1)
        HE_mask = HE_mask.squeeze().view(-1)
        EX_mask = EX_mask.squeeze().view(-1)
        SE_mask = SE_mask.squeeze().view(-1)

        if len(torch.unique((MA_mask))) > 1:
            self.AUC_PR.update(preds[1],MA_mask)
            self.auc_pr[0] += self.AUC_PR.compute()
            self.AUC_PR.reset()
            self.MA_count += 1
        if len(torch.unique((HE_mask))) > 1:
            self.AUC_PR.update(preds[2],HE_mask)
            self.auc_pr[1] += self.AUC_PR.compute()
            self.AUC_PR.reset()
            self.HE_count += 1
        if len(torch.unique((EX_mask))) > 1:
            self.AUC_PR.update(preds[3],EX_mask)
            self.auc_pr[2] += self.AUC_PR.compute()
            self.AUC_PR.reset()
            self.EX_count += 1
        if len(torch.unique((SE_mask))) > 1:
            self.AUC_PR.update(preds[4],SE_mask)
            self.auc_pr[3] += self.AUC_PR.compute()
            self.AUC_PR.reset()
            self.SE_count += 1

    def add_noback(self,preds,MA_mask , HE_mask , EX_mask , SE_mask):
        # preds[preds > 0.5] = 1
        _,num,_,_ = preds.size()
        preds = preds.squeeze()
        preds = preds.view(num,-1)

        MA_mask = MA_mask.squeeze().view(-1)
        HE_mask = HE_mask.squeeze().view(-1)
        EX_mask = EX_mask.squeeze().view(-1)
        SE_mask = SE_mask.squeeze().view(-1)

        if len(MA_mask) > 2:
            self.AUC_PR.update(preds[0],MA_mask)
            self.auc_pr[0] += self.AUC_PR.compute()
            self.AUC_PR.reset()
            self.MA_count += 1
        if len(HE_mask) > 2:
            self.AUC_PR.update(preds[1],HE_mask)
            self.auc_pr[1] += self.AUC_PR.compute()
            self.AUC_PR.reset()
            self.HE_count += 1
        if len(EX_mask) > 2:
            self.AUC_PR.update(preds[2],EX_mask)
            self.auc_pr[2] += self.AUC_PR.compute()
            self.AUC_PR.reset()
            self.EX_count += 1
        if len(SE_mask) > 2:
            self.AUC_PR.update(preds[3],SE_mask)
            self.auc_pr[3] += self.AUC_PR.compute()
            self.AUC_PR.reset()
            self.SE_count += 1
    def my_reset(self):
        self.background_count = 0
        self.MA_count = 0
        self.HE_count = 0
        self.EX_count = 0
        self.SE_count = 0
        self.auc_pr = torch.zeros_like(self.auc_pr)
        self.auc_roc = torch.zeros_like(self.auc_roc)
        self.dice = 0
        self.iou = 0

    def get_metrics(self):
        auc_pr = self.auc_pr
        auc_roc = self.auc_roc

        results = [{
            'MA_AUC_PR': auc_pr[0] / self.MA_count,
            'HE_AUC_PR': auc_pr[1] / self.HE_count,
            'EX_AUC_PR': auc_pr[2] / self.EX_count,
            'SE_AUC_PR': auc_pr[3] / self.SE_count,
        },
        {   'MA_AUC_ROC': auc_roc[0] / self.MA_count,
            'HE_AUC_ROC': auc_roc[1] / self.HE_count,
            'EX_AUC_ROC': auc_roc[2] / self.EX_count,
            'SE_AUC_ROC': auc_roc[3] / self.SE_count,
        },
            {'Dice': 0,
             'IoU': 0,

        }
        ]


        self.my_reset()

        return results


# class DR_metrics(object):
#     def __init__(self,device):
#         self.MA_PR_Curve = PrecisionRecallCurve(task='binary').to(device)
#         self.HE_PR_Curve = PrecisionRecallCurve(task='binary').to(device)
#         self.EX_PR_Curve = PrecisionRecallCurve(task='binary').to(device)
#         self.SE_PR_Curve = PrecisionRecallCurve(task='binary').to(device)
#
#     def add(self,preds,labels):
#         # 先分离不同类别的二值mask
#         MA_preds = torch.zeros_like(preds)
#         HE_preds = torch.zeros_like(preds)
#         EX_preds = torch.zeros_like(preds)
#         SE_preds = torch.zeros_like(preds)
#         #
#         MA_preds[preds == 1] = 1
#         HE_preds[preds == 2] = 1
#         EX_preds[preds == 3] = 1
#         SE_preds[preds == 4] = 1
#
#         MA_labels = torch.zeros_like(labels)
#         HE_labels = torch.zeros_like(labels)
#         EX_labels = torch.zeros_like(labels)
#         SE_labels = torch.zeros_like(labels)
#         #
#         MA_labels[labels == 1] = 1
#         HE_labels[labels == 2] = 1
#         EX_labels[labels == 3] = 1
#         SE_labels[labels == 4] = 1
#
#         #加入
#         self.MA_PR_Curve.update(MA_preds,MA_labels)
#         self.HE_PR_Curve.update(HE_preds,HE_labels)
#         self.EX_PR_Curve.update(EX_preds,EX_labels)
#         self.SE_PR_Curve.update(SE_preds,SE_labels)
#
#     def get_metrics(self):
#         MA_precision,MA_recall,MA_thresholds = self.MA_PR_Curve.compute()
#         HE_precision,HE_recall,HE_thresholds = self.HE_PR_Curve.compute()
#         EX_precision,EX_recall,EX_thresholds = self.EX_PR_Curve.compute()
#         SE_precision,SE_recall,SE_thresholds = self.SE_PR_Curve.compute()
#
#         self.MA_PR_Curve.reset()
#         self.HE_PR_Curve.reset()
#         self.EX_PR_Curve.reset()
#         self.SE_PR_Curve.reset()
#         return [{
#             'MA_PR_Curve':MA_PR_Curve,
#             'HE_PR_Curve':HE_PR_Curve,
#             'EX_PR_Curve':EX_PR_Curve,
#             'SE_PR_Curve':SE_PR_Curve,
#         }]
