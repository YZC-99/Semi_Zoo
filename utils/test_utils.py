from torchmetrics import JaccardIndex,Dice
from sklearn.metrics import auc,roc_auc_score
from torcheval.metrics import MulticlassAUROC,MulticlassAUPRC
from utils.my_metrics import BoundaryIoU
import torch

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

    def add(self,preds,labels):
        preds, labels = preds.squeeze(),labels.squeeze()
        preds = preds.permute(1,2,0)
        preds = preds.view(-1,5)
        labels = labels.view(-1)
        # num_classes = len(torch.unique(labels))
        # num_classes = 5
        # one_hot_tensor = torch.nn.functional.one_hot(labels,num_classes)
        # labels = one_hot_tensor.permute(0,3,1,2)
        self.AUC_PR.update(preds,labels)
        self.AUC_ROC.update(preds,labels)



    def get_metrics(self):
        auc_pr = self.AUC_PR.compute()
        self.AUC_PR.reset()
        auc_roc = self.AUC_ROC.compute()
        self.AUC_ROC.reset()




        return [{
            'MA_AUC_PR': auc_pr[1],
            'HE_AUC_PR': auc_pr[2],
            'EX_AUC_PR': auc_pr[3],
            'SE_AUC_PR': auc_pr[4],
        },
        {   'MA_AUC_ROC': auc_roc[1],
            'HE_AUC_ROC': auc_roc[2],
            'EX_AUC_ROC': auc_roc[3],
            'SE_AUC_ROC': auc_roc[4],
        },
        ]


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
