import torch
import torch.nn.functional as F

def GMM(feat, vecs, pred, true_mask, cls_label):

    # 传入的pred是一个经过clean_mask处理的
    b, k, oh, ow = pred.size()
    # 如果是单一任务的情况下，这里的k就会少一个类别,所以加1
    # TODO 验证是否有效
    k += 1

    # 将mask的大小调整来与feat的相同
    preserve = (true_mask < 255).long().view(b, 1, oh, ow)
    preserve = F.interpolate(preserve.float(), size=feat.size()[-2:], mode='bilinear')
    # 将pred的大小调整来与feat相同
    pred = F.interpolate(pred, size=feat.size()[-2:], mode='bilinear')
    _, _, h, w = pred.size()

    # vecs原本的形状是(b,num_class,c),现在调整为(b,num_class,c,1,1)
    vecs = vecs.view(b, k, -1, 1, 1)
    # feat原来的形状是(b,c,h,w),现在调整为(b,1,c,h,w)
    feat = feat.view(b, 1, -1, h, w)

    """ 255 caused by cropping, using preserve mask """
    # print("feat.shape")
    # print(feat.shape)
    # print("vecs.shape")
    # print(vecs.shape)

    abs = torch.abs(feat - vecs).mean(2) # 对应原论文公式(6)中d的计算
    abs = abs * cls_label.view(b, k, 1, 1) * preserve.view(b, 1, h, w)
    abs = abs.view(b, k, h*w)

    # """ calculate std """
    # pred = pred * preserve
    # num = pred.view(b, k, -1).sum(-1)
    # std = ((pred.view(b, k, -1)*(abs ** 2)).sum(-1)/(num + 1e-6)) ** 0.5
    # std = std.view(b, k, 1, 1).detach()

    # std = ((abs ** 2).sum(-1)/(preserve.view(b, 1, -1).sum(-1)) + 1e-6) ** 0.5
    # std = std.view(b, k, 1, 1).detach()

    abs = abs.view(b, k, h, w)
    res = torch.exp(-(abs * abs)) # 公式(7)里面 -d^2
    # res = torch.exp(-(abs*abs)/(2*std*std + 1e-6))
    res = F.interpolate(res, size=(oh, ow), mode='bilinear')
    res = res * cls_label.view(b, k, 1, 1)
    return res



def GMM_w_std(feat, vecs, pred):
    # 传入的pred是一个经过clean_mask处理的
    b, k, oh, ow = pred.size()

    # 将pred的大小调整来与feat相同
    pred = F.interpolate(pred, size=feat.size()[-2:], mode='bilinear')
    _, _, h, w = pred.size()

    # vecs原本的形状是(b,num_class,c),现在调整为(b,num_class,c,1,1)
    vecs = vecs.view(b, k, -1, 1, 1)
    # feat原来的形状是(b,c,h,w),现在调整为(b,1,c,h,w)
    feat = feat.view(b, 1, -1, h, w)

    """ 255 caused by cropping, using preserve mask """
    abs = torch.abs(feat - vecs).mean(2) # 对应原论文公式(6)中d的计算
    abs = abs.view(b, k, h*w)

    # """ calculate std """
    num = pred.view(b, k, -1).sum(-1)
    std = ((pred.view(b, k, -1)*(abs ** 2)).sum(-1)/(num + 1e-6)) ** 0.5
    std = std.view(b, k, 1, 1).detach()


    abs = abs.view(b, k, h, w)
    # res = torch.exp(-(abs * abs)) # 公式(7)里面 -d^2,精简版，忽略了标准差
    res = torch.exp(-(abs*abs)/(2*std*std + 1e-6))
    res = F.interpolate(res, size=(oh, ow), mode='bilinear')
    # res = res.view(b, k, 1, 1)

    return res

def one_hot_2d(label, nclass):
    h, w = label.size()
    label_cp = label.clone()

    label_cp[label > nclass] = nclass
    label_cp = label_cp.view(1, h*w)

    mask = torch.zeros(nclass+1, h*w).to(label.device)
    mask = mask.scatter_(0, label_cp.long(), 1).view(nclass+1, h, w).float()
    return mask[:-1, :, :]

def one_hot(label, nclass):
    b, h, w = label.size()
    label_cp = label.clone()

    label_cp[label > nclass] = nclass
    label_cp = label_cp.view(b, 1, h*w)

    mask = torch.zeros(b, nclass+1, h*w).to(label.device)
    mask = mask.scatter_(1, label_cp.long(), 1).view(b, nclass+1, h, w).float()
    return mask[:, :-1, :, :]

def build_cur_cls_label(mask, nclass):
    """some point annotations are cropped out, thus the prototypes are partial"""
    b = mask.size()[0]
    mask_one_hot = one_hot(mask, nclass)
    cur_cls_label = mask_one_hot.view(b, nclass, -1).max(-1)[0]
    return cur_cls_label.view(b, nclass, 1, 1)

def cal_protypes(feat, mask, nclass):
    # 将特征上采样与mask大小一致
    feat = F.interpolate(feat, size=mask.size()[-2:], mode='bilinear')
    b, c, h, w = feat.size()
    # 初始化prototypes，形状为(b,class_num,c)
    prototypes = torch.zeros((b, nclass, c),
                           dtype=feat.dtype,
                           device=feat.device)
    # batchsize中的样本逐一处理
    for i in range(b):
        cur_mask = mask[i]
        cur_mask_onehot = one_hot_2d(cur_mask, nclass)

        cur_feat = feat[i]
        cur_prototype = torch.zeros((nclass, c),
                           dtype=feat.dtype,
                           device=feat.device)

        cur_set = list(torch.unique(cur_mask))
        if nclass in cur_set:
            cur_set.remove(nclass)
        if 255 in cur_set:
            cur_set.remove(255)

        for cls in cur_set:# cur_set:0,1,2
            #获取mask中当前类别的像素数量
            m = cur_mask_onehot[cls].view(1, h, w)
            sum = m.sum()
            m = m.expand(c, h, w).view(c, -1)
            # cur_feat:(c,h,w) == > (c,h*w)
            # (cur_feat.view(c, -1)[m == 1]):意思是只取出当前类别的特征，然后按照最后的维度求和，也就是相当于每个通道上的centriiod
            cls_feat = (cur_feat.view(c, -1)[m == 1]).view(c, -1).sum(-1)/(sum + 1e-6)
            cur_prototype[cls, :] = cls_feat

        prototypes[i] += cur_prototype

    cur_cls_label = build_cur_cls_label(mask, nclass).view(b, nclass, 1)
    mean_vecs = (prototypes.sum(0)*cur_cls_label.sum(0))/(cur_cls_label.sum(0)+1e-6)

    loss = proto_loss(prototypes, mean_vecs, cur_cls_label)

    return prototypes.view(b, nclass, c), loss


def proto_loss(prototypes, vecs, cur_cls_label):
    b, nclass, c = prototypes.size()

    # abs = torch.abs(prototypes - vecs).mean(2)
    # positive = torch.exp(-(abs * abs))
    # positive = (positive*cur_cls_label.view(b, nclass)).sum()/(cur_cls_label.sum()+1e-6)
    # positive_loss = 1 - positive

    vecs = vecs.view(nclass, c)
    total_cls_label = (cur_cls_label.sum(0) > 0).long()
    negative = torch.zeros(1,
                           dtype=prototypes.dtype,
                           device=prototypes.device)

    num = 0
    for i in range(nclass):
        # 如果有当前类别
        if total_cls_label[i] == 1:
            for j in range(i+1, nclass):
                if total_cls_label[j] == 1:
                    if i != j:
                        num += 1
                        x, y = vecs[i].view(1, c), vecs[j].view(1, c)
                        abs = torch.abs(x - y).mean(1)
                        negative += torch.exp(-(abs * abs))
                        # print(negative)

    negative = negative/(num+1e-6)
    negative_loss = negative

    return negative_loss


def cal_gmm_loss(pred, res, cls_label, true_mask):
    n, k, h, w = pred.size()
    # 这里对应公式(8)，但是这里将公式(7)的表示也带入其中
    # res表示公式(7)的d^2,貌似还把方差这一项给忽略了,论文中说的是

    # 任务单独划分后，res里面包含了3个类别，但实际上只有两个类别，所以取前两个
    res = res[:,:k,...]
    # print("pred.shape")
    # print(pred.shape)
    # print("res.shape")
    # print(res.shape)

    # print("cls_label")
    # print(cls_label.shape)
    cls_label = cls_label[:,:k,...]

    loss1 = - res * torch.log(pred + 1e-6) - (1 - res) * torch.log(1 - pred + 1e-6)
    loss1 = loss1/2
    loss1 = (loss1*cls_label).sum(1)/(cls_label.sum(1)+1e-6)
    loss1 = loss1[true_mask != 255].mean()

    # 对应原论文公式(9)
    true_mask_one_hot = one_hot(true_mask, k)
    loss2 = - true_mask_one_hot * torch.log(res + 1e-6) \
            - (1 - true_mask_one_hot) * torch.log(1 - res + 1e-6)
    loss2 = loss2/2
    loss2 = (loss2 * cls_label).sum(1) / (cls_label.sum(1) + 1e-6)
    loss2 = loss2[true_mask < k].mean()
    return loss1+loss2
