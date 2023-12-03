import re
import os
import csv

def logs2csv(ex_path=''):
    path = ex_path
    csv_path = os.path.join(path, 'statistic.csv')
    ex_num= 0
    with open(csv_path, 'w', newline='') as csvfile:
        w = csv.writer(csvfile)
        # 写入列头
        w.writerow([
            'experiment',
            'OD_DICE',
            'OD_DICE_iter',
            'OC_DICE',
            'OC_DICE_iter',
            'model',
            'backbone',
            'with_ce',
            'ce_weight',
            'soft_focal',
            'with_dice',
            'ohem',
            'vessel_loss_weight',
            'fuse_type',
            'optim',
            'base_lr',
            'batch_size',
            'max_iterations',
            'lr_decouple',
            'CLAHE',
                    ])
        for root, dirs, files in os.walk(path):
            if 'version' in root and 'log' not in root:
                ex_num += 1
                experiment = root.split(path)[-1]
                OD_DICE = 0
                OD_DICE_iter = 0
                OC_DICE = 0
                OC_DICE_iter = 0
                conf = {}
                for pth in files:
                    if 'OD' in pth:
                        OD_DICE = pth.split('OD_DICE')[-1].split('_')[0]
                        OD_DICE_iter = pth.split('.pth')[0].split('_')[-1]
                    elif 'OC' in pth:
                        OC_DICE = pth.split('OC_DICE')[-1].split('_')[0]
                        OC_DICE_iter = pth.split('.pth')[0].split('_')[-1]
                    elif 'log.txt' in pth:
                        with open(os.path.join(root,pth),'r') as f:
                            conf_str = f.readlines(1)[0].split('Namespace')[-1]
                            # 使用正则表达式提取键值对
                            pairs = re.findall(r'(\w+)=(.*?)(?:,|\))', conf_str)
                            # 构建字典
                            conf = dict(pairs)

                w.writerow([
                    experiment,
                    OD_DICE,
                    OD_DICE_iter,
                    OC_DICE,
                    OC_DICE_iter,
                    conf.get('model', 'N/A'),
                    conf.get('backbone', 'N/A'),
                    conf.get('with_ce', 'True'),  # 如果 'ohem' 不存在，返回 'N/A'
                    conf.get('ce_weight', 'False'),  # 如果 'ohem' 不存在，返回 'N/A'
                    conf.get('with_softfocal', 'False'),  # 如果 'ohem' 不存在，返回 'N/A'
                    conf.get('with_dice', 'False'),  # 如果 'ohem' 不存在，返回 'N/A'
                    conf.get('ohem', 'N/A'),  # 如果 'ohem' 不存在，返回 'N/A'
                    conf.get('vessel_loss_weight', 'N/A'),  # 如果 'ohem' 不存在，返回 'N/A'
                    conf.get('fuse_type', 'N/A'),
                    conf.get('optim', 'N/A'),
                    conf.get('base_lr', 'N/A'),
                    conf.get('batch_size', 'N/A'),
                    conf.get('max_iterations', 'N/A'),
                    conf.get('lr_decouple', 'N/A'),
                    conf.get('CLAHE', 'N/A'),
                ])

    return ex_num






path = "/home/gu721/yzc/segmentation/Semi_Zoo/exp_2d/supervised/"
ex_num = logs2csv(path)
print("收集成功")
print("共计 {} 个实验".format(ex_num))