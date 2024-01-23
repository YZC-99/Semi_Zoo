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
            'EX_AUC-PR',
            'EX_AUC-PR_iter',
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
                conf = {}
                EX_pr = 0
                EX_pr_iter = 0
                for pth in files:
                    if 'EX' in pth:
                        EX_pr = pth.split('EX')[-1].split('_')[0]
                        EX_pr_iter = pth.split('.pth')[0].split('_')[-1]

                    elif 'log.txt' in pth:
                        with open(os.path.join(root,pth),'r') as f:
                            conf_str = f.readlines(1)[0].split('Namespace')[-1]
                            # 使用正则表达式提取键值对
                            """
                            下面这个正则表达式有问题，如果我的str出现了：max_iterations=10000, ce_weight=[0.001, 1.0, 0.1, 0.01, 0.1], ohem=-1.0,
                            这就导致ce_weight=[0.001, 1.0, 0.1, 0.01, 0.1],的匹配有问题，我希望你帮助我重写一个正则表达式
                            """
                            # pairs = re.findall(r'(\w+)=(.*?)(?:,|\))', conf_str)
                            pairs = re.findall(r'(\w+)\s*=\s*((?:\[[^\]]*\]|[^,]+))(?:,|$)', conf_str)
                            # 构建字典
                            conf = dict(pairs)

                w.writerow([
                    experiment,
                    EX_pr,
                    EX_pr_iter,
                    conf.get('model', 'N/A'),
                    conf.get('backbone', 'N/A'),
                    conf.get('with_ce', 'True'),  # 如果 'ohem' 不存在，返回 'N/A'
                    conf.get('ce_weight', 'False'),
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






path = "/root/autodl-tmp/Semi_Zoo/exp_2d_dr"
ex_num = logs2csv(path)
print("收集成功")
print("共计 {} 个实验".format(ex_num))