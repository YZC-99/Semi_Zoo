import re
import os
import csv
import datetime

def logs2csv(ex_path=''):
    path = ex_path
    csv_path = os.path.join(path, 'statistic.csv')
    ex_num= 0
    with open(csv_path, 'w', newline='') as csvfile:
        w = csv.writer(csvfile)
        # 写入列头
        w.writerow([
            'date',
            'experiment',
            'EX_AUC-PR',
            'EX_AUC-PR_iter',
            'MA_pr',
            'MA_pr_iter',
            'SE_pr',
            'SE_pr_iter',
            'HE_pr',
            'HE_pr_iter',
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
                MA_pr = 0
                MA_pr_iter = 0
                SE_pr = 0
                SE_pr_iter = 0
                HE_pr = 0
                HE_pr_iter = 0
                date = None
                for pth in files:
                    if '.txt' in pth:
                        if 'EX' in pth:
                            EX_pr = pth.split('EX')[-1].split('_')[0]
                            EX_pr_iter = pth.split('.txt')[0].split('_')[-1]
                        if 'MA' in pth:
                            MA_pr = pth.split('MA')[-1].split('_')[0]
                            MA_pr_iter = pth.split('.txt')[0].split('_')[-1]
                        if 'SE' in pth:
                            SE_pr = pth.split('SE')[-1].split('_')[0]
                            SE_pr_iter = pth.split('.txt')[0].split('_')[-1]
                        if 'HE' in pth:
                            HE_pr = pth.split('HE')[-1].split('_')[0]
                            HE_pr_iter = pth.split('.txt')[0].split('_')[-1]

                    elif 'log.txt' in pth:
                        log_file_path = os.path.join(root, pth)
                        with open(log_file_path, 'r') as f:
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
                        # Extract and format creation time of 'log.txt'
                        date = os.path.getctime(log_file_path)
                        date = datetime.datetime.fromtimestamp(date).strftime('%m-%d-%H-%M-%S')


                w.writerow([
                    date,
                    experiment,
                    EX_pr,
                    EX_pr_iter,
                    MA_pr,
                    MA_pr_iter,
                    SE_pr,
                    SE_pr_iter,
                    HE_pr,
                    HE_pr_iter,
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
    # Sort CSV based on the 'date' column
    sort_csv(csv_path)
    return ex_num

def sort_csv(csv_path):
    # Read CSV and sort based on the 'date' column
    data = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]

    header = data[0]
    data = data[1:]
    data.sort(key=lambda x: datetime.datetime.strptime(x[0], '%m-%d-%H-%M-%S'))  # Sorting based on 'date'

    # Write sorted data back to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data)




path = "/root/autodl-tmp/Semi_Zoo/exp_2d_dr"
ex_num = logs2csv(path)
print("收集成功")
print("共计 {} 个实验".format(ex_num))