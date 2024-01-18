import re
import os
import csv
import datetime


def logs2csv(ex_path=''):
    path = ex_path
    csv_path = os.path.join(path, 'statistic.csv')
    ex_num = 0
    with open(csv_path, 'w', newline='') as csvfile:
        w = csv.writer(csvfile)
        # 写入列头
        w.writerow([
            'date',  # Moved 'date' column to the first position
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
                date = None
                for pth in files:
                    if 'OD' in pth:
                        OD_DICE = pth.split('OD_DICE')[-1].split('_')[0]
                        OD_DICE_iter = pth.split('.pth')[0].split('_')[-1]
                    elif 'OC' in pth:
                        OC_DICE = pth.split('OC_DICE')[-1].split('_')[0]
                        OC_DICE_iter = pth.split('.pth')[0].split('_')[-1]
                    elif 'log.txt' in pth:
                        log_file_path = os.path.join(root, pth)
                        with open(log_file_path, 'r') as f:
                            conf_str = f.readlines(1)[0].split('Namespace')[-1]
                            # 使用正则表达式提取键值对
                            pairs = re.findall(r'(\w+)=(.*?)(?:,|\))', conf_str)
                            # 构建字典
                            conf = dict(pairs)

                        # Extract and format creation time of 'log.txt'
                        date = os.path.getctime(log_file_path)
                        date = datetime.datetime.fromtimestamp(date).strftime('%m-%d-%H-%M-%S')

                w.writerow([
                    date,
                    experiment,
                    OD_DICE,
                    OD_DICE_iter,
                    OC_DICE,
                    OC_DICE_iter,
                    conf.get('model', 'N/A'),
                    conf.get('backbone', 'N/A'),
                    conf.get('with_ce', 'True'),
                    conf.get('ce_weight', 'False'),
                    conf.get('with_softfocal', 'False'),
                    conf.get('with_dice', 'False'),
                    conf.get('ohem', 'N/A'),
                    conf.get('vessel_loss_weight', 'N/A'),
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


path = "/root/autodl-tmp/Semi_Zoo/exp_2d_odoc"
ex_num = logs2csv(path)
print("收集成功")
print("共计 {} 个实验".format(ex_num))
