import os
import argparse
import torch
from torch.utils.data import DataLoader
from dataloader.fundus import SemiDataset
from tqdm import tqdm
from model.mcnet.unet import MCNet2d_compete_v1,UNet_DTC2d

parser = argparse.ArgumentParser()
parser.add_argument('--image_size',type=int,default=256)


parser.add_argument('--root_path', type=str,
                    default='../data/2018LA_Seg_Training Set', help='Name of Experiment')
parser.add_argument('--model', type=str,
                    default='DTC_16labels', help='model_name')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1,
                    help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1,
                    help='apply NMS post-procssing?')


FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/{}".format(FLAGS.model)

num_classes = 3


def test_calculate_metric():
    net = UNet_DTC2d(in_chns=3,class_num=num_classes).cuda()
    # save_mode_path = os.path.join(
    #     snapshot_path, 'best_model.pth')

    save_mode_path = "/home/gu721/yzc/segmentation/Semi_Zoo/exp_dtc_2d/refuge400/version0/iter_1000.pth"
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    # init dataset
    val_dataset = SemiDataset(name='./dataset/semi_refuge400',
                                  root="/home/gu721/yzc/data/odoc/REFUGE/",
                                  mode='val',
                                  size=FLAGS.image_size)
    labeledtrainloader = DataLoader(val_dataset,batch_size=1)
    iteriter = tqdm(labeledtrainloader)
    for data in iteriter:
        img,label = data['image'].cuda(),data['label'].cuda()
        out_dict = net(img)
        outputs_tanh, outputs = out_dict["output_tanh_1"],out_dict["output1"]
        # print(outputs.shape)
        print(outputs_tanh.shape)
    return 'avg_metric'


if __name__ == '__main__':
    metric = test_calculate_metric()  # 6000
    print(metric)
