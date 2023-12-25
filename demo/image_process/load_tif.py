from PIL import Image
import numpy as np

label_path = "/home/gu721/yzc/data/dr/IDRID/labels_indiv/training/1. Microaneurysms/IDRiD_01_MA.tif"

"""
原始图片是(4288*2848)，是tif的格式
Image.open(label_path)读取出来的高和宽颠倒了，
变成了(2848, 4288)，怎么解决
"""
label = Image.open(label_path)
label.show()
label_arr = np.array(label)
print(label_arr.shape)

