from PIL import Image
import numpy as np
import cv2

# mask_path = "/home/gu721/yzc/data/vessel/HRF/gts_cropped/14_dr.tif"
mask_path = "/home/gu721/yzc/data/vessel/CHASEDB1/gts_cropped/Image_12R_1stHO.png"

mask = Image.open(mask_path)
# mask = mask.point(lambda x: 0 if x < 128 else 255, '1')
# mask.show()
mask_array = np.array(mask)


cv_mask = cv2.imread(mask_path)
# print(np.unique(mask_array))
print(np.unique(cv_mask))
