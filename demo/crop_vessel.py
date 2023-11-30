"""
实现一个使用鼠标点击来进行中心裁剪的交互程序，具体介绍和要求如下：

介绍：img_dir下面的图片和mask_dir下面的mask图片名字一致且分辨率大小一致，只不过是文件后缀不一样
要求：
1、读取img_dir中的图片展示给我，然后我通过鼠标点击，就以我点击的位置为中心在图片上裁剪一个512x512的区域，同时，也要在mask_dir
下找到同名的mask，也在相同的位置以512x512裁剪出来
2、裁剪不影响原图
3、裁剪后的img保存在img_cropped_dir,裁剪后的mask保存在mask_cropped_dir
4、保存过程中如果路径不存在则创建这个文件夹
5、保存的名字和文件类型与原来保存一致
6、执行完当前的图片后，自动切换到下一张图片
7、直到我操作完所有的img后才结束程序
"""


img_dir = "/home/gu721/yzc/data/HRF/images/"
img_cropped_dir = '/home/gu721/yzc/data/HRF/images_cropped'
mask_dir = "/home/gu721/yzc/data/HRF/manual1/"
mask_cropped_dir = '/home/gu721/yzc/data/HRF/gts_cropped'

import cv2
import os

# 设置路径
img_dir = "/home/gu721/yzc/data/HRF/images/"
img_cropped_dir = '/home/gu721/yzc/data/HRF/images_cropped'
mask_dir = "/home/gu721/yzc/data/HRF/manual1/"
mask_cropped_dir = '/home/gu721/yzc/data/HRF/gts_cropped'

# 创建保存裁剪后图片的文件夹
os.makedirs(img_cropped_dir, exist_ok=True)
os.makedirs(mask_cropped_dir, exist_ok=True)

# 获取所有图片文件列表
img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

def crop_image(event, x, y, flags, param):
    global img, mask, img_file, mask_file

    # 如果鼠标左键被点击
    if event == cv2.EVENT_LBUTTONDOWN:
        size = 512

        # 确保裁剪区域在图片内
        if x - size // 2 >= 0 and y - size // 2 >= 0 and x + size // 2 <= img.shape[1] and y + size // 2 <= img.shape[0]:
            # 裁剪图像和 mask
            img_cropped = img[y - size // 2:y + size // 2, x - size // 2:x + size // 2]
            mask_cropped = mask[y - size // 2:y + size // 2, x - size // 2:x + size // 2]

            # 生成保存路径
            img_cropped_path = os.path.join(img_cropped_dir, img_file)
            mask_cropped_path = os.path.join(mask_cropped_dir, mask_file)

            # 保存裁剪后的图像和 mask
            cv2.imwrite(img_cropped_path, img_cropped)
            cv2.imwrite(mask_cropped_path, mask_cropped)

            print(f"Cropped image saved at {img_cropped_path}")
            print(f"Cropped mask saved at {mask_cropped_path}")

            # 跳到下一张图片
            cv2.destroyAllWindows()

    # 如果鼠标右键被点击，跳到下一张图片
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.destroyAllWindows()

# 创建窗口并显示原图
for img_file in img_files:
    # 读取原图
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)

    # 读取对应的 mask 图
    mask_file = os.path.splitext(img_file)[0] + ".tif"
    mask_path = os.path.join(mask_dir, mask_file)

    # 确保 mask 文件存在
    if not os.path.exists(mask_path):
        print(f"Error: Mask file not found - {mask_path}")
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 确保 mask 不为 None
    if mask is not None:
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", crop_image)

        # 调整图像大小以适应屏幕
        max_height = 800
        if img.shape[0] > max_height:
            scale_factor = max_height / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * scale_factor), max_height))
            mask = cv2.resize(mask, (int(mask.shape[1] * scale_factor), max_height))

        cv2.imshow("Image", img)
        cv2.waitKey(0)

# 关闭所有窗口
cv2.destroyAllWindows()
