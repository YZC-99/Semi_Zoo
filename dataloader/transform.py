import numpy as np
from PIL import Image, ImageOps, ImageFilter,ImageEnhance
import random
import torch
from torchvision import transforms
from dataloader.boundary_utils import class2one_hot,one_hot2dist
import cv2


def cartesian_to_polar(img, mask):
    # 将PIL图像转换为NumPy数组
    img_np = np.array(img.convert('RGB'))
    mask_np = np.array(mask)  # 假设mask是灰度图

    # 确保图像是float类型
    img_float = img_np.astype(np.float32)
    mask_float = mask_np.astype(np.float32)
    print(np.unique(mask_float))

    # 计算用于极坐标变换的值，使用图像的短边作为半径
    value = min(img_float.shape[0], img_float.shape[1]) / 2

    # 执行极坐标变换
    polar_img_cv = cv2.linearPolar(img_float, (img_float.shape[1] / 2, img_float.shape[0] / 2), value,
                                   cv2.WARP_FILL_OUTLIERS)
    polar_mask_cv = cv2.linearPolar(mask_float, (mask_float.shape[1] / 2, mask_float.shape[0] / 2), value,
                                    cv2.WARP_FILL_OUTLIERS)

    # 将极坐标图像的数据类型转换为uint8
    polar_img_cv = polar_img_cv.astype(np.uint8)
    polar_mask_cv = polar_mask_cv.astype(np.uint8)

    # 将NumPy数组转换回PIL图像
    polar_img = Image.fromarray(polar_img_cv)
    polar_mask = Image.fromarray(polar_mask_cv, mode="P")

    return polar_img, polar_mask

def dist_transform(mask):
    # mask = np.array(mask)
    # mask_arr_ex = np.expand_dims(mask, axis=0)
    mask_tensor = torch.unsqueeze(mask,dim=0)
    mask_tensor = mask_tensor.to(torch.int64)
    mask_tensor[mask_tensor == 255] = 0
    # mask_tensor = torch.tensor(mask_arr_ex, dtype=torch.int64)
    mask_trans = class2one_hot(mask_tensor, 3)
    mask_trans_arr = mask_trans.cpu().squeeze().numpy()
    bounadry = one_hot2dist(mask_trans_arr, resolution=[1, 1])
    return bounadry


def crop(img, mask, size):
    # padding height or width if smaller than cropping size
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

    # cropping
    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask

def vflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask

def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    return img, mask

def add_salt_pepper_noise(img,mask, p=0.5,noise_level=0.02):
    if random.random() < p:
        img_array = np.array(img)

        h, w, _ = img_array.shape
        num_pixels = int(h * w * noise_level)

        # Add salt noise
        salt_coords = [np.random.randint(0, i - 1, num_pixels) for i in (h, w)]
        img_array[salt_coords[0], salt_coords[1], :] = 255

        # Add pepper noise
        pepper_coords = [np.random.randint(0, i - 1, num_pixels) for i in (h, w)]
        img_array[pepper_coords[0], pepper_coords[1], :] = 0

        img = Image.fromarray(img_array)
    return img,mask

def random_scale(img, mask, min_scale=0.8, p=0.5, max_scale=1.2):
    if random.random() < p:
        w_scale_factor = random.uniform(min_scale, max_scale)
        h_scale_factor = random.uniform(min_scale, max_scale)
        new_width = int(img.width * w_scale_factor)
        new_height = int(img.height * h_scale_factor)

        img = img.resize((new_width, new_height), Image.BILINEAR)
        mask = mask.resize((new_width, new_height), Image.NEAREST)

    return img, mask



def random_scale_and_crop(img, mask, target_size=(512, 512), min_scale=0.8, max_scale=1.2,p=0.5):
    if random.random() < p:
        # random scale (short edge)
        short_size = random.randint(int(img.width  * min_scale), int(img.width  * max_scale))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < target_size[0]:
            padh = target_size[0] - oh if oh < target_size[0] else 0
            padw = target_size[0] - ow if ow < target_size[0] else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - target_size[0])
        y1 = random.randint(0, h - target_size[0])
        img = img.crop((x1, y1, x1 + target_size[0], y1 + target_size[0]))
        mask = mask.crop((x1, y1, x1 + target_size[0], y1 + target_size[0]))

    return img, mask



def random_rotate(img, mask, p=0.5, max_rotation_angle=90):
    if random.random() < p:
        rotation_angle = random.uniform(-max_rotation_angle, max_rotation_angle)
        img = img.rotate(rotation_angle, resample=Image.BILINEAR, expand=True)
        mask = mask.rotate(rotation_angle, resample=Image.NEAREST, expand=True)

    return img, mask

def random_translate(img, mask, p=0.5, max_translate_percent=0.15):
    if random.random() < p:
        img_width, img_height = img.size
        translate_x = random.uniform(-max_translate_percent, max_translate_percent) * img_width
        translate_y = random.uniform(-max_translate_percent, max_translate_percent) * img_height

        img = img.transform(
            img.size, Image.AFFINE, (1, 0, translate_x, 0, 1, translate_y)
        )

        mask = mask.transform(
            mask.size, Image.AFFINE, (1, 0, translate_x, 0, 1, translate_y)
        )

    return img, mask


def color_distortion(img, min_factor=-0.1, max_factor=0.1):
    # 随机生成亮度、对比度和饱和度的调整因子
    brightness_factor = random.uniform(min_factor, max_factor)
    contrast_factor = random.uniform(min_factor, max_factor)
    saturation_factor = random.uniform(min_factor, max_factor)

    # 调整亮度
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1 + brightness_factor)

    # 调整对比度
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1 + contrast_factor)

    # 调整饱和度
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1 + saturation_factor)

    return img


def normalize(img, mask=None,mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0]):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # transforms.Normalize([151.818,74.596,23.749], [28.438,15.263 ,5.225]),
        # Mean: [151.81788834  74.5958448   23.74884842]
        # Std: [28.43763701 15.26303392  5.22472751]
        transforms.Normalize((0.0,), (1.0,))
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img

def resize(img, mask,size,polar=False):
    if polar:
        img,mask = cartesian_to_polar(img,mask)
    img = img.resize((size, size), Image.BILINEAR)
    mask = mask.resize((size, size), Image.NEAREST)
    return img, mask

# class torch_resize(object):
#     def __init__(self, size):
#         self.size = size
#
#     def __call__(self, image, target):
#         size = self.size
#         # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
#         image = F.resize(image, [size,size])
#         # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
#         # 如果是之前的版本需要使用PIL.Image.NEAREST
#         target = F.resize(target, [size,size], interpolation=T.InterpolationMode.NEAREST)
#         return image, target

def randomresize(img, mask, base_size, ratio_range):
    w, h = img.size
    long_side = random.randint(int(base_size * ratio_range[0]), int(base_size * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

def test_blur(img,sigma=0.0):
    img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img



# def cutout(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
#            ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
#     if random.random() < p:
#         img = np.array(img)
#         mask = np.array(mask)
#
#         img_h, img_w, img_c = img.shape
#
#         while True:
#             size = np.random.uniform(size_min, size_max) * img_h * img_w
#             ratio = np.random.uniform(ratio_1, ratio_2)
#             erase_w = int(np.sqrt(size / ratio))
#             erase_h = int(np.sqrt(size * ratio))
#             x = np.random.randint(0, img_w)
#             y = np.random.randint(0, img_h)
#
#             if x + erase_w <= img_w and y + erase_h <= img_h:
#                 break
#
#         if pixel_level:
#             value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
#         else:
#             value = np.random.uniform(value_min, value_max)
#
#         img[y:y + erase_h, x:x + erase_w] = value
#         mask[y:y + erase_h, x:x + erase_w] = 255
#
#         img = Image.fromarray(img.astype(np.uint8))
#         mask = Image.fromarray(mask.astype(np.uint8))
#
#     return img, mask

# def cutout(img, mask, p=0.5, value_min=0, value_max=255, pixel_level=True):
#     if np.random.random() < p:
#         img = np.array(img)
#         mask = np.array(mask)
#
#         # 找到mask中像素值为2的部分
#         mask_2 = (mask == 2).astype(np.uint8)
#
#         # 向内腐蚀50个像素
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
#         mask_2_eroded = cv2.erode(mask_2, kernel)
#
#         # 计算类别2的边缘（原始的类别2减去腐蚀后的结果）
#         mask_edge = mask_2 - mask_2_eroded
#         edge_y, edge_x = np.where(mask_edge > 0)
#
#         # 将边缘掩盖掉
#         for y, x in zip(edge_y, edge_x):
#             if pixel_level:
#                 value = np.random.uniform(value_min, value_max, img[y, x].shape)
#             else:
#                 value = np.random.uniform(value_min, value_max)
#             img[y, x] = value
#             mask[y, x] = 255
#
#         img = Image.fromarray(img.astype(np.uint8))
#         mask = Image.fromarray(mask.astype(np.uint8))
#
#     return img, mask


# v5
# def cutout(img, mask, p=0.5, value_min=0, value_max=255, pixel_level=True):
#     if np.random.random() < p:
#         img = np.array(img)
#         mask = np.array(mask)
#
#         # 找到mask中像素值为2的部分
#         mask_2 = (mask == 2).astype(np.uint8)
#
#         # 向内腐蚀50个像素
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
#         mask_2_eroded = cv2.erode(mask_2, kernel)
#
#         # 计算类别2的边缘（原始的类别2减去腐蚀后的结果）
#         mask_edge = mask_2 - mask_2_eroded
#
#         # 找到mask中像素值为1的部分
#         mask_1 = (mask == 1).astype(np.uint8)
#
#         # 获取mask类别1的区域的边界
#         contours, _ = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         x, y, w, h = cv2.boundingRect(contours[0])
#
#         # 从img中裁剪mask类别1对应的区域
#         patch = img[y:y + h, x:x + w]
#
#         # 创建一个和img大小相同的空图像
#         patch_resized = np.zeros_like(img)
#
#         # 将裁剪出的patch放大/缩小到和被掩盖部分的边缘相同的大小
#         patch_resized_edge = cv2.resize(patch, (mask_edge.shape[1], mask_edge.shape[0]))
#
#         # 使用mask_edge作为模板，将patch_resized_edge粘贴到patch_resized上
#         patch_resized[mask_edge > 0] = patch_resized_edge[mask_edge > 0]
#
#         # 将patch_resized的内容粘贴到img上
#         img[mask_edge > 0] = patch_resized[mask_edge > 0]
#
#         # 将mask的被掩盖部分填充为1
#         mask[mask_edge > 0] = 1
#
#         img = Image.fromarray(img.astype(np.uint8))
#         mask = Image.fromarray(mask.astype(np.uint8))
#
#     return img, mask

# v6
# def cutout(img, mask, p=0.5):
#     if np.random.random() < p:
#         factor = np.random.uniform(0.3, 1.7)
#         enhancer = ImageEnhance.Brightness(img)
#         img = enhancer.enhance(factor)
#     return img, mask

def cutout(source_img, template_img_path='', p=0.5):
    if np.random.random() < p:
        # Convert PIL Image to NumPy array
        source_np = np.array(source_img)

        template_img_path = '/root/autodl-tmp/data/REFUGE/images_cropped/T0062.jpg'
        template = cv2.imread(template_img_path)

        def hist_match(source, template):
            src_hist, bin_edges = np.histogram(source.ravel(), bins=256, density=True)
            src_cdf = src_hist.cumsum()
            src_cdf /= src_cdf[-1]

            tgt_hist, _ = np.histogram(template.ravel(), bins=256, density=True)
            tgt_cdf = tgt_hist.cumsum()
            tgt_cdf /= tgt_cdf[-1]

            inverse_cdf = np.interp(src_cdf, tgt_cdf, bin_edges[:-1])

            return inverse_cdf[source].reshape(source.shape)

        matched_channels = []
        for d in range(source_np.shape[2]):
            matched_c = hist_match(source_np[:, :, d], template[:, :, d])
            matched_channels.append(matched_c)

        matched_np = cv2.merge(matched_channels).astype(np.uint8)

        # Convert the NumPy array back to a PIL Image
        matched_img = Image.fromarray(matched_np)

        return matched_img
    else:
        return source_img
