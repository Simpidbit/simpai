import numpy as np
from PIL import Image
import random
import os
import pickle

import torch

def rgb_to_ycbcr_ndarray(img:np.ndarray) -> np.ndarray:
    '''
    img: RGB, (H,W,3), 0-1
    '''
    matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])
    ycbcr = np.dot(img, matrix.T)
    ycbcr[:, :, [1, 2]] += 0.5
    return ycbcr

def ycbcr_to_rgb_ndarray(img:np.ndarray) -> np.ndarray:
    '''
    img: YCbCr, (H,W,3), 0-1
    '''
    matrix = np.array([
        [1., 0., 1.402],
        [1., -0.344106, -0.714141],
        [1., 1.772, 0.]
    ])
    rgb = img.copy()
    rgb[:, :, [1, 2]] -= 0.5
    rgb = np.dot(rgb, matrix.T)
    return np.clip(rgb, 0, 1)

def rgb_to_ycbcr_tensor(img:torch.Tensor) -> torch.Tensor:
    '''
    img: (B,3,H,W)
    '''
    r = img[:, 0, :, :]
    g = img[:, 1, :, :]
    b = img[:, 2, :, :]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.stack([y, cb, cr], dim = 1)


def filepath_to_ndarray(
    filepath:str,
    transpose:str = 'hwc'
) -> np.ndarray:
    with Image.open(filepath).convert('RGB') as img_file:
        img = np.array(img_file, dtype = 'uint8')
    if transpose == 'hwc':
        pass
    elif transpose == 'chw':
        img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img)


def filepath_to_ndarray_enhancement(
    filepath:str,
    dtype:str           = 'float32',
    enhancement_enable  = True,         # 是否做数据增强（总开关）
    resize_enable:bool  = True,         # Resize
    resize_shape:tuple  = (500, 500),
    crop_enable:bool    = True,         # 随机裁剪
    crop_patch_size:int = 96,
    flip_lr_enable:bool = True,         # 左右翻转
    flip_lr_prob:float  = 0.5,
    flip_ud_enable:bool = True,         # 上下翻转
    flip_ud_prob:float  = 0.5,
    rot_enable:bool     = True,         # 旋转
    rot_prob:float      = 0.5,
    transpose:str       = 'hwc'         # 转换成什么形状
) -> np.ndarray:
    """
    读取图像文件，进行预处理和数据增强，并将其转换为 NumPy 数组。

    该函数读取指定路径的图像，转换为 RGB 模式，并将像素值归一化到 [0, 1] 区间。
    根据 `enhancement_enable` 总开关及各项子配置，函数可执行调整大小 (Resize)、
    随机裁剪 (Random Crop)、随机翻转 (Flip) 和随机旋转 (Rotate) 操作。

    Args:
        filepath (str): 图像文件的路径。
        dtype (str, optional): 输出 NumPy 数组的数据类型。默认为 'float32'。
        enhancement_enable (bool, optional): 数据增强的总开关。
            如果为 False，则 resize、crop、flip 和 rotation 均不会执行。默认为 True。
        resize_enable (bool, optional): 是否启用调整图像大小。需 enhancement_enable 为 True 才生效。默认为 True。
        resize_shape (tuple, optional): 调整后的图像尺寸 (width, height)。默认为 (500, 500)。
        crop_enable (bool, optional): 是否启用随机裁剪。需 enhancement_enable 为 True 才生效。默认为 True。
        crop_patch_size (int, optional): 随机裁剪的正方形边长。默认为 96。
        flip_lr_enable (bool, optional): 是否启用随机左右（水平）翻转。需 enhancement_enable 为 True 才生效。默认为 True。
        flip_lr_prob (float, optional): 触发左右翻转的概率 (0.0 - 1.0)。默认为 0.5。
        flip_ud_enable (bool, optional): 是否启用随机上下（垂直）翻转。需 enhancement_enable 为 True 才生效。默认为 True。
        flip_ud_prob (float, optional): 触发上下翻转的概率 (0.0 - 1.0)。默认为 0.5。
        rot_enable (bool, optional): 是否启用随机旋转 (90/180/270度)。需 enhancement_enable 为 True 才生效。默认为 True。
        rot_prob (float, optional): 触发旋转的概率 (0.0 - 1.0)。默认为 0.5。
        transpose (str, optional): 输出数组的维度顺序。
            'hwc': (高度, 宽度, 通道) - 默认值。
            'chw': (通道, 高度, 宽度) - 常用于 PyTorch 等框架。

    Returns:
        np.ndarray: 处理后的图像数组。
            - 值范围：[0.0, 1.0] (已除以 255.0)。
            - 内存布局：连续 (Contiguous)。
    """

    with Image.open(filepath).convert('RGB') as img_file:
        if resize_enable and enhancement_enable:
            img_file = img_file.resize(resize_shape)
        w, h = img_file.size
        if crop_enable and enhancement_enable:
            if crop_patch_size <= 0:
                raise ValueError(f'crop_patch_size is illegal: {crop_patch_size}')
            if w < crop_patch_size or h < crop_patch_size:
                raise ValueError(f'crop_patch_size: {crop_patch_size} but w, h is {w}, {h}')
            x = random.randint(0, w - crop_patch_size)
            y = random.randint(0, h - crop_patch_size)
            img_file = img_file.crop((x, y, x + crop_patch_size, y + crop_patch_size))

        img = np.array(img_file, dtype = dtype) / 255.0

    if enhancement_enable:
        if not 0 <= flip_lr_prob <= 1:
            raise ValueError(f'flip_lr_prob is {flip_lr_prob}, but it should be a probability!')
        if not 0 <= flip_ud_prob <= 1:
            raise ValueError(f'flip_ud_prob is {flip_ud_prob}, but it should be a probability!')
        if not 0 <= rot_prob <= 1:
            raise ValueError(f'rot_prob is {rot_prob}, but it should be a probability!')
        if flip_lr_enable and random.random() < flip_lr_prob:
            img = np.fliplr(img)
        if flip_ud_enable and random.random() < flip_ud_prob:
            img = np.flipud(img)
        if rot_enable and random.random() < rot_prob:
            img = np.rot90(img, random.randint(1, 3))

    if transpose == 'hwc':
        pass
    elif transpose == 'chw':
        img = np.transpose(img, (2, 0, 1))

    # Make sure that the memory is continuous
    return np.ascontiguousarray(img)

def load_or_build(filepath:str, build_func):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        obj = build_func()
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        return obj
