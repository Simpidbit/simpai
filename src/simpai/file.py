import os
import pickle
import random
import time

from typeguard import typechecked
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as TF

@typechecked
def filepath_to_hwc_rgba_255(filepath: str) -> np.ndarray:
    with Image.open(filepath).convert('RGBA') as img_file:
        img = np.array(img_file, dtype = np.uint8)
    return np.ascontiguousarray(img)

@typechecked
def filepath_to_hwc_rgba_1(filepath: str) -> np.ndarray:
    with Image.open(filepath).convert('RGBA') as img_file:
        img = np.array(img_file, dtype = np.float32)
        img /= 255.
    return np.ascontiguousarray(img)

@typechecked
def filepath_to_chw_rgba_255(filepath: str) -> np.ndarray:
    with Image.open(filepath).convert('RGBA') as img_file:
        img = np.array(img_file, dtype = np.uint8)
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img)

@typechecked
def filepath_to_chw_rgba_1(filepath: str) -> np.ndarray:
    with Image.open(filepath).convert('RGBA') as img_file:
        img = np.array(img_file, dtype = np.float32)
        img /= 255.
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img)

@typechecked
def filepath_to_hwc_rgb_255(filepath: str) -> np.ndarray:
    with Image.open(filepath).convert('RGB') as img_file:
        img = np.array(img_file, dtype = np.uint8)
    return np.ascontiguousarray(img)

@typechecked
def filepath_to_hwc_rgb_1(filepath: str) -> np.ndarray:
    with Image.open(filepath).convert('RGB') as img_file:
        img = np.array(img_file, dtype = np.float32)
        img /= 255.
    return np.ascontiguousarray(img)

@typechecked
def filepath_to_chw_rgb_255(filepath: str) -> np.ndarray:
    with Image.open(filepath).convert('RGB') as img_file:
        img = np.array(img_file, dtype = np.uint8)
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img)

@typechecked
def filepath_to_chw_rgb_1(filepath: str) -> np.ndarray:
    with Image.open(filepath).convert('RGB') as img_file:
        img = np.array(img_file, dtype = np.float32)
        img /= 255.
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img)

@typechecked
def filepath_to_hwc_ycbcr_255(filepath: str) -> np.ndarray:
    with Image.open(filepath).convert('YCbCr') as img_file:
        img = np.array(img_file, dtype = np.uint8)
    return np.ascontiguousarray(img)

@typechecked
def filepath_to_hwc_ycbcr_1(filepath: str) -> np.ndarray:
    with Image.open(filepath).convert('YCbCr') as img_file:
        img = np.array(img_file, dtype = np.float32)
        img /= 255.
    return np.ascontiguousarray(img)

@typechecked
def filepath_to_chw_ycbcr_255(filepath: str) -> np.ndarray:
    with Image.open(filepath).convert('YCbCr') as img_file:
        img = np.array(img_file, dtype = np.uint8)
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img)

@typechecked
def filepath_to_chw_ycbcr_1(filepath: str) -> np.ndarray:
    with Image.open(filepath).convert('YCbCr') as img_file:
        img = np.array(img_file, dtype = np.float32)
        img /= 255.
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img)

@typechecked
def img_enhancement_xxhw(
    img: np.ndarray | torch.Tensor,
    randomly_crop_patch: tuple[int, int] | int | None = None,
    hflip: bool = False,
    vflip: bool = False,
    rot_prob: float = 0.0,
    seed: float | int | None = None
) -> np.ndarray | torch.Tensor:
    h, w = img.shape[-2], img.shape[-1]
    if seed is not None:
        random.seed(seed)

    if randomly_crop_patch is not None:
        if isinstance(randomly_crop_patch, tuple):
            hn, wn = randomly_crop_patch
        else:
            hn = randomly_crop_patch
            wn = randomly_crop_patch
        h0 = random.randint(0, h - hn)
        w0 = random.randint(0, w - wn)
        if len(img.shape) == 2:
            img = img[h0 : h0 + hn, w0 : w0 + wn]
        elif len(img.shape) == 3:
            img = img[:, h0 : h0 + hn, w0 : w0 + wn]
        elif len(img.shape) == 4:
            img = img[:, :, h0 : h0 + hn, w0 : w0 + wn]

    ifrot = random.random() < rot_prob
    if isinstance(img, torch.Tensor): 
        if hflip and random.random() < 0.5: img = img.flip(dims = [-1])
        if vflip and random.random() < 0.5: img = img.flip(dims = [-2])
        if ifrot: img = torch.rot90(img, k = random.randint(1, 3), dims = (-2, -1))
    else:
        if hflip: img = np.flip(img, axis = -1)
        if vflip: img = np.flip(img, axis = -2)
        if ifrot: img = np.rot90(img, k = random.randint(1, 3), axes = (-2, -1))

    if seed is not None:
        random.seed(os.urandom(1)[0])
    return img


@typechecked
def filepath_to_ndarray_enhancement(
    filepath:str,
    dtype:str           = 'float32',
    enhancement_enable  = True,         # Master switch for data augmentation
    resize_enable:bool  = True,         # Enable resize operation
    resize_shape:tuple  = (500, 500),
    crop_enable:bool    = True,         # Enable random crop
    crop_patch_size:int = 96,
    flip_lr_enable:bool = True,         # Enable left-right flip
    flip_lr_prob:float  = 0.5,
    flip_ud_enable:bool = True,         # Enable up-down flip
    flip_ud_prob:float  = 0.5,
    rot_enable:bool     = True,         # Enable rotation
    rot_prob:float      = 0.5,
    transpose:str       = 'hwc'         # Output dimension order
) -> np.ndarray:
    """
    Load an image file with preprocessing and data augmentation capabilities.

    This function reads an image from the specified path, converts it to RGB mode,
    and normalizes pixel values to the [0, 1] range. Based on the `enhancement_enable`
    master switch and sub-configurations, the function can perform resize, random crop,
    random flip, and random rotation operations.

    Args:
        filepath: Path to the image file.
        dtype: Output numpy array data type. Default is 'float32'.
        enhancement_enable: Master switch for data augmentation.
            If False, resize, crop, flip, and rotation will not execute. Default is True.
        resize_enable: Whether to enable image resizing.
            Only effective when enhancement_enable is True. Default is True.
        resize_shape: Target image dimensions after resize (width, height). Default is (500, 500).
        crop_enable: Whether to enable random cropping.
            Only effective when enhancement_enable is True. Default is True.
        crop_patch_size: Side length of the square crop patch. Default is 96.
        flip_lr_enable: Whether to enable random left-right (horizontal) flip.
            Only effective when enhancement_enable is True. Default is True.
        flip_lr_prob: Probability of triggering left-right flip (0.0 - 1.0). Default is 0.5.
        flip_ud_enable: Whether to enable random up-down (vertical) flip.
            Only effective when enhancement_enable is True. Default is True.
        flip_ud_prob: Probability of triggering up-down flip (0.0 - 1.0). Default is 0.5.
        rot_enable: Whether to enable random rotation (90/180/270 degrees).
            Only effective when enhancement_enable is True. Default is True.
        rot_prob: Probability of triggering rotation (0.0 - 1.0). Default is 0.5.
        transpose: Dimension order of the output array.
            - 'hwc': (height, width, channels) - default
            - 'chw': (channels, height, width) - common for PyTorch and similar frameworks

    Returns:
        np.ndarray: Processed image array.
            - Value range: [0.0, 1.0] (normalized by dividing by 255.0)
            - Memory layout: contiguous

    Raises:
        ValueError: If crop_patch_size is invalid or exceeds image dimensions.
        ValueError: If any probability parameter is not in [0, 1] range.
    """

    # Open image file and convert to RGB mode
    with Image.open(filepath).convert('RGB') as img_file:
        # Apply resize if enabled
        if resize_enable and enhancement_enable:
            img_file = img_file.resize(resize_shape)
        w, h = img_file.size
        # Apply random crop if enabled
        if crop_enable and enhancement_enable:
            if crop_patch_size <= 0:
                raise ValueError(f'crop_patch_size is illegal: {crop_patch_size}')
            if w < crop_patch_size or h < crop_patch_size:
                raise ValueError(f'crop_patch_size: {crop_patch_size} but w, h is {w}, {h}')
            # Randomly select crop position
            x = random.randint(0, w - crop_patch_size)
            y = random.randint(0, h - crop_patch_size)
            img_file = img_file.crop((x, y, x + crop_patch_size, y + crop_patch_size))

        # Convert to numpy array and normalize to [0, 1]
        img = np.array(img_file, dtype = dtype) / 255.0

    # Apply data augmentation if enabled
    if enhancement_enable:
        # Validate probability parameters
        if not 0 <= flip_lr_prob <= 1:
            raise ValueError(f'flip_lr_prob is {flip_lr_prob}, but it should be a probability!')
        if not 0 <= flip_ud_prob <= 1:
            raise ValueError(f'flip_ud_prob is {flip_ud_prob}, but it should be a probability!')
        if not 0 <= rot_prob <= 1:
            raise ValueError(f'rot_prob is {rot_prob}, but it should be a probability!')
        # Apply left-right flip with probability
        if flip_lr_enable and random.random() < flip_lr_prob:
            img = np.fliplr(img)
        # Apply up-down flip with probability
        if flip_ud_enable and random.random() < flip_ud_prob:
            img = np.flipud(img)
        # Apply random rotation with probability
        if rot_enable and random.random() < rot_prob:
            img = np.rot90(img, random.randint(1, 3))

    # Transpose to channel-first format if requested
    if transpose == 'hwc':
        pass
    elif transpose == 'chw':
        img = np.transpose(img, (2, 0, 1))

    # Ensure contiguous memory layout
    return np.ascontiguousarray(img)

@typechecked
def load_or_build(filepath:str, build_func):
    """
    Load an object from a file if it exists, otherwise build it and save it.

    This utility function implements a lazy loading pattern with caching.
    If the file at the specified path exists, it loads and returns the object
    using pickle. Otherwise, it calls the build_func to create the object,
    saves it to the file, and returns it.

    Args:
        filepath: Path to the file where the object is stored/will be stored.
        build_func: Callable function that builds the object if the file doesn't exist.
            The function should take no arguments and return the object to be cached.

    Returns:
        The loaded or newly built object.
    """
    # Load from cache if file exists
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    # Build new object, cache it, and return
    else:
        obj = build_func()
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        return obj
