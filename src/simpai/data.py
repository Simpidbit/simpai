from typing import Any
from threading import Thread
from queue import Queue
from copy import deepcopy

import numpy as np
import torch
from typeguard import typechecked

from simpai import logger

_simpai_data_ckpt_queue: Queue[tuple[str, Any]] = Queue(maxsize = 0)
def _simpai_data_worker() -> None:
    global _simpai_data_ckpt_queue
    while True:
        filename, meta_dict = _simpai_data_ckpt_queue.get(block = True)
        if meta_dict is None:
            # For `wait_for_ckpt_io`
            return
        torch.save(meta_dict, filename)
_simpai_data_worker_thread = Thread(target = _simpai_data_worker, daemon = True)
_simpai_data_worker_thread.start()

def rgb_to_ycbcr_ndarray(img:np.ndarray) -> np.ndarray:
    """
    Convert RGB image to YCbCr color space.

    Args:
        img: RGB image as numpy array with shape (H, W, 3).
            Pixel values should be in the range [0, 1].

    Returns:
        np.ndarray: YCbCr image with the same shape (H, W, 3).
            - Channel 0: Y (luminance)
            - Channel 1: Cb (chrominance blue)
            - Channel 2: Cr (chrominance red)
    """
    # Transformation matrix for RGB to YCbCr conversion
    matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])
    # Apply matrix multiplication to convert color space
    ycbcr = np.dot(img, matrix.T)
    # Offset Cb and Cr channels to [0, 1] range
    ycbcr[:, :, [1, 2]] += 0.5
    return ycbcr

def ycbcr_to_rgb_ndarray(img:np.ndarray) -> np.ndarray:
    """
    Convert YCbCr image to RGB color space.

    Args:
        img: YCbCr image as numpy array with shape (H, W, 3).
            Pixel values should be in the range [0, 1].

    Returns:
        np.ndarray: RGB image with the same shape (H, W, 3).
            Pixel values are clipped to [0, 1] range.
    """
    # Transformation matrix for YCbCr to RGB conversion
    matrix = np.array([
        [1., 0., 1.402],
        [1., -0.344106, -0.714141],
        [1., 1.772, 0.]
    ])
    # Create a copy to avoid modifying the input
    rgb = img.copy()
    # Remove the offset from Cb and Cr channels
    rgb[:, :, [1, 2]] -= 0.5
    # Apply matrix multiplication to convert color space
    rgb = np.dot(rgb, matrix.T)
    # Clip values to valid [0, 1] range
    return np.clip(rgb, 0, 1)

def rgb_to_ycbcr_tensor(img:torch.Tensor) -> torch.Tensor:
    """
    Convert RGB tensor to YCbCr color space.

    Args:
        img: RGB image as PyTorch tensor with shape (B, 3, H, W).
            - B: batch size
            - 3: RGB channels
            - H: height
            - W: width

    Returns:
        torch.Tensor: YCbCr tensor with the same shape (B, 3, H, W).
            - Channel 0: Y (luminance)
            - Channel 1: Cb (chrominance blue)
            - Channel 2: Cr (chrominance red)
    """
    # Extract individual color channels
    r = img[:, 0, :, :]
    g = img[:, 1, :, :]
    b = img[:, 2, :, :]
    # Calculate Y (luminance) channel
    y = 0.299 * r + 0.587 * g + 0.114 * b
    # Calculate Cb (chrominance blue) channel with offset
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    # Calculate Cr (chrominance red) channel with offset
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    # Stack channels back into tensor
    return torch.stack([y, cb, cr], dim = 1)


def _simpai_data_make_meta_dict_tuple(*args: tuple[str, Any], **kwargs) -> tuple[dict]:
    ckpt_meta_dict_list: list = list()
    for arg in args:
        if isinstance(arg[1], torch.nn.Module):
            ckpt_meta_dict_list.append({
                'typeid': 'torch.nn.Module',
                'meta_data': arg[1].state_dict()
            })
        elif isinstance(arg[1], torch.optim.Optimizer):
            ckpt_meta_dict_list.append({
                'typeid': 'torch.optim.Optimizer',
                'meta_data': arg[1].state_dict()
            })
    return tuple(ckpt_meta_dict_list)


@typechecked
def save_ckpt(*args: tuple[str, Any], **kwargs) -> None:
    ckpt_meta_dict_tuple = _simpai_data_make_meta_dict_tuple(*args, **kwargs)
    for i in range(len(ckpt_meta_dict_tuple)):
        torch.save(ckpt_meta_dict_tuple[i], args[i][0])

@typechecked
def save_ckpt_multithread(*args: tuple[str, Any], **kwargs) -> None:
    global _simpai_data_ckpt_queue
    ckpt_meta_dict_tuple = _simpai_data_make_meta_dict_tuple(*args, **kwargs)
    for i in range(len(ckpt_meta_dict_tuple)):
        _simpai_data_ckpt_queue.put((args[i][0], deepcopy(ckpt_meta_dict_tuple[i])))

@typechecked
def wait_for_ckpt_io() -> None:
    _simpai_data_ckpt_queue.put((str(), None))
    _simpai_data_worker_thread.join()

@typechecked
def load_ckpt(
        *args: str | tuple[str, torch.nn.Module | torch.optim.Optimizer], 
        device: torch.device = torch.device('cpu')
) -> tuple:
    result_list: list = list()
    for arg in args:
        if isinstance(arg, str):
            try:
                ckpt_meta_dict = torch.load(arg, map_location = device)
                result_list.append(ckpt_meta_dict)
            except FileNotFoundError:
                logger.warning(f'load_ckpt: {arg} is not found. Pass.')
        else:
            try:
                ckpt_meta_dict = torch.load(arg[0], map_location = device)
                # It may be torch.nn.Module or torch.optim.Optimizer!
                # But both have `load_state_dict` method.
                if isinstance(arg[1], torch.nn.Module):
                    arg[1].load_state_dict(ckpt_meta_dict['meta_data'], strict = False)
                else:
                    arg[1].load_state_dict(ckpt_meta_dict['meta_data'])
                result_list.append(ckpt_meta_dict)
            except FileNotFoundError:
                logger.warning(f'load_ckpt: {arg[0]} is not found. Pass.')

    return tuple(result_list)
