import numpy as np

import torch
from typeguard import typechecked

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


