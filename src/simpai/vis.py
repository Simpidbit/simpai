import torch
import numpy as np
import matplotlib.pyplot as plt

def show_hw_ndarray(img:np.ndarray,
                    title:str           = '',
                    figsize:tuple       = (-1, -1)) -> None:
    """
    Display a 2D grayscale image from a numpy array.

    Args:
        img: Grayscale image as numpy array with shape (H, W).
            Values should be in the range [0, 1].
        title: Optional title for the plot. Default is empty string.
        figsize: Figure size as (width, height) tuple.
            If both values are -1, the size is automatically calculated
            based on image dimensions. Default is (-1, -1).

    Raises:
        TypeError: If img is not a numpy array.
        ValueError: If img does not have 2 dimensions.
    """
    # Validate input type
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img must be a np.ndarray, got {type(img)}")

    # Validate input dimensions
    if img.ndim != 2:
        raise ValueError(f"img must have shape (H,W). Got {tuple(img.shape)}")

    h, w = img.shape
    # Auto-calculate figure size if not specified
    if figsize[0] == -1 and figsize[1] == -1:
        width = w / 100
        height = h / 100
        figsize = (width, height)

    # Create and display the plot
    plt.figure(figsize = figsize)
    if title != '':
        plt.title(title)
    plt.axis("off")
    plt.imshow(img, cmap = "gray", vmin = 0.0, vmax = 1.0)
    plt.show()

def show_hwc_ndarray(img:np.ndarray,
                     title:str          = '',
                     figsize:tuple      = (-1, -1)) -> None:
    """
    Display an image from a numpy array with HWC (height, width, channels) format.

    Args:
        img: Image as numpy array with shape (H, W, C).
            C can be 1 (grayscale), 3 (RGB), or 4 (RGBA).
            Values should be in the range [0, 1].
        title: Optional title for the plot. Default is empty string.
        figsize: Figure size as (width, height) tuple.
            If both values are -1, the size is automatically calculated
            based on image dimensions. Default is (-1, -1).

    Raises:
        TypeError: If img is not a numpy array.
        ValueError: If img does not have 3 dimensions.
        ValueError: If channel count is not 1, 3, or 4.
    """
    # Validate input type
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img must be a np.ndarray, got {type(img)}")

    # Validate input dimensions
    if img.ndim != 3:
        raise ValueError(f"img must have shape (H,W,C). Got {tuple(img.shape)}")

    h, w, c = img.shape
    # Validate channel count
    if c not in (1, 3, 4):
        raise ValueError(f"Channel C must be 1, 3, or 4. Got C = {c}")

    # Auto-calculate figure size if not specified
    if figsize[0] == -1 and figsize[1] == -1:
        width = w / 100
        height = h / 100
        figsize = (width, height)

    # Create and display the plot
    plt.figure(figsize = figsize)
    if title != '':
        plt.title(title)
    plt.axis("off")

    # Display based on channel count
    if c == 1:
        plt.imshow(img[..., 0], cmap = "gray", vmin = 0.0, vmax = 1.0)
    else:
        plt.imshow(img)  # RGB/RGBA
    plt.show()

def show_chw_ndarray(img:np.ndarray,
                     title:str          = '',
                     figsize:tuple      = (-1, -1)) -> None:
    """
    Display an image from a numpy array with CHW (channels, height, width) format.

    This function transposes the array from CHW to HWC format and delegates
    to show_hwc_ndarray for display.

    Args:
        img: Image as numpy array with shape (C, H, W).
            C can be 1 (grayscale), 3 (RGB), or 4 (RGBA).
            Values should be in the range [0, 1].
        title: Optional title for the plot. Default is empty string.
        figsize: Figure size as (width, height) tuple.
            If both values are -1, the size is automatically calculated
            based on image dimensions. Default is (-1, -1).

    Note:
        See show_hwc_ndarray for detailed error conditions.
    """
    # Transpose from CHW to HWC format and display
    show_hwc_ndarray(np.transpose(img, (1, 2, 0)), title, figsize)

def show_chw_tensor(img:torch.Tensor,
                    title:str           = '',
                    figsize:tuple       = (-1, -1)) -> None:
    """
    Display an image from a PyTorch tensor with CHW (channels, height, width) format.

    This function converts the tensor to a numpy array and delegates
    to show_hwc_ndarray for display.

    Args:
        img: Image as PyTorch tensor with shape (C, H, W).
            C can be 1 (grayscale), 3 (RGB), or 4 (RGBA).
        title: Optional title for the plot. Default is empty string.
        figsize: Figure size as (width, height) tuple.
            If both values are -1, the size is automatically calculated
            based on image dimensions. Default is (-1, -1).

    Raises:
        TypeError: If img is not a PyTorch tensor.

    Note:
        See show_hwc_ndarray for detailed error conditions after conversion.
    """
    # Validate input type
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"img must be a torch.Tensor, got {type(img)}")
    # Convert tensor to numpy array: detach from graph, move to CPU, permute to HWC
    x = img.detach().to("cpu").float().permute(1, 2, 0).numpy()
    show_hwc_ndarray(x, title, figsize)

def show_hwc_tensor(img:torch.Tensor,
                    title:str           = '',
                    figsize:tuple       = (-1, -1)) -> None:
    """
    Display an image from a PyTorch tensor with HWC (height, width, channels) format.

    This function converts the tensor to a numpy array and delegates
    to show_hwc_ndarray for display.

    Args:
        img: Image as PyTorch tensor with shape (H, W, C).
            C can be 1 (grayscale), 3 (RGB), or 4 (RGBA).
        title: Optional title for the plot. Default is empty string.
        figsize: Figure size as (width, height) tuple.
            If both values are -1, the size is automatically calculated
            based on image dimensions. Default is (-1, -1).

    Raises:
        TypeError: If img is not a PyTorch tensor.

    Note:
        See show_hwc_ndarray for detailed error conditions after conversion.
    """
    # Validate input type
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"img must be a torch.Tensor, got {type(img)}")
    # Convert tensor to numpy array: detach from graph, move to CPU
    x = img.detach().to("cpu").float().numpy()
    show_hwc_ndarray(x, title, figsize)

