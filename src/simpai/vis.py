"""
Visualization utilities for displaying images.

This module provides functions to display images in various tensor/array formats
commonly used in deep learning workflows (BHWC, BCHW, HWC, CHW, HW, BHW).

All functions support both numpy.ndarray and torch.Tensor inputs.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typeguard import typechecked


@typechecked
def plot_bhwc(
    img: np.ndarray | torch.Tensor,
    nrows: int                              = -1,
    ncols: int                              = -1,
    title: tuple[str, ...]                  = tuple(),
    figsize: tuple[int|float,int|float]     = (-1., -1.),
    save_as: str                            = str(),
    save_dpi: int                           = 300,
    show: bool                              = True,
) -> None:
    """
    Plot a batch of images in BHWC format in a grid layout.

    Args:
        img: Image batch with shape (B, H, W, C). Supports both numpy.ndarray
            and torch.Tensor. C can be 1 (grayscale), 3 (RGB), or 4 (RGBA).
        nrows: Number of rows in the grid. If -1, automatically calculated.
            Default is -1.
        ncols: Number of columns in the grid. If -1, automatically calculated.
            Default is -1.
        title: Tuple of titles for each image. Length must match batch size B.
            Default is empty tuple (no titles).
        figsize: Figure size as (width, height). If either value is <= 0,
            automatically calculated based on image dimensions. Default is (-1, -1).
        save_as: File path to save the figure. If empty string, figure is not saved.
            Default is empty string.
        save_dpi: DPI (dots per inch) for saving the figure. Default is 300.
        show: Whether to display the figure. Default is True.

    Raises:
        ValueError: If img does not have 4 dimensions.
        ValueError: If number of titles does not match batch size.
        ValueError: If grid size (nrows * ncols) is smaller than batch size.

    Note:
        If both nrows and ncols are -1, the grid dimensions are automatically
        calculated to form a roughly square layout.
    """
    # Convert torch.Tensor to numpy array
    if isinstance(img, torch.Tensor):
        img = img.detach().clone().to('cpu').numpy()

    # Validate input dimensions
    if img.ndim != 4:
        raise ValueError(f'img must have shape (B,H,W,C), but got {img.shape}')

    b, h, w, c = img.shape

    # Validate title count matches batch size
    if len(title) > 0 and len(title) != b:
        raise ValueError(f'{b} image(s) but {len(title)} titles')

    # Calculate grid dimensions if not specified
    if nrows < 0 and ncols < 0:
        ncols = int(np.ceil(np.sqrt(b).item()).item())
        nrows = int(np.ceil(b / ncols).item())
    elif nrows < 0:
        nrows = int(np.ceil(b / ncols).item())
    elif ncols < 0:
        ncols = int(np.ceil(b / nrows).item())
    elif nrows * ncols < b:
        raise ValueError(f'nrows = {nrows}, ncols = {ncols}, but there\'re {b} image(s)')

    print(f'nrows = {nrows}, ncols = {ncols}')

    # Calculate figure size
    figsize = (
        (figsize[0] if figsize[0] > 0 else w / 100) * ncols,
        (figsize[1] if figsize[1] > 0 else h / 100) * nrows,
    )

    # Create subplot grid
    _, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize, squeeze = False)

    # Plot each image
    for i, ax in enumerate(axes.flatten()):
        if i < b:
            if c == 1:
                ax.imshow(img[i], cmap = 'gray')
            else:
                ax.imshow(img[i])

            if len(title) != 0:
                ax.set_title(title[i])

        ax.axis('off')

    plt.tight_layout()

    # Save figure if path is specified
    if len(save_as) != 0:
        plt.savefig(save_as, dpi = save_dpi, bbox_inches = 'tight')

    # Display figure if requested
    if show:
        plt.show()

    # Warn if figure is neither shown nor saved
    if (not show) and (len(save_as) == 0):
        print('plot_bhwc(): [WARNING] Ploted images aren\'t showed or saved.')


@typechecked
def show_bhw(
    img: np.ndarray | torch.Tensor,
    title: tuple[str, ...]                  = tuple(),
    figsize: tuple[int|float, int|float]    = (-1, -1),
) -> None:
    """
    Display a batch of grayscale images in BHW format.

    This function adds a channel dimension and delegates to plot_bhwc.

    Args:
        img: Grayscale image batch with shape (B, H, W). Supports both
            numpy.ndarray and torch.Tensor.
        title: Tuple of titles for each image. Length must match batch size B.
            Default is empty tuple (no titles).
        figsize: Figure size as (width, height). If either value is <= 0,
            automatically calculated based on image dimensions. Default is (-1, -1).

    Raises:
        ValueError: If img does not have 3 dimensions.
    """
    # Validate input dimensions
    if img.ndim != 3:
        raise ValueError(f"img must have shape (B, H,W). Got {tuple(img.shape)}")

    # Convert torch.Tensor to numpy array
    if isinstance(img, torch.Tensor):
        img = img.detach().clone().to('cpu').numpy()

    # Add channel dimension and plot
    plot_bhwc(
        img     = np.expand_dims(img, axis = (3,)),
        title   = title,
        figsize = figsize,
        show    = True,
    )


@typechecked
def show_hw(
    img: np.ndarray | torch.Tensor,
    title: str                              = str(),
    figsize: tuple[int|float, int|float]    = (-1, -1),
) -> None:
    """
    Display a single grayscale image in HW format.

    This function adds batch and channel dimensions and delegates to plot_bhwc.

    Args:
        img: Grayscale image with shape (H, W). Supports both numpy.ndarray
            and torch.Tensor.
        title: Title for the image. Default is empty string (no title).
        figsize: Figure size as (width, height). If either value is <= 0,
            automatically calculated based on image dimensions. Default is (-1, -1).

    Raises:
        ValueError: If img does not have 2 dimensions.
    """
    # Validate input dimensions
    if img.ndim != 2:
        raise ValueError(f"img must have shape (H,W). Got {tuple(img.shape)}")

    # Convert torch.Tensor to numpy array
    if isinstance(img, torch.Tensor):
        img = img.detach().clone().to('cpu').numpy()

    # Add batch and channel dimensions, then plot
    plot_bhwc(
        img     = np.expand_dims(img, axis = (0, 3)),
        title   = (title, ),
        figsize = figsize,
        show    = True,
    )


@typechecked
def show_bhwc(
    img: np.ndarray | torch.Tensor,
    title: tuple[str, ...]                  = tuple(),
    figsize: tuple[int|float, int|float]    = (-1, -1),
) -> None:
    """
    Display a batch of images in BHWC format.

    This function delegates directly to plot_bhwc.

    Args:
        img: Image batch with shape (B, H, W, C). Supports both numpy.ndarray
            and torch.Tensor. C can be 1 (grayscale), 3 (RGB), or 4 (RGBA).
        title: Tuple of titles for each image. Length must match batch size B.
            Default is empty tuple (no titles).
        figsize: Figure size as (width, height). If either value is <= 0,
            automatically calculated based on image dimensions. Default is (-1, -1).
    """
    plot_bhwc(
        img     = img,
        title   = title,
        figsize = figsize,
        show    = True,
    )


@typechecked
def show_hwc(
    img: np.ndarray | torch.Tensor,
    title: str                              = str(),
    figsize: tuple[int|float, int|float]    = (-1, -1),
) -> None:
    """
    Display a single image in HWC format.

    This function adds a batch dimension and delegates to plot_bhwc.

    Args:
        img: Image with shape (H, W, C). Supports both numpy.ndarray
            and torch.Tensor. C can be 1 (grayscale), 3 (RGB), or 4 (RGBA).
        title: Title for the image. Default is empty string (no title).
        figsize: Figure size as (width, height). If either value is <= 0,
            automatically calculated based on image dimensions. Default is (-1, -1).

    Raises:
        ValueError: If img does not have 3 dimensions.
    """
    # Validate input dimensions
    if img.ndim != 3:
        raise ValueError(f"img must have shape (H,W,C). Got {tuple(img.shape)}")

    # Convert torch.Tensor to numpy array
    if isinstance(img, torch.Tensor):
        img = img.detach().clone().to('cpu').numpy()

    # Add batch dimension and plot
    plot_bhwc(
        img     = np.expand_dims(img, axis = (0,)),
        title   = (title, ),
        figsize = figsize,
        show    = True,
    )


@typechecked
def show_bchw(
    img: np.ndarray | torch.Tensor,
    title: tuple[str, ...]                  = tuple(),
    figsize: tuple[int|float, int|float]    = (-1, -1),
) -> None:
    """
    Display a batch of images in BCHW format.

    This function transposes from BCHW to BHWC format and delegates to plot_bhwc.

    Args:
        img: Image batch with shape (B, C, H, W). Supports both numpy.ndarray
            and torch.Tensor. C can be 1 (grayscale), 3 (RGB), or 4 (RGBA).
        title: Tuple of titles for each image. Length must match batch size B.
            Default is empty tuple (no titles).
        figsize: Figure size as (width, height). If either value is <= 0,
            automatically calculated based on image dimensions. Default is (-1, -1).
    """
    if isinstance(img, np.ndarray):
        # Transpose from (B, C, H, W) to (B, H, W, C)
        plot_bhwc(
            img     = img.transpose(0, 2, 3, 1),
            title   = title,
            figsize = figsize,
            show    = True,
        )
    else:
        # `type(img)` is <class 'torch.Tensor'>
        # Permute from (B, C, H, W) to (B, H, W, C)
        plot_bhwc(
            img     = img.permute(0, 2, 3, 1),
            title   = title,
            figsize = figsize,
            show    = True,
        )


@typechecked
def show_chw(
    img: np.ndarray | torch.Tensor,
    title: str                              = str(),
    figsize: tuple[int|float, int|float]    = (-1, -1),
) -> None:
    """
    Display a single image in CHW format.

    This function transposes from CHW to HWC format, adds a batch dimension,
    and delegates to plot_bhwc.

    Args:
        img: Image with shape (C, H, W). Supports both numpy.ndarray
            and torch.Tensor. C can be 1 (grayscale), 3 (RGB), or 4 (RGBA).
        title: Title for the image. Default is empty string (no title).
        figsize: Figure size as (width, height). If either value is <= 0,
            automatically calculated based on image dimensions. Default is (-1, -1).
    """
    if isinstance(img, np.ndarray):
        # Transpose from (C, H, W) to (H, W, C), add batch dimension, and plot
        plot_bhwc(
            img     = np.expand_dims(img.transpose(1, 2, 0), axis = (0, )),
            title   = (title, ),
            figsize = figsize,
            show    = True,
        )
    else:
        # `type(img)` is <class 'torch.Tensor'>
        # Permute from (C, H, W) to (H, W, C), add batch dimension, and plot
        plot_bhwc(
            img     = torch.unsqueeze(img.permute(1, 2, 0), dim = 0),
            title   = (title, ),
            figsize = figsize,
            show    = True,
        )


if __name__ == '__main__':
    from simpai import data
    show_chw(torch.rand(3, 90, 100))
