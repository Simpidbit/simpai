import torch
import numpy as np
import matplotlib.pyplot as plt

def show_hw_ndarray(img:np.ndarray,
                    title:str           = '',
                    figsize:tuple       = (-1, -1)) -> None:
    # 判断类型是否正确
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img must be a np.ndarray, got {type(img)}")

    # 判断维度是否正确
    if img.ndim != 2:
        raise ValueError(f"img must have shape (H,W). Got {tuple(img.shape)}")

    h, w = img.shape
    if figsize[0] == -1 and figsize[1] == -1:
        width = w / 100
        height = h / 100
        figsize = (width, height)

    plt.figure(figsize = figsize)
    if title != '':
        plt.title(title)
    plt.axis("off")

    plt.imshow(img, cmap = "gray", vmin = 0.0, vmax = 1.0)
    plt.show()

def show_hwc_ndarray(img:np.ndarray,
                     title:str          = '', 
                     figsize:tuple      = (-1, -1)) -> None:
    # 判断类型是否正确
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img must be a np.ndarray, got {type(img)}")

    # 判断维度是否正确
    if img.ndim != 3:
        raise ValueError(f"img must have shape (H,W,C). Got {tuple(img.shape)}")

    h, w, c = img.shape
    if c not in (1, 3, 4):
        raise ValueError(f"Channel C must be 1, 3, or 4. Got C = {c}")


    if figsize[0] == -1 and figsize[1] == -1:
        width = w / 100
        height = h / 100
        figsize = (width, height)
    
    plt.figure(figsize = figsize)
    if title != '':
        plt.title(title)
    plt.axis("off")

    if c == 1:
        plt.imshow(img[..., 0], cmap = "gray", vmin = 0.0, vmax = 1.0)
    else:
        plt.imshow(img)  # RGB/RGBA
    plt.show()

def show_chw_ndarray(img:np.ndarray,
                     title:str          = '', 
                     figsize:tuple      = (-1, -1)) -> None:
    show_hwc_ndarray(np.transpose(img, (1, 2, 0)), title, figsize)

def show_chw_tensor(img:torch.Tensor,
                    title:str           = '', 
                    figsize:tuple       = (-1, -1)) -> None:
    # 判断是否是张量
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"img must be a torch.Tensor, got {type(img)}")
    x = img.detach().to("cpu").float().permute(1, 2, 0).numpy()
    show_hwc_ndarray(x, title, figsize)

def show_hwc_tensor(img:torch.Tensor,
                    title:str           = '', 
                    figsize:tuple       = (-1, -1)) -> None:
    # 判断是否是张量
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"img must be a torch.Tensor, got {type(img)}")
    x = img.detach().to("cpu").float().numpy()
    show_hwc_ndarray(x, title, figsize)

