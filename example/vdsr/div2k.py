import simpai

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors

import os
from PIL import Image

class SP_DIV2K_Dataset(Dataset):
    """
    PyTorch Dataset for loading DIV2K image pairs for super-resolution training.

    The DIV2K dataset is a high-quality image dataset commonly used for
    single image super-resolution tasks. This class loads paired high-resolution
    (HR) and low-resolution (LR) images with support for caching and data augmentation.

    Attributes:
        HR_filepaths: Sorted list of paths to HR images.
        LR_filepaths: Sorted list of paths to LR images.
        cache_rate: Fraction of dataset to cache in memory.
        cache_end: Index up to which images are cached.
        HR_files: Cached HR image data (as numpy arrays).
        LR_files: Cached LR image data (as numpy arrays).
        is_train: Whether this is a training dataset.
        patch_size: Size of random crop patches for training.
    """

    def __init__(self, HR_path, LR_path, patch_size = 64, is_train = True):
        """
        Initialize the DIV2K dataset.

        Args:
            HR_path: Path to directory containing high-resolution images.
            LR_path: Path to directory containing low-resolution images.
            patch_size: Size of random crop patches for training. Default is 64.
            is_train: Whether this dataset is for training. Default is True.
        """
        # Collect all PNG file paths
        self.HR_filepaths = [os.path.join(HR_path, filename) for filename in os.listdir(HR_path) if filename.endswith('.png')]
        self.LR_filepaths = [os.path.join(LR_path, filename) for filename in os.listdir(LR_path) if filename.endswith('.png')]

        # Ensure consistent ordering between HR and LR images
        self.HR_filepaths.sort()
        self.LR_filepaths.sort()

        # Configure caching
        self.cache_rate = 1.0
        self.cache_end = int(len(self.HR_filepaths) * self.cache_rate)

        # Load or build cached HR images
        self.HR_files = simpai.data.load_or_build(
            f'HR_files_{self.cache_end}_{is_train}.pkl',
            lambda: [
                simpai.data.filepath_to_ndarray(fp, transpose = 'chw')
                for fp in self.HR_filepaths[:self.cache_end]
            ]
        )
        # Load or build cached LR images
        self.LR_files = simpai.data.load_or_build(
            f'LR_files_{self.cache_end}_{is_train}.pkl',
            lambda: [
                simpai.data.filepath_to_ndarray(fp, transpose = 'chw')
                for fp in self.LR_filepaths[:self.cache_end]
            ]
        )
        self.is_train = is_train
        self.patch_size = patch_size
        print('SP_DIV2K_Dataset.__init__() is called!')

    def __len__(self):
        """
        Return the number of image pairs in the dataset.

        Returns:
            int: Number of HR/LR image pairs.
        """
        return len(self.HR_filepaths)

    def __getitem__(self, idx):
        """
        Get a single HR/LR image pair with optional augmentation.

        Args:
            idx: Index of the image pair to retrieve.

        Returns:
            tuple: (LR_tensor, HR_tensor) where both are PyTorch tensors
                with shape (channels, height, width) and dtype float32.
        """
        # Load images from cache or disk
        if idx < self.cache_end:
            HR_img = self.HR_files[idx]
            LR_img = self.LR_files[idx]
        else:
            HR_img = simpai.data.filepath_to_ndarray(self.HR_filepaths[idx], transpose = 'chw')
            LR_img = simpai.data.filepath_to_ndarray(self.LR_filepaths[idx], transpose = 'chw')

        # Convert to PyTorch tensors
        HR_tensor = torch.from_numpy(HR_img)
        LR_tensor = torch.from_numpy(LR_img)

        # Define augmentation transforms
        transforms = v2.Compose([
            v2.RandomCrop((self.patch_size, self.patch_size)),
            v2.RandomHorizontalFlip(p = 0.5),
            v2.ToDtype(torch.float32, scale = True)
        ])

        # Apply transforms consistently to both HR and LR images
        HR_tensor = tv_tensors.Image(HR_tensor)
        LR_tensor = tv_tensors.Image(LR_tensor)
        HR_tensor, LR_tensor = transforms(HR_tensor, LR_tensor)

        return LR_tensor, HR_tensor
