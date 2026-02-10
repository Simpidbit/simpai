import simpai

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors

import os
from PIL import Image

class SP_DIV2K_Dataset(Dataset):
    def __init__(self, HR_path, LR_path, patch_size = 64, is_train = True):
        self.HR_filepaths = [os.path.join(HR_path, filename) for filename in os.listdir(HR_path) if filename.endswith('.png')]
        self.LR_filepaths = [os.path.join(LR_path, filename) for filename in os.listdir(LR_path) if filename.endswith('.png')]

        self.HR_filepaths.sort()
        self.LR_filepaths.sort()

        self.cache_rate = 1.0
        self.cache_end = int(len(self.HR_filepaths) * self.cache_rate)

        self.HR_files = simpai.data.load_or_build(
            f'HR_files_{self.cache_end}_{is_train}.pkl',
            lambda: [
                simpai.data.filepath_to_ndarray(fp, transpose = 'chw')
                for fp in self.HR_filepaths[:self.cache_end]
            ]
        )
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
        return len(self.HR_filepaths)

    def __getitem__(self, idx):
        if idx < self.cache_end:
            HR_img = self.HR_files[idx]
            LR_img = self.LR_files[idx]
        else:
            HR_img = simpai.data.filepath_to_ndarray(self.HR_filepaths[idx], transpose = 'chw')
            LR_img = simpai.data.filepath_to_ndarray(self.LR_filepaths[idx], transpose = 'chw')

        HR_tensor = torch.from_numpy(HR_img)
        LR_tensor = torch.from_numpy(LR_img)

        transforms = v2.Compose([
            v2.RandomCrop((self.patch_size, self.patch_size)),
            v2.RandomHorizontalFlip(p = 0.5),
            v2.ToDtype(torch.float32, scale = True)
        ])

        HR_tensor = tv_tensors.Image(HR_tensor)
        LR_tensor = tv_tensors.Image(LR_tensor)
        HR_tensor, LR_tensor = transforms(HR_tensor, LR_tensor)

        return LR_tensor, HR_tensor
