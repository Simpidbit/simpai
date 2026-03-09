import os
import random

import torch
import numpy as np
from torch.utils.data import Dataset

from simpai import file
from simpai import logger
from simpai import visual
from simpai import hyperparam as hp

class Our485(Dataset):
    def __init__(self, is_train: bool) -> None:
        self.is_train: bool = is_train
        self.high_imgs: list[torch.Tensor] = list()
        self.low_imgs:  list[torch.Tensor] = list()
        self._read_imgs()

    def __len__(self) -> int:
        return len(self.high_imgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = os.urandom(1)[0]
        high_img_enhanced = file.img_enhancement_xxhw(
            self.high_imgs[idx],
            randomly_crop_patch = hp.get_hp('patch_size'),
            hflip = True,
            vflip = False,
            rot_prob = 0.5,
            seed = seed
        )

        low_img_enhanced = file.img_enhancement_xxhw(
            self.low_imgs[idx],
            randomly_crop_patch = hp.get_hp('patch_size'),
            hflip = True,
            vflip = False,
            rot_prob = 0.5,
            seed = seed
        )

        return low_img_enhanced, high_img_enhanced


    def _read_imgs(self) -> None:
        our485_dir = '/data/our485/'
        high_dir = our485_dir + 'high/'
        low_dir = our485_dir + 'low/'

        high_paths = list()
        low_paths = list()
        for f in os.listdir(high_dir):
            if f.split('.')[-1] == 'png':
                high_paths.append(high_dir + f)
                low_paths.append(low_dir + f)
        logger.debug(f'high_paths = {high_paths}')
        logger.debug(f'low_paths = {low_paths}')

        if self.is_train:
            sample = random.sample(
                range(len(high_paths)), 
                int(len(high_paths) * 0.9)
            )
        else:
            sample = random.sample(
                range(len(high_paths)),
                len(high_paths) - int(len(high_paths) * 0.9)
            )
        sample.sort()

        logger.info(f'Reading images...')
        self.high_imgs: list[torch.Tensor] = list()
        self.low_imgs:  list[torch.Tensor] = list()
        for i in range(len(high_paths)):
            if i == sample[0]:
                self.high_imgs.append(torch.from_numpy(
                    file.filepath_to_chw_rgb_1(high_paths[i])
                ))
                self.low_imgs.append(torch.from_numpy(
                    file.filepath_to_chw_rgb_1(low_paths[i])
                ))
                sample.pop(0)
                if len(sample) == 0: break
        logger.info('OK')

if __name__ == '__main__':
    hp.set_hp_begin()
    hp.set_hp('device', torch.device('cpu'))
    hp.set_hp('patch_size', 200)
    hp.set_hp_end()

    our485 = Our485(True)
    for _ in range(10):
        i = random.randint(0, len(our485.high_imgs) - 1)
        visual.show_bchw(np.array(list(
            our485[i]
        )))
    logger.wait_for_log_io()
