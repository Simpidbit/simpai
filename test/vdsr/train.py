from simpai import hyperparam as hp
from simpai import constant as cst
from simpai import data as dt
from simpai.trainer import Trainer
from simpai import vis

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import os

# 超参数
hp.set_hp_begin()
hp.set_hp('device', torch.device('cuda'))
hp.set_hp('epoch_num', 100)
hp.set_hp('batch_size', 20)
hp.set_hp('dataset_dir', '/media/simpidbit/Home/BaiduNetdiskDownload/DIV2K')
hp.set_hp('patch_size', 100)
hp.set_hp('loss_fn', torch.nn.MSELoss())
hp.set_hp_end()

# 网络结构
from vdsr import VDSR
model = VDSR(1).to(hp.get_hp('device'))
hp.set_hp_begin()
hp.set_hp('optimizer', torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 1e-4))
hp.set_hp_end()

# 数据集
from div2k import SP_DIV2K_Dataset
train_HR_path = f'{hp.get_hp("dataset_dir")}{cst.SEP}DIV2K_train_HR'
test_HR_path = f'{hp.get_hp("dataset_dir")}{cst.SEP}DIV2K_valid_HR'
train_LR_path = f'{hp.get_hp("dataset_dir")}{cst.SEP}DIV2K_train_LR'
test_LR_path = f'{hp.get_hp("dataset_dir")}{cst.SEP}DIV2K_valid_LR'

train_dataset = SP_DIV2K_Dataset(train_HR_path, train_LR_path, patch_size = hp.get_hp('patch_size'), is_train = True)
test_dataset  = SP_DIV2K_Dataset(test_HR_path,  test_LR_path,  patch_size = hp.get_hp('patch_size'), is_train = False)
train_loader  = DataLoader(train_dataset, batch_size = hp.get_hp('batch_size'), shuffle = True,
                           num_workers = 20,
                           pin_memory = True,
                           persistent_workers = True)
test_loader   = DataLoader(test_dataset, batch_size = hp.get_hp('batch_size'), shuffle = True,
                           num_workers = 20,
                           pin_memory = True,
                           persistent_workers = True)

# 训练
trainer = Trainer(model, train_loader)

#hp.get_hp('optimizer').load_state_dict(trainer.load_checkpoint('checkpoint.pt', hp.get_hp('device')))

psnr_metric = PeakSignalNoiseRatio(data_range = 1.0).to(hp.get_hp('device'))
ssim_metric = StructuralSimilarityIndexMeasure(data_range = 1.0).to(hp.get_hp('device'))

@trainer.set_step
def step_fn(epoch_idx, model, tqdm, data):
    x, truth_y = data

    x = x.to(hp.get_hp('device'))
    truth_y = truth_y.to(hp.get_hp('device'))

    x = dt.rgb_to_ycbcr_tensor(x)[:, 0, :, :].unsqueeze(1)
    truth_y = dt.rgb_to_ycbcr_tensor(truth_y)[:, 0, :, :].unsqueeze(1)

    predict_y = model(x)

    loss = hp.get_hp('loss_fn')(predict_y, truth_y)

    hp.get_hp('optimizer').zero_grad()
    loss.mean().backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        0.2 / hp.get_hp('optimizer').param_groups[0]['lr']
    )

    hp.get_hp('optimizer').step()

    return loss.mean()

@trainer.set_eval
def eval_fn(epoch_idx, model, tqdm):
    model.to(hp.get_hp('device'))
    model.eval()

    total_loss = 0.
    total_psnr = 0.
    total_ssim = 0.
    for x, truth_y in test_loader:
        x = x.to(hp.get_hp('device'))
        truth_y = truth_y.to(hp.get_hp('device'))

        x = dt.rgb_to_ycbcr_tensor(x)[:, 0, :, :].unsqueeze(1)
        truth_y = dt.rgb_to_ycbcr_tensor(truth_y)[:, 0, :, :].unsqueeze(1)

        with torch.no_grad():
            predict_y = model(x)

        total_psnr += psnr_metric(truth_y, predict_y)
        total_ssim += ssim_metric(truth_y, predict_y)

        total_loss += hp.get_hp('loss_fn')(predict_y, truth_y).item()
    mean_loss = total_loss / len(test_loader)
    mean_psnr = total_psnr / len(test_loader)
    mean_ssim = total_ssim / len(test_loader)
    tqdm.write(f'Epoch {epoch_idx + 1}: loss = {mean_loss:.5g}, psnr = {mean_psnr:.5g}, ssim = {mean_ssim:.5g}')

trainer.train(hp.get_hp('epoch_num'), 'checkpoint.pt', hp.get_hp('optimizer'))
