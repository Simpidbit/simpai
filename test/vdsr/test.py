import torch
from vdsr import VDSR
from div2k import SP_DIV2K_Dataset
from simpai import constant as cst
from simpai import vis
from simpai import data as dt
from torch.utils.data import DataLoader

model = VDSR(1).to(torch.device('cuda'))
checkpoint = torch.load('checkpoint.pt', map_location = torch.device('cuda'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dataset_dir = '/media/simpidbit/Home/BaiduNetdiskDownload/DIV2K'
test_HR_path = f'{dataset_dir}{cst.SEP}DIV2K_valid_HR'
test_LR_path = f'{dataset_dir}{cst.SEP}DIV2K_valid_LR'

test_dataset = SP_DIV2K_Dataset(test_HR_path, test_LR_path, patch_size = 1000, is_train = False)

test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = True,
                         num_workers = 20,
                         pin_memory = True,
                         persistent_workers = True)

if __name__ == '__main__':
    for x, truth_y in test_loader:
        x = x.to(torch.device('cuda'))
        truth_y = truth_y.to(torch.device('cuda'))

        x_ycbcr = dt.rgb_to_ycbcr_tensor(x)
        truth_y_ycbcr = dt.rgb_to_ycbcr_tensor(truth_y)

        with torch.no_grad():
            predict_y_y = model(x_ycbcr[:, 0, :, :].unsqueeze(1))
        vis.show_chw_tensor(x[0])
        vis.show_chw_tensor(truth_y[0])

        output_ycbcr = torch.stack([
            predict_y_y[:, 0, :, :],
            x_ycbcr[:, 1, :, :],
            x_ycbcr[:, 2, :, :]
        ], dim = 1)[0]
        output_rgb = output_ycbcr.detach().to('cpu').float().permute(1, 2, 0).numpy()
        output_rgb = dt.ycbcr_to_rgb_ndarray(output_rgb)
        vis.show_hwc_ndarray(output_rgb)
