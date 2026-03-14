from os import wait
import typing
import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import gc

from simpai import utils
if utils.is_notebook():
    import tqdm.notebook as tqdm
else:
    import tqdm


from simpai import hyperparam as hp
from simpai import train, visual, logger, data, wait_for_io

from retinex import RetinexNet
from dataset import Our485

hp.set_hp_begin()
hp.set_hp('device', torch.device('cpu'))
hp.set_hp('batch_size', 16)
hp.set_hp('patch_size', 300)
hp.set_hp('mode', 'eval')
hp.set_hp_end()



def eval(
    model: torch.nn.Module,
    user_data: dict
) -> None:
    model = typing.cast(RetinexNet, model)
    try:
        logger.info(f'loss_Decom: {model.loss_Decom}, loss_Relight: {model.loss_Relight}', output = True)
    except AttributeError:
        pass
    for x, y in user_data['test_dataloader']:
        x = x.to(hp.get_hp('device'))
        y = y.to(hp.get_hp('device'))
        model(x, y)
        visual.show_bchw(torch.cat((
            x.to(torch.device('cpu')),
            model.output_S,
            y.to(torch.device('cpu')),
            model.output_R_low,
            model.output_I_delta,
        ), dim = 0), nrows = 5, ncols = 16)

if __name__ == '__main__':
    test_dataloader =  DataLoader(
        Our485(is_train = False),
        batch_size = hp.get_hp('batch_size'),
        shuffle = True,
        num_workers = 20,
        pin_memory = True,
        persistent_workers = True,
    )

    retinex_net = RetinexNet()
    retinex_net.eval()
    for p in retinex_net.parameters():
        p.requires_grad_(False)

    data.load_ckpt(
        ('model_state_dict.pt', retinex_net),
    )

    with torch.no_grad():
        eval(retinex_net, {'test_dataloader': test_dataloader})


    wait_for_io()

    del test_dataloader
    gc.collect()
