import typing
import torch
from torch.utils.data import DataLoader
import gc
import sys

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
hp.set_hp('epoch_num', 1)  # small
hp.set_hp('batch_size', 2)
hp.set_hp('patch_size', 96)
hp.set_hp('lr', 0.001)
hp.set_hp_end()

def eval_fn(
    epoch_idx: int,
    model: torch.nn.Module,
    user_data: dict,
    bar: tqdm.tqdm):
    model = typing.cast(RetinexNet, model)
    try:
        logger.info(f'loss_Decom: {model.loss_Decom}, loss_Relight: {model.loss_Relight}', output = True)
    except AttributeError:
        pass

def step_fn(
    epoch_idx: int,
    model: torch.nn.Module,
    user_data: dict,
    bar: tqdm.tqdm,
    data: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    model = typing.cast(RetinexNet, model)
    x, y = data
    x = x.to(hp.get_hp('device'))
    y = y.to(hp.get_hp('device'))

    logger.debug(f'Train step, epoch {epoch_idx}')

    model(x, y)
    if hp.get_hp('train_phase') == 'decom':

        if epoch_idx > 20:
            for param_group in user_data['decom_op'].param_groups:
                param_group['lr'] = 0.0001
        else:
            for param_group in user_data['decom_op'].param_groups:
                param_group['lr'] = 0.001

        user_data['decom_op'].zero_grad()
        model.loss_Decom.backward()
        user_data['decom_op'].step()
        train_loss = model.loss_Decom
    elif hp.get_hp('train_phase') == 'relight':

        if epoch_idx > 20:
            for param_group in user_data['relight_op'].param_groups:
                param_group['lr'] = 0.0001
        else:
            for param_group in user_data['relight_op'].param_groups:
                param_group['lr'] = 0.001

        user_data['relight_op'].zero_grad()
        model.loss_Relight.backward()
        user_data['relight_op'].step()
        train_loss = model.loss_Relight
    else: raise RuntimeError
    return train_loss

if __name__ == '__main__':
    train_dataloader = DataLoader(
        Our485(is_train = True),
        batch_size = hp.get_hp('batch_size'),
        shuffle = True,
        num_workers = 0,  # reduce workers for quick test
        pin_memory = True,
        persistent_workers = False,
    )
    test_dataloader =  DataLoader(
        Our485(is_train = False),
        batch_size = hp.get_hp('batch_size'),
        shuffle = True,
        num_workers = 0,
        pin_memory = True,
        persistent_workers = False,
    )

    retinex_net = RetinexNet()

    trainer = train.Trainer(retinex_net, train_dataloader)
    trainer.set_step(step_fn = step_fn)
    trainer.set_eval(eval_fn = eval_fn)

    decom_optimizer = torch.optim.Adam(
        typing.cast(RetinexNet, trainer.model).decom_net.parameters(),
        lr = hp.get_hp('lr'),
        betas = (0.9, 0.999),
    )

    relight_optimizer = torch.optim.Adam(
        typing.cast(RetinexNet, trainer.model).relight_net.parameters(),
        lr = hp.get_hp('lr'),
        betas = (0.9, 0.999),
    )

    trainer.set_user_data({
        'decom_op': decom_optimizer,
        'relight_op': relight_optimizer,
        'test_dataloader': test_dataloader,
        'train_dataloader': train_dataloader,
    })

    print("Loading checkpoint...")
    data.load_ckpt(
        ('model_state_dict.pt', trainer.model),
        ('decom_optimizer_state_dict.pt', decom_optimizer),
        ('relight_optimizer_state_dict.pt', relight_optimizer)
    )
    print("Checkpoint loaded successfully.")

    # Test one training step
    hp.set_hp_begin()
    hp.set_hp('train_phase', 'decom')
    hp.set_hp_end()
    print("Running one training step...")
    # We'll manually call step_fn with a batch
    batch = next(iter(train_dataloader))
    loss = step_fn(0, trainer.model, trainer.user_data, None, batch)
    print(f"Step loss: {loss}")
    print("Success! No errors.")
    sys.exit(0)