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
hp.set_hp('epoch_num', 1)
hp.set_hp('batch_size', 2)
hp.set_hp('patch_size', 96)
hp.set_hp('mode', 'train')
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
    if epoch_idx == hp.get_hp('epoch_num') - 1:
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
            break

def _flatten_grad_vector(
    grads: tuple[torch.Tensor | None, ...],
    device: torch.device,
) -> torch.Tensor:
    grad_parts = [grad.reshape(-1) for grad in grads if grad is not None]
    if len(grad_parts) == 0:
        return torch.zeros(1, dtype = torch.float32, device = device)
    return torch.cat(grad_parts)

def _log_decom_grad_conflict(model: RetinexNet) -> None:
    shared_params = tuple(param for param in model.decom_net.parameters() if param.requires_grad)
    g_ori = torch.autograd.grad(
        model.original_loss_Decom,
        shared_params,
        retain_graph = True,
        allow_unused = True,
    )
    g_qwen = torch.autograd.grad(
        0.01 * model.qwen_score,
        shared_params,
        retain_graph = True,
        allow_unused = True,
    )

    v_ori = _flatten_grad_vector(g_ori, model.device)
    v_qwen = _flatten_grad_vector(g_qwen, model.device)

    dot = torch.dot(v_ori, v_qwen)
    ori_norm = v_ori.norm()
    qwen_norm = v_qwen.norm()
    cosine = torch.tensor(0.0, device = dot.device)
    if ori_norm.item() > 0 and qwen_norm.item() > 0:
        cosine = dot / (ori_norm * qwen_norm)

    logger.debug(
        'decom grad stats '
        f'ori_norm={ori_norm.item():.6e}, '
        f'qwen_norm={qwen_norm.item():.6e}, '
        f'dot={dot.item():.6e}, '
        f'cosine={cosine.item():.6f}, '
        f'norm_ratio={(qwen_norm / (ori_norm + 1e-12)).item():.6e}'
    )

def _log_relight_grad_conflict(model: RetinexNet) -> None:
    shared_params = tuple(param for param in model.decom_net.parameters() if param.requires_grad)
    g_ori = torch.autograd.grad(
        model.original_loss_Relight,
        shared_params,
        retain_graph = True,
        allow_unused = True,
    )
    g_qwen = torch.autograd.grad(
        0.01 * model.qwen_score,
        shared_params,
        retain_graph = True,
        allow_unused = True,
    )

    v_ori = _flatten_grad_vector(g_ori, model.device)
    v_qwen = _flatten_grad_vector(g_qwen, model.device)

    dot = torch.dot(v_ori, v_qwen)
    ori_norm = v_ori.norm()
    qwen_norm = v_qwen.norm()
    cosine = torch.tensor(0.0, device = dot.device)
    if ori_norm.item() > 0 and qwen_norm.item() > 0:
        cosine = dot / (ori_norm * qwen_norm)

    logger.debug(
        'decom grad stats '
        f'ori_norm={ori_norm.item():.6e}, '
        f'qwen_norm={qwen_norm.item():.6e}, '
        f'dot={dot.item():.6e}, '
        f'cosine={cosine.item():.6f}, '
        f'norm_ratio={(qwen_norm / (ori_norm + 1e-12)).item():.6e}'
    )

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

        _log_decom_grad_conflict(model)

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

        _log_relight_grad_conflict(model)

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
        num_workers = 20,
        pin_memory = True,
        persistent_workers = True,
    )
    test_dataloader =  DataLoader(
        Our485(is_train = False),
        batch_size = hp.get_hp('batch_size'),
        shuffle = True,
        num_workers = 20,
        pin_memory = True,
        persistent_workers = True,
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

    data.load_ckpt(
        ('model_state_dict.pt', trainer.model),
        ('decom_optimizer_state_dict.pt', decom_optimizer),
        ('relight_optimizer_state_dict.pt', relight_optimizer)
    )

    hp.set_hp_begin()
    hp.set_hp('train_phase', 'decom')
    hp.set_hp_end()
    trainer.train(hp.get_hp('epoch_num'), interrupt_feedback = lambda x, y: wait_for_io())

    '''
    hp.set_hp_begin()
    hp.set_hp('train_phase', 'relight')
    hp.set_hp_end()
    trainer.train(hp.get_hp('epoch_num'), interrupt_feedback = lambda x, y: wait_for_io())
    '''

    data.save_ckpt(
        ('model_state_dict.pt', trainer.model),
        ('decom_optimizer_state_dict.pt', decom_optimizer),
        ('relight_optimizer_state_dict.pt', relight_optimizer)
    )

    wait_for_io()

    del train_dataloader
    del test_dataloader
    del trainer
    gc.collect()
