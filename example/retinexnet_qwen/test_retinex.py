import torch
from torch.utils.data import DataLoader

from simpai import hyperparam as hp
from simpai import trainer
from simpai import visual

from retinex import RetinexNet
from dataset import Our485

hp.set_hp_begin()
hp.set_hp('device', torch.device('cuda:0'))
hp.set_hp('epoch_num', 30)
hp.set_hp('batch_size', 10)
hp.set_hp('patch_size', 100)
hp.set_hp('lr', 0.001)
hp.set_hp_end()

retinex_net = RetinexNet()

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
trainer = trainer.Trainer(retinex_net, train_dataloader)

@trainer.set_eval
def eval_fn(epoch_idx: int, model, tqdm):
    try:
        tqdm.write(f'loss_Decom: {model.loss_Decom}, loss_Relight: {model.loss_Relight}')
    except AttributeError:
        pass
    if epoch_idx > 25:
        for x, y in test_dataloader:
            x = x.to(hp.get_hp('device'))
            y = y.to(hp.get_hp('device'))
            model(x, y)
            visual.show_bchw(model.output_S)
            break

decom_optimizer = torch.optim.Adam(
    retinex_net.decom_net.parameters(),
    lr = hp.get_hp('lr'),
    betas = (0.9, 0.999),
)

relight_optimizer = torch.optim.Adam(
    retinex_net.relight_net.parameters(),
    lr = hp.get_hp('lr'),
    betas = (0.9, 0.999),
)
@trainer.set_step
def step_fn(
    epoch_idx: int,
    model: RetinexNet,
    tqdm,
    data: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    x, y = data
    x = x.to(hp.get_hp('device'))
    y = y.to(hp.get_hp('device'))

    model(x, y)
    if hp.get_hp('train_phase') == 'decom':
        decom_optimizer.zero_grad()
        model.loss_Decom.backward()
        decom_optimizer.step()
        return model.loss_Decom
    elif hp.get_hp('train_phase') == 'relight':
        relight_optimizer.zero_grad()
        model.loss_Relight.backward()
        relight_optimizer.step()
        return model.loss_Relight
    else: raise RuntimeError

hp.set_hp_begin()
hp.set_hp('train_phase', 'decom')
hp.set_hp_end()
trainer.train(hp.get_hp('epoch_num'), optimizer = decom_optimizer)

hp.set_hp_begin()
hp.set_hp('train_phase', 'relight')
hp.set_hp_end()
trainer.train(hp.get_hp('epoch_num'), optimizer = relight_optimizer)
