import torch
import typing
from simpai import hyperparam as hp

hp.set_hp_begin()
hp.set_hp('device', torch.device('cpu'))
hp.set_hp('epoch_num', 10)
hp.set_hp('batch_size', 2)
hp.set_hp('patch_size', 96)
hp.set_hp('lr', 0.001)
hp.set_hp_end()

from retinex import RetinexNet
from simpai import data

model = RetinexNet()
decom_optimizer = torch.optim.Adam(
    typing.cast(RetinexNet, model).decom_net.parameters(),
    lr = hp.get_hp('lr'),
    betas = (0.9, 0.999),
)
relight_optimizer = torch.optim.Adam(
    typing.cast(RetinexNet, model).relight_net.parameters(),
    lr = hp.get_hp('lr'),
    betas = (0.9, 0.999),
)

print("Loading model and optimizers...")
try:
    data.load_ckpt(
        ('model_state_dict.pt', model),
        ('decom_optimizer_state_dict.pt', decom_optimizer),
        ('relight_optimizer_state_dict.pt', relight_optimizer)
    )
    print("Load succeeded")
except Exception as e:
    print(f"Load failed: {e}")
    import traceback
    traceback.print_exc()