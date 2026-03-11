import torch
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
print("Model created")
try:
    data.load_ckpt(('model_state_dict.pt', model))
    print("Load succeeded")
except Exception as e:
    print(f"Load failed: {e}")
    import traceback
    traceback.print_exc()