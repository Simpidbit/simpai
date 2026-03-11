import torch
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