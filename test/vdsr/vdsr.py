import math
import torch.nn as nn
import torch

class VDSR(nn.Module):
    def __init__(self, num_channels = 3):
        super(VDSR, self).__init__()

        # 初始卷积层：输入通道 -> 64 通道
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size = 3, padding = 1, bias = False),
            nn.ReLU(inplace = True)
        )

        # 中间特征提取层：18层相同的 (卷积 + ReLU)
        # VDSR 共 20 层，减去首尾各 1 层
        layers_tmp = []
        for _ in range(18):
            layers_tmp += [
                nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias = False),
                nn.ReLU(inplace = True)
            ]
        self.layer2 = nn.Sequential(*layers_tmp)

        # 最后一层
        self.layer3 = nn.Conv2d(64, num_channels, kernel_size = 3, padding = 1, bias = False)

        # 权重初始化：He
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        
    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = torch.add(out, residual)
        return out
