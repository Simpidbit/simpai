import math
import torch.nn as nn
import torch

class VDSR(nn.Module):
    """
    Very Deep Super Resolution (VDSR) network for image super-resolution.

    VDSR is a deep CNN architecture for single image super-resolution.
    It uses a global residual learning framework where the network learns
    to predict the residual (difference) between the low-resolution and
    high-resolution images, which is then added to the input.

    The architecture consists of:
    - Input layer: Conv2d + ReLU (num_channels -> 64)
    - Middle layers: 18 x (Conv2d + ReLU) (64 -> 64)
    - Output layer: Conv2d (64 -> num_channels)

    Args:
        num_channels: Number of input/output image channels. Default is 3 (RGB).

    Reference:
        Kim, J., Lee, J. K., & Lee, K. M. (2016).
        "Accurate Image Super-Resolution Using Very Deep Convolutional Networks"
        CVPR 2016.
    """
    def __init__(self, num_channels = 3):
        """
        Initialize the VDSR network.

        Args:
            num_channels: Number of input/output image channels. Default is 3 (RGB).
        """
        super(VDSR, self).__init__()

        # Initial convolution layer: input channels -> 64 channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size = 3, padding = 1, bias = False),
            nn.ReLU(inplace = True)
        )

        # Middle feature extraction layers: 18 identical (Conv2d + ReLU) layers
        # VDSR has 20 layers total, minus 1 input layer and 1 output layer
        layers_tmp = []
        for _ in range(18):
            layers_tmp += [
                nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias = False),
                nn.ReLU(inplace = True)
            ]
        self.layer2 = nn.Sequential(*layers_tmp)

        # Output layer: 64 channels -> num_channels
        self.layer3 = nn.Conv2d(64, num_channels, kernel_size = 3, padding = 1, bias = False)

        # Weight initialization: He initialization (Kaiming)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        
    def forward(self, x):
        """
        Forward pass of the VDSR network.

        Args:
            x: Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of the same shape as x.
                The output is the sum of the input (residual) and the
                network's prediction (residual learning).
        """
        # Store input for residual connection
        residual = x

        # Pass through all layers
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        # Add residual to final output (global residual learning)
        out = torch.add(out, residual)
        return out
