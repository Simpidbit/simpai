import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from typeguard import typechecked

from simpai import hyperparam as hp
from simpai import logger

from decom import DecomNet
from relight import RelightNet
from qwen import QwenScorer

class RetinexNet(nn.Module):
    @typechecked
    def __init__(self) -> None:
        super(RetinexNet, self).__init__()
        self.device         = hp.get_hp('device')
        self.decom_net      = DecomNet().to(self.device)
        self.relight_net    = RelightNet().to(self.device)
        self.qwen_scorer    = QwenScorer(self.device)

    def forward(
        self,
        input_low: torch.Tensor,
        input_high: torch.Tensor
    ) -> None:
        # Forward DecompNet
        R_low, I_low   = self.decom_net(input_low)
        R_high, I_high = self.decom_net(input_high)

        # Forward RelightNet
        I_delta = self.relight_net(I_low, R_low)

        # Other variables
        I_low_3  = torch.cat((I_low, I_low, I_low), dim=1)
        I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)
        I_delta_3= torch.cat((I_delta, I_delta, I_delta), dim=1)

        # Compute losses
        self.recon_loss_low  = F.l1_loss(R_low * I_low_3,  input_low)
        self.recon_loss_high = F.l1_loss(R_high * I_high_3, input_high)
        self.recon_loss_mutal_low  = F.l1_loss(R_high * I_low_3, input_low)
        self.recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, input_high)
        self.equal_R_loss = F.l1_loss(R_low,  R_high.detach())
        self.relight_loss = F.l1_loss(R_low * I_delta_3, input_high)

        self.Ismooth_loss_low   = self._smooth(I_low, R_low)
        self.Ismooth_loss_high  = self._smooth(I_high, R_high)
        self.Ismooth_loss_delta = self._smooth(I_delta, R_low)

        self.output_S_device = R_low * I_delta_3


        self.original_loss_Decom = self.recon_loss_low + \
                          self.recon_loss_high + \
                          0.001 * self.recon_loss_mutal_low + \
                          0.001 * self.recon_loss_mutal_high + \
                          0.1 * self.Ismooth_loss_low + \
                          0.1 * self.Ismooth_loss_high + \
                          0.01 * self.equal_R_loss
        self.original_loss_Relight = self.relight_loss + \
                            3 * self.Ismooth_loss_delta

        if hp.get_hp('mode') == 'train':
            self.qwen_score = self.qwen_scorer(self.output_S_device)

            logger.debug(f'Qwen Score: {self.qwen_score}')
            self.loss_Decom = self.original_loss_Decom + 0.01 * self.qwen_score
            self.loss_Relight = self.original_loss_Relight# + 0.01 * self.qwen_score
        elif hp.get_hp('mode') == 'eval':
            self.loss_Decom = self.original_loss_Decom# + 0.01 * self.qwen_score
            self.loss_Relight = self.original_loss_Relight# + 0.01 * self.qwen_score

        logger.debug(f'Loss_Decom: {self.loss_Decom}, Loss_Relight: {self.loss_Relight}')


        if hp.get_hp('mode') == 'train':
            self.qwen_score.retain_grad()
            self.original_loss_Decom.retain_grad()
            self.original_loss_Relight.retain_grad()

        self.output_R_low   = R_low.detach().cpu()
        self.output_I_low   = I_low_3.detach().cpu()
        self.output_I_delta = I_delta_3.detach().cpu()
        self.output_S       = R_low.detach().cpu() * I_delta_3.detach().cpu()

    def _gradient(
        self,
        input_tensor: torch.Tensor,
        direction: str
    ) -> torch.Tensor:
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).to(self.device)
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == 'x':
            kernel = self.smooth_kernel_x
        elif direction == 'y':
            kernel = self.smooth_kernel_y
        else:
            raise RuntimeError

        return torch.abs(F.conv2d(
            input_tensor,
            kernel,
            stride = 1,
            padding = 1
        ))

    def _ave_gradient(
        self,
        input_tensor: torch.Tensor,
        direction: str,
    ) -> torch.Tensor:
        return F.avg_pool2d(
            self._gradient(input_tensor, direction),
            kernel_size = 3,
            stride = 1,
            padding = 1,
        )

    def _smooth(
        self,
        input_I: torch.Tensor,
        input_R: torch.Tensor,
    ) -> torch.Tensor:
        input_R = 0.299 * input_R[:, 0, :, :] + \
                  0.587 * input_R[:, 1, :, :] + \
                  0.114 * input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim = 1)
        return torch.mean(
            self._gradient(input_I, 'x') * torch.exp(-10 * self._ave_gradient(input_R, 'x'))
                +
            self._gradient(input_I, 'y') * torch.exp(-10 * self._ave_gradient(input_R, 'y'))
        )
