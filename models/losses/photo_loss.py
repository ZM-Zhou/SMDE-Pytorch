import torch
import torch.nn as nn

from utils import platform_manager


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images from
    https://github.com/nianticlabs/monodepth2."""
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) *\
            (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


@platform_manager.LOSSES.add_module
class PhotoLoss(nn.Module):
    def __init__(self,
                 pred_n,
                 target_n,
                 l1_rate=1,
                 l2_rate=0,
                 ssim_rate=0,
                 other_side=False,
                 device='cpu'):
        super().__init__()
        self.init_opts = locals()
        self.pred_n = pred_n
        self.target_n = target_n

        self.l1_rate = l1_rate
        self.l2_rate = l2_rate
        self.ssim_rate = ssim_rate
        assert l1_rate + l2_rate + ssim_rate == 1,\
            'the sum of losses rate must be 1.'
        self.other_side = other_side

        if self.ssim_rate != 0:
            self.ssim = SSIM().to(device)
        else:
            self.ssim = None

    def forward(self, outputs, side):
        pred = outputs[self.pred_n.format(side)]
        if self.other_side:
            target = outputs[self.target_n.format('s' if side == 'o' else 'o')]
        else:
            target = outputs[self.target_n.format(side)]
        loss_map = torch.zeros_like(target)[:, 0, ...].unsqueeze(1)

        if self.l1_rate != 0:
            l1_loss_map = torch.abs(pred - target).mean(1, True)
            loss_map += self.l1_rate * l1_loss_map
        if self.l2_rate != 0:
            l2_loss_map = ((pred - target)**2).mean(1, True)
            loss_map += self.l2_rate * l2_loss_map
        if self.ssim_rate != 0:
            ssim_loss_map = self.ssim(pred, target).mean(1, True)
            loss_map += self.ssim_rate * ssim_loss_map

        return loss_map
