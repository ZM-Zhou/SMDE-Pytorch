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
class HintsLoss(nn.Module):
    def __init__(self,
                 pred_n,
                 hints_n,
                 target_n,
                 pred_depth_n,
                 hints_depth_n,
                 recons_rate=1,
                 hints_rate=1,
                 hints_loss_mode='L1',
                 device='cpu'):
        super().__init__()
        self.init_opts = locals()
        self.pred_n = pred_n
        self.hints_n = hints_n
        self.target_n = target_n
        self.pred_depth_n = pred_depth_n
        self.hints_depth_n = hints_depth_n
        self.recons_rate = recons_rate
        self.hints_rate = hints_rate
        self.hints_loss_mode = hints_loss_mode

        self.ssim = SSIM().to(device)

    def forward(self, outputs, side):
        pred = outputs[self.pred_n.format(side)]
        hints = outputs[self.hints_n.format(side)]
        target = outputs[self.target_n.format(side)]
        pred_depth = outputs[self.pred_depth_n.format(side)]
        hints_depth = outputs[self.hints_depth_n.format(side)]

        loss_map = torch.zeros_like(pred)[:, 0, ...].unsqueeze(1)

        error_pred = (0.15 * torch.abs(pred - target).mean(1, True) +
                      0.85 * self.ssim(pred, target).mean(1, True))
        error_hints = (0.15 * torch.abs(hints - target).mean(1, True) +
                       0.85 * self.ssim(hints, target).mean(1, True))
        
        hints_mask = (error_hints < error_pred + 1e-5)
        hints_mask *= error_hints < 0.2

        if self.hints_loss_mode == 'L1':
            hints_loss = torch.abs(pred_depth - hints_depth)
        elif self.hints_loss_mode == 'berHu':
            abs_loss = torch.abs(pred_depth - hints_depth)
            abs_flatten = abs_loss.flatten(1)
            delta = (0.2 * abs_flatten.max(
                dim=1, keepdim=True)[0]).unsqueeze(2).unsqueeze(2)
            L2_mask = (abs_loss > delta).to(torch.float)
            l2_loss = ((pred_depth - hints_depth)**2 + delta**2) / (2 * delta)
            hints_loss = abs_loss * (1 - L2_mask) + l2_loss * L2_mask

        loss_map = (self.recons_rate * error_pred +
                    self.hints_rate * hints_mask * hints_loss)

        return loss_map, hints_mask.to(torch.int) * 2
