import torch
import torch.nn as nn

from utils import platform_manager

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
       from https://github.com/nianticlabs/monodepth2
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) *\
            (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

@platform_manager.LOSSES.add_module
class MD2_PhotoLoss(nn.Module):
    def __init__(self, preds_n, idents_n, target_n,
                 automask=True, minproj=True,
                 device="cpu"):
        super().__init__()
        self.init_opts = locals()
        self.preds_n = preds_n
        self.idents_n = idents_n
        self.target_n = target_n

        self.automask = automask
        self.minproj = minproj
        self.device = device

        self.ssim = SSIM().to(device)
 
    def forward(self, outputs, side):
        self.target = outputs[self.target_n.format(side)]
        pred_loss_maps = []
        if self.automask:
            ident_loss_maps = [] 
        for idx, pred_name in enumerate(self.preds_n):
            pred = outputs[pred_name.format(side)]
            loss_map = self._compute_photometric(pred)
            pred_loss_maps.append(loss_map)
            if self.automask:
                ident = outputs[self.idents_n[idx].format(side)]
                ident_map = self._compute_photometric(ident)
                ident_loss_maps.append(ident_map)
        
        if self.automask:
            ident_maps = torch.cat(ident_loss_maps, dim=1)
            ident_maps += torch.randn(ident_maps.shape).to(self.device) * 0.00001
            if self.minproj:
                loss_maps = torch.cat([ident_maps] + pred_loss_maps, dim=1)
                final_loss_map, select_idx = loss_maps.min(dim=1, keepdim=True)
                auto_mask = (select_idx > ident_maps.shape[1] - 1).float()
                final_loss_map = final_loss_map * auto_mask.detach()
                # outputs["auto_mask_{}".format(pred_name[-3:])] = auto_mask
            else:
                ident_maps = ident_maps.mean(dim=1, keepdim=True)
                pred_maps = torch.cat(pred_loss_maps, dim=1).mean(dim=1, keepdim=True)
                loss_maps = torch.cat([ident_maps, pred_maps], dim=1)
                final_loss_map, select_idx = loss_maps.min(dim=1, keepdim=True)
                auto_mask = (select_idx > ident_maps.shape[1] - 1).float()
                final_loss_map = final_loss_map * auto_mask.detach()
                # outputs["auto_mask_{}".format(pred_name[-3:])] = auto_mask
        else:
            loss_maps = torch.cat(pred_loss_maps, dim=1)
            if self.minproj:
                final_loss_map, _ = loss_maps.min(dim=1, keepdim=True)
            else:
                final_loss_map = loss_maps.mean(dim=1, keepdim=True)
        return final_loss_map
    
    def _compute_photometric(self, pred):
        abs_diff = torch.abs(pred - self.target)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, self.target).mean(1, True)
        photometric_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return photometric_loss
