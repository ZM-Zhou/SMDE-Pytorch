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
class DepthHints_PhotoLoss(nn.Module):
    def __init__(self, preds_n, idents_n, target_n,
                 hints_n=None, hints_depth=None, pred_depth=None,
                 automask=True, minproj=True,
                 device="cpu"):
        super().__init__()
        self.init_opts = locals()
        self.preds_n = preds_n
        self.idents_n = idents_n
        self.target_n = target_n
        self.hints_n = hints_n
        self.hints_depth = hints_depth
        self.pred_depth = pred_depth

        self.automask = automask
        self.minproj = minproj
        self.device = device

        self.ssim = SSIM().to(device)
 
    def forward(self, outputs, side):
        self.target = outputs[self.target_n.format(side)]
        # read the reprojection image computed with depth hints
        # and mask the loss without hints (hints depth <= 0)
        if self.hints_n is not None:
            depthhints = outputs[self.hints_n.format(side)]
            hints_loss_map = self._compute_photometric(depthhints)
            hints_depth = outputs[self.hints_depth.format(side)]
            pred_depth = outputs[self.pred_depth.format(side)]
            hints_mask = hints_depth > 0
            hints_loss_map[~hints_mask] = 1000
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
        
        # compute the min reprojection losses in the depth hints way
        pred_loss_maps = torch.cat(pred_loss_maps, dim=1)
        if self.minproj:
            pred_loss_map, _ = pred_loss_maps.min(dim=1, keepdim=True)
        else:
            pred_loss_map, _ = pred_loss_maps.mean(dim=1, keepdim=True)    
        if self.automask:
            ident_loss_maps = torch.cat(ident_loss_maps, dim=1)
            if self.minproj:
                ident_loss_map, _ = ident_loss_maps.min(dim=1, keepdim=True)
            else:
                ident_loss_map = ident_loss_maps.mean(dim=1, keepdim=True)
        
        # auto|hints|losses
        #   v |  v  |use ph. loss if the pred or hints error is smaller than the ident error
        #   v |     |use ph. loss if the pred is smaller than the ident error
        #     |  v  |use all ph. loss 
        #     |     |use all ph. loss
        if self.automask:
            ident_loss_map += torch.randn(ident_loss_map.shape).to(self.device) * 0.00001
            loss_maps = torch.cat([ident_loss_map, pred_loss_map], dim=1)
            
            if self.hints_n is not None:
                loss_maps = torch.cat([loss_maps, hints_loss_map], dim=1)
                _, select_idx = loss_maps.min(dim=1, keepdim=True)
                auto_mask = (select_idx != 0).float()
                hints_mask = (select_idx == 2).float()
                final_loss_map = pred_loss_map * auto_mask.detach()
                final_loss_map += self._compute_hints_proxy(pred_depth, hints_depth, hints_mask)
            else:
                _, select_idx = loss_maps.min(dim=1, keepdim=True)
                auto_mask = (select_idx != 0).float()
                final_loss_map = pred_loss_map * auto_mask.detach()
        else:
            loss_maps = pred_loss_map
            if self.hints_n is not None:
                loss_maps = torch.cat([loss_maps, hints_loss_map], dim=1)
                _, select_idx = loss_maps.min(dim=1, keepdim=True)
                hints_mask = (select_idx == 1).float()
                final_loss_map = pred_loss_map
                final_loss_map += self._compute_hints_proxy(pred_depth, hints_depth, hints_mask)
            else:
                final_loss_map = pred_loss_map 

        return final_loss_map
    
    def _compute_photometric(self, pred):
        abs_diff = torch.abs(pred - self.target)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, self.target).mean(1, True)
        photometric_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return photometric_loss
    
    def _compute_hints_proxy(self, pred, target, mask):
        depth_hint_loss = torch.log(torch.abs(target - pred) + 1) * mask
        return depth_hint_loss
