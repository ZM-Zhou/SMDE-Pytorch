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
class EPCDepth_PhotoLoss(nn.Module):
    '''
    preds_n: 
    '''
    def __init__(self, preds_n, target_n,
                 pred_depths_n=None,
                 hints_n=None, hints_depth=None,
                 ident_n=None, automask=True,
                 spp_distill_rate=1,
                 device="cpu"):
        super().__init__()
        self.init_opts = locals()
        self.preds_n = preds_n
        self.target_n = target_n
        self.pred_depths_n = pred_depths_n
        self.hints_n = hints_n
        self.hints_depth = hints_depth
        self.ident_n = ident_n
        self.automask = automask
        self.spp_distill_rate = spp_distill_rate
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
            hints_mask = hints_depth > 0
            hints_loss_map[~hints_mask] = 1000

        # compute the min reprojection losses in the EPCDepth way
        if self.automask:
            ident = outputs[self.ident_n.format(side)]
            ident_loss_map = self._compute_photometric(ident)
        
        total_loss_map = 0
        depth_best = None
        decoder_depth_best = None
        reproj_loss_min = None
        for idx in range(len(self.preds_n)):
            all_loss_maps = []
            pred = outputs[self.preds_n[idx].format(side)]
            depth = outputs[self.pred_depths_n[idx].format(side)]
            pred_loss_map = self._compute_photometric(pred)
            all_loss_maps.append(pred_loss_map)
            if self.automask:
                all_loss_maps.append(ident_loss_map)
            if self.hints_n:
                all_loss_maps.append(hints_loss_map)
            all_loss_maps = torch.cat(all_loss_maps, dim=1)
            _, select_idx = all_loss_maps.min(dim=1, keepdim=True)
            auto_mask = (select_idx != 1).float()
            hints_mask = (select_idx == 2).float()
            final_loss_map = pred_loss_map * auto_mask.detach()
            final_loss_map += self._compute_hints_proxy(depth, hints_depth, hints_mask)
            total_loss_map += final_loss_map

            if self.spp_distill_rate:
                if idx == 0:
                    depth_best = depth
                    reproj_loss_min = pred_loss_map
                elif idx == 5:
                    decoder_depth_best = depth_best.clone()
                    depth_best = depth
                    reproj_loss_min = pred_loss_map
                else:
                    depth_best = torch.where(pred_loss_map < reproj_loss_min, depth, depth_best)
                    reproj_loss_min, _ = torch.cat([pred_loss_map, reproj_loss_min], dim=1).min(dim=1, keepdim=True)
        total_loss_map /= len(self.preds_n)  

        if self.spp_distill_rate:
            if decoder_depth_best is not None:
                decoder_depth_best = decoder_depth_best.detach()
                encoder_depth_best = depth_best.detach()
            else:
                decoder_depth_best = depth_best.detach()
            spp_loss_map = 0
            for idx in range(len(self.preds_n)):
                depth = outputs[self.pred_depths_n[idx].format(side)]
                depth_best = decoder_depth_best if idx < 5 else encoder_depth_best
                spp_loss_map += torch.log(torch.abs(1 / depth_best - 1 / depth) + 1)
            
        total_loss_map += self.spp_distill_rate * spp_loss_map / len(self.preds_n)  
        
        return total_loss_map
    
    def _compute_photometric(self, pred):
        abs_diff = torch.abs(pred - self.target)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, self.target).mean(1, True)
        photometric_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return photometric_loss
    
    def _compute_hints_proxy(self, pred, target, mask):
        depth_hint_loss = torch.log(torch.abs(target - pred) + 1) * mask
        return depth_hint_loss
