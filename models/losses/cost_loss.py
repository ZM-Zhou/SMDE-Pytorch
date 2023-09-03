import torch
import torch.nn as nn
import torch.nn.functional as F

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
class CostLoss(nn.Module):
    def __init__(self,
                 pred_n,
                 targets_n,
                 pred_img_n=None,
                 target_img_n=None,
                 real_img_n=None,
                 able_mask_n=None,
                 use_confi=True,
                 t_confi=0,
                 t_l1 = 0,
                 a_smol1=0.05,
                 l1_rate=0, 
                 ce_rate=0,   
                 t_ce=1,   
                 smol1_rate=1,
                 total_tl1=False,
                 device='cpu'):
        super().__init__()
        self.init_opts = locals()
        self.pred_n = pred_n
        self.targets_n = targets_n
        self.pred_img_n = pred_img_n
        self.target_img_n = target_img_n
        self.real_img_n = real_img_n
        self.able_mask_n = able_mask_n
        self.use_confi = use_confi
        self.t_confi = t_confi
        self.t_l1 = t_l1
        self.l1_rate = l1_rate
        self.ce_rate = ce_rate
        self.t_ce = t_ce
        self.smol1_rate = smol1_rate
        self.a_smol1 = a_smol1
        self.total_tl1 = total_tl1
        
        assert l1_rate + ce_rate + smol1_rate == 1,\
            'the sum of losses rate must be 1.'
        if len(self.targets_n) > 1:
            assert use_confi,\
                'confi should be used for selecting the target'

        if pred_img_n is not None:
            assert len(self.targets_n) == 1,\
                'hints mask could not used with multi-target'
            self.ssim = SSIM().to(device)

    def forward(self, outputs, side):
        pred = outputs[self.pred_n.format(side)]
        targets = []
        confis = []
        for target_n in self.targets_n:
            target = outputs[target_n.format(side)]
            targets.append(target.unsqueeze(1))
            
            if self.use_confi:
                if self.t_ce != 1:
                    target = torch.softmax(target, dim=1)
                max_label_idx = target.max(dim=1, keepdim=True)[1]
                bins_range = torch.cat([max_label_idx - 1, max_label_idx, max_label_idx + 1], dim=1)
                bins_mask = (bins_range >= 0) & (bins_range <= target.shape[1] - 1)
                bins_range[bins_range < 0] = 0
                bins_range[bins_range > target.shape[1] - 1] = target.shape[1] - 1
                local_bins = torch.gather(target, 1, bins_range)
                confi = (local_bins * bins_mask.to(torch.float)).sum(dim=1, keepdim=True)
                confis.append(confi)
        
        loss_map = torch.zeros_like(pred)[:, 0, ...].unsqueeze(1)
        
        if self.use_confi:
            confis = torch.cat(confis, dim=1)
            max_confi, select_label = confis.max(dim=1, keepdim=True)
            confi_mask = max_confi
            confi_mask[confi_mask < self.t_confi] = 0
            label = torch.gather(torch.cat(targets, dim=1),
                                 1,
                                 select_label.unsqueeze(2).repeat(1, 1, target.shape[1], 1, 1))
            label = label.squeeze(1)
        else:
            label = targets[0].squeeze(1)
            confi_mask = torch.ones_like(loss_map)
        
        if self.pred_img_n is not None:
            pred_img = outputs[self.pred_img_n.format(side)]
            target_img = outputs[self.target_img_n.format(side)]
            real_img = outputs[self.real_img_n.format(side)]
            
            error_pred = (0.15 * torch.abs(pred_img - real_img).mean(1, True) +
                0.85 * self.ssim(pred_img, real_img).mean(1, True))
            error_target = (0.15 * torch.abs(target_img - real_img).mean(1, True) +
                0.85 * self.ssim(target_img, real_img).mean(1, True))

            hints_mask = (error_target < error_pred + 1e-5).to(torch.float)
            if self.able_mask_n is not None:
                able_mask = outputs[self.able_mask_n.format(side)]
                hints_mask = torch.cat([able_mask, hints_mask], dim=1).max(dim=1, keepdim=True)[0]
        else:
            hints_mask = torch.ones_like(loss_map)

        if self.ce_rate > 0:
            if self.t_ce != 1:
                pred = torch.softmax(pred / self.t_ce, dim=1)
                label = torch.softmax(label / self.t_ce, dim=1)
            log_softmax = torch.log(pred + 1e-5)
            loss_map += self.ce_rate * ( -(log_softmax * label).sum(dim=1, keepdim=True) + (torch.log(label + 1e-5) * label).sum(dim=1, keepdim=True))
        if self.l1_rate > 0:
            l1_loss_map = torch.abs(pred - label)
            if self.t_l1 > 0:
                if self.total_tl1:
                    l1_loss_map = l1_loss_map.sum(1, True)
                    l1_loss_map[l1_loss_map < self.t_l1] = 0
                else:
                    l1_loss_map[l1_loss_map < self.t_l1] = 0
                    l1_loss_map = l1_loss_map.sum(1, True)
            loss_map += self.l1_rate * l1_loss_map
        if self.smol1_rate > 0:
            l1_loss_map = torch.abs(pred - target).sum(1, True)
            if isinstance(self.a_smol1, str) and 'max' in self.a_smol1:
                thres = l1_loss_map.max() * float(self.a_smol1.replace('max', ''))
            else:
                thres = self.a_smol1
            l1_mask = (l1_loss_map > thres).to(torch.float)
            l1_loss_map = thres * l1_loss_map - 0.5 * thres ** 2
            l2_loss_map = 0.5 * ((pred - target) ** 2).sum(1, True)
            
            loss_map += self.smol1_rate * (l1_mask * l1_loss_map + (1-l1_mask) * l2_loss_map)
        loss_map = loss_map * confi_mask * hints_mask


        return loss_map
