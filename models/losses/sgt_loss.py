from matplotlib.pyplot import margins
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import platform_manager

@platform_manager.LOSSES.add_module
class GuideTripletLoss(nn.Module):
    def __init__(self,
                 pred_n,
                 target_n,
                 kernel_size=5,
                 margin=0.3, 
                 device='cpu'):
        super().__init__()
        self.init_opts = locals()
        self.pred_n = pred_n
        self.target_n = target_n
        self.kernel_size = kernel_size
        self.margin=margin

    def forward(self, outputs, side):
        pred = outputs[self.pred_n.format(side)]
        target = outputs[self.target_n.format(side)]
        loss_map = torch.zeros_like(target)[:, 0, ...].unsqueeze(1)

        _, _, h, w = target.shape
        
        pad = self.kernel_size // 2
        h, w = pred.shape[2:]
        seg = F.interpolate(target, size=(h, w), mode='nearest')
        center = seg
        padded = F.pad(center, [pad] * 4, value=-1)
        aggregated_label = torch.zeros(*(center.shape + (self.kernel_size, self.kernel_size))).to(center.device)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                shifted = padded[:, :, 0 + i: h + i, 0 + j:w + j]
                label = center == shifted
                aggregated_label[:, :, :, :, i, j] = label
        aggregated_label = aggregated_label.float()
        pos_idx = (aggregated_label == 1).float()
        neg_idx = (aggregated_label == 0).float()
        pos_idx_num = pos_idx.sum(dim=-1).sum(dim=-1)
        neg_idx_num = neg_idx.sum(dim=-1).sum(dim=-1)

        boundary_region = (pos_idx_num >= self.kernel_size - 1) & (
                neg_idx_num >= self.kernel_size - 1)
        # non_boundary_region = (pos_idx_num != 0) & (neg_idx_num == 0)
        # if s == min(self.opt.sgt_layers):
        #     outputs[('boundary', s)] = boundary_region.data
        #     outputs[('non_boundary', s)] = non_boundary_region.data

        affinity = self._compute_affinity(pred, kernel_size=self.kernel_size)
        pos_dist = (pos_idx * affinity).sum(dim=-1).sum(dim=-1)[boundary_region] / \
                    pos_idx.sum(dim=-1).sum(dim=-1)[boundary_region]
        neg_dist = (neg_idx * affinity).sum(dim=-1).sum(dim=-1)[boundary_region] / \
                    neg_idx.sum(dim=-1).sum(dim=-1)[boundary_region]
        zeros = torch.zeros(pos_dist.shape).to(pos_dist.device)
        loss_map = torch.max(zeros, pos_dist - neg_dist + self.margin)

        return loss_map

    def _compute_affinity(self, feature, kernel_size):
        pad = kernel_size // 2
        feature = F.normalize(feature, dim=1)
        unfolded = F.pad(feature, [pad] * 4).unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        feature = feature.unsqueeze(-1).unsqueeze(-1)
        similarity = (feature * unfolded).sum(dim=1, keepdim=True)
        eps = torch.zeros(similarity.shape).to(similarity.device) + 1e-9
        affinity = torch.max(eps, 2 - 2 * similarity).sqrt()
        return affinity
