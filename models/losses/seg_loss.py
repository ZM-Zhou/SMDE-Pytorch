import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import platform_manager

@platform_manager.LOSSES.add_module
class SegmentLoss(nn.Module):
    def __init__(self,
                 pred_n,
                 target_n,
                 device='cpu'):
        super().__init__()
        self.init_opts = locals()
        self.pred_n = pred_n
        self.target_n = target_n

    def forward(self, outputs, side):
        pred = outputs[self.pred_n.format(side)]
        target = outputs[self.target_n.format(side)].to(torch.long).squeeze(1)
        # weights = target.sum(1, keepdim=True).float()
        # ignore_mask = (weights == 0)
        # weights[ignore_mask] = 1
        loss_map = F.cross_entropy(pred, target, reduction='none')

        return loss_map
