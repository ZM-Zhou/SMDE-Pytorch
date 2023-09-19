import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import platform_manager


@platform_manager.LOSSES.add_module
class GradLoss(nn.Module):
    def __init__(self,
                 pred_n,
                 target_n,
                 t_grad=0,
                 device='cpu'):
        super().__init__()
        self.init_opts = locals()
        self.pred_n = pred_n
        self.target_n = target_n
        self.t_grad = t_grad

        self.kernels = {}
        k = torch.Tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
        k = k.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
        self.kernels['d_w'] = k.to(device)

        k = torch.Tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
        k = k.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
        self.kernels['d_h'] = k.to(device)
        
        self.replpad = nn.ReplicationPad2d(1)

    def forward(self, outputs, side):
        pred = outputs[self.pred_n.format(side)]
        target = outputs[self.target_n.format(side)].detach()
       
        pad_pred = self.replpad(pred)
        pad_target = self.replpad(target)
        pred_w = torch.abs(
            F.conv2d(pad_pred,
                     self.kernels['d_w'],
                     padding=0,
                     stride=1,
                     groups=1))
        pred_h = torch.abs(
            F.conv2d(pad_pred,
                     self.kernels['d_h'],
                     padding=0,
                     stride=1,
                     groups=1))

        target_w = torch.abs(
            F.conv2d(pad_target,
                     self.kernels['d_w'],
                     padding=0,
                     stride=1,
                     groups=1))
        target_h = torch.abs(
            F.conv2d(pad_target,
                     self.kernels['d_h'],
                     padding=0,
                     stride=1,
                     groups=1))
        
        pred_w = pred_w.mean(dim=1, keepdim=True)
        pred_h = pred_h.mean(dim=1, keepdim=True)

        target_w = target_w.mean(dim=1, keepdim=True)
        target_h = target_h.mean(dim=1, keepdim=True)

        loss_map = torch.abs(target_w - pred_w) + torch.abs(target_h - pred_h)
        if self.t_grad > 0:
            loss_map = loss_map - self.t_grad
            loss_map[loss_map < 0] = 0
        return loss_map
