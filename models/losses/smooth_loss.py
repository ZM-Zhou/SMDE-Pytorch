import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import platform_manager


@platform_manager.LOSSES.add_module
class SmoothLoss(nn.Module):
    def __init__(self,
                 pred_n,
                 image_n=None,
                 more_kernel=False,
                 map_ch=1,
                 gamma_rate=1,
                 gray_img=False,
                 relative_smo=False,
                 device='cpu'):
        super().__init__()
        self.init_opts = locals()
        self.pred_n = pred_n
        self.image_n = image_n
        self.more_kernel = more_kernel
        self.map_ch = map_ch
        self.gray_img = gray_img
        self.gamma_rate = gamma_rate
        self.relative_smo = relative_smo

        if gray_img:
            img_ch = 1
        else:
            img_ch = 3

        self.kernels = {}
        if more_kernel:
            k = torch.Tensor([[0, 0, 0], [-1, 2, -1], [0, 0, 0]])
            k = k.unsqueeze(0).unsqueeze(0).repeat(img_ch, 1, 1, 1)
            self.kernels['img_w'] = k.to(device)

            k = torch.Tensor([[0, -1, 0], [0, 2, 0], [0, -1, 0]])
            k = k.unsqueeze(0).unsqueeze(0).repeat(img_ch, 1, 1, 1)
            self.kernels['img_h'] = k.to(device)

            k = torch.Tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
            k = k.unsqueeze(0).unsqueeze(0).repeat(self.map_ch, 1, 1, 1)
            self.kernels['d_w2'] = k.to(device)

            k = torch.Tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
            k = k.unsqueeze(0).unsqueeze(0).repeat(self.map_ch, 1, 1, 1)
            self.kernels['d_h2'] = k.to(device)

        else:
            k = torch.Tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
            k = k.unsqueeze(0).unsqueeze(0).repeat(img_ch, 1, 1, 1)
            self.kernels['dimg_w'] = k.to(device)

            k = torch.Tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
            k = k.unsqueeze(0).unsqueeze(0).repeat(img_ch, 1, 1, 1)
            self.kernels['dimg_h'] = k.to(device)

        k = torch.Tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
        k = k.unsqueeze(0).unsqueeze(0).repeat(self.map_ch, 1, 1, 1)
        self.kernels['d_w'] = k.to(device)

        k = torch.Tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
        k = k.unsqueeze(0).unsqueeze(0).repeat(self.map_ch, 1, 1, 1)
        self.kernels['d_h'] = k.to(device)
        self.replpad = nn.ReplicationPad2d(1)

    def forward(self, outputs, side):
        pred = outputs[self.pred_n.format(side)]
        if self.relative_smo:
            mean_pred = pred.mean(2, True).mean(3, True)
            pred = pred / (mean_pred + 1e-7)

        pad_pred = self.replpad(pred)
        if self.image_n is not None:
            img = outputs[self.image_n.format(side)].clone()
            m_rgb = torch.ones_like(img)
            m_rgb[:, 0, :, :] = 0.411 * m_rgb[:, 0, :, :]
            m_rgb[:, 1, :, :] = 0.432 * m_rgb[:, 1, :, :]
            m_rgb[:, 2, :, :] = 0.45 * m_rgb[:, 2, :, :]
            img = img + m_rgb
            if self.gray_img:
                img[:, 0, :, :] = 0.299 * img[:, 0, :, :] +\
                                  0.587 * img[:, 1, :, :] +\
                                  0.114 * img[:, 2, :, :]
                img = img[:, 0, :, :].unsqueeze(1)
            img = self.replpad(img)
        grad_w = torch.abs(
            F.conv2d(pad_pred,
                     self.kernels['d_w'],
                     padding=0,
                     stride=1,
                     groups=self.map_ch))
        grad_h = torch.abs(
            F.conv2d(pad_pred,
                     self.kernels['d_h'],
                     padding=0,
                     stride=1,
                     groups=self.map_ch))
        if self.more_kernel:
            grad_w += torch.abs(
                F.conv2d(pad_pred,
                         self.kernels['d_w2'],
                         padding=0,
                         stride=1,
                         groups=self.map_ch))
            grad_h += torch.abs(
                F.conv2d(pad_pred,
                         self.kernels['d_h2'],
                         padding=0,
                         stride=1,
                         groups=self.map_ch))
            if self.image_n is not None:
                grad_img_w = F.conv2d(img,
                                      self.kernels['img_w'],
                                      padding=0,
                                      stride=1,
                                      groups=img.shape[1])
                grad_img_h = F.conv2d(img,
                                      self.kernels['img_h'],
                                      padding=0,
                                      stride=1,
                                      groups=img.shape[1])
        else:
            if self.image_n is not None:
                grad_img_w = F.conv2d(img,
                                      self.kernels['dimg_w'],
                                      padding=0,
                                      stride=1,
                                      groups=img.shape[1])
                grad_img_h = F.conv2d(img,
                                      self.kernels['dimg_h'],
                                      padding=0,
                                      stride=1,
                                      groups=img.shape[1])
        grad_w = grad_w.mean(dim=1, keepdim=True)
        grad_h = grad_h.mean(dim=1, keepdim=True)
        if self.image_n is not None:
            grad_img_w = torch.abs(grad_img_w).mean(dim=1, keepdim=True)
            grad_img_h = torch.abs(grad_img_h).mean(dim=1, keepdim=True)
            grad_w *= torch.exp(-self.gamma_rate * grad_img_w)
            grad_h *= torch.exp(-self.gamma_rate * grad_img_h)

        return grad_w + grad_h
