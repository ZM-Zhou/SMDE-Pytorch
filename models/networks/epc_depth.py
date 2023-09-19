import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.platform_manager as platform_manager
from models.backbones.resnet import ResNet_Backbone
from models.base_model import Base_of_Model
from models.decoders.epcdepth_rsu import RSUDecoder


@platform_manager.MODELS.add_module
class EPCDepth_Net(Base_of_Model):
    def _initialize_model(self,
                          backbone='Res50',
                          max_depth=100,
                          min_depth=0.1,
                          data_graft=True):
        self.init_opts = locals()

        self.backbone = backbone
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.data_graft = data_graft

        # Initialize architecture
        self.net_modules = {}
        if 'Res' in self.backbone:
            layer_num = int(self.backbone[3:])
            self.net_modules['enc'] = ResNet_Backbone(layer_num=layer_num)
            if layer_num == 18:
                num_ch_enc = [64, 64, 128, 256, 512]
            elif layer_num == 50:
                num_ch_enc = [64, 256, 512, 1024, 2048]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self.net_modules['dec'] = RSUDecoder(num_output_channels=1,
                                             use_encoder_disp=True,
                                             encoder_layer_channels=num_ch_enc)

        self._networks = nn.ModuleList(list(self.net_modules.values()))

    def forward(self, x, outputs, **kargs):
        features = self.net_modules['enc'](x)
        norm_disps = self.net_modules['dec'](features)
        _, pred_depth = self._disp_to_depth(norm_disps[0])
        
        outputs[('depth', 's')] = pred_depth * 5.4
        outputs['norm_disps_s'] = norm_disps
        pred = pred_depth * 5.4

        return pred, outputs

    def _preprocess_inputs(self):
        if self.is_train:
            if self.data_graft:
                self._do_data_graft()
            aug = '_aug'
        else:
            aug = ''

        x = (self.inputs['color_s{}'.format(aug)] - 0.45) / 0.225
        return x

    def _postprocess_outputs(self, outputs):
        loss_sides = ['s']
        img_h, img_w = self.inputs['color_s_aug'].shape[2:]
        sou_img = self.inputs['color_o']
        K =self.inputs['K']
        T = self.inputs['T'].clone()
        T[:, 0, 3] = T[:, 0, 3] / 5.4

        norm_disps = outputs['norm_disps_s']
        for i in range(len(norm_disps)):
            pred_disp = F.interpolate(norm_disps[i], [img_h, img_w],
                                        mode='bilinear',
                                        align_corners=False)
            _, pred_depth = self._disp_to_depth(pred_disp)
            outputs['disp_{}_s'.format(i)] = pred_disp.detach()
            outputs['depth_{}_s'.format(i)] = pred_depth * 5.4

            outputs['synth_{}_s'.format(i)] = self._generate_warp_image(
                sou_img, K, T, pred_depth)
        
        outputs['synth_hints_s'] = self._generate_warp_image(
            sou_img, K, self.inputs['T'], self.inputs['hints_s'])
    
        return loss_sides, outputs

    def _do_data_graft(self):
        rand_w = random.randint(0, 4) / 5
        b, c, h, w = self.inputs['color_s'].shape
        if int(rand_w * h) == 0:
            return
        l_num = self.inputs['direct'][self.inputs['direct'] > 0].shape[0]
        r_num = self.inputs['direct'][self.inputs['direct'] < 0].shape[0]
        l_graft_idx = torch.randperm(l_num).to(self.device)
        r_graft_idx = torch.randperm(r_num).to(self.device)
        graft_h = int(rand_w * h)
        flip = random.random()
        for name in self.inputs:
            if 'color' in name or 'hints' in name:
                self.inputs[name][self.inputs['direct'] > 0, :,
                                  graft_h:, :] = self.inputs[name][
                                      self.inputs['direct'] > 0].clone()[
                                          l_graft_idx, :, graft_h:, :]
                self.inputs[name][self.inputs['direct'] < 0, :,
                                  graft_h:, :] = self.inputs[name][
                                      self.inputs['direct'] < 0].clone()[
                                          r_graft_idx, :, graft_h:, :]
                if flip < 0.5:
                    d = self.inputs[name].clone()
                    self.inputs[name][:, :, :-graft_h] = d[:, :, graft_h:]
                    self.inputs[name][:, :, -graft_h:] = d[:, :, :graft_h]

    def _disp_to_depth(self, disp):
        """Convert network's sigmoid output into depth prediction The formula
        for this conversion is given in the 'additional considerations' section
        of the paper."""
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth

    def _generate_warp_image(self, img, K, T, D):
        batch_size, _, height, width = img.shape
        eps = 1e-7
        inv_K = torch.from_numpy(np.linalg.pinv(K.cpu().numpy())).type_as(D)

        meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        id_coords = nn.Parameter(torch.from_numpy(id_coords),
                                 requires_grad=False).type_as(D)

        ones = nn.Parameter(torch.ones(batch_size, 1, height * width),
                            requires_grad=False).type_as(D)

        pix_coords = torch.unsqueeze(
            torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0)
        pix_coords = pix_coords.repeat(batch_size, 1, 1)
        pix_coords = nn.Parameter(torch.cat([pix_coords, ones], 1),
                                  requires_grad=False).type_as(D)

        cam_points = torch.matmul(inv_K[:, :3, :3], pix_coords)
        cam_points = D.view(batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, ones], 1)

        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, cam_points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) +
                                             eps)
        pix_coords = pix_coords.view(batch_size, 2, height, width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= width - 1
        pix_coords[..., 1] /= height - 1
        pix_coords = (pix_coords - 0.5) * 2

        warp_img = torch.nn.functional.grid_sample(img,
                                                   pix_coords,
                                                   padding_mode='border')
        return warp_img
