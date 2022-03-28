import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from torchvision.models.vgg import vgg19

import utils.platform_manager as platform_manager
from models.backbones.resnet import ResNet_Backbone
from models.base_net import Base_of_Network
from models.decoders.epcdepth_rsu import RSUDecoder


@platform_manager.MODELS.add_module
class EPCDepth_Net(Base_of_Network):
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

    def forward(self, inputs, is_train=True):
        self.inputs = inputs
        outputs = {}
        if is_train:
            losses = {}
            losses['loss'] = 0
            if self.data_graft:
                self._do_data_graft()

            img_h, img_w = self.inputs['color_s_aug']
            input_image = (self.inputs['color_s_aug'] - 0.45) / 0.225
            features = self.net_modules['enc'](input_image)
            norm_disps = self.net_modules['dec'](features)
            for i in range(len(norm_disps)):
                pred_disp = F.interpolate(norm_disps[i], [img_h, img_w],
                                          mode='bilinear',
                                          align_corners=False)
                _, pred_depth = self._disp_to_depth(pred_disp)
                outputs['disp_{}'.format(i)] = pred_disp
                outputs['depth_{}'.format(i)] = pred_depth

            outputs = self._get_warp_img(self.inputs['color_o_aug'],
                                         self.inputs['K'], self.inputs['T'],
                                         outputs)

            self._compute_losses(outputs, 's', losses, add_loss=False)
            self._add_final_losses('s', losses)
            return outputs, losses

        else:
            input_image = (self.inputs['color_s'] - 0.45) / 0.225
            features = self.net_modules['enc'](input_image)
            norm_disps = self.net_modules['dec'](features)
            _, pred_depth = self._disp_to_depth(norm_disps[0])
            outputs[('depth', 's')] = pred_depth * 5.4
            return outputs

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
            if 'color' in name:
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

    def _get_warp_img(self, sou_img, K, T, outputs, depth_hints=None):
        if depth_hints:
            outputs[('synth_hints')] = self._generate_warp_image(
                sou_img, K, T, depth_hints)
        for k, v in outputs.items():
            if 'depth' in k:
                outputs['synth_{}'.format(
                    k.split('_')[-1])] = self._generate_warp_image(
                        sou_img, K, T, v)
        return outputs

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

    def _spp_distillate(self, data, predicts):
        with torch.no_grad():
            disp_best = None
            decoder_disp_best = None
            reproj_loss_min = None
            for scale, disp in enumerate(predicts['disparity']):
                reproj_loss = self.compute_reprojection_loss(
                    predicts['warp_from_other_side'][scale], data['curr'])
                if scale == 0:
                    disp_best = disp
                    reproj_loss_min = reproj_loss
                elif scale == 5:
                    decoder_disp_best = disp_best.clone()
                    disp_best = disp
                    reproj_loss_min = reproj_loss
                else:
                    disp_best = torch.where(reproj_loss < reproj_loss_min,
                                            disp, disp_best)
                    reproj_loss_min, _ = torch.cat(
                        [reproj_loss, reproj_loss_min],
                        dim=1).min(dim=1, keepdim=True)

            if decoder_disp_best is not None:
                decoder_disp_best = decoder_disp_best.detach()
                # encoder_disp_best = disp_best.detach()
            else:
                decoder_disp_best = disp_best.detach()
