import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.resnet import ResNet_Backbone
from models.base_net import Base_of_Network
from models.decoders.msfm_dec import R_MSFM6, R_MSFM3
from models.decoders.pose import PoseDecoder
from utils import platform_manager


@platform_manager.MODELS.add_module
class R_MSFM(Base_of_Network):
    def _initialize_model(
            self,
            encoder_layer=18,
            decoder_mode='6',
            min_depth=0.1,
            max_depth=100,
            image_size=[192, 640],
            data_mode=[1, -1],  # ["o"]
            
    ):
        self.init_opts = locals()

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.image_size = image_size
        self.data_mode = data_mode
        self.decoder_mode = decoder_mode

        self.mono_train = (1 in data_mode or -1 in data_mode)

        self.net_module = {}
        self.net_module['encoder'] = ResNet_Backbone(encoder_layer, ignore_last2=True)
        if self.decoder_mode == '6':
            self.net_module['decoder'] = R_MSFM6(False)
        elif self.decoder_mode == '3':
            self.net_module['decoder'] = R_MSFM3(False)
        elif self.decoder_mode == '6x':
            self.net_module['decoder'] = R_MSFM6(True)
        else:
            raise NotImplementedError


        if self.mono_train:
            self.net_module['pose_encoder'] = ResNet_Backbone(18, in_ch=6)
            enc_ch_num = [64, 64, 128, 256, 512]
            self.net_module['pose_decoder'] = PoseDecoder(
                enc_ch_num, num_input_features=1, num_frames_to_predict_for=2)

        self._networks = nn.ModuleList(list(self.net_module.values()))

        self.projector = {}
        self.projector[0] = DepthProjector(image_size).to(self.device)

    def forward(self, inputs, is_train=True):
        self.inputs = inputs
        outputs = {}
        if is_train:
            losses = {}
            losses['loss'] = 0
            train_side = 's'

            x = (inputs['color_{}_aug'.format(train_side)] - 0.45) / 0.225
            features = self.net_module['encoder'](x)
            disp_outputs = self.net_module['decoder'](features,
                                                      x.shape)

            K = self.inputs['K']
            inv_K = self.inputs['inv_K']
            if self.mono_train:
                outputs.update(self._get_poses())
            for scale in range(len(disp_outputs)-1, -1, -1):
                disp = disp_outputs[scale]
                outputs['disp_{}_{}'.format(scale, train_side)] = disp
                _, depth = self._disp2depth(disp)
                for id_frame in self.data_mode:
                    source_img = self.inputs[('color_{}'.format(id_frame))]
                    if id_frame != 'o':
                        T = outputs['T_{}'.format(id_frame)]
                    else:
                        T = inputs['T'].clone()
                        T[:, 0, 3] = T[:, 0, 3] / 5.4
                    projected_img, _ = self.projector[0](depth, inv_K, T, K,
                                                         source_img, False)
                    outputs['proj_img_{}_{}_{}'.format(
                        id_frame, scale, train_side)] = projected_img

            self._compute_losses(outputs, train_side, losses)
            return outputs, losses

        else:
            x = (inputs['color_s'] - 0.45) / 0.225
            features = self.net_module['encoder'](x)
            disp_outputs = self.net_module['decoder'](features,
                                                      x.shape)
            
            disp = disp_outputs[0]
            _, pred_depth = self._disp2depth(disp)
            if not self.mono_train:
                pred_depth = pred_depth * 5.4         
            outputs[('depth', 's')] = pred_depth

            return outputs

    def _get_poses(self):
        outputs = {}
        pose_feats = {
            f_i: self.inputs['color_{}_aug'.format(f_i)]
            for f_i in self.data_mode
        }
        pose_feats[0] = self.inputs['color_s_aug']
        for f_i in self.data_mode:
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]
            pose_inputs = [
                self.net_module['pose_encoder'](torch.cat(pose_inputs, 1))
            ]
            axisangle, translation = self.net_module['pose_decoder'](
                pose_inputs)
            outputs['axisangle_{}'.format(f_i)] = axisangle
            outputs['translation_{}'.format(f_i)] = translation
            outputs['T_{}'.format(f_i)] = self._T_from_params(axisangle[:, 0],
                                                              translation[:,
                                                                          0],
                                                              invert=(f_i < 0))
        return outputs

    def _T_from_params(self, axisangle, translation, invert=False):
        """Convert the network's (axisangle, translation) output into a 4x4
        matrix."""
        R = rot_from_axisangle(axisangle)
        t = translation.clone()
        if invert:
            R = R.transpose(1, 2)
            t *= -1
        T = get_translation_matrix(t)
        if invert:
            M = torch.matmul(R, T)
        else:
            M = torch.matmul(T, R)

        return M

    def _disp2depth(self, disp):
        """Convert network's sigmoid output into depth prediction."""
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix."""
    T = torch.zeros(translation_vector.shape[0], 4,
                    4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix (adapted
    from https://github.com/Wallacoloo/printipi) Input 'vec' has to be
    Bx1x3."""
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class DepthProjector(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.height = image_size[0]
        self.width = image_size[1]

        meshgrid = np.meshgrid(range(self.width),
                               range(self.height),
                               indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(
            torch.stack(
                [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0),
            0)
        # self.pix_coords = self.pix_coords.repeat(1, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones],
                                                 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K, T, K, img, is_mask=False):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(depth.shape[0], 1, -1) * cam_points
        cam_points = torch.cat(
            [cam_points, self.ones.repeat(depth.shape[0], 1, 1)], 1)

        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, cam_points)

        pix_coords = cam_points[:, :2, :] \
            / (cam_points[:, 2, :].unsqueeze(1) + 1e-8)
        pix_coords = pix_coords.view(cam_points.shape[0], 2, self.height,
                                     self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2

        warped_img = F.grid_sample(img,
                                   pix_coords,
                                   mode='bilinear',
                                   padding_mode='border',
                                   align_corners=True)

        if is_mask:
            mask = ((pix_coords >= -1) & (pix_coords <= 1)).to(torch.float)
            mask = torch.min(mask, dim=3, keepdim=True)[0].permute(0, 3, 1, 2)
        else:
            mask = None

        return warped_img, mask
