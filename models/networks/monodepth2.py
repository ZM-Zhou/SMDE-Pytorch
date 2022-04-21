import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.resnet import ResNet_Backbone
from models.backbones.packnet_enc import PackEncoder
from models.base_net import Base_of_Network
from models.decoders.packnet_dec import PackDecoder
from models.decoders.pose import PoseDecoder, PoseNet
from models.decoders.upsample import UpSample_Layers
from utils import platform_manager


@platform_manager.MODELS.add_module
class Monodepth2(Base_of_Network):
    def _initialize_model(
            self,
            encoder_layer=18,
            min_depth=0.1,
            max_depth=100,
            image_size=[192, 640],
            data_mode=[1, -1],  # ["o"]
            use_depthhints=False,
            use_packnet=False,
            set_SCALE=None,
    ):
        self.init_opts = locals()

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.image_size = image_size
        self.data_mode = data_mode
        self.use_depthhints = use_depthhints
        self.use_packnet = use_packnet
        self.set_SCALE = set_SCALE

        self.mono_train = (1 in data_mode or -1 in data_mode)

        self.net_module = {}
        if self.use_packnet:
            self.net_module['encoder'] = PackEncoder()
            self.net_module['decoder'] = PackDecoder()

            if self.mono_train:
                self.net_module['pose_encoder'] = nn.Identity()
                self.net_module['pose_decoder'] = PoseNet(1)
        else:
            self.net_module['encoder'] = ResNet_Backbone(encoder_layer)
            if encoder_layer == 18:
                enc_ch_num = [64, 64, 128, 256, 512]
            else:
                enc_ch_num = [64, 256, 512, 1024, 2048]
            self.net_module['decoder'] = UpSample_Layers(enc_ch_num)

            if self.mono_train:
                self.net_module['pose_encoder'] = ResNet_Backbone(encoder_layer,
                                                                  in_ch=6)
                self.net_module['pose_decoder'] = PoseDecoder(
                    enc_ch_num, num_input_features=1, num_frames_to_predict_for=2)

        self._networks = nn.ModuleList(list(self.net_module.values()))

        self.projector = {}
        for scale in range(1):
            temp_scale = [length // 2**scale for length in image_size]
        self.projector[scale] = DepthProjector(temp_scale).to(self.device)

    def forward(self, inputs, is_train=True):
        self.inputs = inputs
        outputs = {}
        if is_train:
            losses = {}
            losses['loss'] = 0
            train_side = 's'

            x = (inputs['color_{}_aug'.format(train_side)] - 0.45) / 0.225
            features = self.net_module['encoder'](x)
            disp_outputs = self.net_module['decoder'](features, x.shape)

            K = self.inputs['K']
            inv_K = self.inputs['inv_K']
            if self.mono_train:
                outputs.update(self._get_poses())
            for scale in range(4):
                if self.use_packnet:
                    disp = disp_outputs[scale]
                else:
                    disp = torch.sigmoid(disp_outputs[scale])
                outputs['disp_raw_{}_{}'.format(scale, train_side)] = disp
                disp = F.interpolate(disp,
                                     self.image_size,
                                     mode='bilinear',
                                     align_corners=False)
                outputs['disp_{}_{}'.format(scale, train_side)] = disp
                _, depth = self._disp2depth(disp)
                if self.use_depthhints:
                    outputs['depth_{}_{}'.format(scale, train_side)] = depth * 5.4
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
            if self.use_depthhints:
                source_img = self.inputs['color_o']
                depth = inputs['hints_{}'.format(train_side)]
                T = inputs['T'].clone()
                projected_img, _ = self.projector[0](depth, inv_K, T, K,
                                                     source_img, False)
                outputs['proj_img_hints_{}'.format(
                        train_side)] = projected_img

            self._compute_losses(outputs, train_side, losses)
            return outputs, losses

        else:
            if self.use_packnet:
                x = inputs['color_s']
            else:
                x = (inputs['color_s'] - 0.45) / 0.225
            features = self.net_module['encoder'](x)
            disp_outputs = self.net_module['decoder'](features, x.shape)
            if self.use_packnet:
                disp = disp_outputs[0]
            else:
                disp = torch.sigmoid(disp_outputs[0])
            _, pred_depth = self._disp2depth(disp)
            if not self.mono_train:
                pred_depth = pred_depth * 5.4
            if self.set_SCALE is not None:
                pred_depth = pred_depth * self.set_SCALE
            outputs[('depth', 's')] = pred_depth
            # for scale in range(4):
            #     disp = torch.sigmoid(disp_outputs[scale])
            #     disp = F.interpolate(disp,
            #                          self.image_size,
            #                          mode='bilinear',
            #                          align_corners=False)
            #     outputs['disp_{}_{}'.format(scale, 's')] = disp

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
