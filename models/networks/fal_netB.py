import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg19

from models.backbones.residualconv import Residual_Conv
from models.backbones.resnet import ResNet_Backbone
from models.base_net import Base_of_Network
from models.decoders.upsample import UpSample_Layers
from models.decoders.upsample_v2 import UpSample_Layers_v2
from utils import platform_manager


@platform_manager.MODELS.add_module
class FAL_NetB(Base_of_Network):
    def _initialize_model(
        self,
        out_num=49,
        encoder='FALB',
        decoder='Upv2',
        num_ch_dec = [64, 128, 256, 256, 256],
        out_scales = [0],
        min_disp=2,
        max_disp=300,
        image_size=[192, 640],
        mom_module=False,
        raw_fal_arch=False,
    ):
        self.init_opts = locals()
        self.out_num = out_num
        self.encoder = encoder
        self.decoder = decoder
        self.num_ch_dec = num_ch_dec
        self.out_scales = out_scales
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.image_size = image_size
        self.mom_module = mom_module
        self.raw_fal_arch = raw_fal_arch

        # Initialize architecture
        self.convblocks = {}
        if self.encoder == 'FALB':
            num_ch_enc = [32, 64, 128, 256, 256, 256, 512]
            self.convblocks['enc'] = Residual_Conv(num_ch_enc=num_ch_enc,
                                                   input_flow=self.raw_fal_arch)
        elif 'Res' in self.encoder:
            encoder_layer = int(self.encoder[3:])
            self.convblocks['enc'] = ResNet_Backbone(encoder_layer)
            if encoder_layer <= 34:
                num_ch_enc = [64, 64, 128, 256, 512]
            else:
                num_ch_enc = [64, 256, 512, 1024, 2048]
        
        if self.decoder == 'Upv2':
            assert self.encoder == 'FALB',\
                'Upv2 decoder must be used with FALB encoder'
            self.convblocks['dec'] = UpSample_Layers_v2(
                num_ch_enc,
                self.num_ch_dec,
                output_ch=out_num,
                raw_fal_arch=self.raw_fal_arch)
        elif self.decoder == 'Upv1':
            # When use Upv1 with FALB, please set
            # num_ch_dec = [64, 64, 128, 256, 256, 256]
            self.convblocks['dec'] = UpSample_Layers(num_ch_enc,
                                                     self.num_ch_dec,
                                                     output_ch=out_num,
                                                     out_scales=[0])
        self._networks = nn.ModuleList(list(self.convblocks.values()))

        # Initialize transformer
        disp_range = []
        rel_disp = self.max_disp / self.min_disp
        for disp_idx in range(self.out_num):
            index_num = rel_disp**(disp_idx / (self.out_num - 1) - 1)
            disp = self.max_disp * index_num
            disp_range.append(disp)

        volume_dk = torch.tensor(disp_range).unsqueeze(1).unsqueeze(1)
        volume_dk = volume_dk.unsqueeze(0).unsqueeze(0)

        self.volume_dk = volume_dk.to(self.device)

        # side(s) transformer means warped to LEFT
        # other-side(o) transformer means warped to RIGHT
        self.transformer = {}
        self.transformer['o'] = DispTransformer(self.image_size, disp_range,
                                                self.device)
        self.transformer['s'] = DispTransformer(self.image_size,
                                                [-disp for disp in disp_range],
                                                self.device)

        self.feat_net = []
        # load pretrained vgg19 module
        vgg = vgg19(pretrained=True, progress=False).features.to(self.device)
        vgg_feats = list(vgg.modules())
        vgg_layer_num = [5, 5, 9]
        read_module_num = 0
        for module_num in vgg_layer_num:
            self.feat_net.append(nn.Sequential())
            for _ in range(module_num):
                self.feat_net[-1].add_module(str(read_module_num),
                                             vgg_feats[read_module_num + 1])
                read_module_num += 1

        self.train_sides = ['s']
        if self.mom_module:
            self.train_sides.append('o')
            self.fix_network = {}
            self.fix_network['enc'] = copy.deepcopy(self.convblocks['enc']).to(self.device)
            self.fix_network['dec'] = copy.deepcopy(self.convblocks['dec']).to(self.device)
            self.loaded_flag = False
        else:
            self.loaded_flag = True

    def forward(self, inputs, is_train=True):
        self.inputs = inputs
        outputs = {}
        if is_train:
            losses = {}
            losses['loss'] = 0
            if not self.loaded_flag:
                self.fix_network['enc'].load_state_dict(
                    self.convblocks['enc'].state_dict().copy())
                self.fix_network['dec'].load_state_dict(
                    self.convblocks['dec'].state_dict().copy())
                self.loaded_flag = True

            directs = self.inputs['direct']
            directs = directs.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            for train_side in self.train_sides:
                # oside is used to select the disparity transformer
                oside = 'o' if train_side == 's' else 's'
                # process inputs
                t_img_aug = self.inputs['color_{}_aug'.format(train_side)]

                # generate the train side disparity
                raw_volume = self._get_dispvolume_from_img(train_side)
                d_volume = raw_volume.unsqueeze(1)

                p_volume = self._get_probvolume_from_dispvolume(d_volume)
                pred_disp = self._get_disp_from_probvolume(p_volume, directs)
                w_d_volume = self.transformer[oside]\
                    .get_warped_volume(d_volume, directs)
                w_p_volume = self._get_probvolume_from_dispvolume(w_d_volume)
                # biuld the outputs
                outputs['disp_{}'.format(train_side)] = pred_disp
                outputs['p_volume_{}'.format(train_side)] = p_volume.detach()
                outputs['w_p_volume_{}'.format(
                    train_side)] = w_p_volume.detach()
                if self.mom_module:
                    outputs['depth_{}'.format(
                        train_side)] = self._get_depth_from_disp(pred_disp)
                    with torch.no_grad():
                        if train_side == 's':
                            t_img_aug_f = torch.flip(t_img_aug, [3])
                        else:
                            t_img_aug_f = t_img_aug
                        if self.raw_fal_arch:
                            B, C, H, W = t_img_aug_f.shape
                            flow = torch.ones(B, 1, H, W).type(t_img_aug_f.type())
                            flow[:, 0, :, :] = self.max_disp * flow[:, 0, :, :] / 100
                            x_f = t_img_aug_f
                        else:
                            x_f = t_img_aug_f / 0.225
                            flow = None
                        features_f = self.fix_network['enc'](x_f, flow)
                        raw_volume_f = self.fix_network['dec'](
                            features_f, t_img_aug_f.shape)
                        d_volume_f = raw_volume_f.unsqueeze(1)
                        p_volume_f = self._get_probvolume_from_dispvolume(
                            d_volume_f)
                        pred_disp_f = self._get_disp_from_probvolume(
                            p_volume_f, directs)
                        pred_depth_f = self._get_depth_from_disp(pred_disp_f)
                        if train_side == 's':
                            pred_depth_ff = torch.flip(pred_depth_f, [3])
                        else:
                            pred_depth_ff = pred_depth_f
                        outputs['disp_f_{}'.format(train_side)] = pred_disp_f
                        outputs['depth_ff_{}'.format(
                            train_side)] = pred_depth_ff
                        mask = torch.ones_like(pred_depth_ff)
                        mask = mask / pred_depth_ff.max()
                        outputs['ff_mask_{}'.format(oside)] = mask

                # generate the synthetic image in right side
                # named side but its synthesized the image of other side
                w_img = self.transformer[oside]\
                    .get_warped_frame(t_img_aug, directs)
                synth_img = (w_p_volume * w_img).sum(dim=2)
                outputs['synth_img_{}'.format(train_side)] = synth_img

            # compute the occlusion mask
            if self.mom_module:
                for train_side in self.train_sides:
                    oside = 'o' if train_side == 's' else 's'
                    p_volume = outputs['p_volume_{}'.format(train_side)]
                    cyc_p_volume = self.transformer[oside]\
                        .get_warped_volume(p_volume, directs)
                    occ_mask1 = cyc_p_volume.sum(dim=2)
                    w_p_volume = outputs['w_p_volume_{}'.format(oside)]
                    cyc_w_p_volume = self.transformer[oside]\
                        .get_warped_volume(w_p_volume, directs)
                    occ_mask2 = cyc_w_p_volume.sum(dim=2)
                    occ_mask = (occ_mask1 * occ_mask2).clamp(0, 1)
                    outputs['mask_{}'.format(train_side)] = occ_mask
                    outputs['inv_mask_{}'.format(train_side)] = (1 - occ_mask)
                    outputs['ff_mask_{}'.format(oside)] = (
                        1 -
                        occ_mask) * outputs['ff_mask_{}'.format(train_side)]

            # extract features by vgg
            for train_side in self.train_sides:
                oside = 'o' if train_side == 's' else 's'
                raw_img = self.inputs['color_{}_aug'.format(oside)]
                synth_img = outputs['synth_img_{}'.format(train_side)]
                if self.mom_module:
                    occ_mask = outputs['mask_{}'.format(train_side)]
                    inv_occ_mask = outputs['inv_mask_{}'.format(train_side)]
                    synth_img = synth_img * occ_mask + raw_img * inv_occ_mask
                    outputs['synth_img_{}'.format(train_side)] = synth_img

                with torch.no_grad():
                    raw_feats = self._get_conv_feats_from_image(raw_img)
                synth_feats = self._get_conv_feats_from_image(synth_img)

                for feat_idx in range(3):
                    rawf_name = 'raw_feats_{}_{}'.format(feat_idx, train_side)
                    outputs[rawf_name] = raw_feats[feat_idx]
                    synthf_name = 'synth_feats_{}_{}'.format(
                        feat_idx, train_side)
                    outputs[synthf_name] = synth_feats[feat_idx]

            # compute the losses
            for train_side in self.train_sides:
                self._compute_losses(outputs,
                                     train_side,
                                     losses,
                                     add_loss=False)
                self._add_final_losses(train_side, losses)
            return outputs, losses

        else:
            raw_volume = self._get_dispvolume_from_img('s', aug='')
            d_volume = raw_volume.unsqueeze(1)

            p_volume = self._get_probvolume_from_dispvolume(d_volume)
            pred_disp = self._get_disp_from_probvolume(p_volume)
            pred_depth = self._get_depth_from_disp(pred_disp)
            outputs[('depth', 's')] = pred_depth
            return outputs

    def _get_dispvolume_from_img(self, side, aug='_aug'):
        input_img = self.inputs['color_{}{}'.format(side, aug)].clone()
        if side == 'o':
            input_img = torch.flip(input_img, dims=[3])

        if self.raw_fal_arch:
            B, C, H, W = input_img.shape
            flow = torch.ones(B, 1, H, W).type(input_img.type())
            flow[:, 0, :, :] = self.max_disp * flow[:, 0, :, :] / 100
            x = input_img
            features = self.convblocks['enc'](x, flow)
        else:
            x = input_img
            features = self.convblocks['enc'](x)

        out_volume = self.convblocks['dec'](features, input_img.shape)
        if side == 'o':
            out_volume = torch.flip(out_volume, dims=[3])
        return out_volume

    def _upsample(self, x, shape):
        return F.interpolate(x, size=shape[2:], mode='nearest')

    def _get_probvolume_from_dispvolume(self, volume):
        return F.softmax(volume, dim=2)

    def _get_disp_from_probvolume(self, volume, directs=None):
        disp = (volume * self.volume_dk).sum(dim=2)
        if directs is not None:
            disp = disp * torch.abs(directs)
        return disp

    def _get_mask_from_probvolume(self, volume):
        raw_mask = volume.sum(dim=2)
        return raw_mask.clamp(0, 1)

    def _get_depth_from_disp(self, disp):
        if 'disp_k' in self.inputs:
            k = self.inputs['disp_k'].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:
            k = torch.tensor([721.54 * 0.54], dtype=torch.float)
            k = k.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(disp)
        depth = torch.abs(k) / disp
        return depth

    def _get_conv_feats_from_image(self, raw_img):
        feats = []
        x = raw_img
        for block_idx in range(len(self.feat_net)):
            x = self.feat_net[block_idx](x)
            feats.append(x)
        return feats


class Conv3x3(nn.Module):
    """Layer to pad and convolve input from
    https://github.com/nianticlabs/monodepth2."""
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU from
    https://github.com/nianticlabs/monodepth2."""
    def __init__(self, in_channels, out_channels, bn=False, nonlin=True):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        if nonlin:
            self.nonlin = nn.ELU(inplace=True)
        else:
            self.nonlin = None
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)
        if self.nonlin is not None:
            out = self.nonlin(out)
        return out


class AtrousConv(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 apply_bn_first=True):
        super().__init__()
        self.atrous_conv = torch.nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module(
                'first_bn',
                nn.BatchNorm2d(in_channels,
                               momentum=0.01,
                               affine=True,
                               track_running_stats=True,
                               eps=1.1e-5))

        self.atrous_conv.add_module(
            'aconv_sequence',
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels * 2,
                          bias=False,
                          kernel_size=1,
                          stride=1,
                          padding=0),
                nn.BatchNorm2d(out_channels * 2,
                               momentum=0.01,
                               affine=True,
                               track_running_stats=True), nn.ReLU(),
                nn.Conv2d(in_channels=out_channels * 2,
                          out_channels=out_channels,
                          bias=False,
                          kernel_size=3,
                          stride=1,
                          padding=(dilation, dilation),
                          dilation=dilation)))

    def forward(self, x):
        return self.atrous_conv.forward(x)


class DispTransformer(object):
    def __init__(self, image_size, disp_range, device='cuda'):
        i_tetha = torch.zeros(1, 2, 3)
        i_tetha[:, 0, 0] = 1
        i_tetha[:, 1, 1] = 1
        normal_coord = F.affine_grid(i_tetha,
                                     [1, 1, image_size[0], image_size[1]],
                                     align_corners=True)
        self.base_coord = normal_coord.to(device)

        # add disparity
        self.normal_disp_bunch = []
        zeros = torch.zeros_like(self.base_coord)
        for disp in disp_range:
            disp_map = zeros.clone()
            disp_map[..., 0] = disp_map[..., 0] + disp
            normal_disp_map = self._get_normalize_coord(disp_map, image_size)
            normal_disp_map = normal_disp_map.to(torch.device(device))
            self.normal_disp_bunch.append(normal_disp_map)

        self.ch_num = len(disp_range)

    def _get_normalize_coord(self, coord, image_size):
        """TODO."""
        coord[..., 0] /= (image_size[1] / 2)
        coord[..., 1] /= (image_size[0] / 2)
        return coord

    def get_warped_volume(self, volume, directs):
        """Warp the volume by disparity range with zeros padding."""
        # bs = volume.shape[0]
        new_volume = []
        for ch_idx in range(self.ch_num):
            normal_disp = self.normal_disp_bunch[ch_idx]# .repeat(bs, 1, 1, 1)
            # To adapt flip data augment
            grid_coord = normal_disp * directs + self.base_coord
            warped_frame = F.grid_sample(volume[:, :, ch_idx, ...],
                                         grid_coord,
                                         mode='bilinear',
                                         padding_mode='zeros',
                                         align_corners=True)
            new_volume.append(warped_frame.unsqueeze(2))
        return torch.cat(new_volume, dim=2)

    def get_warped_frame(self, x, directs, base_coord=None, coords_k=None):
        """Warp the images by disparity range with border padding."""
        if base_coord is None:
            base_coord = self.base_coord
            bs, ch, h, w = x.shape
        else:
            bs, h, w, _ = base_coord.shape
            ch = x.shape[1]
            directs *= coords_k
        # frame_volume = torch.zeros((bs, ch, self.ch_num, h, w)).to(x)
        frame_volume = []
        for ch_idx in range(self.ch_num):
            normal_disp = self.normal_disp_bunch[ch_idx]# .repeat(bs, 1, 1, 1)
            # To adapt flip data augment
            grid_coord = normal_disp * directs + base_coord
            warped_frame = F.grid_sample(x,
                                         grid_coord,
                                         mode='bilinear',
                                         padding_mode='border',
                                         align_corners=True)
            frame_volume.append(warped_frame.unsqueeze(2))
        return torch.cat(frame_volume, dim=2)


class DepthProjecter(nn.Module):
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


class SelfOccluMask(nn.Module):
    def __init__(self, maxDisp=21, device='cpu'):
        super(SelfOccluMask, self).__init__()
        self.maxDisp = maxDisp
        self.device = device
        self.init_kernel()

    def init_kernel(self):
        self.convweights = torch.zeros(self.maxDisp, 1, 3,
                                       self.maxDisp + 2).to(self.device)
        self.occweights = torch.zeros(self.maxDisp, 1, 3,
                                      self.maxDisp + 2).to(self.device)
        self.convbias = (torch.arange(self.maxDisp).type(torch.FloatTensor) +
                         1).to(self.device)
        self.padding = nn.ReplicationPad2d((0, self.maxDisp + 1, 1, 1))
        for i in range(0, self.maxDisp):
            self.convweights[i, 0, :, 0:2] = 1 / 6
            self.convweights[i, 0, :, i + 2:i + 3] = -1 / 3
            self.occweights[i, 0, :, i + 2:i + 3] = 1 / 3

    def forward(self, dispmap, bsline):
        maskl = self.computeMask(dispmap, 'l')
        maskr = self.computeMask(dispmap, 'r')
        lind = bsline < 0
        rind = bsline > 0
        mask = torch.zeros_like(dispmap)
        mask[lind, :, :, :] = maskl[lind, :, :, :]
        mask[rind, :, :, :] = maskr[rind, :, :, :]
        return mask

    def computeMask(self, dispmap, direction):
        with torch.no_grad():
            if direction == 'l':
                padmap = self.padding(dispmap)
                output = nn.functional.conv2d(padmap, self.convweights,
                                              self.convbias)
                output = torch.abs(output)
                mask, min_idx = torch.min(output, dim=1, keepdim=True)
                mask = mask.clamp(0, 1)
            elif direction == 'r':
                dispmap_opp = torch.flip(dispmap, dims=[3])
                padmap = self.padding(dispmap_opp)
                output = nn.functional.conv2d(padmap, self.convweights,
                                              self.convbias)
                output = torch.abs(output)
                mask, min_idx = torch.min(output, dim=1, keepdim=True)
                mask = mask.clamp(0, 1)
                mask = torch.flip(mask, dims=[3])
        return mask
