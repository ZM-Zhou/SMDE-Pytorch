import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.backbones.resnet import ResNet_Backbone
from models.base_model import Base_of_Model
from models.decoders.upsample_aspp import UpSample_withDenseASPP
from utils import platform_manager


@platform_manager.MODELS.add_module
class OCFD_Net(Base_of_Model):
    """Network for learning occlusion-aware coarse-to-fine depth map.

    Args:
        out_num (int): Nmber of discrete disparity levels. Default: 49.
        min_disp (int): Minimum disparity. Default: 2.
        max_disp (int): Maximum disparity. Default: 300.
        image_size (list): Image size at the  training stage.
            Default: [192, 640].
        fix_residual (int): Weight of the scene depth residual.
            Default: 10.
        occ_mask_mode (str | None): Mode of the cclusion mask. Default: 'mv'.
        pred_out (bool): Disable the scene depth residual. Default: False.
    """
    def _initialize_model(self,
                          out_num=49,
                          min_disp=2,
                          max_disp=300,
                          image_size=[192, 640],
                          fix_residual=10,
                          occ_mask_mode='vm',
                          pred_out=False,
                          new_decode=False,
                          set_SCALE=1):
        self.init_opts = locals()

        self.out_num = out_num
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.image_size = image_size
        self.fix_residual = fix_residual
        self.occ_mask_mode = occ_mask_mode
        self.pred_out = pred_out
        self.new_decode = new_decode
        self.set_SCALE = set_SCALE

        # Initialize architecture
        self.convblocks = {}
        num_ch_enc = [64, 256, 512, 1024, 2048]
        num_ch_dec = [64, 64, 64, 128, 256]
        self.convblocks['enc'] = ResNet_Backbone()
        self.convblocks['dec'] = UpSample_withDenseASPP(
            num_ch_enc, num_ch_dec, self.out_num + 1,
            unfold_upsample=self.new_decode)

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

        # l transformer means warped to LEFT
        self.transformer = {}
        self.transformer['o'] = DispTransformer(self.image_size, disp_range,
                                                self.device)
        self.transformer['s'] = DispTransformer(self.image_size,
                                                [-disp for disp in disp_range],
                                                self.device)

        # depth projecter
        self.transformer['proj'] = DepthProjecter(image_size)\
            .to(self.device)

        # load the pretrained res18 model
        self.res_block = []
        resnet = models.resnet18(pretrained=True,
                                 progress=False).to(self.device)
        self.res_block.append(nn.Sequential())
        self.res_block[-1].add_module('conv1', resnet.conv1)
        self.res_block[-1].add_module('bn1', resnet.bn1)
        self.res_block[-1].add_module('relu1', resnet.relu)
        self.res_block.append(nn.Sequential())
        self.res_block[-1].add_module('max_pool', resnet.maxpool)
        self.res_block[-1].add_module('layer1', resnet.layer1)
        self.res_block.append(nn.Sequential())
        self.res_block[-1].add_module('layer2', resnet.layer2)

        # load the occlusion mask builder
        if self.occ_mask_mode is not None:
            self.occ_computer = {}
            self.occ_computer[0] = SelfOccluMask(41, device=self.device)

        self.train_sides = ['s']

    def forward(self, x, outputs, **kargs):
        features = self.convblocks['enc'](x)
        raw_volume = self.convblocks['dec'](features, x.shape)
        d_volume = raw_volume[:, :self.out_num, ...].unsqueeze(1)
        extra_normal_depth = torch.sigmoid(raw_volume[:, self.out_num,
                                                        ...].unsqueeze(1))
        p_volume = self._get_probvolume_from_dispvolume(d_volume)
        pred_disp = self._get_disp_from_probvolume(p_volume)
        pred_depth = 401.55 / pred_disp * self.set_SCALE

        d_weight = self.fix_residual
        residual_depth = (extra_normal_depth - 0.5) * d_weight
        fine_depth = (pred_depth + residual_depth).clamp(1e-3, 1e5)

        if self.pred_out:
            outputs[('depth', 's')] = pred_depth
            pred = pred_depth
        else:
            outputs[('depth', 's')] = fine_depth
            pred = fine_depth
        
        if self.is_train:
            outputs['depth_s'] = pred_depth
            outputs['fine_depth_s'] = fine_depth
            outputs['d_volume_s'] = d_volume
            outputs['disp_s'] = pred_disp
            outputs['extra_normal_depth_s'] = extra_normal_depth
    
        return pred, outputs

    def _preprocess_inputs(self):
        if self.is_train:
            aug = '_aug'
        else:
            aug = ''
        x = self.inputs['color_s{}'.format(aug)] / 0.225
        return x

    def _postprocess_outputs(self, outputs):
        inv_K = self.inputs['inv_K']
        K = self.inputs['K']
        T = self.inputs['T']
        loss_sides = ['s']
        train_side = 's'
        oside = 'o'

        t_img_aug = self.inputs['color_{}_aug'.format(train_side)]
        s_img_aug = self.inputs['color_{}_aug'.format(oside)]
        directs = self.inputs['direct']
        directs = directs.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        d_volume = outputs['d_volume_{}'.format(train_side)]
        w_d_volume = self.transformer[oside]\
            .get_warped_volume(d_volume, directs)
        w_p_volume = self._get_probvolume_from_dispvolume(w_d_volume)

        # generate the synthetic image in other side
        # named side but its synthesized the image of other side
        w_img = self.transformer[oside]\
            .get_warped_frame(t_img_aug, directs)
        synth_img = (w_p_volume * w_img).sum(dim=2)
        outputs['synth_img_{}'.format(train_side)] = synth_img

        source_img = self.inputs['color_{}'.format(oside)]
        if self.pred_out:
            pred_depth = outputs['depth_{}'.format(train_side)]
            projected_img, blend_mask = self.transformer['proj'](
                pred_depth, inv_K, T, K, source_img, True)
        else:
            fine_depth = outputs['fine_depth_{}'.format(train_side)]
            extra_normal_depth = outputs['extra_normal_depth_{}'.format(train_side)]
            projected_img, blend_mask = self.transformer['proj'](
                fine_depth, inv_K, T, K, source_img, True)
            fine_disp = self._trans_depth_and_disp(fine_depth)
            outputs['residual_depth_{}'.format(train_side)] = (
                extra_normal_depth - 0.5)
            outputs['fine_disp_{}'.format(train_side)] = fine_disp
        outputs['proj_img_{}'.format(train_side)] = projected_img
        outputs['mask_{}'.format(train_side)] = blend_mask.detach()

        # compute the occlusion mask
        occ_mask = 1
        if self.occ_mask_mode is not None:
            if 'v' in self.occ_mask_mode:
                ww_p_volume = self.transformer[train_side]\
                    .get_warped_volume(w_p_volume.detach(), directs)
                warp_mask = ww_p_volume.sum(dim=2).clamp(0, 1)
                occ_mask = occ_mask * warp_mask.detach()
            if 'm' in self.occ_mask_mode:
                pred_disp = outputs['disp_{}'.format(train_side)]
                occ_mask1 = self.occ_computer[0].forward(
                    pred_disp, T[:, 0, 3])
                occ_mask = occ_mask * occ_mask1.detach()
            outputs['mask_{}'.format(train_side)] = outputs[
                'mask_{}'.format(train_side)] * occ_mask
            outputs['smo_mask_{}'.format(train_side)] = 2 - outputs[
                'mask_{}'.format(train_side)]

        # extract features by res18
        raw_img = s_img_aug
        synth_img = outputs['synth_img_{}'.format(train_side)]

        with torch.no_grad():
            raw_feats = self._get_conv_feats_from_image(raw_img)
        synth_feats = self._get_conv_feats_from_image(synth_img)

        for feat_idx in range(3):
            rawf_name = 'raw_feats_{}_{}'.format(feat_idx, train_side)
            outputs[rawf_name] = raw_feats[feat_idx]
            synthf_name = 'synth_feats_{}_{}'.format(
                feat_idx, train_side)
            outputs[synthf_name] = synth_feats[feat_idx]

        return loss_sides, outputs

    def _get_probvolume_from_dispvolume(self, volume):
        return F.softmax(volume, dim=2)

    def _get_disp_from_probvolume(self, volume, directs=None):
        disp = (volume * self.volume_dk).sum(dim=2)
        if directs is not None:
            disp = disp * torch.abs(directs)
        return disp

    def _trans_depth_and_disp(self, in_d):
        k = self.inputs['disp_k'].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        out_d = torch.abs(k) / in_d
        return out_d

    def _get_conv_feats_from_image(self, raw_img):
        feats = []
        x = raw_img
        for block_idx in range(len(self.res_block)):
            x = self.res_block[block_idx](x)
            feats.append(x)
        return feats


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
