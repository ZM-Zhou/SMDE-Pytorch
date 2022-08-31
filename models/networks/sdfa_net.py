import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg19

from models.base_net import Base_of_Network
from models.backbones.resnet import ResNet_Backbone
from models.backbones.swin import get_orgwintrans_backbone
from models.decoders.sdfa_dec import SDFA_Decoder

from utils import platform_manager

@platform_manager.MODELS.add_module
class SDFA_Net(Base_of_Network):
    """Network with the SDFA module for self-supervised monocular deoth
       estimation.

       Args:
        backbone (str): Name of the backbone (encoder). Default: 'orgSwin-T-s2'
        decoder (str):  Name of the decoder. Default: 'SDFA'
        out_num (int): Nmber of discrete disparity levels. Default: 49.
        min_disp (int): Minimum disparity. Default: 2.
        max_disp (int): Maximum disparity. Default: 300.
        image_size (list): Image size at the training stage.
            Default: [192, 640].
        feat_mode (str): Mode of the network for the perception loss.
            Default: 'vgg19'.
        distill_offset: train with self-distill training strategy.
            Default: True.
        do_flip_distill: flip the features during the self-distilled step.
            Default: True.
    """
    def _initialize_model(self,
                          encoder='orgSwin-T-s2',
                          decoder='SDFA',
                          out_num=49,
                          min_disp=2,
                          max_disp=300,
                          image_size=[192, 640],
                          feat_mode='vgg19',
                          distill_offset=True,
                          do_flip_distill=True):
        self.init_opts = locals()

        self.out_num = out_num
        self.encoder = encoder
        self.decoder = decoder
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.image_size = image_size
        self.feat_mode = feat_mode
        self.distill_offset = distill_offset 
        self.do_flip_distill = do_flip_distill

        # Initialize architecture
        self.net_modules = {}
        if 'orgSwin' in self.encoder:
            self.net_modules['enc'], num_ch_enc = get_orgwintrans_backbone(
                self.encoder, True)
            num_ch_enc = copy.deepcopy(num_ch_enc)
            
            num_ch_dec = [64, 64, 128, 256]
            if (self.encoder == 'orgSwin-T'):
                num_ch_dec = [64] + num_ch_dec
        elif 'Res' in self.encoder:
            layer_num = int(self.encoder[3:])
            self.net_modules['enc'] = ResNet_Backbone(layer_num=layer_num)
            num_ch_enc = [64, 64, 128, 256, 512]
            if layer_num > 34:
                num_ch_enc = [ch_num * 4 for ch_num in num_ch_enc]
                num_ch_enc[0] = 64
            
            num_ch_dec = [64, 64, 64, 128, 256]
        else:
            raise NotImplementedError

        if 'OA' in self.decoder:
             self.net_modules['dec'] = SDFA_Decoder(
                num_ch_enc=num_ch_enc,
                num_ch_dec=num_ch_dec,
                output_ch=self.out_num,
                insert_sdfa=[i_f + 1 for i_f in range(len(num_ch_enc) - 1)])

        elif 'SDFA' in self.decoder:
            self.net_modules['dec'] = SDFA_Decoder(
                num_ch_enc=num_ch_enc,
                num_ch_dec=num_ch_dec,
                output_ch=self.out_num,
                insert_sdfa=[i_f + 1 for i_f in range(len(num_ch_enc) - 1)],
                sdfa_mode='SDFA',
                out_mode='two')
        else:
            raise NotImplementedError

        self._networks = nn.ModuleList(list(self.net_modules.values()))

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

        # load pretrained vgg19 module
        self.feat_net = Feat_Net(net_mode=self.feat_mode, device=self.device)

        if self.distill_offset:
            self.mask_builder = {}
            self.mask_builder[0] = SelfOccluMask(61, device=self.device)

        self.image_base = torch.ones((1, 3, *self.image_size)).to(self.device)
        self.image_base[:, 0, :, :] = 0.411 * self.image_base[:, 0, :, :]
        self.image_base[:, 1, :, :] = 0.432 * self.image_base[:, 1, :, :]
        self.image_base[:, 2, :, :] = 0.45 * self.image_base[:, 2, :, :]

        # Initialize the train side
        self.train_sides = ['s']

    def forward(self, inputs, is_train=True):
        self.inputs = inputs
        outputs = {}
        if is_train:
            losses = {}
            losses['loss'] = 0

            directs = self.inputs['direct']
            directs = directs.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            outputs['color_s_norm'] = self.image_base + self.inputs['color_s']
            outputs['color_o_norm'] = self.image_base + self.inputs['color_o']
            for train_side in self.train_sides:
                # oside is used to select the disparity transformer
                oside = 'o' if train_side == 's' else 's'

                # generate the train side disparity
                raw_volume = self._get_dispvolume_from_img(train_side)
                if isinstance(raw_volume, tuple):
                    raw_volume = self._process_net_outputs(
                        raw_volume, outputs, losses, train_side)

                d_volume = raw_volume.unsqueeze(1)

                p_volume = self._get_probvolume_from_dispvolume(d_volume)
                pred_disp = self._get_disp_from_probvolume(p_volume, directs)
                w_d_volume = self.transformer[oside]\
                    .get_warped_volume(d_volume, directs)
                w_p_volume = self._get_probvolume_from_dispvolume(w_d_volume)
                
                # biuld the outputs
                outputs['disp_{}'.format(train_side)] = pred_disp
                outputs['tar_disp_{}'.format(train_side)] = pred_disp.detach()
                outputs['depth_{}'.format(train_side)] =\
                    self._trans_depth_and_disp(pred_disp)

                t_img_aug = self.inputs['color_{}_aug'.format(train_side)]

                # generate the synthetic image in other side
                # named side but its synthesized the image of other side
                w_img = self.transformer[oside]\
                    .get_warped_frame(t_img_aug, directs)
                synth_img = (w_p_volume * w_img).sum(dim=2)
                outputs['synth_img_{}'.format(train_side)] = synth_img
                
                if self.distill_offset:
                    d_volume_fine = outputs['fine_volume'].unsqueeze(1)
                    p_volume_fine = self._get_probvolume_from_dispvolume(
                        d_volume_fine)
                    pred_disp_fine = self._get_disp_from_probvolume(
                        p_volume_fine, directs)
                    pred_depth_fine = self._trans_depth_and_disp(
                        pred_disp_fine)

                    side_flag = 1 if train_side == 's' else -1
                    outputs['fine_disp_{}'.format(train_side)] = pred_disp_fine
                    outputs['fine_depth_{}'.format(
                        train_side)] = pred_depth_fine
                    outputs['tar_depth_{}'.format(
                        train_side)] = self._trans_depth_and_disp(outputs[
                            'tar_disp_{}'.format(train_side)].detach())

                    source_image = outputs['color_{}_norm'.format(oside)]
                    warp_img_fine, blend_mask = self.transformer[
                        train_side].get_warp_with_disp(
                            -pred_disp_fine, source_image,
                            side_flag * directs)
                    warp_img, _ = self.transformer[
                        train_side].get_warp_with_disp(
                            -pred_disp.detach(), source_image,
                            side_flag * directs)

                    occ_mask = self.mask_builder[0](
                        pred_disp_fine, -side_flag * self.inputs['direct'])
                    outputs['warp_img_fine_{}'.format(
                        train_side)] = warp_img_fine
                    outputs['warp_img_{}'.format(train_side)] = warp_img
                    outputs['occ_fine_{}'.format(train_side)] = (
                        (occ_mask > 0.5) * blend_mask).to(torch.float)


            # extract features by vgg
            for train_side in self.train_sides:
                oside = 'o' if train_side == 's' else 's'
                raw_img = self.inputs['color_{}_aug'.format(oside)]
                synth_img = outputs['synth_img_{}'.format(train_side)]

                with torch.no_grad():
                    raw_feats = self.feat_net.get_feats(raw_img)
                synth_feats = self.feat_net.get_feats(synth_img)

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
                                     add_loss=True)
            return outputs, losses

        else:
            raw_volume = self._get_dispvolume_from_img('s', aug='')
            if isinstance(raw_volume, tuple):
                raw_volume = self._process_net_outputs(raw_volume, outputs, {},
                                                       's')
            if self.distill_offset:
                d_volume = outputs['fine_volume'].unsqueeze(1)
            else:
                d_volume = raw_volume.unsqueeze(1)

            p_volume = self._get_probvolume_from_dispvolume(d_volume)
            pred_disp = self._get_disp_from_probvolume(p_volume)
            if 'disp_k' in self.inputs:
                pred_depth = self._trans_depth_and_disp(pred_disp)
            else:
                pred_depth = 401.55 / pred_disp
            

            outputs[('depth', 's')] = pred_depth


            return outputs

    def _get_dispvolume_from_img(self, side, aug='_aug'):
        # pass image to the encoder
        input_img = self.inputs['color_{}{}'.format(side, aug)].clone()
        x = input_img / 0.225
        features = self.net_modules['enc'](x)
        if not (not self.training and self.distill_offset):
            out_volume = self.net_modules['dec'](features, input_img.shape)
            if isinstance(out_volume, torch.Tensor):
                out_volume = [out_volume]
        else:
            out_volume = (None, {}, {})
       
        # pass for step2
        # at the training stage, flip and pass the
        # features to decoder in the second step
        if self.distill_offset:
            if aug == '_aug' and self.do_flip_distill:
                new_features = []
                for idx_f in range(len(features)):
                    new_features.append(
                        torch.flip(features[idx_f].clone(), dims=[3]))
                out_volume_raw = self.net_modules['dec'](new_features,
                                                         input_img.shape,
                                                         switch=True)
                
                for k, v in out_volume_raw[1].items():
                    if k not in out_volume[1]:
                        v = torch.flip(v, dims=[3])
                        v[:, 0, ...] = -v[:, 0, ...]
                        out_volume[1][k] = v
                out_volume_raw = out_volume_raw[0]
                out_volume_raw = torch.flip(out_volume_raw, dims=[3])

            else:
                out_volume_raw = self.net_modules['dec'](features,
                                                         input_img.shape,
                                                         switch=True)
                for k, v in out_volume_raw[1].items():
                    if k not in out_volume[1]:
                        out_volume[1][k] = v
                out_volume_raw = out_volume_raw[0]

        else:
            out_volume_raw = None

        return (*out_volume, out_volume_raw)

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

    def _process_net_outputs(self, raw_volume, outputs, losses, train_side):
        if ('SDFA' in self.decoder
                or 'OA' in self.decoder):
            for k, v in raw_volume[1].items():
                outputs['delta_{}_{}_{}'.format(k[0], k[1], train_side)] = v
                losses['value-delta_{}_{}_{}'.format(
                    k[0], k[1], train_side)] = torch.abs(v.detach()).mean()

        if self.distill_offset:
            outputs['fine_volume'] = raw_volume[-1]

        raw_volume = raw_volume[0]
        return raw_volume

    def _compute_losses(self, outputs, train_side, losses, add_loss=True):
        loss_inputs = {}
        for out_key, out_vlaue in outputs.items():
            loss_inputs[out_key] = out_vlaue
        for in_key, in_vlaue in self.inputs.items():
            loss_inputs[in_key] = in_vlaue

        for used_loss in self.loss_options['types']:
            loss_name = used_loss['name']
            loss_rate = used_loss['rate']
            loss = self.loss_computer[loss_name](loss_inputs, train_side)
            if isinstance(loss, tuple):
                idx_mask = loss[1]
                losses['choice_mask/{}'.format(train_side)] = idx_mask.to(
                    torch.float) / 2
                total_num = (idx_mask >= 0).sum()
                losses['{}/self-value'.format(
                    train_side)] = (idx_mask == 0).sum() / total_num
                losses['{}/pred-value'.format(
                    train_side)] = (idx_mask == 1).sum() / total_num
                losses['{}/hints-value'.format(
                    train_side)] = (idx_mask == 2).sum() / total_num
                loss = loss[0]
            if 'mask' in used_loss:
                mask = loss_inputs[used_loss['mask'].format(train_side)]
                mask = mask.to(torch.float)
                loss = loss * mask

            loss_value = loss_rate * loss.mean()
            if add_loss:
                losses['loss'] += loss_value

            losses['{}/{}'.format(loss_name, train_side)] = loss
            losses['{}/{}-value'.format(train_side, loss_name)] = loss_value


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
    
    def get_warp_with_disp(self, disp, x, directs):
        disp_map = torch.zeros_like(self.base_coord)
        disp_map = disp_map.repeat(disp.shape[0], 1, 1, 1)
        disp_map[..., 0] = disp_map[..., 0] + disp[:, 0, ...]
        image_size = disp.shape[2:]
        normal_disp_map = self._get_normalize_coord(disp_map, image_size)
        normal_disp_map = normal_disp_map
        grid_coord = normal_disp_map * directs + self.base_coord
        warped_x = F.grid_sample(x,
                                 grid_coord,
                                 mode='bilinear',
                                 padding_mode='border',
                                 align_corners=True)
        mask = ((grid_coord >= -1) & (grid_coord <= 1)).to(torch.float)
        mask = torch.min(mask, dim=3, keepdim=True)[0].permute(0, 3, 1, 2)
        return warped_x, mask.detach()

class Feat_Net(object):
    def __init__(self, net_mode='vgg19', device='cpu'):
        self.feat_net = []
        if net_mode == 'vgg19':
            vgg = vgg19(pretrained=True, progress=False).features.to(device)
            vgg_feats = list(vgg.modules())
            vgg_layer_num = [5, 5, 9]
            read_module_num = 0
            for module_num in vgg_layer_num:
                self.feat_net.append(nn.Sequential())
                for _ in range(module_num):
                    self.feat_net[-1].add_module(
                        str(read_module_num), vgg_feats[read_module_num + 1])
                    read_module_num += 1
        else:
            raise NotImplementedError

    def get_feats(self, input_img):
        feats = []
        x = input_img
        for block_idx in range(len(self.feat_net)):
            x = self.feat_net[block_idx](x)
            feats.append(x)
        return feats


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
