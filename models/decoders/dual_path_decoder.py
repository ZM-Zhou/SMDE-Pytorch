import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from mmcv.ops.deform_conv import DeformConv2dFunction

class DPDecoder(nn.Module):
    def __init__(self,
                 num_ch_enc,
                 norm_disp_range,
                 num_ch_dec = [16, 32, 64, 128, 256],
                 sfg_scales = [2],
                 sfg_mode='MFM', # stereo feature generator in [MFM, Cat, Attn]
                 db_scales = [2],
                 db_mode='V0', # distilled branch in [SDFA, DeformConv]
                 dec_mode='',
                 stereo_out_ch=49,
                 mono_out_ch=49,
                 mid_out_ch=49,
                 disable_disbranch=False,
                 image_size=None
                 ):
        super().__init__()
        self.convblocks = {}
        self.num_layers = len(num_ch_dec) - 1
        self.num_enc_feats = len(num_ch_enc) - 1
        self.sfg_scales = sfg_scales
        self.sfg_mode = sfg_mode
        self.db_scales = db_scales
        self.db_mode = db_mode
        self.dec_mode = dec_mode
        self.disable_disbranch = disable_disbranch
        self.image_size = image_size

        self.warps = {}
        # Bulid decoder
        idx_feats = self.num_enc_feats
    
        if isinstance(stereo_out_ch, int):
            output_ch = [[mono_out_ch, stereo_out_ch] for _ in [0]]
        
        dec_enc_dict = {}
        dec_conv_in_dict = {}
        dec_conv_out_dict = {}
        for i in range(self.num_layers, -1, -1):
            # upconv_0
            if i == self.num_layers:
                num_ch_in = num_ch_enc[idx_feats]
            else:
                num_ch_in = num_ch_dec[i + 1]
            if i == self.num_layers:
                idx_feats -= 1
            num_ch_out = num_ch_dec[i]
            dec_conv_in_dict[(i, 0)] = num_ch_in
            dec_conv_out_dict[(i, 0)] = num_ch_out
            self.convblocks[("dec", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            # upconv_1
            num_ch_in = num_ch_dec[i]
            if idx_feats >= 0:
                dec_enc_dict[i] = idx_feats
                if self.dec_mode == '':
                    num_ch_in += num_ch_enc[idx_feats]
                else:

                    self.convblocks[("dec-sdfa", i, 0)] = ConvBlock(num_ch_enc[idx_feats],
                                                                    num_ch_in)
                    if self.dec_mode == 'SDFA':
                        self.convblocks[("dec-sdfa", i, 1)] = nn.Sequential(
                            nn.Conv2d(num_ch_in * 2, num_ch_in, kernel_size=1, bias=False),
                            nn.BatchNorm2d(num_ch_in),
                            nn.ELU(),
                            Conv3x3(num_ch_in, 2, bias=False))
                        self.convblocks[("dec-sdfa", i, 2)] = nn.Sequential(
                            nn.Conv2d(num_ch_in * 2, num_ch_in, kernel_size=1, bias=False),
                            nn.BatchNorm2d(num_ch_in),
                            nn.ELU(),
                            Conv3x3(num_ch_in, 2, bias=False))
                    elif self.dec_mode == 'SDFAReLU':
                        self.convblocks[("dec-sdfa", i, 1)] = nn.Sequential(
                            nn.Conv2d(num_ch_in * 2, num_ch_in, kernel_size=1, bias=False),
                            nn.BatchNorm2d(num_ch_in),
                            nn.ReLU(),
                            Conv3x3(num_ch_in, 2, bias=False))
                        self.convblocks[("dec-sdfa", i, 2)] = nn.Sequential(
                            nn.Conv2d(num_ch_in * 2, num_ch_in, kernel_size=1, bias=False),
                            nn.BatchNorm2d(num_ch_in),
                            nn.ReLU(),
                            Conv3x3(num_ch_in, 2, bias=False))
                    self.convblocks[("dec-sdfa", i, 1)][3].conv.weight.data.zero_()
                    self.convblocks[("dec-sdfa", i, 2)][3].conv.weight.data.zero_()
                idx_feats -= 1
            num_ch_out = num_ch_dec[i]
            dec_conv_in_dict[(i, 1)] = num_ch_in
            dec_conv_out_dict[(i, 1)] = num_ch_out
            self.convblocks[("dec", i, 1)] = ConvBlock(num_ch_in,
                                                       num_ch_out)
        
        self.convblocks[("out", 0)] = Conv3x3(num_ch_dec[0], output_ch[0][0])
        
        for i in range(self.num_layers, -1, -1):
            num_ch_out = num_ch_dec[i]
            if i in self.sfg_scales:
                if self.sfg_mode == 'Attn':
                    self.convblocks[("sfg", i, 0)] = CA_Block_V0(num_ch_out, norm_disp_range, image_size)
                elif self.sfg_mode == 'MFM':
                    self.convblocks[("sfg", i, 0)] = CA_Block_V2(num_ch_out, norm_disp_range, image_size)
                elif self.sfg_mode == 'Cat':
                    self.convblocks[("sfg", i, 0)] = CA_Block_Cat(num_ch_out, norm_disp_range, image_size)
        
        if stereo_out_ch != mono_out_ch:
            self.convblocks[("out-stereo", 0)] = nn.Sequential(
                ConvBlock(num_ch_dec[0], num_ch_dec[0]),
                Conv3x3(num_ch_dec[0], output_ch[0][1]))
            self.use_stereo_out_layer=True
        else:
            self.use_stereo_out_layer=False
        
        for i in range(self.num_layers, -1, -1):
            if i in self.db_scales:
                num_ch_out = num_ch_dec[i]
                if db_mode == 'SDFA':
                    if self.dec_mode == 'SDFA':
                        self.convblocks[("midout", i)] = nn.Sequential(
                            nn.Conv2d(dec_conv_in_dict[(i, 1)] * 2, dec_conv_in_dict[(i, 1)], kernel_size=1, bias=False),
                            nn.BatchNorm2d(dec_conv_in_dict[(i, 1)]),
                            nn.ELU(),
                            Conv3x3(dec_conv_in_dict[(i, 1)], 2, bias=False))
                    self.convblocks[("midout", i)][3].conv.weight.data.zero_()
                
                elif db_mode == 'DeformConv':
                    self.convblocks[("midout", i, 0)] = DeformOffsetConv(dec_conv_in_dict[(i, 0)])
                    self.convblocks[("midout", i, 1)] = DeformOffsetConv(dec_conv_in_dict[(i, 1)])

        self.convblocks[("midout-out", 0)] = Conv3x3(num_ch_dec[0], mid_out_ch)
        
        self._convs = nn.ModuleList(list(self.convblocks.values()))
    
    def forward(self, features, img_shape, directs=None, out_two_side=False, with_mo=True):
        if isinstance(features[0], list):
            outputs, costs= self.forward_stereo(features, img_shape, directs, out_two_side)
        else:
            outputs = self.forward_mono(features, img_shape, with_mo)
            costs = {}
            for k in list(outputs.keys()):
                if isinstance(k, str) and 'delta' in k:
                    costs[k] = outputs.pop(k).detach()
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs, costs
    
    def forward_stereo(self, features, img_shape, directs, out_two_side=False):
        outputs = {}
        costs = {} 
        x_s = features[-1][0]
        x_o = features[-1][1]
        idx_feats = self.num_enc_feats - 1
        for i in range(self.num_layers, -1, -1):
            x_s = self.convblocks[("dec", i, 0)](x_s)
            x_o = self.convblocks[("dec", i, 0)](x_o)
            if idx_feats >= 0:
                tar_shape = features[idx_feats][0].shape
            elif i == 0:
                tar_shape = img_shape
            else:
                tar_shape = [s * 2 for s in x_s.shape]
            x_s = [self._upsample(x_s, tar_shape)]
            x_o = [self._upsample(x_o, tar_shape)]
            if idx_feats >= 0:
                if self.dec_mode == '':
                    x_s += [features[idx_feats][0]]
                    x_o += [features[idx_feats][1]]
                else:
                    x_s = x_s[0]
                    x_s_enc = self.convblocks[("dec-sdfa", i, 0)](features[idx_feats][0])
                    x_s_con = torch.cat((x_s, x_s_enc), 1)
                    delta1_s = self.convblocks[("dec-sdfa", i, 1)](x_s_con)
                    x_s = self.bilinear_interpolate_torch_gridsample(
                        x_s, x_s.shape[2:], delta1_s)
                    delta2_s_enc = self.convblocks[("dec-sdfa", i, 2)](x_s_con)
                    x_s_enc = self.bilinear_interpolate_torch_gridsample(
                        x_s_enc, x_s_enc.shape[2:], delta2_s_enc)
                    x_s = [x_s + x_s_enc]

                    x_o = x_o[0]
                    x_o_enc = self.convblocks[("dec-sdfa", i, 0)](features[idx_feats][1])
                    x_o_con = torch.cat((x_o, x_o_enc), 1)
                    delta1_o = self.convblocks[("dec-sdfa", i, 1)](x_o_con)
                    x_o = self.bilinear_interpolate_torch_gridsample(
                        x_o, x_o.shape[2:], delta1_o)
                    delta2_o_enc = self.convblocks[("dec-sdfa", i, 2)](x_o_con)
                    x_o_enc = self.bilinear_interpolate_torch_gridsample(
                        x_o_enc, x_o_enc.shape[2:], delta2_o_enc)
                    x_o = [x_o + x_o_enc]

                idx_feats -= 1
            x_s = torch.cat(x_s, 1)
            x_o = torch.cat(x_o, 1)
            
            x_s = self.convblocks[("dec", i, 1)](x_s)
            x_o = self.convblocks[("dec", i, 1)](x_o)
            
            if i in self.sfg_scales:
                new_x_s, cost_s = self.convblocks[("sfg", i, 0)]([x_s, x_o], directs, img_shape)
                new_x_o, cost_o = self.convblocks[("sfg", i, 0)]([x_o, x_s], -directs, img_shape)
                if not cost_s is None and not cost_o is None:
                    costs['{}-s'.format(i)] = cost_s
                    costs['{}-o'.format(i)] = cost_o
                x_s = new_x_s
                x_o = new_x_o

            if i == 0:
                if self.use_stereo_out_layer:
                    out_layer = self.convblocks[("out-stereo", 0)]
                else:
                    out_layer = self.convblocks[("out", i)]
            
                if out_two_side:
                    out_disp_s = out_layer(x_s)
                    outputs['{}-s'.format(i)] = out_disp_s
                    out_disp_o = out_layer(x_o)
                    outputs['{}-o'.format(i)] = out_disp_o

                else:
                    out_disp = out_layer(x_s)
                    outputs[i] = out_disp

        return outputs, costs
    
    def forward_mono(self, features, img_shape, with_mo):
        outputs = {}
        x_s = features[-1]
        idx_feats = self.num_enc_feats - 1
        for i in range(self.num_layers, -1, -1):
            if i in self.db_scales and with_mo and self.db_mode == 'DeformConv':
                offset = self.convblocks[("midout", i, 0)](x_s)
                deform_conv2d = DeformConv2dFunction.apply
                x_s = deform_conv2d(x_s, offset, 
                                    self.convblocks[("dec", i, 0)].conv.conv.weight,
                                    self.convblocks[("dec", i, 0)].conv.conv.stride,
                                    1, # padding
                                    self.convblocks[("dec", i, 0)].conv.conv.dilation,
                                    self.convblocks[("dec", i, 0)].conv.conv.groups,
                                    self.convblocks[("midout", i, 0)].deform_groups)
                x_s = self.convblocks[("dec", i, 0)](x_s, wo_conv=True)
            else:
                x_s = self.convblocks[("dec", i, 0)](x_s)

            if idx_feats >= 0:
                tar_shape = features[idx_feats].shape
            elif i == 0:
                tar_shape = img_shape
            else:
                tar_shape = [s * 2 for s in x_s.shape]
            x_s = [self._upsample(x_s, tar_shape)]
            if idx_feats >= 0:
                if self.dec_mode == '':
                    x_s += [features[idx_feats]]
                else:
                    x_s = x_s[0]
                    x_s_enc = self.convblocks[("dec-sdfa", i, 0)](features[idx_feats])
                    x_s_con = torch.cat((x_s, x_s_enc), 1)
                    delta1_s = self.convblocks[("dec-sdfa", i, 1)](x_s_con)
                    x_s = self.bilinear_interpolate_torch_gridsample(
                        x_s, x_s.shape[2:], delta1_s)
                    outputs['delta_{}-1'.format(i)] = delta1_s
                    
                    if i in self.db_scales and with_mo and self.db_mode == 'SDFA':
                        delta2_s_enc = self.convblocks[("midout", i)](x_s_con)
                        x_s_enc = self.bilinear_interpolate_torch_gridsample(
                            x_s_enc, x_s_enc.shape[2:], delta2_s_enc)
                        x_s = [x_s + x_s_enc]
                        outputs['delta_{}-3'.format(i)] = delta2_s_enc
                    else:
                        delta2_s_enc = self.convblocks[("dec-sdfa", i, 2)](x_s_con)
                        x_s_enc = self.bilinear_interpolate_torch_gridsample(
                            x_s_enc, x_s_enc.shape[2:], delta2_s_enc)
                        x_s = [x_s + x_s_enc]
                        outputs['delta_{}-2'.format(i)] = delta2_s_enc

                idx_feats -= 1
            x_s = torch.cat(x_s, 1)

            if i in self.db_scales and with_mo and self.db_mode == 'DeformConv':
                offset = self.convblocks[("midout", i, 1)](x_s)
                deform_conv2d = DeformConv2dFunction.apply
                x_s = deform_conv2d(x_s, offset, 
                                    self.convblocks[("dec", i, 1)].conv.conv.weight,
                                    self.convblocks[("dec", i, 1)].conv.conv.stride,
                                    1, # padding
                                    self.convblocks[("dec", i, 1)].conv.conv.dilation,
                                    self.convblocks[("dec", i, 1)].conv.conv.groups,
                                    self.convblocks[("midout", i, 1)].deform_groups)
                x_s = self.convblocks[("dec", i, 1)](x_s, wo_conv=True)

            else:
                x_s = self.convblocks[("dec", i, 1)](x_s)  
            
            if i == 0:
                out_disp = self.convblocks[("out", i)](x_s)
                outputs[i] = out_disp
                if with_mo:
                    if not self.disable_disbranch:
                        out_middisp = self.convblocks[("midout-out", 0)](x_s)
                    else:
                        out_middisp = self.convblocks[("out", 0)](x_s)
                    outputs['{}-mid'.format(i)] = out_middisp

        return outputs
    
    def _upsample(self, x, shape, is_bilinear=False):
        if is_bilinear:
            return F.interpolate(x, size=shape[2:], mode="bilinear", align_corners=False)
        else:
            return F.interpolate(x, size=shape[2:], mode="nearest")
        
    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 2.0
        norm = torch.tensor([[[[(out_w - 1) / s, (out_h - 1) / s]]]
                             ]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
       from https://github.com/nianticlabs/monodepth2
    """
    def __init__(self, in_channels, out_channels, use_refl=True, bias=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, bias=bias)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
       from https://github.com/nianticlabs/monodepth2
    """
    def __init__(self, in_channels, out_channels, bn=False, nonlin=True,):
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

    def forward(self, x, wo_conv=False):
        if not wo_conv:
            out = self.conv(x)
        else:
            out = x
        if self.bn is not None:
            out = self.bn(out)
        if self.nonlin is not None:
            out = self.nonlin(out)
        return out

class CA_Block(nn.Module):
    def __init__(self, in_channels, norm_disp_range, train_image_size=None):
        super().__init__()
        self.norm_disp_range = norm_disp_range
        cost_channels = len(norm_disp_range)
        self.train_image_size = train_image_size
        self.q_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1))
        self.k_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1))
        self.v_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1))
        self.redu = nn.Sequential(
            nn.Conv2d(in_channels + cost_channels, in_channels, 1),
            nn.ELU())
       
        self.post_cost = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels // 4, cost_channels, 3, padding=1))
    
    def forward(self, feats, directs, image_shape):
        t_feat = feats[0]
        s_feat = feats[1]

        q = self.q_conv(t_feat)
        k = self.k_conv(s_feat)
        v = self.v_conv(s_feat)
        warped_feat = self._get_warped_frame(k, directs, image_shape)
        warped_v = self._get_warped_frame(v, directs, image_shape)
        cost = (q.unsqueeze(2) * warped_feat).sum(dim=1) / (q.shape[1] ** 0.5)
        norm_cost = torch.softmax(cost, dim=1)
        warped_v = (norm_cost.unsqueeze(1) * warped_v).sum(dim=2)
        res_cost = self.post_cost(torch.cat([q, warped_v], dim=1))
        cost = (cost + res_cost) / 2
        norm_cost = torch.softmax(cost, dim=1)
        x = self.redu(torch.cat([t_feat, norm_cost], dim=1))

        return x, cost

    def _get_warped_frame(self, x, directs, image_shape):
        """Warp the images by disparity range with border padding."""
        i_tetha = torch.zeros(1, 2, 3)
        i_tetha[:, 0, 0] = 1
        i_tetha[:, 1, 1] = 1
        normal_coord = F.affine_grid(i_tetha,
                                     [1, 1, x.shape[2], x.shape[3]],
                                     align_corners=True)
        base_coord = normal_coord.to(x)
        zeros = torch.zeros_like(base_coord)
        frame_volume = []
        if self.train_image_size[1] != image_shape[3]:
            rel_scale = self.train_image_size[1] / image_shape[3]
        else:
            rel_scale = 1
        for ch_idx in range(len(self.norm_disp_range)):
            disp_map = zeros.clone()
            disp_map[..., 0] = disp_map[..., 0] + self.norm_disp_range[ch_idx] * 2 * rel_scale
            grid_coords = disp_map * directs + base_coord
            warped_frame = F.grid_sample(x,
                                         grid_coords,
                                         mode='bilinear',
                                         padding_mode='border',
                                         align_corners=True)
            frame_volume.append(warped_frame.unsqueeze(2))
        return torch.cat(frame_volume, dim=2)

class CA_Block_V0(nn.Module):
    def __init__(self, in_channels, norm_disp_range, train_image_size=None):
        super().__init__()
        self.norm_disp_range = norm_disp_range
        cost_channels = len(norm_disp_range)
        self.train_image_size = train_image_size
        
        self.q_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1))
        self.k_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1))
       
        self.redu = nn.Sequential(
            nn.Conv2d(in_channels + cost_channels, in_channels, 1),
            nn.ELU())
    
    def forward(self, feats, directs, image_shape):
        t_feat = feats[0]
        s_feat = feats[1]

        q = self.q_conv(t_feat)
        k = self.k_conv(s_feat)
        warped_feat = self._get_warped_frame(k, directs, image_shape)
        cost = (q.unsqueeze(2) * warped_feat).sum(dim=1) / (q.shape[1] ** 0.5)
        norm_cost = torch.softmax(cost, dim=1)
        x = self.redu(torch.cat([t_feat, norm_cost], dim=1))

        return x, cost

    def _get_warped_frame(self, x, directs, image_shape):
        """Warp the images by disparity range with border padding."""
        i_tetha = torch.zeros(1, 2, 3)
        i_tetha[:, 0, 0] = 1
        i_tetha[:, 1, 1] = 1
        normal_coord = F.affine_grid(i_tetha,
                                     [1, 1, x.shape[2], x.shape[3]],
                                     align_corners=True)
        base_coord = normal_coord.to(x)
        zeros = torch.zeros_like(base_coord)
        frame_volume = []
        if self.train_image_size[1] != image_shape[3]:
            rel_scale = self.train_image_size[1] / image_shape[3]
        else:
            rel_scale = 1
        for ch_idx in range(len(self.norm_disp_range)):
            disp_map = zeros.clone()
            disp_map[..., 0] = disp_map[..., 0] + self.norm_disp_range[ch_idx] * 2 * rel_scale
            grid_coords = disp_map * directs + base_coord
            warped_frame = F.grid_sample(x,
                                         grid_coords,
                                         mode='bilinear',
                                         padding_mode='border',
                                         align_corners=True)
            frame_volume.append(warped_frame.unsqueeze(2))
        return torch.cat(frame_volume, dim=2)

class CA_Block_V2(nn.Module):
    def __init__(self, in_channels, norm_disp_range, train_image_size=None):
        super().__init__()
        self.norm_disp_range = norm_disp_range
        cost_channels = len(norm_disp_range)
        self.train_image_size = train_image_size
        
        self.q_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1))
        self.k_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1))
       
        self.redu = nn.Sequential(
            Conv3x3(in_channels + cost_channels, in_channels),
            SELayer(in_channels),
            nn.ELU())
    
    def forward(self, feats, directs, image_shape):
        t_feat = feats[0]
        s_feat = feats[1]

        q = self.q_conv(t_feat)
        k = self.k_conv(s_feat)
        warped_feat = self._get_warped_frame(k, directs, image_shape)
        cost = (q.unsqueeze(2) * warped_feat).sum(dim=1) / (q.shape[1] ** 0.5)
        norm_cost = torch.softmax(cost, dim=1)
        x = self.redu(torch.cat([t_feat, norm_cost], dim=1))

        return x, cost

    def _get_warped_frame(self, x, directs, image_shape):
        """Warp the images by disparity range with border padding."""
        i_tetha = torch.zeros(1, 2, 3)
        i_tetha[:, 0, 0] = 1
        i_tetha[:, 1, 1] = 1
        normal_coord = F.affine_grid(i_tetha,
                                     [1, 1, x.shape[2], x.shape[3]],
                                     align_corners=True)
        base_coord = normal_coord.to(x)
        zeros = torch.zeros_like(base_coord)
        frame_volume = []
        if self.train_image_size[1] != image_shape[3]:
            rel_scale = self.train_image_size[1] / image_shape[3]
        else:
            rel_scale = 1
        for ch_idx in range(len(self.norm_disp_range)):
            disp_map = zeros.clone()
            disp_map[..., 0] = disp_map[..., 0] + self.norm_disp_range[ch_idx] * 2 * rel_scale
            grid_coords = disp_map * directs + base_coord
            warped_frame = F.grid_sample(x,
                                         grid_coords,
                                         mode='bilinear',
                                         padding_mode='border',
                                         align_corners=True)
            frame_volume.append(warped_frame.unsqueeze(2))
        return torch.cat(frame_volume, dim=2)

class CA_Block_Cat(nn.Module):
    def __init__(self, in_channels, norm_disp_range, train_image_size=None):
        super().__init__()
        
        self.q_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 1))
        self.k_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 1))
        
        self.redu = nn.Sequential(
            Conv3x3(in_channels, in_channels),
            SELayer(in_channels),
            nn.ELU())
    
    def forward(self, feats, directs, image_shape):
        t_feat = feats[0]
        s_feat = feats[1]

        process_feats = []
        for b_idx in range(t_feat.shape[0]):
            if directs[b_idx] < 0:
                t_feat_sample = self.q_conv(t_feat[b_idx:b_idx + 1, ...])
                s_feat_sample = self.k_conv(s_feat[b_idx:b_idx + 1, ...])
                cat_sample = torch.cat([t_feat_sample, s_feat_sample], dim=1)
                out_feast_sample = self.redu(cat_sample)
                process_feats.append(out_feast_sample)
            else:
                t_feat_sample = self.q_conv(torch.flip(t_feat[b_idx:b_idx + 1, ...], dims=[3]))
                s_feat_sample = self.k_conv(torch.flip(s_feat[b_idx:b_idx + 1, ...], dims=[3]))
                cat_sample = torch.cat([t_feat_sample, s_feat_sample], dim=1)
                out_feast_sample = self.redu(cat_sample)
                process_feats.append(torch.flip(out_feast_sample, dims=[3]))

        return torch.cat(process_feats, dim=0), None

class SA_Block(nn.Module):
    def __init__(self, in_channels, cost_channels, woELU=False, cat_raw_feat=False):
        super().__init__()
        self.cat_raw_feat = cat_raw_feat
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
                                   nn.ELU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=1),
                                   nn.ELU(),
                                   nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=1))

        if woELU:
            self.pred = nn.Sequential(
                nn.Conv2d(in_channels//4, cost_channels, 1))
        else:
            self.pred = nn.Sequential(
                nn.Conv2d(in_channels//4, cost_channels, 1),
                nn.ELU())

        self.redu = nn.Sequential(
            nn.Conv2d(in_channels + cost_channels if cat_raw_feat else in_channels//4 + cost_channels,
                in_channels, 1),
            nn.ELU())
    
    def forward(self, x,):
        x_redu = self.conv1(x)
        x_redu = F.elu(self.conv2(x_redu) + x_redu)
        cost  = self.pred(x_redu)
        norm_cost = torch.softmax(cost, dim=1)
        cat_x = x if self.cat_raw_feat else x_redu
        x = self.redu(torch.cat([cat_x, norm_cost], dim=1))
        return x, cost

class Res_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
                                   nn.ELU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
                                   nn.ELU(),
                                   nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
                                   nn.ELU())

    def forward(self, x, x_last):
        if x_last is not None:
            x = torch.cat([x, x_last], dim=1)
        x_redu = self.conv1(x)
        x_redu = F.elu(self.conv2(x_redu) + x_redu)
        return self.conv3(x_redu)

class CC_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.cca = CC_module(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, in_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.ELU(in_channels)
            # nn.BatchNorm2d(in_channels),
            # nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        for _ in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output

class CC2_Block(nn.Module):
    def __init__(self, in_channels, use_rel_pos=False, feat_shape=None):
        super().__init__()
        inter_channels = in_channels
        self.conva = nn.Sequential(Conv3x3(in_channels, inter_channels, bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.cca = CC_module(inter_channels, use_rel_pos=use_rel_pos, feat_shape=feat_shape)
        self.convb = nn.Sequential(Conv3x3(inter_channels, inter_channels, bias=False),
                                   nn.ELU())

        # self.bottleneck = nn.Sequential(
        #     Conv3x3(in_channels+inter_channels, in_channels),
        #     nn.ELU()
        #     # nn.BatchNorm2d(in_channels),
        #     # nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        #     )

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        for _ in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        # output = self.bottleneck(torch.cat([x, output], 1))
        return output

class CC2ELU_Block(nn.Module):
    def __init__(self, in_channels, use_rel_pos=False, feat_shape=None):
        super().__init__()
        inter_channels = in_channels // 2
        self.conva = nn.Sequential(Conv3x3(in_channels, inter_channels, bias=False),
                                   nn.ELU())
        self.cca = CC_module(inter_channels, use_rel_pos=use_rel_pos, feat_shape=feat_shape)
        self.convb = nn.Sequential(Conv3x3(inter_channels, inter_channels, bias=False),
                                   nn.ELU())

        self.bottleneck = nn.Sequential(
            Conv3x3(in_channels+inter_channels, in_channels),
            nn.ELU()
            # nn.BatchNorm2d(in_channels),
            # nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        for _ in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output

class CC3_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(Conv3x3(in_channels, in_channels // 4),
                                   nn.ELU())
        
        self.conv2 = nn.Sequential(Conv3x3(in_channels // 4, in_channels // 4),
                                   nn.ELU(),
                                   Conv3x3(in_channels // 4, in_channels // 4))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=3, dilation=3),
                                   nn.ELU(),
                                   Conv3x3(in_channels // 4, in_channels // 4))
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=6, dilation=6),
                                   nn.ELU(),
                                   Conv3x3(in_channels // 4, in_channels // 4))
        
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=9, dilation=9),
                                   nn.ELU(),
                                   Conv3x3(in_channels // 4, in_channels // 4))

        self.redu = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ELU())
    
    def forward(self, x,):
        x_redu = self.conv1(x)
        x_out = []
        x_out.append(self.conv2(x_redu))
        x_out.append(self.conv3(x_redu))
        x_out.append(self.conv4(x_redu))
        x_out.append(self.conv5(x_redu))
        x_out = F.elu(torch.cat(x_out, dim=1))
        x_out = self.redu(x_out)
        return x_out

class CC4_Block(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.out_channels = out_channels
        self.aspp = {}
        if out_channels is not None:
            self.conv_in = nn.Conv2d(in_channels, out_channels,3,1,1)
            in_channels = out_channels
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, 1),
                                   nn.ELU(),
                                   nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=3, dilation=3, groups=in_channels // 4),
                                   nn.ELU())
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels + in_channels // 4, in_channels // 2, 1),
                                   nn.ELU(),
                                   nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=6, dilation=6, groups=in_channels // 4),
                                   nn.ELU())

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels + in_channels // 4 * 2, in_channels // 2, 1),
                                   nn.ELU(),
                                   nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=12, dilation=12, groups=in_channels // 4),
                                   nn.ELU())
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels + in_channels // 4 * 3, in_channels // 2, 1),
                                   nn.ELU(),
                                   nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=18, dilation=18, groups=in_channels // 4),
                                   nn.ELU())
        
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels + in_channels // 4 * 4, in_channels // 2, 1),
                                   nn.ELU(),
                                   nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=24, dilation=24, groups=in_channels // 4),
                                   nn.ELU())

        self.redu = nn.Sequential(
            nn.Conv2d(in_channels + in_channels // 4 * 5,
                      in_channels,
                      1),
            nn.ELU())
    
    def forward(self, x,):
        if self.out_channels:
            x = self.conv_in(x)
        x_redu = self.conv1(x)
        x_redu = torch.cat([x, x_redu], dim=1)
        x_tmp = self.conv2(x_redu)
        x_redu = torch.cat([x_redu, x_tmp], dim=1)
        x_tmp = self.conv3(x_redu)
        x_redu = torch.cat([x_redu, x_tmp], dim=1)
        x_tmp = self.conv4(x_redu)
        x_redu = torch.cat([x_redu, x_tmp], dim=1)
        x_tmp = self.conv5(x_redu)
        x_redu = torch.cat([x_redu, x_tmp], dim=1)
        x_out = self.redu(x_redu)
        return x_out

def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class CC_module(nn.Module):
    def __init__(self, in_dim, in_dim_v=None, use_rel_pos=False, feat_shape=None):
        super(CC_module, self).__init__()
        if in_dim_v is None:
            in_dim_v = in_dim
        self.q_convuery_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.k_convey_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.v_convalue_conv = nn.Conv2d(in_channels=in_dim_v, out_channels=in_dim_v, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            self.rel_pos = nn.Parameter(torch.zeros((1, 1, 1, feat_shape[0]+ feat_shape[1])))
            trunc_normal_(self.rel_pos, std=.02)
    
    def forward(self, x, x_v=None):
        if x_v is None:
            x_v = x
        m_batchsize, _, height, width = x.size()
        proj_query = self.q_convuery_conv(x)
        #                         [B, W, C, H]                  [BW, C, H]                        [BW, H, C]
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        #                         [B, H, C, W]                  [BH, C, W]                        [BH, W, C]
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.k_convey_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.v_convalue_conv(x_v)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = torch.cat([energy_H, energy_W], 3)
        if self.use_rel_pos:
            concate = concate + self.rel_pos
        concate = self.softmax(concate)

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x_v

class PCV3(nn.Module):
    def __init__(self, features, cost_channels, cat_raw_feat=False):
        super().__init__()

        self.conv0 = nn.Sequential(nn.Conv2d(features, features // 2, 3, padding=1),
                                   nn.ELU())
        self.convimg1 = nn.Sequential(nn.Conv2d(3, features // 4, 3, padding=1),
                                     nn.ELU())
        self.convimg2 = nn.Sequential(nn.Conv2d(features // 4, features // 4, 3, padding=1),
                                   nn.ELU(),
                                   nn.Conv2d(features // 4, features // 4, 3, padding=1))
        
        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(features // 4 * 3, features//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(features//2),
            nn.ELU(),
            Conv3x3(features//2, 2, bias=False))
        self.delta_gen1[3].conv.weight.data.zero_()


        self.conv1 = nn.Sequential(nn.Conv2d(features, features // 4, 3, padding=1),
                                   nn.ELU())                    
        self.conv2 = nn.Sequential(nn.Conv2d(features // 4, features // 4, 3, padding=1),
                                   nn.ELU(),
                                   nn.Conv2d(features // 4, features // 4, 3, padding=1))

        self.pred = nn.Sequential(
            nn.Conv2d(features//4, cost_channels, 1))

        self.redu = nn.Sequential(
            nn.Conv2d(features + cost_channels if cat_raw_feat else features//4 + cost_channels,
                      features, 1),
            nn.ELU())

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 2.0
        norm = torch.tensor([[[[(out_w - 1) / s, (out_h - 1) / s]]]
                             ]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def forward(self, x, img):
        h, w = x.shape[2:]
        img = F.interpolate(img, [h, w], mode='bilinear', align_corners=False)
        img_x = self.convimg1(img)
        img_x =  F.elu(self.convimg2(img_x) + img_x)
        redu_x = self.conv0(x)

        concat = torch.cat((redu_x, img_x), 1)

        delta1 = self.delta_gen1(concat)
        x = self.bilinear_interpolate_torch_gridsample(
            x, x.shape[2:], delta1)
        
        pred_x = self.conv1(x)
        pred_x = F.elu(self.conv2(pred_x) + pred_x)
        cost  = self.pred(pred_x)
        norm_cost = torch.softmax(cost, dim=1)
        x = self.redu(torch.cat([x, norm_cost], dim=1))

        return x, cost, delta1

class PFV6(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv0 = nn.Sequential(nn.Conv2d(features, features // 2, 3, padding=1),
                                   nn.ELU())
        self.convimg1 = nn.Sequential(nn.Conv2d(3, features // 4, 3, padding=1),
                                     nn.ELU())
        self.convimg2 = nn.Sequential(nn.Conv2d(features // 4, features // 4, 3, padding=1),
                                   nn.ELU(),
                                   nn.Conv2d(features // 4, features // 4, 3, padding=1))
        
        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(features // 4 * 3, features//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(features//2),
            nn.ReLU(),
            Conv3x3(features//2, 2, bias=False))
        self.delta_gen1[3].conv.weight.data.zero_()


        # self.conv1 = nn.Sequential(nn.Conv2d(features, features, 3, padding=1),
        #                            nn.ELU())                    
        self.conv2 = nn.Sequential(nn.Conv2d(features, features // 2, 3, padding=1),
                                   nn.ELU(),
                                   nn.Conv2d(features // 2, features, 3, padding=1))

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 2.0
        norm = torch.tensor([[[[(out_w - 1) / s, (out_h - 1) / s]]]
                             ]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def forward(self, x, img):
        h, w = x.shape[2:]
        img = F.interpolate(img, [h, w], mode='bilinear', align_corners=False)
        img_x = self.convimg1(img)
        img_x =  F.elu(self.convimg2(img_x) + img_x)
        
        x = F.elu(self.conv2(x) + x)
        
        redu_x = self.conv0(x)
        concat = torch.cat((redu_x, img_x), 1)

        delta1 = self.delta_gen1(concat)
        x = self.bilinear_interpolate_torch_gridsample(
            x, x.shape[2:], delta1)
        
        # pred_x = self.conv1(x)
        return x, delta1

class PFV7(nn.Module):
    def __init__(self, high_feat_dim, low_feat_dim):
        super().__init__()

        self.proj_high = nn.Conv2d(high_feat_dim, high_feat_dim, 1)
        self.conv_high = nn.Sequential(nn.Conv2d(high_feat_dim, high_feat_dim, 3, padding=1),
                                       nn.ELU(),
                                       nn.Conv2d(high_feat_dim, high_feat_dim, 3, padding=1))
        
        self.redu_high = nn.Conv2d(high_feat_dim, high_feat_dim, 1)
        
        self.proj_low = nn.Conv2d(low_feat_dim, low_feat_dim, 1)
        self.conv_low = nn.Sequential(nn.Conv2d(low_feat_dim, low_feat_dim, 3, padding=1),
                                       nn.ELU(),
                                       nn.Conv2d(low_feat_dim, low_feat_dim, 3, padding=1))
        
        self.redu_low = nn.Conv2d(low_feat_dim, high_feat_dim, 1)

        
        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(low_feat_dim + high_feat_dim, (low_feat_dim + high_feat_dim)//2, kernel_size=1, bias=False),
            nn.BatchNorm2d((low_feat_dim + high_feat_dim)//2),
            nn.ReLU(),
            Conv3x3((low_feat_dim + high_feat_dim)//2, 2, bias=False))
        self.delta_gen1[3].conv.weight.data.zero_()

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 2.0
        norm = torch.tensor([[[[(out_w - 1) / s, (out_h - 1) / s]]]
                             ]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def forward(self, x, high_feat):

        x = F.elu(self.conv_low(x) + self.proj_low(x))
        high_feat = F.elu(self.conv_high(high_feat) + self.proj_high(high_feat))

    
        h, w = high_feat.shape[2:]
        x = F.interpolate(x, [h, w], mode='bilinear', align_corners=False)
        concat = torch.cat((x, high_feat), 1)
        delta1 = self.delta_gen1(concat)
        x = self.bilinear_interpolate_torch_gridsample(
            x, x.shape[2:], delta1)

        redu_high = self.redu_high(high_feat)
        redu_low = self.redu_low(x)

        x = F.elu(redu_high + redu_low)
        return x, delta1

class MidOut_Module(nn.Module):
    def __init__(self, in_ch, scale=0, out_ch=1):
        super().__init__()
        self.up_scale = 2 ** scale
        self.out_ch = out_ch
        self.conv_out = nn.Sequential(ConvBlock(in_ch, in_ch),
                                      Conv3x3(in_ch, out_ch))

        self.mask = nn.Sequential(
                    ConvBlock(in_ch, in_ch * 2),
                    nn.Conv2d(in_ch * 2,((2**scale) ** 2)*9, 1))
    
    
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, self.up_scale, self.up_scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3,3], padding=1)
        up_flow = up_flow.view(N, self.out_ch, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, self.out_ch, self.up_scale*H, self.up_scale*W)
    
    def forward(self, x):
        mask = 0.25 * self.mask(x)
        x = self.conv_out(x)
        return self.upsample_flow(x, mask)

class DASPP(nn.Module):
    def __init__(self, ch_num, out_ch=None):
        super().__init__()
        self.out_ch = out_ch
        self.aspp = {}
        if out_ch is not None:
            self.aspp_in = nn.Conv2d(ch_num, out_ch,3,1,1)
            ch_num = out_ch
        self.aspp[0] = AtrousConv(ch_num, ch_num // 2, 3, apply_bn_first=False)
        self.aspp[1] = AtrousConv(ch_num + ch_num // 2, ch_num // 2, 6)
        self.aspp[2] = AtrousConv(ch_num * 2, ch_num // 2, 12)
        self.aspp[3] = AtrousConv(ch_num * 2 + ch_num // 2, ch_num // 2, 18)
        self.aspp[4] = AtrousConv(ch_num * 3, ch_num // 2, 24)
        self.aspp_out= torch.nn.Sequential(
            nn.Conv2d(ch_num + (ch_num // 2) * 5,
                      ch_num,
                      3,
                      1,
                      1,
                      bias=False), nn.ELU())
        
        self.aspp_conv = nn.ModuleList(list(self.aspp.values()))
    
    def forward(self, x):
        if self.out_ch:
            x = self.aspp_in(x)
        for a_idx in range(5):
            a_x = self.aspp[a_idx](x)
            x = torch.cat([x, a_x], dim=1)
        x = self.aspp_out(x)

        return x

class AtrousConv(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 apply_bn_first=True):
        super().__init__()
        self.atrous_conv = nn.Sequential()
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

class OA(nn.Module):
    """ReLU -> ELU InPlaceASB -> BN + ELU."""
    def __init__(self,
                 ch_num_enc,
                 ch_num_dec,
                 bn=True,
                 relu=False,
                 refl_pad=True):
        super().__init__()

        self.conv_enc = ConvBlock(ch_num_enc, ch_num_dec, True, True)

        if relu:
            act_func = nn.ReLU
        else:
            act_func = nn.ELU

        if bn:
            bn_func = nn.BatchNorm2d
        else:
            bn_func = nn.Identity

        if refl_pad:
            self.delta_gen1 = nn.Sequential(
                nn.Conv2d(ch_num_dec * 2,
                            ch_num_dec,
                            kernel_size=1,
                            bias=False), bn_func(ch_num_dec), act_func(),
                Conv3x3(ch_num_dec, 2, bias=False))
            self.delta_gen1[3].conv.weight.data.zero_()

            self.delta_gen2 = nn.Sequential(
                nn.Conv2d(ch_num_dec * 2,
                            ch_num_dec,
                            kernel_size=1,
                            bias=False), bn_func(ch_num_dec), act_func(),
                Conv3x3(ch_num_dec, 2, bias=False))
            self.delta_gen2[3].conv.weight.data.zero_()
        else:
            self.delta_gen1 = nn.Sequential(
                nn.Conv2d(ch_num_dec * 2,
                            ch_num_dec,
                            kernel_size=1,
                            bias=False), bn_func(ch_num_dec), act_func(),
                nn.Conv2d(ch_num_dec,
                            2,
                            kernel_size=3,
                            padding=1,
                            bias=False))
            self.delta_gen1[3].weight.data.zero_()
            self.delta_gen2 = nn.Sequential(
                nn.Conv2d(ch_num_dec * 2,
                            ch_num_dec,
                            kernel_size=1,
                            bias=False), bn_func(ch_num_dec), act_func(),
                nn.Conv2d(ch_num_dec,
                            2,
                            kernel_size=3,
                            padding=1,
                            bias=False))
            self.delta_gen2[3].weight.data.zero_()

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 2.0
        norm = torch.tensor([[[[(out_w - 1) / s, (out_h - 1) / s]]]
                             ]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def forward(self, enc_feat, dec_feat):
        enc_feat2 = self.conv_enc(enc_feat)
        delta_dict = {}
        h, w = enc_feat2.size(2), enc_feat2.size(3)
        dec_feat = F.interpolate(input=dec_feat,
                                   size=(h, w),
                                   mode='bilinear',
                                   align_corners=True)

        concat = torch.cat((enc_feat2, dec_feat), 1)

        delta1 = self.delta_gen1(concat)
        dec_feat = self.bilinear_interpolate_torch_gridsample(
            dec_feat, (h, w), delta1)
        delta_dict[1] = delta1
       
        delta2 = self.delta_gen2(concat)
        enc_feat = self.bilinear_interpolate_torch_gridsample(
            enc_feat, (h, w), delta2)
        delta_dict[2] = delta2

        fuse_dec_feat = torch.cat((enc_feat, dec_feat), 1)

        return fuse_dec_feat, delta_dict

class DeformOffsetConv(nn.Module):
    """Layer to perform a convolution followed by ELU
       from https://github.com/nianticlabs/monodepth2
    """
    def __init__(self, in_channels, deform_groups=1):
        super(DeformOffsetConv, self).__init__()
        
        self.kernel_size = [3, 3]
        self.deform_groups = deform_groups
        
        self.conv_offset = nn.Conv2d(
            in_channels,
            deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=1,
            padding=1,
            dilation=1,
            bias=True)

        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return offset

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)