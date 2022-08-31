import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDFA_Decoder(nn.Module):
    def __init__(
            self,
            num_ch_enc,
            num_ch_dec=[64, 64, 64, 128, 256],
            output_ch=49,
            insert_sdfa=[],  # [1, 2, 3]
            sdfa_mode='OA',
            out_mode=''):

        super().__init__()
        self.insert_sdfa = insert_sdfa
        self.sdfa_mode = sdfa_mode
        self.out_mode = out_mode

        self.num_layers = len(num_ch_dec) - 1
        self.num_enc_feats = len(num_ch_enc) - 1
        self.convblocks = {}
        num_ch_enc = copy.deepcopy(num_ch_enc)

        # Build SDFA
        if self.insert_sdfa:
            for idx_insert in self.insert_sdfa:
                in_ch_num_h = num_ch_enc[self.num_enc_feats - idx_insert]
                in_ch_num_l = num_ch_dec[self.num_layers - idx_insert + 1]
                self._build_alingfa(idx_insert, in_ch_num_h, in_ch_num_l)
        
        # Build decoder
        idx_feats = self.num_enc_feats
        forward_layer_idx = 1
        for i in range(self.num_layers, -1, -1):
            # conv_0
            if i == self.num_layers:
                num_ch_in = num_ch_enc[idx_feats]
                idx_feats -= 1
            else:
                num_ch_in = num_ch_dec[i + 1]
            num_ch_out = num_ch_dec[i]
            self.convblocks[('dec', i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            # conv_1
            num_ch_in = num_ch_dec[i]
            if idx_feats >= 0:
                if forward_layer_idx not in self.insert_sdfa:
                    num_ch_in += num_ch_enc[idx_feats]
                idx_feats -= 1
            num_ch_out = num_ch_dec[i]
            self.convblocks[('dec', i, 1)] = ConvBlock(num_ch_in, num_ch_out)

            forward_layer_idx += 1
        
        if self.out_mode == '':
            self.convblocks[('out', 0)] = Conv3x3(num_ch_dec[0], output_ch)
        elif self.out_mode == 'two':
            self.convblocks[('out', 0)] = Conv3x3(num_ch_dec[0], output_ch)
            self.convblocks[('out_fine', 0)] = Conv3x3(num_ch_dec[0],
                                                       output_ch)
        else:
            raise NotImplementedError
        
        self._convs = nn.ModuleList(list(self.convblocks.values()))
    
    def forward(self, features, img_shape, switch=False):
        all_delta_dict = {}
        multi_scale_out = {}
        x = features[-1]

        idx_feats = self.num_enc_feats - 1
        forward_layer_idx = 1
        for i in range(self.num_layers, -1, -1):
            # conv_0
            x = self.convblocks[('dec', i, 0)](x)
            # upsample and feature fuse
            # Insert SDFA
            if forward_layer_idx in self.insert_sdfa:
                x, half_x, delta_dict = self._forward_sdfa(
                    forward_layer_idx, x, features[idx_feats], switch)
                for k, v in delta_dict.items():
                    all_delta_dict[(forward_layer_idx, k)] = v
                idx_feats -= 1
            else:
                if idx_feats >= 0:
                    tar_shape = features[idx_feats].shape
                elif i == 0:
                    tar_shape = img_shape
                else:
                    tar_shape = [s * 2 for s in x.shape]

                x = [self._upsample(x, tar_shape)]

                if idx_feats >= 0:
                    x += [features[idx_feats]]
                    idx_feats -= 1
                x = torch.cat(x, 1)
            
            # conv_1
            x = self.convblocks[('dec', i, 1)](x)
            forward_layer_idx += 1
        
        final_out = self.convblocks[('out', 0)](x)
        if self.out_mode == 'two' and switch:
            final_out = self.convblocks[('out_fine', 0)](x)
        
        return (final_out, all_delta_dict, multi_scale_out)
    
    def _build_alingfa(self, idx_sdfa, ch_h, ch_l):
        self.convblocks['SDFA-conv',
                        idx_sdfa] = ConvBlock(ch_h, ch_l, True, True)

        if self.sdfa_mode == 'OA':
            self.convblocks['SDFA-fuse', idx_sdfa] = OA(ch_l)
        
        elif self.sdfa_mode == 'SDFA':
            self.convblocks['SDFA-fuse', idx_sdfa] = SDFA(ch_l)
        else:
            raise NotImplementedError

    def _forward_sdfa(self, idx_sdfa, h_x, l_x, switch=False):
        l_x = self.convblocks['SDFA-conv', idx_sdfa](l_x)
        tmp_out = self.convblocks['SDFA-fuse',idx_sdfa](h_x, l_x,
                                                              switch)
        out_x, sampled_high_stage, delta_dict = tmp_out
        return out_x, sampled_high_stage, delta_dict

    def _upsample(self, x, shape, mode='nearest'):
        return F.interpolate(x, size=shape[2:], mode=mode)

class Conv3x3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation=1,
                 padding=1,
                 bias=True,
                 use_refl=True,
                 name=None):
        super().__init__()
        self.name = name
        if use_refl:
            # self.pad = nn.ReplicationPad2d(padding)
            self.pad = nn.ReflectionPad2d(padding)
        else:
            self.pad = nn.ZeroPad2d(padding)
        conv = nn.Conv2d(int(in_channels),
                         int(out_channels),
                         3,
                         dilation=dilation,
                         bias=bias)
        if self.name:
            setattr(self, self.name, conv)
        else:
            self.conv = conv

    def forward(self, x):
        out = self.pad(x)
        if self.name:
            use_conv = getattr(self, self.name)
        else:
            use_conv = self.conv
        out = use_conv(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False, nonlin=True):
        super().__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        if nonlin:
            self.nonlin = nn.ELU(inplace=True)
        else:
            self.nonlin = None

    def forward(self, x):
        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)
        if self.nonlin is not None:
            out = self.nonlin(out)
        return out

class OA(nn.Module):
    """ReLU -> ELU InPlaceASB -> BN + ELU."""
    def __init__(self,
                 features,
                 disable_gen1=False,
                 disable_gen2=False,
                 bn=True,
                 relu=False,
                 refl_pad=True):
        super().__init__()
        self.disable_gen1 = disable_gen1
        self.disable_gen2 = disable_gen2

        if relu:
            act_func = nn.ReLU
        else:
            act_func = nn.ELU

        if bn:
            bn_func = nn.BatchNorm2d
        else:
            bn_func = nn.Identity

        if refl_pad:
            if not self.disable_gen1:
                self.delta_gen1 = nn.Sequential(
                    nn.Conv2d(features * 2,
                              features,
                              kernel_size=1,
                              bias=False), bn_func(features), act_func(),
                    Conv3x3(features, 2, bias=False))
                self.delta_gen1[3].conv.weight.data.zero_()

            if not self.disable_gen2:
                self.delta_gen2 = nn.Sequential(
                    nn.Conv2d(features * 2,
                              features,
                              kernel_size=1,
                              bias=False), bn_func(features), act_func(),
                    Conv3x3(features, 2, bias=False))
                self.delta_gen2[3].conv.weight.data.zero_()
        else:
            if not self.disable_gen1:
                self.delta_gen1 = nn.Sequential(
                    nn.Conv2d(features * 2,
                              features,
                              kernel_size=1,
                              bias=False), bn_func(features), act_func(),
                    nn.Conv2d(features,
                              2,
                              kernel_size=3,
                              padding=1,
                              bias=False))
                self.delta_gen1[3].weight.data.zero_()

            if not self.disable_gen2:
                self.delta_gen2 = nn.Sequential(
                    nn.Conv2d(features * 2,
                              features,
                              kernel_size=1,
                              bias=False), bn_func(features), act_func(),
                    nn.Conv2d(features,
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

    def forward(self, high_stage, low_stage):
        delta_dict = {}
        h, w = low_stage.size(2), low_stage.size(3)
        high_stage = F.interpolate(input=high_stage,
                                   size=(h, w),
                                   mode='bilinear',
                                   align_corners=True)

        concat = torch.cat((low_stage, high_stage), 1)

        if not self.disable_gen1:
            delta1 = self.delta_gen1(concat)
            high_stage = self.bilinear_interpolate_torch_gridsample(
                high_stage, (h, w), delta1)
            delta_dict[1] = delta1
        if not self.disable_gen2:
            delta2 = self.delta_gen2(concat)
            low_stage = self.bilinear_interpolate_torch_gridsample(
                low_stage, (h, w), delta2)
            delta_dict[2] = delta2

        fuse_high_stage = high_stage + low_stage

        return fuse_high_stage, high_stage, delta_dict

class SDFA(nn.Module):
    """ReLU -> ELU InPlaceASB -> BN + ELU."""
    def __init__(self, features):
        super().__init__()

        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features), nn.ELU(), Conv3x3(features,
                                                        2,
                                                        bias=False))
        self.delta_gen1[3].conv.weight.data.zero_()

        self.delta_gen2 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features), nn.ELU(), Conv3x3(features,
                                                        2,
                                                        bias=False))
        self.delta_gen2[3].conv.weight.data.zero_()

        self.delta_gen3 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features), nn.ELU(), Conv3x3(features,
                                                        2,
                                                        bias=False))
        self.delta_gen3[3].conv.weight.data.zero_()

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

    def forward(self, high_stage, low_stage, switch=False):
        delta_dict = {}
        h, w = low_stage.size(2), low_stage.size(3)
        high_stage = F.interpolate(input=high_stage,
                                   size=(h, w),
                                   mode='bilinear',
                                   align_corners=True)

        concat = torch.cat((low_stage, high_stage), 1)

        delta1 = self.delta_gen1(concat)
        high_stage = self.bilinear_interpolate_torch_gridsample(
            high_stage, (h, w), delta1)
        delta_dict[1] = delta1

        if not switch:
            delta2 = self.delta_gen2(concat)
            low_stage = self.bilinear_interpolate_torch_gridsample(
                low_stage, (h, w), delta2)
            delta_dict[2] = delta2
        else:
            delta3 = self.delta_gen3(concat)
            low_stage = self.bilinear_interpolate_torch_gridsample(
                low_stage, (h, w), delta3)
            delta_dict[3] = delta3

        fuse_high_stage = high_stage + low_stage

        return fuse_high_stage, high_stage, delta_dict