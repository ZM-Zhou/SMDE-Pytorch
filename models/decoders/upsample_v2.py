import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSample_Layers_v2(nn.Module):
    def __init__(self,
                 num_ch_enc,
                 num_ch_dec=[64, 128, 256, 256, 256],
                 output_ch=49,
                 raw_fal_arch=False):
        super().__init__()
        self.convblocks = {}
        real_num_ch_dec = [output_ch] + num_ch_dec
        self.num_layers = len(real_num_ch_dec) - 1
        self.num_enc_feats = len(num_ch_enc) - 1
        self.raw_fal_arch = raw_fal_arch
        # Build decoder
        idx_feats = self.num_enc_feats
        for i in range(self.num_layers, -1, -1):
            # upconv_0
            num_ch_in = num_ch_enc[
                idx_feats] if i == self.num_layers else real_num_ch_dec[i + 1]
            if i == self.num_layers:
                idx_feats -= 1
            if raw_fal_arch and i == 0:
                num_ch_out = num_ch_in
            else:
                num_ch_out = num_ch_in // 2
            self.convblocks[('dec', i, 0)] = ConvBlock(num_ch_in,
                                                       num_ch_out,
                                                       bias=False)
            # upconv_1
            num_ch_in = num_ch_out
            if idx_feats >= 0:
                num_ch_in += num_ch_enc[idx_feats]
                idx_feats -= 1
            if i > 0:
                num_ch_out = real_num_ch_dec[i]
                self.convblocks[('dec', i,
                                 1)] = ConvBlock(num_ch_in, num_ch_out)
            else:
                self.convblocks[('dec', i, 1)] = Conv3x3(num_ch_in,
                                                         output_ch,
                                                         bias=False)

        if self.raw_fal_arch:
            self.convblocks['conv0'] = nn.Conv2d(output_ch, output_ch, 1)

        self._convs = nn.ModuleList(list(self.convblocks.values()))

    def forward(self, features, img_shape):
        x = features[-1]
        idx_feats = self.num_enc_feats - 1
        for i in range(self.num_layers, -1, -1):
            if idx_feats >= 0:
                tar_shape = features[idx_feats].shape
            else:
                tar_shape = [s * 2 for s in x.shape]
            x = self._upsample(x, tar_shape)
            x = self.convblocks[('dec', i, 0)](x)
            if idx_feats >= 0:
                x = [x, features[idx_feats]]
                idx_feats -= 1
                x = torch.cat(x, 1)
            x = self.convblocks[('dec', i, 1)](x)
        if self.raw_fal_arch:
            x = self.convblocks['conv0'](x)
        return x

    def _upsample(self, x, shape):
        return F.interpolate(x, size=shape[2:], mode='nearest')


class Conv3x3(nn.Module):
    """Layer to pad and convolve input from
    https://github.com/nianticlabs/monodepth2."""
    def __init__(self, in_channels, out_channels, use_refl=True, bias=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels),
                              int(out_channels),
                              3,
                              bias=bias)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU from
    https://github.com/nianticlabs/monodepth2."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 bn=False,
                 nonlin=True):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels, bias=bias)
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
