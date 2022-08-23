import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSample_withDenseASPP(nn.Module):
    def __init__(self,
                 num_ch_enc,
                 num_ch_dec=[64, 64, 64, 128, 256],
                 output_ch=49,
                 first_fuse=False,
                 unfold_upsample=False):
        super().__init__()
        self.first_fuse = first_fuse
        self.unfold_upsample = unfold_upsample
        self.convblocks = {}
        # Build Dense-ASPP module
        if self.first_fuse:
            self.embed_conv = nn.Conv2d(num_ch_enc[-1],
                                        num_ch_dec[-1],
                                        1,
                                        bias=False)
        num_feats = num_ch_dec[-1]

        self.convblocks['aspp', 0] = AtrousConv(num_feats,
                                                num_feats // 2,
                                                3,
                                                apply_bn_first=False)
        self.convblocks['aspp', 1] = AtrousConv(num_feats + num_feats // 2,
                                                num_feats // 2, 6)
        self.convblocks['aspp', 2] = AtrousConv(num_feats * 2,
                                                num_feats // 2, 12)
        self.convblocks['aspp',
                        3] = AtrousConv(num_feats * 2 + num_feats // 2,
                                        num_feats // 2, 18)
        self.convblocks['aspp', 4] = AtrousConv(num_feats * 3,
                                                num_feats // 2, 24)
        self.convblocks['aspp_out', 0] = torch.nn.Sequential(
            nn.Conv2d(num_feats + (num_feats // 2) * 5,
                      num_feats,
                      3,
                      1,
                      1,
                      bias=False), nn.ELU())

        # Build decoder
        self.num_layers = len(num_ch_dec) - 1
        self.num_enc_feats = len(num_ch_enc) - 1
        idx_feats = self.num_enc_feats
        if self.first_fuse:
            first_in_ch = num_ch_dec[-1]
        else:
            first_in_ch = num_ch_enc[idx_feats]

        for i in range(self.num_layers, -1, -1):
            # upconv_0
            num_ch_in = first_in_ch if i == self.num_layers else num_ch_dec[i +
                                                                            1]
            if i == self.num_layers:
                idx_feats -= 1
            num_ch_out = num_ch_dec[i]
            self.convblocks[('dec', i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            if self.unfold_upsample:
                self.convblocks[('up', i)] = nn.Conv2d(num_ch_out, int(num_ch_out * 4), 1)
            # upconv_1
            num_ch_in = num_ch_dec[i]
            if idx_feats >= 0:
                num_ch_in += num_ch_enc[idx_feats]
                idx_feats -= 1
            num_ch_out = num_ch_dec[i]
            if i == self.num_layers:
                self.convblocks[('dec', i, 1)] = ConvBlock(num_ch_in,
                                                           num_ch_out,
                                                           nonlin=False)
            else:
                self.convblocks[('dec', i,
                                 1)] = ConvBlock(num_ch_in, num_ch_out)

        self.convblocks[('out', 0)] = Conv3x3(num_ch_dec[0], output_ch)

        self._convs = nn.ModuleList(list(self.convblocks.values()))
        if self.unfold_upsample:
            self.unfold = nn.PixelShuffle(2)

    def forward(self, features, img_shape):
        x = features[-1]
        delta = []
        if self.first_fuse:
            x = self.embed_conv(x)
            # a_feats = [x]
            for a_idx in range(5):
                a_x = self.convblocks['aspp', a_idx](x)
                if isinstance(a_x, tuple):
                    delta.append(a_x[1].detach())
                    a_x = a_x[0]
                # a_feats.append(a_x)
                x = torch.cat([x, a_x], dim=1)
            # x = torch.cat(a_feats, dim=1)
            x = self.convblocks['aspp_out', 0](x)

        idx_feats = self.num_enc_feats - 1
        for i in range(self.num_layers, -1, -1):
            x = self.convblocks[('dec', i, 0)](x)
            if idx_feats >= 0:
                tar_shape = features[idx_feats].shape
            elif i == 0:
                tar_shape = img_shape
            else:
                tar_shape = [s * 2 for s in x.shape]
            if self.unfold_upsample:
                x = self.convblocks[('up', i)](x)
                x = [self._upsample_unfold(x, tar_shape)]
            else:
                x = [self._upsample(x, tar_shape)]
            if idx_feats >= 0:
                x += [features[idx_feats]]
                idx_feats -= 1
            x = torch.cat(x, 1)
            x = self.convblocks[('dec', i, 1)](x)
            # print(x.shape)
            if not self.first_fuse and i == self.num_layers:
                # a_feats = [x]
                for a_idx in range(5):
                    a_x = self.convblocks['aspp', a_idx](x)
                    # a_feats.append(a_x)
                    if isinstance(a_x, tuple):
                        delta.append(a_x[1].detach())
                        a_x = a_x[0]
                    x = torch.cat([x, a_x], dim=1)
                # x = torch.cat(a_feats, dim=1)
                x = self.convblocks['aspp_out', 0](x)
        return self.convblocks[('out', 0)](x)

    def _upsample(self, x, shape):
        return F.interpolate(x, size=shape[2:], mode='nearest')
    
    def _upsample_unfold(self, x, shape):
        assert shape[2] / x.shape[2] == 2 and x.shape[1] % 4 == 0,\
            'Shuffle could not be used in {} to {}'.format(x.shape, shape)
        return self.unfold(x)


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
