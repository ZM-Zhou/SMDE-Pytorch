import torch
import torch.nn as nn
import torch.nn.functional as F

class UpSample_Layers(nn.Module):
    def __init__(self,
                 num_ch_enc,
                 num_ch_dec = [16, 32, 64, 128, 256],
                 output_ch = 1,
                 out_scales = [0, 1, 2, 3]):
        super().__init__()
        self.out_scales = out_scales
        self.convblocks = {}
        self.num_layers = len(num_ch_dec) - 1
        self.num_enc_feats = len(num_ch_enc) - 1

        # Bulid decoder
        idx_feats = self.num_enc_feats
        for i in range(self.num_layers, -1, -1):
            # upconv_0
            num_ch_in = num_ch_enc[idx_feats] if i == self.num_layers else num_ch_dec[i + 1]
            if i == self.num_layers:
                idx_feats -= 1
            num_ch_out = num_ch_dec[i]
            self.convblocks[("dec", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            # upconv_1
            num_ch_in = num_ch_dec[i]
            if idx_feats >= 0:
                num_ch_in += num_ch_enc[idx_feats]
                idx_feats -= 1
            num_ch_out = num_ch_dec[i]
            self.convblocks[("dec", i, 1)] = ConvBlock(num_ch_in,
                                                       num_ch_out)
        for scale in self.out_scales:
            self.convblocks[("out", scale)] = Conv3x3(num_ch_dec[scale], output_ch)
        self._convs = nn.ModuleList(list(self.convblocks.values()))
    
    def forward(self, features, img_shape):
        outputs = {}
        x = features[-1]
        idx_feats = self.num_enc_feats - 1
        for i in range(self.num_layers, -1, -1):
            x = self.convblocks[("dec", i, 0)](x)
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
            x = self.convblocks[("dec", i, 1)](x)
            if i in self.out_scales:
                out_disp = self.convblocks[("out", i)](x)
                outputs[i] = out_disp
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs
    
    def _upsample(self, x, shape):
        return F.interpolate(x, size=shape[2:], mode="nearest")


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
       from https://github.com/nianticlabs/monodepth2
    """
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
    """Layer to perform a convolution followed by ELU
       from https://github.com/nianticlabs/monodepth2
    """
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
        

