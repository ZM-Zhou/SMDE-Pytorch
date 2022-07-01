import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoders.upsample import UpSample_Layers, ConvBlock


class CMA(nn.Module):
    def __init__(self, num_ch_enc):
        super(CMA, self).__init__()

        self.scales = [0, 1, 2, 3]
        self.cma_layers = [3, 2, 1]
        self.num_ch_dec = num_ch_enc
        self.sgt = 0.1
        in_channels_list = [32, 64, 128, 256, 16]

        num_output_channels = 1

        self.depth_decoder = UpSample_Layers(num_ch_enc,
                                             output_ch=num_output_channels)
        self.seg_decoder = UpSample_Layers(num_ch_enc,
                                           output_ch=19,
                                           out_scales=[0])

        att_d_to_s = {}
        att_s_to_d = {}
        for i in self.cma_layers:
            att_d_to_s[str(i)] = MultiEmbedding(in_channels=in_channels_list[i],
                                                num_head=4,
                                                ratio=2)
            att_s_to_d[str(i)] = MultiEmbedding(in_channels=in_channels_list[i],
                                                num_head=4,
                                                ratio=2)
        self.att_d_to_s = nn.ModuleDict(att_d_to_s)
        self.att_s_to_d = nn.ModuleDict(att_s_to_d)

    def forward(self, input_features, image_shape=None):

        outputs = {}
        x = input_features[-1]
        x_d = None
        x_s = None
        for i in range(4, -1, -1):
            if x_d is None:
                x_d = self.depth_decoder._convs[-2 * i + 8](x)
            else:
                x_d = self.depth_decoder._convs[-2 * i + 8](x_d)

            if x_s is None:
                x_s = self.seg_decoder._convs[-2 * i + 8](x)
            else:
                x_s = self.seg_decoder._convs[-2 * i + 8](x_s)

            x_d = [upsample(x_d)]
            x_s = [upsample(x_s)]

            if i > 0:
                x_d += [input_features[i - 1]]
                x_s += [input_features[i - 1]]

            x_d = torch.cat(x_d, 1)
            x_s = torch.cat(x_s, 1)

            x_d = self.depth_decoder._convs[-2 * i + 9](x_d)
            x_s = self.seg_decoder._convs[-2 * i + 9](x_s)

            if (i - 1) in self.cma_layers:
                if len(self.cma_layers) == 1:
                    x_d_att = self.att_d_to_s(x_d, x_s)
                    x_s_att = self.att_s_to_d(x_s, x_d)
                    x_d = x_d_att
                    x_s = x_s_att
                else:
                    x_d_att = self.att_d_to_s[str(i - 1)](x_d, x_s)
                    x_s_att = self.att_s_to_d[str(i - 1)](x_s, x_d)
                    x_d = x_d_att
                    x_s = x_s_att

            if self.sgt:
                outputs[('d_feature', i)] = x_d
                outputs[('s_feature', i)] = x_s
            if i in self.scales:
                outs = self.depth_decoder._convs[10 + i](x_d)
                outputs[i] =outs
                if i == 0:
                    outs = self.seg_decoder._convs[10 + i](x_s)
                    outputs[("seg_logits", i)] = outs[:, :19, :, :]

        return outputs

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

def W(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels)

    )

class MultiEmbedding(nn.Module):
    def __init__(self, in_channels, num_head, ratio):
        super(MultiEmbedding, self).__init__()
        self.in_channels = in_channels

        self.num_head = num_head
        self.out_channel = int(num_head * in_channels * ratio)
        self.query_conv = nn.Conv2d(in_channels, self.out_channel, kernel_size=1, bias=True)
        self.key_conv = nn.Conv2d(in_channels, self.out_channel, kernel_size=1, bias=True)
        self.value_conv = nn.Conv2d(in_channels, self.out_channel, kernel_size=1, bias=True)
        self.W = W(int(in_channels * ratio), in_channels)

        self.fuse = nn.Sequential(ConvBlock(in_channels * 2, in_channels),
                                  nn.Conv2d(in_channels, in_channels, kernel_size=1))

    def forward(self, key, query):

        batch, channels, height, width = query.size()
        q_out = self.query_conv(query).contiguous().view(batch, self.num_head, -1, height, width)
        k_out = self.key_conv(key).contiguous().view(batch, self.num_head, -1, height, width)
        v_out = self.value_conv(key).contiguous().view(batch, self.num_head, -1, height, width)

        att = (q_out * k_out).sum(dim=2) / np.sqrt(self.out_channel)

        if self.num_head == 1:
            softmax = att.unsqueeze(dim=2)
        else:
            softmax = F.softmax(att, dim=1).unsqueeze(dim=2)

        weighted_value = v_out * softmax
        weighted_value = weighted_value.sum(dim=1)
        out = self.W(weighted_value)

        return self.fuse(torch.cat([key, out], dim=1))
