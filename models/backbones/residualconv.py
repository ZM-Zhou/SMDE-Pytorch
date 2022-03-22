import torch
import torch.nn as nn


class Residual_Conv(nn.Module):
    def __init__(self,
                 num_ch_input=3,
                 num_ch_enc=[32, 64, 128, 256, 256, 256, 512],
                 is_bn=False,
                 input_flow=False,
                 output_all=True):
        super().__init__()

        self.num_layers = len(num_ch_enc)
        self.input_flow = input_flow
        self.output_all = output_all

        for idx, num_ch in enumerate(num_ch_enc):
            if idx == 0:
                setattr(self, 'conv{}'.format(idx),
                        conv_elu(is_bn, num_ch_input, num_ch, kernel_size=3))
            else:
                if idx == 1 and self.input_flow:
                    in_ch = num_ch_enc[idx - 1] + 1
                else:
                    in_ch = num_ch_enc[idx - 1]
                setattr(
                    self, 'conv{}'.format(idx),
                    conv_elu(is_bn,
                             in_ch,
                             num_ch,
                             kernel_size=3,
                             pad=1,
                             stride=2))
            setattr(self, 'conv{}_1'.format(idx), residual_block(num_ch))

        # Initialize conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight.data
                )  # initialize weights with normal distribution
                if m.bias is not None:
                    m.bias.data.zero_()  # initialize bias as zero
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, flow=None):
        feats = []
        for idx in range(self.num_layers):
            layer_0 = getattr(self, 'conv{}'.format(idx))
            layer_1 = getattr(self, 'conv{}_1'.format(idx))
            if idx == 1 and flow is not None:
                x = torch.cat([x, flow], dim=1)
            x = layer_1(layer_0(x))
            if not self.output_all and idx == 0:
                continue
            feats.append(x)
        return feats


def conv_elu(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, pad=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=pad,
                      bias=False), nn.BatchNorm2d(out_planes),
            nn.ELU(inplace=True))

    else:
        return nn.Sequential(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=pad,
                      bias=True), nn.ELU(inplace=True))


class residual_block(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(residual_block, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes,
                               in_planes,
                               kernel_size=kernel_size,
                               padding=(kernel_size - 1) // 2,
                               bias=False)
        self.conv2 = nn.Conv2d(in_planes,
                               in_planes,
                               kernel_size=kernel_size,
                               padding=(kernel_size - 1) // 2,
                               bias=False)

    def forward(self, x):
        x = self.elu(self.conv2(self.elu(self.conv1(x))) + x)
        return x
