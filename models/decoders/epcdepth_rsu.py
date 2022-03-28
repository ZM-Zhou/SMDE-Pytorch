import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDisp(nn.Module):
    def __init__(self, bott_channels, out_channels, bottleneck):
        super(EncoderDisp, self).__init__()
        self.bottleneck = bottleneck
        self.disp = nn.Sequential(
            nn.Conv2d(bott_channels,
                      out_channels,
                      3,
                      1,
                      1,
                      padding_mode='reflect'), nn.Sigmoid())

    def forward(self, inputs):
        features = self.bottleneck(inputs)
        out = self.disp(features)
        return out


class RSUDecoder(nn.Module):
    def __init__(self,
                 encoder_layer_channels,
                 num_output_channels=1,
                 use_encoder_disp=False):
        super(RSUDecoder, self).__init__()

        self.use_encoder_disp = use_encoder_disp
        decoder_layer_channels = [256, 128, 64, 32, 16]

        # decoder
        self.stage5d = RSU3(
            encoder_layer_channels[-1] + encoder_layer_channels[-2], 64,
            decoder_layer_channels[0], False)

        self.stage4d = RSU4(
            decoder_layer_channels[0] + encoder_layer_channels[-3], 32,
            decoder_layer_channels[1], False)

        self.stage3d = RSU5(
            decoder_layer_channels[1] + encoder_layer_channels[-4], 16,
            decoder_layer_channels[2], False)

        self.stage2d = RSU6(
            decoder_layer_channels[2] + encoder_layer_channels[-5], 8,
            decoder_layer_channels[3], False)

        self.stage1d = RSU7(decoder_layer_channels[3], 4,
                            decoder_layer_channels[4], False)

        if use_encoder_disp:
            self.encoder_disps = nn.ModuleList()
            bottlenecks = [RSU7, RSU6, RSU5, RSU4, RSU3]
            mid_channels = [32, 32, 64, 128, 256]
            in_channels = encoder_layer_channels
            out_channels = [64, 64, 128, 256, 512]
            for c, mid_c, bott_c, bottleneck in zip(in_channels, mid_channels,
                                                    out_channels, bottlenecks):
                self.encoder_disps.append(
                    EncoderDisp(bott_c, num_output_channels,
                                bottleneck(c, mid_c, bott_c, False)))

        self.disps = nn.ModuleList()
        for channel in decoder_layer_channels:
            self.disps.append(
                nn.Sequential(
                    nn.Conv2d(channel,
                              num_output_channels,
                              3,
                              1,
                              1,
                              padding_mode='reflect'), nn.Sigmoid()))

        self.upsamp = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, inputs):

        hx6up = self.upsamp(inputs[-1])
        hx5d = self.stage5d(torch.cat((hx6up, inputs[-2]), 1))

        hx5dup = self.upsamp(hx5d)
        hx4d = self.stage4d(torch.cat((hx5dup, inputs[-3]), 1))

        hx4dup = self.upsamp(hx4d)
        hx3d = self.stage3d(torch.cat((hx4dup, inputs[-4]), 1))

        hx3dup = self.upsamp(hx3d)
        hx2d = self.stage2d(torch.cat((hx3dup, inputs[-5]), 1))

        hx2dup = self.upsamp(hx2d)
        hx1d = self.stage1d(hx2dup)

        disp_features = [hx5d, hx4d, hx3d, hx2d, hx1d]
        disps = []
        for i in range(len(disp_features)):
            disps.append(self.disps[i](disp_features[i]))

        if self.use_encoder_disp:
            encoder_disps = []
            for i in range(len(inputs)):
                encoder_disps.append(self.encoder_disps[i](inputs[i]))
            disps = encoder_disps + disps

        return disps[::-1]


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(ConvBnRelu, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,
                                 out_ch,
                                 3,
                                 padding=1 * dirate,
                                 dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


class ConvElu(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(ConvElu, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,
                                 out_ch,
                                 3,
                                 padding=1 * dirate,
                                 dilation=1 * dirate,
                                 padding_mode='reflect')
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.elu(self.conv_s1(hx))

        return xout


# upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')

    return src


# RSU-7 #
class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, encoder=True):
        super(RSU7, self).__init__()

        ConvBlock = ConvBnRelu if encoder else ConvElu

        self.rebnconvin = ConvBlock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = ConvBlock(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = ConvBlock(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = ConvBlock(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ConvBlock(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-6 #
class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, encoder=True):
        super(RSU6, self).__init__()

        ConvBlock = ConvBnRelu if encoder else ConvElu

        self.rebnconvin = ConvBlock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = ConvBlock(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = ConvBlock(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = ConvBlock(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ConvBlock(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-5 #
class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, encoder=True):
        super(RSU5, self).__init__()

        ConvBlock = ConvBnRelu if encoder else ConvElu

        self.rebnconvin = ConvBlock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = ConvBlock(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = ConvBlock(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = ConvBlock(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ConvBlock(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-4 #
class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, encoder=True):
        super(RSU4, self).__init__()

        ConvBlock = ConvBnRelu if encoder else ConvElu

        self.rebnconvin = ConvBlock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = ConvBlock(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = ConvBlock(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = ConvBlock(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ConvBlock(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-4F #
class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, encoder=True):
        super(RSU4F, self).__init__()

        ConvBlock = ConvBnRelu if encoder else ConvElu

        self.rebnconvin = ConvBlock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = ConvBlock(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = ConvBlock(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = ConvBlock(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = ConvBlock(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = ConvBlock(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = ConvBlock(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = ConvBlock(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


# RSU-3 #
class RSU3(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, encoder=True):
        super(RSU3, self).__init__()

        ConvBlock = ConvBnRelu if encoder else ConvElu

        self.rebnconvin = ConvBlock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = ConvBlock(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = ConvBlock(mid_ch, mid_ch, dirate=1)

        self.rebnconv3 = ConvBlock(mid_ch, mid_ch, dirate=2)

        self.rebnconv2d = ConvBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = ConvBlock(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)

        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)

        hx3 = self.rebnconv3(hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3, hx2), 1))

        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-2 #
class RSU2(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, encoder=True):
        super(RSU2, self).__init__()

        ConvBlock = ConvBnRelu if encoder else ConvElu

        self.rebnconvin = ConvBlock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = ConvBlock(out_ch, mid_ch, dirate=1)

        self.rebnconv2 = ConvBlock(mid_ch, mid_ch, dirate=2)

        self.rebnconv1d = ConvBlock(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)

        hx2 = self.rebnconv2(hx1)

        hx1d = self.rebnconv1d(torch.cat((hx1, hx2), 1))

        return hx1d + hxin
