import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DIFFNetDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, mobile_encoder=False):
        super(DIFFNetDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()
        
        # decoder
        self.convs = nn.ModuleDict()
        
        # adaptive block
        if self.num_ch_dec[0] < 16:
            self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1],self.num_ch_dec[0])
            self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0],self.num_ch_dec[0])
        
        # adaptive block
            self.convs["72"] = Attention_Module(2 * self.num_ch_dec[4],  2 * self.num_ch_dec[4]  , self.num_ch_dec[4])
            self.convs["36"] = Attention_Module(self.num_ch_dec[4], 3 * self.num_ch_dec[3], self.num_ch_dec[3])
            self.convs["18"] = Attention_Module(self.num_ch_dec[3], self.num_ch_dec[2] * 3 + 64 , self.num_ch_dec[2])
            self.convs["9"] = Attention_Module(self.num_ch_dec[2], 64, self.num_ch_dec[1])
        else: 
            self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1],self.num_ch_dec[0])
            self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0],self.num_ch_dec[0])
            self.convs["72"] = Attention_Module(self.num_ch_enc[4]  , self.num_ch_enc[3] * 2, 256)
            self.convs["36"] = Attention_Module(256, self.num_ch_enc[2] * 3, 128)
            self.convs["18"] = Attention_Module(128, self.num_ch_enc[1] * 3 + 64 , 64)
            self.convs["9"] = Attention_Module(64, 64, 32)
        for i in range(4):
            self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, image_size=None):
        outputs = {}
        feature144 = input_features[4]
        feature72 = input_features[3]
        feature36 = input_features[2]
        feature18 = input_features[1]
        feature64 = input_features[0]
        x72 = self.convs["72"](feature144, feature72)
        x36 = self.convs["36"](x72 , feature36)
        x18 = self.convs["18"](x36 , feature18)
        x9 = self.convs["9"](x18,[feature64])
        x6 = self.convs["up_x9_1"](upsample(self.convs["up_x9_0"](x9)))
        
        outputs[0] = self.convs["dispConvScale0"](x6)
        outputs[1] = self.convs["dispConvScale1"](x9)
        outputs[2] = self.convs["dispConvScale2"](x18)
        outputs[3] = self.convs["dispConvScale3"](x36)
        return outputs
        
def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
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
    
class Attention_Module(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel = None):
        super(Attention_Module, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        channel = in_channel
        self.ca = ChannelAttention(channel)
        #self.sa = SpatialAttention()
        #self.cs = CS_Block(channel)
        self.conv_se = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, stride = 1, padding = 1 )
        self.relu = nn.ReLU(inplace = True)

    def forward(self, high_features, low_features):

        features = [upsample(high_features)]
        features += low_features
        features = torch.cat(features, 1)

        features = self.ca(features)
        #features = self.sa(features)
        #features = self.cs(features)
        
        return self.relu(self.conv_se(features))

## ChannelAttetion
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Linear(in_planes,in_planes // ratio, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(in_planes // ratio, in_planes, bias = False)
        )
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, in_feature):
        x = in_feature
        b, c, _, _ = in_feature.size()
        avg_out = self.fc(self.avg_pool(x).view(b,c)).view(b, c, 1, 1)
        out = avg_out
        return self.sigmoid(out).expand_as(in_feature) * in_feature