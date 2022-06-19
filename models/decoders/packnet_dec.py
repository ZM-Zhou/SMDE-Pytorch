import torch
import torch.nn as nn


class PackDecoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        out_channels = 1
        ni, no = 64, out_channels
        n1, n2, n3, n4, n5 = 64, 64, 128, 256, 512

        n1o, n1i = n1, n1 + ni + no
        n2o, n2i = n2, n2 + n1 + no
        n3o, n3i = n3, n3 + n2 + no
        n4o, n4i = n4, n4 + n3
        n5o, n5i = n5, n5 + n4
        unpack_kernel = [3, 3, 3, 3, 3]
        iconv_kernel = [3, 3, 3, 3, 3]

        self.unpack5 = UnpackLayerConv3d(n5, n5o, unpack_kernel[0])
        self.unpack4 = UnpackLayerConv3d(n5, n4o, unpack_kernel[1])
        self.unpack3 = UnpackLayerConv3d(n4, n3o, unpack_kernel[2])
        self.unpack2 = UnpackLayerConv3d(n3, n2o, unpack_kernel[3])
        self.unpack1 = UnpackLayerConv3d(n2, n1o, unpack_kernel[4])

        self.iconv5 = Conv2D(n5i, n5, iconv_kernel[0], 1)
        self.iconv4 = Conv2D(n4i, n4, iconv_kernel[1], 1)
        self.iconv3 = Conv2D(n3i, n3, iconv_kernel[2], 1)
        self.iconv2 = Conv2D(n2i, n2, iconv_kernel[3], 1)
        self.iconv1 = Conv2D(n1i, n1, iconv_kernel[4], 1)

        # Depth Layers
        self.unpack_disp4 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        self.disp4_layer = InvDepth(n4, out_channels=out_channels)
        self.disp3_layer = InvDepth(n3, out_channels=out_channels)
        self.disp2_layer = InvDepth(n2, out_channels=out_channels)
        self.disp1_layer = InvDepth(n1, out_channels=out_channels)
    
        self.init_weights()

    def init_weights(self):
        """Initializes network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, feats, image_size=None):
        skip1, skip2, skip3, skip4, skip5, x5p = feats
        outputs = {}
        
        unpack5 = self.unpack5(x5p)
        concat5 = torch.cat((unpack5, skip5), 1)
        iconv5 = self.iconv5(concat5)

        unpack4 = self.unpack4(iconv5)
        concat4 = torch.cat((unpack4, skip4), 1)
        iconv4 = self.iconv4(concat4)

        disp4 = self.disp4_layer(iconv4)
        udisp4 = self.unpack_disp4(disp4)

        unpack3 = self.unpack3(iconv4)
        concat3 = torch.cat((unpack3, skip3, udisp4), 1)
        iconv3 = self.iconv3(concat3)
        
        disp3 = self.disp3_layer(iconv3)
        udisp3 = self.unpack_disp3(disp3)

        unpack2 = self.unpack2(iconv3)
        concat2 = torch.cat((unpack2, skip2, udisp3), 1)
        iconv2 = self.iconv2(concat2)

        disp2 = self.disp2_layer(iconv2)
        udisp2 = self.unpack_disp2(disp2)

        unpack1 = self.unpack1(iconv2)
        concat1 = torch.cat((unpack1, skip1, udisp2), 1)
        iconv1 = self.iconv1(concat1)

        disp1 = self.disp1_layer(iconv1)
        
        outputs[3] = disp4
        outputs[2] = disp3
        outputs[1] = disp2
        outputs[0] = disp1

        return outputs


class UnpackLayerConv3d(nn.Module):
    """
    Unpacking layer with 3d convolutions. Takes a [B,C,H,W] tensor, convolves it
    to produce [B,(r^2)C,H,W] and then unpacks it to produce [B,C,rH,rW].
    """
    def __init__(self, in_channels, out_channels, kernel_size, r=2, d=8):
        """
        Initializes a UnpackLayerConv3d object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        d : int
            Number of 3D features
        """
        super().__init__()
        self.conv = Conv2D(in_channels, out_channels * (r ** 2) // d, kernel_size, 1)
        self.unpack = nn.PixelShuffle(r)
        self.conv3d = nn.Conv3d(1, d, kernel_size=(3, 3, 3),
                                stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        """Runs the UnpackLayerConv3d layer."""
        x = self.conv(x)
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        b, c, d, h, w = x.shape
        x = x.view(b, c * d, h, w)
        x = self.unpack(x)
        return x

class InvDepth(nn.Module):
    """Inverse depth layer"""
    def __init__(self, in_channels, out_channels=1, min_depth=0.5):
        """
        Initializes an InvDepth object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        min_depth : float
            Minimum depth value to calculate
        """
        super().__init__()
        self.min_depth = min_depth
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.pad = nn.ConstantPad2d([1] * 4, value=0)
        self.activ = nn.Sigmoid()

    def forward(self, x):
        """Runs the InvDepth layer."""
        x = self.conv1(self.pad(x))
        # return x
        return self.activ(x) / self.min_depth

class Conv2D(nn.Module):
    """
    2D convolution with GroupNorm and ELU
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Kernel size
    stride : int
        Stride
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.pad = nn.ConstantPad2d([kernel_size // 2] * 4, value=0)
        self.normalize = torch.nn.GroupNorm(16, out_channels)
        self.activ = nn.ELU(inplace=True)

    def forward(self, x):
        """Runs the Conv2D layer."""
        x = self.conv_base(self.pad(x))
        return self.activ(self.normalize(x))