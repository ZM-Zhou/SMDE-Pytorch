import torch
import torch.nn as nn
from functools import partial


class PackEncoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        in_channels = 3
        ni = 64
        n1, n2, n3, n4, n5 = 64, 64, 128, 256, 512
        pack_kernel = [5, 3, 3, 3, 3]
        num_blocks = [2, 2, 3, 3]
        dropout = None

        self.pre_calc = Conv2D(in_channels, ni, 5, 1)

        self.pack1 = PackLayerConv3d(n1, pack_kernel[0])
        self.pack2 = PackLayerConv3d(n2, pack_kernel[1])
        self.pack3 = PackLayerConv3d(n3, pack_kernel[2])
        self.pack4 = PackLayerConv3d(n4, pack_kernel[3])
        self.pack5 = PackLayerConv3d(n5, pack_kernel[4])

        self.conv1 = Conv2D(ni, n1, 7, 1)
        self.conv2 = ResidualBlock(n1, n2, num_blocks[0], 1, dropout=dropout)
        self.conv3 = ResidualBlock(n2, n3, num_blocks[1], 1, dropout=dropout)
        self.conv4 = ResidualBlock(n3, n4, num_blocks[2], 1, dropout=dropout)
        self.conv5 = ResidualBlock(n4, n5, num_blocks[3], 1, dropout=dropout)
    
        self.init_weights()

    def init_weights(self):
        """Initializes network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        x = self.pre_calc(x)
        # Encoder
        x1 = self.conv1(x)
        x1p = self.pack1(x1)
        x2 = self.conv2(x1p)
        x2p = self.pack2(x2)
        x3 = self.conv3(x2p)
        x3p = self.pack3(x3)
        x4 = self.conv4(x3p)
        x4p = self.pack4(x4)
        x5 = self.conv5(x4p)
        x5p = self.pack5(x5)

        return [x, x1p, x2p, x3p, x4p, x5p]




class PackLayerConv3d(nn.Module):
    """
    Packing layer with 3d convolutions. Takes a [B,C,H,W] tensor, packs it
    into [B,(r^2)C,H/r,W/r] and then convolves it to produce [B,C,H/r,W/r].
    """
    def __init__(self, in_channels, kernel_size, r=2, d=8):
        """
        Initializes a PackLayerConv3d object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        d : int
            Number of 3D features
        """
        super().__init__()
        self.conv = Conv2D(in_channels * (r ** 2) * d, in_channels, kernel_size, 1)
        self.pack = partial(packing, r=r)
        self.conv3d = nn.Conv3d(1, d, kernel_size=(3, 3, 3),
                                stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        """Runs the PackLayerConv3d layer."""
        x = self.pack(x)
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        b, c, d, h, w = x.shape
        x = x.view(b, c * d, h, w)
        x = self.conv(x)
        return x

def packing(x, r=2):
    """
    Takes a [B,C,H,W] tensor and returns a [B,(r^2)C,H/r,W/r] tensor, by concatenating
    neighbor spatial pixels as extra channels. It is the inverse of nn.PixelShuffle
    (if you apply both sequentially you should get the same tensor)
    Parameters
    ----------
    x : torch.Tensor [B,C,H,W]
        Input tensor
    r : int
        Packing ratio
    Returns
    -------
    out : torch.Tensor [B,(r^2)C,H/r,W/r]
        Packed tensor
    """
    b, c, h, w = x.shape
    out_channel = c * (r ** 2)
    out_h, out_w = h // r, w // r
    x = x.contiguous().view(b, c, out_h, r, out_w, r)
    return x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)

def ResidualBlock(in_channels, out_channels, num_blocks, stride, dropout=None):
    """
    Returns a ResidualBlock with various ResidualConv layers.
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    num_blocks : int
        Number of residual blocks
    stride : int
        Stride
    dropout : float
        Dropout value
    """
    layers = [ResidualConv(in_channels, out_channels, stride, dropout=dropout)]
    for i in range(1, num_blocks):
        layers.append(ResidualConv(out_channels, out_channels, 1, dropout=dropout))
    return nn.Sequential(*layers)

class ResidualConv(nn.Module):
    """2D Convolutional residual block with GroupNorm and ELU"""
    def __init__(self, in_channels, out_channels, stride, dropout=None):
        """
        Initializes a ResidualConv object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        stride : int
            Stride
        dropout : float
            Dropout value
        """
        super().__init__()
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride)
        self.conv2 = Conv2D(out_channels, out_channels, 3, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.normalize = torch.nn.GroupNorm(16, out_channels)
        self.activ = nn.ELU(inplace=True)

        if dropout:
            self.conv3 = nn.Sequential(self.conv3, nn.Dropout2d(dropout))

    def forward(self, x):
        """Runs the ResidualConv layer."""
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        shortcut = self.conv3(x)
        return self.activ(self.normalize(x_out + shortcut))

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