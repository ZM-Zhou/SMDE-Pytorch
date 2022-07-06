import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by LeakyReLU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.LeakyReLU(inplace=True)

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


class dispHead(nn.Module):
    def __init__(self):
        super(dispHead, self).__init__()
        outD = 1

        self.covd1 = torch.nn.Sequential(nn.ReflectionPad2d(1),
                                         torch.nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=1,
                                                         padding=0, bias=True),
                                         torch.nn.LeakyReLU(inplace=True))

        self.covd2 = torch.nn.Sequential(nn.ReflectionPad2d(1),
                                         torch.nn.Conv2d(in_channels=256, out_channels=outD, kernel_size=3, stride=1,
                                                         padding=0, bias=True))

    def forward(self, x):
        return self.covd2(self.covd1(x))


class BasicMotionEncoder(nn.Module):
    def __init__(self):
        super(BasicMotionEncoder, self).__init__()
        # inD = 1

        self.convc1 = ConvBlock(128, 160)
        self.convc2 = ConvBlock(160, 128)

        self.convf1 = torch.nn.Sequential(
            nn.ReflectionPad2d(3),
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=0, bias=True),
            torch.nn.LeakyReLU(inplace=True))
        self.convf2 = torch.nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=0, bias=True),
            torch.nn.LeakyReLU(inplace=True))

        self.conv = ConvBlock(128 + 32, 192 - 1)

    def forward(self, depth, corr):
        cor = self.convc1(corr)
        cor = self.convc2(cor)

        dep = self.convf1(depth)
        dep = self.convf2(dep)

        cor_depth = torch.cat([cor, dep], dim=1)
        out = self.conv(cor_depth)
        return torch.cat([out, depth], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder()

        self.flow_head = dispHead()

        self.mask = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(192, 324, 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(324, 64 * 9, 1, padding=0))

    def forward(self, net, corr, depth):
        net = self.encoder(depth, corr)
        delta_depth = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)

        return net, mask, delta_depth

class SepConvGRU(nn.Module):
    def __init__(self):
        super(SepConvGRU, self).__init__()
        hidden_dim = 128
        catt = 256

        self.convz1 = nn.Conv2d(catt, hidden_dim, (1, 3), padding=(0, 1))
        self.convr1 = nn.Conv2d(catt, hidden_dim, (1, 3), padding=(0, 1))
        self.convq1 = nn.Conv2d(catt, hidden_dim, (1, 3), padding=(0, 1))

        self.convz2 = nn.Conv2d(catt, hidden_dim, (3, 1), padding=(1, 0))
        self.convr2 = nn.Conv2d(catt, hidden_dim, (3, 1), padding=(1, 0))
        self.convq2 = nn.Conv2d(catt, hidden_dim, (3, 1), padding=(1, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class R_MSFM3(nn.Module):
    def __init__(self, x):
        super(R_MSFM3, self).__init__()

        self.convX11 = torch.nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=0, bias=True),
            torch.nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=0, bias=True),
            torch.nn.Tanh())
        if x:  
            self.convX21 = torch.nn.Sequential(
                nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=0, bias=True),
                torch.nn.Tanh())
            self.convX31 = torch.nn.Sequential(
                nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),
                torch.nn.Tanh())
        else:
            self.convX21 = torch.nn.Sequential(
                nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0, bias=True),
                torch.nn.Tanh())
            self.convX31 = torch.nn.Sequential(
                nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),
                torch.nn.Tanh())

        self.sigmoid = nn.Sigmoid()

        self.update_block = BasicUpdateBlock()
        self.gruc = SepConvGRU()
    def upsample_depth(self, flow, mask):
        """ Upsample depth field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, 8 * H, 8 * W)

    def forward(self, features, image_size=None, iters=3):
        """ Estimate depth for a single image """

        x1, x2, x3 = features

        disp_predictions = {}
        b, c, h, w = x3.shape
        dispFea = torch.zeros([b, 1, h, w], requires_grad=True).to(x1.device)
        net = torch.zeros([b, 256, h, w], requires_grad=True).to(x1.device)

        for itr in range(iters):
            if itr in [0]:
                corr = self.convX31(x3)
            elif itr in [1]:
                corrh = corr
                corr = self.convX21(x2)
                corr = self.gruc(corrh, corr)
            elif itr in [2]:
                corrh = corr
                corr = self.convX11(x1)
                corr = self.gruc(corrh, corr)

            net, up_mask, delta_disp = self.update_block(net, corr, dispFea)
            dispFea = dispFea + delta_disp

            disp = self.sigmoid(dispFea)
            # upsample predictions
            if self.training:
                disp_up = self.upsample_depth(disp, up_mask)
                disp_predictions[iters - itr - 1] = disp_up
            else:
                if (iters-1)==itr:
                    disp_up = self.upsample_depth(disp, up_mask)
                    disp_predictions[iters - itr - 1] = disp_up


        return disp_predictions


class R_MSFM6(nn.Module):
    def __init__(self,x):
        super(R_MSFM6, self).__init__()

        self.convX11 = torch.nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=0, bias=True),
            torch.nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=0, bias=True),
            torch.nn.Tanh())

        self.convX12 = torch.nn.Sequential(
            nn.Conv2d(128, 128, (1, 3), padding=(0, 1)),
            torch.nn.Tanh(),
            nn.Conv2d(128, 128, (3, 1), padding=(1, 0)),
            torch.nn.Tanh())


        if x:
            self.convX21 = torch.nn.Sequential(
                nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=0, bias=True),
                torch.nn.Tanh())
            self.convX31 = torch.nn.Sequential(
                nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),
                torch.nn.Tanh())
        else:
            self.convX21 = torch.nn.Sequential(
                nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0, bias=True),
                torch.nn.Tanh())
            self.convX31 = torch.nn.Sequential(
                nn.ReflectionPad2d(1),
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1,
                                bias=True),
                torch.nn.Tanh())



        self.convX22 = torch.nn.Sequential(
            nn.Conv2d(128, 128, (1, 3), padding=(0, 1)),
            torch.nn.Tanh(),
            nn.Conv2d(128, 128, (3, 1), padding=(1, 0)),
            torch.nn.Tanh())

        self.convX32 = torch.nn.Sequential(
            nn.Conv2d(128, 128, (1, 3), padding=(0, 1)),
            torch.nn.Tanh(),
            nn.Conv2d(128, 128, (3, 1), padding=(1, 0)),
            torch.nn.Tanh())

        self.sigmoid = nn.Sigmoid()
        self.gruc = SepConvGRU()
        self.update_block = BasicUpdateBlock()

    def upsample_depth(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, 8 * H, 8 * W)

    def forward(self, features, image_size=None, iters=6):
        """ Estimate depth for a single image """

        x1, x2, x3 = features

        disp_predictions = {}
        b, c, h, w = x3.shape
        dispFea = torch.zeros([b, 1, h, w], requires_grad=True).to(x1.device)
        net = torch.zeros([b, 256, h, w], requires_grad=True).to(x1.device)

        for itr in range(iters):
            if itr in [0]:
                corr = self.convX31(x3)
            elif itr in [1]:
                corrh = corr
                corr = self.convX32(corr)
                corr = self.gruc(corrh, corr)
            elif itr in [2]:
                corrh = corr
                corr = self.convX21(x2)
                corr = self.gruc(corrh, corr)
            elif itr in [3]:
                corrh = corr
                corr = self.convX22(corr)
                corr = self.gruc(corrh, corr)
            elif itr in [4]:
                corrh = corr
                corr = self.convX11(x1)
                corr = self.gruc(corrh, corr)
            elif itr in [5]:
                corrh = corr
                corr = self.convX12(corr)
                corr = self.gruc(corrh, corr)

            net, up_mask, delta_disp = self.update_block(net, corr, dispFea)
            dispFea = dispFea + delta_disp

            disp = self.sigmoid(dispFea)
            # upsample predictions
   
            if self.training:
                disp_up = self.upsample_depth(disp, up_mask)
                disp_predictions[iters - itr - 1] = disp_up
            else:
                if (iters-1)==itr:
                    disp_up = self.upsample_depth(disp, up_mask)
                    disp_predictions[iters - itr - 1] = disp_up


        return disp_predictions
