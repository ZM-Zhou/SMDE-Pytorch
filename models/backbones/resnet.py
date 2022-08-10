import torch.nn as nn
import torchvision.models as models


class ResNet_Backbone(nn.Module):
    def __init__(self, layer_num=50, pretrained=True, in_ch=3, ignore_last2=None):
        super().__init__()
        self.ignore_last2 = ignore_last2

        if layer_num == 50:
            encoder = models.resnet50(pretrained)
        elif layer_num == 18:
            encoder = models.resnet18(pretrained)
        else:
            raise NotImplementedError

        if in_ch == 3:
            first_conv = encoder.conv1
        else:
            first_conv = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)

        self.layers0 = nn.Sequential(first_conv, encoder.bn1, encoder.relu)
        self.layers1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layers2 = encoder.layer2
        if not self.ignore_last2:
            self.layers3 = encoder.layer3
            self.layers4 = encoder.layer4

    def forward(self, x, *args, **kargs):
        feats = []
        x = self.layers0(x)
        feats.append(x)
        x = self.layers1(x)
        feats.append(x)
        x = self.layers2(x)
        feats.append(x)
        if not self.ignore_last2:
            x = self.layers3(x)
            feats.append(x)
            x = self.layers4(x)
            feats.append(x)

        return feats
