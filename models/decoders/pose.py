# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

class PoseNet(nn.Module):
    """Pose network """

    def __init__(self, nb_ref_imgs=2, rotation_mode='euler', **kwargs):
        super().__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.rotation_mode = rotation_mode

        conv_channels = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv_gn(3 * (1 + self.nb_ref_imgs), conv_channels[0], kernel_size=7)
        self.conv2 = conv_gn(conv_channels[0], conv_channels[1], kernel_size=5)
        self.conv3 = conv_gn(conv_channels[1], conv_channels[2])
        self.conv4 = conv_gn(conv_channels[2], conv_channels[3])
        self.conv5 = conv_gn(conv_channels[3], conv_channels[4])
        self.conv6 = conv_gn(conv_channels[4], conv_channels[5])
        self.conv7 = conv_gn(conv_channels[5], conv_channels[6])

        self.pose_pred = nn.Conv2d(conv_channels[6], 6 * self.nb_ref_imgs,
                                   kernel_size=1, padding=0)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image, context=None):
        # assert (len(context) == self.nb_ref_imgs)
        # input = [image]
        # input.extend(context)
        # input = torch.cat(input, 1)
        out_conv1 = self.conv1(image[0])
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 1, 6)
        axisangle = pose[..., :3]
        translation = pose[..., 3:]

        return axisangle, translation

def conv_gn(in_planes, out_planes, kernel_size=3):
    """
    Convolutional block with GroupNorm
    Parameters
    ----------
    in_planes : int
        Number of input channels
    out_planes : int
        Number of output channels
    kernel_size : int
        Convolutional kernel size
    Returns
    -------
    layers : nn.Sequential
        Sequence of Conv2D + GroupNorm + ReLU
    """
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  padding=(kernel_size - 1) // 2, stride=2),
        nn.GroupNorm(16, out_planes),
        nn.ReLU(inplace=True)
    )

