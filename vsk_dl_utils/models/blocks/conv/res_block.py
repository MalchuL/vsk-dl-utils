import copy
import math

import torch
from torch import nn

from vsk_dl_utils.layers import conv1x1
from vsk_dl_utils.models.blocks.conv.convbnrelu import Conv2dBNReLU
from vsk_dl_utils.models.blocks.weighted_sum import WeightedSum
from vsk_dl_utils.utils.act_layers import get_act


class ResidualBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=None,
                 stride=1,
                 dilation=1,
                 groups=1,
                 last_act=False,
                 norm_layer='batch',
                 act_layer='relu'):
        super(ResidualBlock, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = (kernel_size - 1) // 2
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 is not supported!')
        self.conv1 = Conv2dBNReLU(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  bias=False,
                                  dilation=dilation,
                                  groups=groups,
                                  norm_layer=norm_layer,
                                  act_layer=act_layer)
        self.conv2 = Conv2dBNReLU(out_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  bias=False,
                                  dilation=dilation,
                                  groups=groups,
                                  norm_layer=norm_layer,
                                  act_layer=None)

        self.downsample = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.downsample = conv1x1(in_planes=in_channels, out_planes=out_channels,
                                      stride=stride, bias=True)

        self.out_channels = out_channels
        self.alpha_blend = WeightedSum()
        self.last_act = nn.Identity()
        if last_act:
            self.last_act = get_act(act_layer)(inplace=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        residual = self.downsample(residual)
        out = self.alpha_blend(out, residual)
        out = self.last_act(out)
        return out
