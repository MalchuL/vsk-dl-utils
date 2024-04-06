import math
from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .abstract_autoencoder import AbstractAutoEncoder
from ..blocks.conv.convbnrelu import BNReLUConv2d, Conv2dBNReLU
from ..blocks.conv.res_block import ResidualBlock
from ..blocks.conv.res_block_v2 import ResidualBlockV2
from ...layers import conv3x3, conv1x1, Downsample, Upsample, SobelFilter
from ...utils.norm_layers import get_norm_layer

PADDING_MODE = 'reflect'
BIAS = False

INV_SQRT2 = 1 / math.sqrt(2)


class StableAutoencoderV2(AbstractAutoEncoder):
    def __init__(self, in_channels, ch, ch_mult=(1, 2, 4, 4), num_res_blocks=2, z_channels=4, double_z=True,
                 norm_name='batch', apply_tanh=False, out_channels=None, activation='silu'):
        act_layer = activation
        norm_layer = norm_name

        if out_channels is None:
            out_channels = in_channels
        encoder = Encoder(in_channels=in_channels, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                          z_channels=z_channels, double_z=double_z, norm_layer=norm_layer, act_layer=act_layer)
        decoder = Decoder(ch=ch, out_ch=out_channels, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                          z_channels=z_channels, tanh_out=apply_tanh, norm_layer=norm_layer, act_layer=act_layer)
        super().__init__(encoder, decoder, z_channels)

    def reset_embedders(self):
        self.encoder.enc[-1].reset_parameters()  # Reset last conv of encoder to kaiming_uniform


def conv7x7(in_planes, out_planes, stride=1, bias=BIAS, use_padding=True, groups=1):
    "3x3 convolution with padding"
    conv = nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=7,
                     stride=stride,
                     padding=3 if use_padding else 0,
                     bias=bias,
                     padding_mode=PADDING_MODE,
                     groups=groups)
    return conv


class Encoder(nn.Module):
    def __init__(self, *, in_channels, ch, ch_mult=(1, 2, 4, 4), num_res_blocks=2,
                 z_channels=4, double_z=True,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
        super().__init__()

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.double_z = double_z
        self.z_channels = z_channels
        # downsampling
        modules = [conv7x7(in_channels, self.ch),
                   BNReLUConv2d(self.ch, self.ch, kernel_size=3, padding=1,
                                norm_layer=norm_layer, act_layer=act_layer)]

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult

        for i_level in range(self.num_resolutions):
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                modules.append(ResidualBlockV2(in_channels=block_in,
                                               out_channels=block_out,
                                               norm_layer=norm_layer,
                                               act_layer=act_layer))
                block_in = block_out

            if i_level != self.num_resolutions - 1:
                modules.append(Downsample(block_in))

        modules.append(ResidualBlock(in_channels=block_in,
                                     out_channels=block_in,
                                     norm_layer=norm_layer,
                                     act_layer=act_layer))

        modules.append(
            Conv2dBNReLU(block_in, block_in, kernel_size=3, padding=1, norm_layer=norm_layer, act_layer=act_layer))
        modules.append(conv3x3(block_in, 2 * self.z_channels if self.double_z else self.z_channels, bias=True))

        self.enc = nn.Sequential(*modules)

    def forward(self, x):
        # downsampling
        out = self.enc(x)
        if self.double_z:
            return out[:, :self.z_channels], out[:, self.z_channels:]
        else:
            return out, None


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 4), num_res_blocks=2,
                 z_channels, tanh_out=False, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        modules = [conv3x3(z_channels, block_in, stride=1, bias=True),
                   BNReLUConv2d(block_in, block_in, kernel_size=3, padding=1, norm_layer=norm_layer,
                                act_layer=act_layer)]

        modules.append(ResidualBlockV2(in_channels=block_in,
                                       out_channels=block_in,
                                       norm_layer=norm_layer,
                                       act_layer=act_layer))

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                modules.append(ResidualBlockV2(in_channels=block_in,
                                               out_channels=block_out,
                                               norm_layer=norm_layer,
                                               act_layer=act_layer))
                block_in = block_out

            if i_level != 0:
                modules.append(Upsample(block_in))

        modules.append(ResidualBlock(in_channels=block_in,
                                     out_channels=block_in,
                                     norm_layer=norm_layer,
                                     act_layer=act_layer))

        modules.append(
            Conv2dBNReLU(block_in, block_in, kernel_size=3, padding=1, norm_layer=norm_layer, act_layer=act_layer))
        # end
        modules.append(conv7x7(block_in, out_ch, bias=True))
        self.dec = nn.Sequential(*modules)

    def forward(self, z):
        # z to block_in
        out = self.dec(z)
        if self.tanh_out:
            out = torch.tanh(out)
        return out


class ImprovedAEV2(StableAutoencoderV2):
    def __init__(self, in_channels, ch, ch_mult=(1, 2, 4, 4), num_res_blocks=2, z_channels=4, double_z=True,
                 norm_name='batch', activation='silu', apply_tanh=False, sobel_size=5):
        super().__init__(in_channels=in_channels + 3, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                         z_channels=z_channels, double_z=double_z, norm_name=norm_name, activation=activation,
                         apply_tanh=apply_tanh,
                         out_channels=in_channels)
        self.sobel_filter = SobelFilter(k_sobel=sobel_size)

    def forward_encoder(self, x) -> List[torch.Tensor]:
        sobel_feats = self.sobel_filter(x)
        additional_layer = torch.sqrt(sobel_feats[:, 0:1] ** 2 + sobel_feats[:, 1:2] ** 2 + 1e-3)
        new_x = torch.cat([x, sobel_feats, additional_layer], dim=1)
        return super().forward_encoder(new_x)
