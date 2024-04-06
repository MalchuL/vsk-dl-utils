import math
from typing import List

import torch
from torch import nn

from .abstract_autoencoder import AbstractAutoEncoder
from vsk_dl_utils.layers import conv3x3, conv1x1, SobelFilter, Downsample, Upsample
from vsk_dl_utils.utils.norm_layers import get_norm_layer

PADDING_MODE = 'reflect'
BIAS = False

INV_SQRT2 = 1 / math.sqrt(2)


def get_act(name):
    if name.lower() == 'silu':
        return lambda: nn.SiLU(inplace=True)
    elif name.lower() == 'relu6':
        return lambda: nn.ReLU6(inplace=True)
    elif name.lower() == 'relu':
        return lambda: nn.ReLU(inplace=True)
    elif name.lower() == 'leaky_relu':
        return lambda: nn.LeakyReLU(negative_slope=0.05, inplace=True)
    else:
        raise ValueError(f'{name} is not supported')


class StableAutoencoder(AbstractAutoEncoder):
    def __init__(self, in_channels, ch, ch_mult=(1, 2, 4, 4), num_res_blocks=2, z_channels=4, double_z=True,
                 norm_name='batch', apply_tanh=False, out_channels=None, activation='silu'):
        act_layer = get_act(activation)
        norm_layer = get_norm_layer(norm_name)

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


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.conv1 = conv3x3(in_channels, out_channels)
        self.norm1 = norm_layer(out_channels)

        self.activation = act_layer()
        self.norm2 = norm_layer(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, bias=True)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv3x3(in_channels, out_channels, bias=True)
            else:
                self.nin_shortcut = conv1x1(in_channels, out_channels, bias=True)

        self.input_scale = nn.Parameter(torch.ones([]), requires_grad=True)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.norm1(h)
        h = self.activation(h)

        h = self.conv2(h)
        h = self.norm2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        # Residual connection
        h = (h + x * self.input_scale) * INV_SQRT2
        out = self.activation(h)
        return out


class Encoder(nn.Module):
    def __init__(self, *, in_channels, ch, ch_mult=(1, 2, 4, 4), num_res_blocks=2,
                 z_channels=4, double_z=True,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
        super().__init__()

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.activation = act_layer()
        self.double_z = double_z
        self.z_channels = z_channels
        # downsampling
        modules = [conv7x7(in_channels, self.ch), norm_layer(self.ch), act_layer()]

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult

        for i_level in range(self.num_resolutions):
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                modules.append(ResnetBlock(in_channels=block_in,
                                           out_channels=block_out,
                                           norm_layer=norm_layer,
                                           act_layer=act_layer))
                block_in = block_out

            if i_level != self.num_resolutions - 1:
                modules.append(Downsample(block_in))

        modules.append(ResnetBlock(in_channels=block_in,
                                   out_channels=block_in,
                                   norm_layer=norm_layer,
                                   act_layer=act_layer))
        # middle
        modules.append(ResnetBlock(in_channels=block_in,
                                   out_channels=block_in,
                                   norm_layer=norm_layer,
                                   act_layer=act_layer))

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

        self.activation = act_layer()

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        modules = [conv3x3(z_channels, block_in),
                   norm_layer(block_in),
                   act_layer()]

        modules.append(ResnetBlock(in_channels=block_in,
                                   out_channels=block_in,
                                   norm_layer=norm_layer,
                                   act_layer=act_layer))
        modules.append(ResnetBlock(in_channels=block_in,
                                   out_channels=block_in,
                                   norm_layer=norm_layer,
                                   act_layer=act_layer))

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                modules.append(ResnetBlock(in_channels=block_in,
                                           out_channels=block_out,
                                           norm_layer=norm_layer,
                                           act_layer=act_layer))
                block_in = block_out

            if i_level != 0:
                modules.append(Upsample(block_in))

        # end
        modules.append(conv7x7(block_in, out_ch, bias=True))
        self.dec = nn.Sequential(*modules)

    def forward(self, z):
        # z to block_in
        out = self.dec(z)
        if self.tanh_out:
            out = torch.tanh(out)
        return out


class ImprovedAE(StableAutoencoder):
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
