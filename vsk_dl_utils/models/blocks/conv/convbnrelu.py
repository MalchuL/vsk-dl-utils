from torch import nn

from vsk_dl_utils.layers.conv_layers import BIAS, PADDING_MODE
from vsk_dl_utils.utils.act_layers import get_act
from vsk_dl_utils.utils.norm_layers import get_norm_layer


class Conv2dBNReLU(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = BIAS,
                 padding_mode: str = PADDING_MODE,
                 # Parameters for norm layers
                 norm_layer: str = 'batch',
                 act_layer: str = 'relu',
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.groups = groups

        self.norm_layer = norm_layer
        self.act_layer = act_layer

        self.module = self.construct_module()

    def construct_module(self):
        return nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                       dilation=self.dilation, groups=self.groups, bias=self.bias,
                                       padding_mode=self.padding_mode),
                             get_norm_layer(self.norm_layer)(self.out_channels),
                             get_act(self.act_layer)(inplace=True),
                             )

    def forward(self, x):
        return self.module(x)


class BNReLUConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = BIAS,
                 padding_mode: str = PADDING_MODE,
                 # Parameters for norm layers
                 norm_layer: str = 'batch',
                 act_layer: str = 'relu',
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.groups = groups

        self.norm_layer = norm_layer
        self.act_layer = act_layer

        self.module = self.construct_module()

    def construct_module(self):
        return nn.Sequential(get_norm_layer(self.norm_layer)(self.in_channels),
                             get_act(self.act_layer)(inplace=True),
                             nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                       dilation=self.dilation, groups=self.groups, bias=self.bias,
                                       padding_mode=self.padding_mode)
                             )

    def forward(self, x):
        return self.module(x)
