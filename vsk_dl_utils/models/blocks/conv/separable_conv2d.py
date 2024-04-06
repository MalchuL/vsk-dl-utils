from torch import nn

from vsk_dl_utils.layers.conv_layers import BIAS, PADDING_MODE


class SeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = BIAS,
                 padding_mode: str = PADDING_MODE,
                 sep_mode='default'):

        super().__init__()
        assert sep_mode in ['inception', 'default']
        self.sep_mode = sep_mode

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else self.kernel_size // 2
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

        self.module = self.construct_module()

    def construct_module(self):
        sep_mode = self.sep_mode
        if self.kernel_size == 1:
            pointwise_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=1, stride=self.stride,
                                       dilation=1, groups=self.groups, bias=self.bias)
            module = nn.Sequential(pointwise_conv)
        elif sep_mode == 'default':
            depthwise_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                                       kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                       dilation=self.dilation, groups=self.in_channels, bias=False,
                                       padding_mode=self.padding_mode)
            pointwise_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=1, stride=1,
                                       dilation=1, groups=self.groups, bias=self.bias)
            module = nn.Sequential(depthwise_conv, pointwise_conv)
        elif sep_mode == 'inception':
            pointwise_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=1, stride=1,
                                       dilation=1, groups=self.groups, bias=False)
            depthwise_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels,
                                       kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                       dilation=self.dilation, groups=self.out_channels, bias=self.bias,
                                       padding_mode=self.padding_mode)
            module = nn.Sequential(pointwise_conv, depthwise_conv)
        else:
            raise ValueError("sep_mode not in ['inception', 'default']")

        return module

    def forward(self, x):
        return self.module(x)
