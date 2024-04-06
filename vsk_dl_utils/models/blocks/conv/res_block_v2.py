from torch import nn

from vsk_dl_utils.layers import conv1x1
from vsk_dl_utils.models.blocks.conv.convbnrelu import BNReLUConv2d
from vsk_dl_utils.models.blocks.weighted_sum import WeightedSum


class ResidualBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3,
                 padding=None,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm_layer='batch',
                 act_layer='relu'):
        super(ResidualBlockV2, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = (kernel_size - 1) // 2
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 is not supported!')
        self.conv1 = BNReLUConv2d(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  bias=False,
                                  dilation=dilation,
                                  groups=groups,
                                  norm_layer=norm_layer,
                                  act_layer=act_layer)
        self.conv2 = BNReLUConv2d(out_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  bias=False,
                                  dilation=dilation,
                                  groups=groups,
                                  norm_layer=norm_layer,
                                  act_layer=None)
        # self.conv3 = BNReLUConv2(out_channels, out_channels, context=context, bias=True)

        self.downsample = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.downsample = conv1x1(in_planes=in_channels, out_planes=out_channels,
                                      stride=stride, bias=True)

        self.out_channels = out_channels
        self.alpha_blend = WeightedSum()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        # out = self.conv3(out, inference_context)
        residual = self.downsample(residual)
        out = self.alpha_blend(out, residual)
        return out
