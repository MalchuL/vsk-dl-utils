from torch import nn

from vsk_dl_utils.layers import conv3x3


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves

        self.pool = nn.AvgPool2d(2, stride=2)
        self.conv = conv3x3(in_channels, in_channels, bias=False)

    def forward(self, x, inference_context):
        x = self.pool(x)
        x = x * 0.5  # Because after avgpool dispersion will increases by sqrt(4)
        x = self.conv(x, inference_context=inference_context)
        return x
