import torch.nn.functional as F
from torch import nn

from vsk_dl_utils.layers import conv3x3


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels=None, interpolate_mode="nearest"):
        super().__init__()
        self.interpolate_mode = interpolate_mode
        out_channels = out_channels or in_channels
        self.conv = conv3x3(in_channels, out_channels, bias=False)

    def forward(self, x):
        # Firstly upsample, after activate
        x = F.interpolate(x, scale_factor=2.0, mode=self.interpolate_mode)
        x = self.conv(x)
        return x
