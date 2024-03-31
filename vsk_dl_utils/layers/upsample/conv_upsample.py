import torch.nn.functional as F
from torch import nn

from vsk_dl_utils.layers import conv3x3


class Upsample(nn.Module):
    def __init__(self, in_channels, interpolate_mode="nearest"):
        super().__init__()
        self.interpolate_mode = interpolate_mode
        self.conv = conv3x3(in_channels, in_channels, bias=False)

    def forward(self, x, inference_context):
        # Firstly upsample, after activate
        x = F.interpolate(x, scale_factor=2.0, mode=self.interpolate_mode)
        x = self.conv(x, inference_context=inference_context)
        return x
