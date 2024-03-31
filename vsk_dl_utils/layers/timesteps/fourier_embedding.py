import math

import numpy as np
import torch
import torch.nn as nn

SQRT_2 = math.sqrt(2)


class FourierEmbedding(nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        freqs = torch.randn(num_channels // 2) * scale
        multiplier = torch.tensor(2 * np.pi).to(freqs.dtype)

        self.register_buffer('freqs', freqs)
        self.register_buffer('multiplier', multiplier)

    def forward(self, x):
        """
        Calculates fourier embeddings
        :param x: 1D tensor with batch_size
        :return: Nxnum_channels tensor
        """
        x = x.outer(self.freqs * self.multiplier)
        x = torch.cat([x.cos(), x.sin()], dim=1) * SQRT_2
        return x

