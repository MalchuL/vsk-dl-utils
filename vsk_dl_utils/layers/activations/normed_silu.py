import torch
import torch.nn as nn
from torch import Tensor


class NormedSiLU(nn.SiLU):
    DIVIDER = 1 / 0.596

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input) * self.DIVIDER
