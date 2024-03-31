from typing import Mapping, Optional

import torch
import torch.nn as nn

from .loss_wrapper import LossWrapper


class MergingLossWrapper(nn.Module):
    def __init__(
        self, losses: Mapping[str, nn.Module], weights: Optional[Mapping[str, float]] = None
    ):
        super().__init__()
        if weights is None:
            weights = {k: 1 for k in losses.keys()}

        self.losses = nn.ModuleList(
            [LossWrapper(loss, weight=weights[k]) for k, loss in losses.items()]
        )

    def forward(self, *args, **kwargs):
        loss = 0
        for module in self.losses:
            loss += module(*args, **kwargs)
        return loss
