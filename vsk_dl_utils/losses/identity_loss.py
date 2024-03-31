import torch.nn as nn

from .loss_wrapper import LossWrapper


class IdentityLoss(LossWrapper):
    def __init__(self):
        super().__init__(self, weight=0)

    def forward(self, *args, **kwargs):
        return 0.0
