import torch
import torch.nn as nn


# Refer to https://arxiv.org/pdf/1806.05764.pdf
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self._eps_square = self.eps ** 2
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, x, y):
        loss = torch.sqrt(self.mse(x, y) + self._eps_square).mean()
        return loss
