import torch
import torch.nn as nn


# Trainable param will increase memory consumption by 5%
class WeightedSum(nn.Module):
    def __init__(self, init_blend=0.0, use_sqrt=False, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(init_blend), dtype=torch.float32))
        self.use_sqrt = use_sqrt
        self.eps = eps

    def forward(self, x, y):
        alpha = torch.sigmoid(self.weight)
        if self.use_sqrt:
            alpha = alpha * (1 - 2 * self.eps) + self.eps
            out = (x * torch.sqrt(alpha + self.eps) + y * torch.sqrt((1 - self.eps) - alpha))
        else:
            out = (x * alpha + y * (1 - alpha))
        return out

    def extra_repr(self):
        alpha = torch.sigmoid(self.weight)
        if self.use_sqrt:
            alpha = alpha * (1 - 2 * self.eps) + self.eps
            return 'blend_value={}'.format(torch.sqrt(alpha + self.eps).item())
        else:
            return 'blend_value={}'.format(alpha.item())

