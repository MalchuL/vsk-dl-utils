import math

import torch
import torch.nn as nn


def logit(value):
    return math.log(value) - math.log(1 - value)


class GANLoss(nn.Module):
    def __init__(self, criterion=nn.BCELoss(), is_logit=False, clip=None):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.use_clip = clip is not None and clip > 0
        if self.use_clip:
            assert clip < 0.5
            self.clip_min = clip
            self.clip_max = 1 - clip
            if is_logit:
                self.clip_min = logit(self.clip_min)
                self.clip_max = logit(self.clip_max)
        self.is_logit = is_logit
        self.base_loss = criterion

    def clip_tensor(self, pred):
        if self.use_clip:
            pred = torch.clip(pred, min=self.clip_min, max=self.clip_max)
        return pred

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def forward(self, pred, target_is_real, use_clip=False):
        if use_clip:
            pred = self.clip_tensor(pred)
        return self.base_loss(pred, self.get_target_tensor(pred, target_is_real))


