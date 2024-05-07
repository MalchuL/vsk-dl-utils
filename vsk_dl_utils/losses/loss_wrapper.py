import torch
import torch.nn as nn

from vsk_dl_utils.utils.interpolation.interpolator import Interpolator


class LossWrapper(nn.Module):
    def __init__(
            self, loss, weight=1, warmup_num_steps=None, warmup_method="easeInExpo", inverse=False
    ):
        """Loss wrapper to make it more stable and generic for training :param loss: nn.Module loss
        :param weight: Weight to multiply value.

        Useful to avoid params storing in class :param warmup_num_steps: Warmup steps numbers
        :param warmup_method: Pytweening warmup method name. Look at https://pypi.org/project/pytweening/
        :param warmup_num_steps: Warmup steps numbers :param warmup_method: Pytweening warmup
        method name. Look at
        :param inverse: If true starts from 1 else from 0. False by default
        """
        super().__init__()
        if weight <= 0:
            self.loss = lambda *args, **kwargs: 0
        else:
            self.loss = loss
        self.weight = weight

        self.interpolator = None
        self.register_buffer("num_steps", torch.zeros([], dtype=torch.long))
        if warmup_num_steps is not None and warmup_num_steps > 0:
            self.interpolator = Interpolator(
                method=warmup_method, num_steps=warmup_num_steps, inverse=inverse
            )

    def forward(self, *args, **kwargs):
        warmup_weight = 1
        if self.training and self.interpolator is not None:
            num_steps = self.num_steps.item()
            warmup_weight = self.interpolator(num_steps)
            self.num_steps += 1
        if self.weight > 0:
            loss = self.loss(*args, **kwargs)
            if isinstance(loss, tuple):
                loss, *out_values = loss
                return loss * (self.weight * warmup_weight), *out_values
            else:
                return loss * (self.weight * warmup_weight)
        else:
            return 0

    def denorm_loss(self, loss_value):
        if self.weight > 0:
            return loss_value / self.weight
        else:
            return 0

    def extra_repr(self) -> str:
        if self.interpolator is not None:
            return "weight={}, warmup_num_steps={}, warmup_method={}".format(
                self.weight, self.interpolator.num_steps, self.interpolator.method
            )
        else:
            return "weight={}".format(self.weight)
