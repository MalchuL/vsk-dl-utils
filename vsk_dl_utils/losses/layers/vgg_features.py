import gc
import math
from functools import partial

import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg19
from torchvision.models import vgg19_bn as vgg19_bn

from vsk_dl_utils.utils.clipping_utils import z_clip


class Scale(nn.Module):
    def __init__(self, module, scale):
        super().__init__()
        self.module = module
        self.register_buffer("scale", torch.tensor(scale))

    def extra_repr(self):
        return f"(scale): {self.scale.item():g}"

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs) * self.scale


class VGGFeatures(nn.Module):
    poolings = {"max": nn.MaxPool2d, "average": nn.AvgPool2d, "l2": partial(nn.LPPool2d, 2)}
    # pooling_scales = {'max': 1., 'average': 2., 'l2': 0.78}
    pooling_scales = {"max": 1.0, "average": 1, "l2": 0.78}
    layers_mapping = {"vgg16": 21, "vgg19": 25, "vgg19_bn": 36}

    def __init__(
        self,
        network="vgg19_bn",
        layers=None,
        mean=None,
        std=None,
        fix_pad=False,
        pooling="max",
        z_clipping=None,
    ):
        """VGG features extractor :param network: VGG network name :param layers: Layers numbers to
        extract. :param mean: Mean list to normalize image. By default as in ImageNet :param std:
        Std list to normalize image. By default as in ImageNet :param fix_pad: Makes confolutions
        with Reflect padding.

        :param pooling: set pooling layers
        :param z_clipping: apply clipping. Better to use value between [1.5 - 3], will fix some normalizations
        """
        super().__init__()
        self.z_clipping = z_clipping
        self.pooling = pooling

        assert network in ["vgg16", "vgg19", "vgg19_bn"]
        if layers is None or len(layers) == 0:
            layers = [self.layers_mapping[network]]
        layers = sorted(set(layers))
        self.layers = layers

        if mean or std:
            assert mean and std, (mean, std)
        else:
            # Imagenet statistics
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

        network_model = globals()[network]
        perception = list(network_model(pretrained=True).features)[: layers[-1] + 1]
        pool_scale = self.pooling_scales[pooling]
        for i, layer in enumerate(perception):
            if pooling != "max" and isinstance(layer, nn.MaxPool2d):
                # Changing the pooling type from max results in the scale of activations
                # changing, so rescale them. Gatys et al. (2015) do not do this.
                perception[i] = Scale(self.poolings[pooling](2), pool_scale)
        self.perception = nn.Sequential(*perception).eval()

        if fix_pad:
            self.fix_padding(self.perception)

        for param in self.perception.parameters():
            param.requires_grad = False

        self.perception.requires_grad_(False)

    def fix_padding(self, model: nn.Module, padding="reflect"):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                module.padding_mode = padding

    def forward(self, x):
        """Calculates VGG features :param x: 4D images tensor.

        Inputs must be normalized in [0..1] range
        :return: list of VGG features defined in layers
        """
        self.perception.eval()
        feats = {"x": x}
        x = (x - self.mean) / self.std

        for i, module in enumerate(self.perception):
            x = module(x)
            if self.z_clipping is not None and isinstance(module, nn.Conv2d):
                x = z_clip(x, self.z_clipping)
            if i in self.layers:
                feats[i] = x
        feats["output"] = x
        return feats

    def extra_repr(self) -> str:
        return (
            f"Norm: [mean: {self.mean.view(3)}, std: {self.std.view(3)}]\n"
            + f"Layers {self.layers} {[(i, self.perception[i]) for i in self.layers]}\n"
            + f"Clip {self.z_clipping}"
        )
