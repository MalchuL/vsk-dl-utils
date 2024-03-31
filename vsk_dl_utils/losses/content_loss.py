import torch
import torch.nn as nn

from vsk_dl_utils.losses.charbonnier_loss import CharbonnierLoss
from vsk_dl_utils.losses.layers.vgg_features import VGGFeatures

VGG16_LAYERS = [2, 7, 14, 21]
VGG19_LAYERS = [2, 7, 16, 25]
VGG19_BN_LAYERS = [3, 10, 23, 36]


def PerceptualLossSimple(
    model_name="vgg19_bn",
    loss_type="charbonnier",
    use_last_layers=None,
    z_clip=3,
    fix_pad=True,
    apply_norm=True,
):
    name2layers = {"vgg16": VGG16_LAYERS, "vgg19": VGG19_LAYERS, "vgg19_bn": VGG19_BN_LAYERS}
    layers = name2layers[model_name]
    if use_last_layers:
        assert len(layers) >= use_last_layers > 0
        layers = layers[-use_last_layers:]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return PerceptualLoss(
        model_name=model_name,
        layers=layers,
        apply_norm=apply_norm,
        fix_pad=fix_pad,
        mean=mean,
        std=std,
        loss_type=loss_type,
        z_clip=z_clip,
    )


class PerceptualLoss(nn.Module):
    # Layers from https://towardsdatascience.com/implementing-neural-style-transfer-using-pytorch-fd8d43fb7bfa
    def __init__(
        self,
        model_name="vgg19",
        layers=(),
        weight_scaler=2,
        apply_norm=False,
        fix_pad=False,
        mean=None,
        std=None,
        reverse_weights=False,
        loss_type="smooth_l1",
        z_clip=None,
    ):
        super().__init__()
        assert (
            layers is not None and len(layers) > 0
        ), "Please add layers to content loss. i.e. [25] for vgg19 or [36] for vgg19"
        self.mean = mean
        self.std = std
        self.apply_norm = apply_norm
        assert apply_norm in [True, False, "both"]
        assert loss_type in ["l1", "smooth_l1", "l2", "charbonnier"]
        self.base_loss = self.get_loss(loss_type)
        self.z_clip = z_clip
        self.perception = self.get_model(
            model_name=model_name, layers=layers, fix_pad=fix_pad, mean=self.mean, std=self.std
        )

        weights = list(reversed([1 / (weight_scaler**i) for i in range(len(layers))]))
        if reverse_weights:
            weights = list(reversed(weights))
        sum_weight = sum(weights)
        weights = [weight / sum_weight for weight in weights]
        self.layers = dict(zip(list(layers), weights))
        self.norm = self.get_norm(512)

    def get_model(self, model_name, layers=(), fix_pad=False, mean=None, std=None):
        return VGGFeatures(
            network=model_name,
            layers=layers,
            fix_pad=fix_pad,
            mean=mean,
            std=std,
            z_clipping=self.z_clip,
        )

    def get_loss(self, loss_type):
        loss_type = loss_type.lower()
        if loss_type == "l1":
            return nn.L1Loss()
        elif loss_type == "smooth_l1":
            beta = 0.2
            return nn.SmoothL1Loss(beta=beta)
        elif loss_type == "l2":
            return nn.MSELoss()
        elif loss_type == "charbonnier":
            return CharbonnierLoss()
        else:
            raise ValueError(f"Error with loss type {loss_type}")

    def get_norm(self, num_channels):
        return nn.InstanceNorm2d(
            num_channels, affine=False, track_running_stats=False
        )  # Please don't touch eps in norm, it affects image gray content

    def forward(self, pred, target):
        self.norm.eval()
        pred = self.perception(pred)
        with torch.no_grad():
            target = self.perception(target)
        loss = 0
        for layer, weight in self.layers.items():
            pred_i = pred[layer]
            target_i = target[layer]
            if self.apply_norm == "both":
                loss += self.base_loss(pred_i, target_i) * weight
            if self.apply_norm or self.apply_norm == "both":
                pred_i = self.norm(pred_i)
                target_i = self.norm(target_i)
            loss += self.base_loss(pred_i, target_i) * weight
        return loss

    def extra_repr(self) -> str:
        return f"Layers_weights: {self.layers}\n" + f"Apply norm: {self.apply_norm}\n"
