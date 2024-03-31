import torch

from vsk_dl_utils.losses import PerceptualLossSimple


def test_perceptual_loss():
    loss = PerceptualLossSimple()
    x = torch.rand(10, 3, 256, 256)
    y = torch.rand(10, 3, 256, 256)
    print(loss)
    print(loss(x, y))
    loss = PerceptualLossSimple(model_name="vgg16", use_last_layers=1)
    print(loss)
    print(loss(x, y))
    loss = PerceptualLossSimple(model_name="vgg19_bn", apply_norm=False)
    print(loss)
    print(loss(x, y))
