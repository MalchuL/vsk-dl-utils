import torch

from vsk_dl_utils.layers.conv.convbnrelu import Conv2dBNReLU, BNReLUConv2d
from vsk_dl_utils.layers.conv.resnet.res_block import ResidualBlock
from vsk_dl_utils.layers.conv.resnet.res_block_v2 import ResidualBlockV2
from vsk_dl_utils.layers.conv.separable_conv2d import SeparableConv2d


@torch.no_grad()
def test_separable_conv():
    x = torch.randn(10, 3, 25, 25)
    # Test default config without padding
    obj = SeparableConv2d(3, 2, kernel_size=3, stride=1, padding=0)
    assert obj(x).shape == (10, 2, 23, 23)
    # Test default config with padding
    obj = SeparableConv2d(3, 2, kernel_size=3, stride=1, padding=1)
    assert obj(x).shape == (10, 2, 25, 25)
    # Test default config with padding and stride
    obj = SeparableConv2d(3, 2, stride=2, padding=1)
    assert obj(x).shape == (10, 2, 13, 13)
    print(obj(x).shape)


@torch.no_grad()
def test_conv_bn_relu():
    x = torch.randn(10, 3, 25, 25)
    # Test default config without padding
    obj = Conv2dBNReLU(3, 2, kernel_size=3, stride=1, padding=0)
    assert obj(x).shape == (10, 2, 23, 23)
    # Test default config with padding
    obj = Conv2dBNReLU(3, 2, kernel_size=3, stride=1, padding=1)
    assert obj(x).shape == (10, 2, 25, 25)
    # Test default config with padding and stride
    obj = Conv2dBNReLU(3, 2, stride=2, padding=1, norm_layer='instance')
    assert obj(x).shape == (10, 2, 13, 13)
    print(obj)


@torch.no_grad()
def test_bn_relu_conv():
    x = torch.randn(10, 3, 25, 25)
    # Test default config without padding
    obj = BNReLUConv2d(3, 2, kernel_size=3, stride=1, padding=0)
    assert obj(x).shape == (10, 2, 23, 23)
    # Test default config with padding
    obj = BNReLUConv2d(3, 2, kernel_size=3, stride=1, padding=1)
    assert obj(x).shape == (10, 2, 25, 25)
    # Test default config with padding and stride
    obj = BNReLUConv2d(3, 2, stride=2, padding=1, norm_layer='instance')
    assert obj(x).shape == (10, 2, 13, 13)
    print(obj)


@torch.no_grad()
def test_residual_block():
    obj = ResidualBlock(3, 2)
    x = torch.randn(10, 3, 25, 25)
    assert obj(x).shape == (10, 2, 25, 25)
    obj = ResidualBlock(3, 2, stride=2)
    assert obj(x).shape == (10, 2, 13, 13)
    obj = ResidualBlock(3, 16, stride=2, kernel_size=5, dilation=1, act_layer='leaky_relu', norm_layer='group', last_act=True)
    assert obj(x).shape == (10, 16, 13, 13)
    print(obj)


@torch.no_grad()
def test_residual_v2_block():
    obj = ResidualBlockV2(3, 2)
    x = torch.randn(10, 3, 25, 25)
    assert obj(x).shape == (10, 2, 25, 25)
    obj = ResidualBlockV2(3, 2, stride=2)
    assert obj(x).shape == (10, 2, 13, 13)
    x = torch.randn(10, 12, 25, 25)
    obj = ResidualBlockV2(12, 16, stride=2, kernel_size=5, dilation=1, act_layer='leaky_relu', norm_layer='group')
    assert obj(x).shape == (10, 16, 13, 13)
    print(obj)
