from torch import nn, Tensor

from vsk_dl_utils.layers.activations.normed_silu import NormedSiLU


def get_act(name):
    if name is None or name.lower() == 'none':
        return lambda inplace=False, in_channels=None: nn.Identity()
    elif name.lower() in ['silu', 'swish']:
        return lambda inplace=False, in_channels=None: NormedSiLU(inplace=inplace)
    elif name.lower() == 'relu6':
        return lambda inplace=False, in_channels=None: nn.ReLU6(inplace=inplace)
    elif name.lower() == 'relu':
        return lambda inplace=False, in_channels=None: nn.ReLU(inplace=inplace)
    elif name.lower() == 'leaky_relu':
        return lambda inplace=False, in_channels=None: nn.LeakyReLU(negative_slope=0.05, inplace=inplace)
    else:
        raise ValueError(f'{name} is not supported')