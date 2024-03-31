import torch.nn as nn
import functools

from vsk_dl_utils.layers import GroupNorm
from vsk_dl_utils.layers.group_norm import GroupNorm8


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'group':
        norm_layer = GroupNorm
    elif norm_type == 'group8':
        norm_layer = GroupNorm8
    elif norm_type == 'none':
        norm_layer = nn.Identity
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
