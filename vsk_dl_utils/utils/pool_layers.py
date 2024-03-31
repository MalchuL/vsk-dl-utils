import torch.nn as nn


def get_pool_layer(pool_type):
    assert pool_type in ['max', 'avg']
    if pool_type == 'max':
        pool_layer = nn.MaxPool2d
    elif pool_type == 'avg':
        pool_layer = nn.AvgPool2d
    else:
        raise NotImplementedError('pool layer [%s] is not found' % pool_type)
    return pool_layer
