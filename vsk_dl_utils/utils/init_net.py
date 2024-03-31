import math

import torch
import torch.nn as nn
from torch.nn import init


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'uniform':
                # You can convert to it from normal distribution by gain = normal_gain * sqrt(12)
                init.uniform_(m.weight.data, a=-gain, b=gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                # gain converts to = math.sqrt(2.0 / (1 + gain ** 2))
                init.kaiming_normal_(m.weight.data, a=gain, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'zeros':
                init.zeros_(m.weight.data)
            elif init_type == 'orthogonal':
                # https://hjweide.github.io/orthogonal-initialization-in-convolutional-layerss
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (classname.find('BatchNorm2d') != -1 or classname.find('GroupNorm') != -1) and hasattr(m,
                                                                                                    'weight') and m.weight is not None:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='kaiming', init_gain=math.sqrt(5)):
    if init_type is None:
        return net
    init_weights(net, init_type, gain=init_gain)
    return net


def init_gan_net(net):
    return init_net(net, init_type='normal', init_gain=0.02)
