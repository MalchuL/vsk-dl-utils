import torch
import torch.nn as nn

from vsk_dl_utils.layers.conv_layers import conv1x1, conv3x3
from vsk_dl_utils.layers.gan_layers.style_representation import StyleRepresentation

NEG_SLOPE = 0.2
LAST_BIAS = False


class StdPool(nn.Module):
    def forward(self, x):
        return torch.std(x, dim=(2, 3), keepdim=True)


class MinPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        return self.pool(-x)


class ChannelsPool(nn.Module):

    def __init__(self, in_channels, hidden_layers, out_layers, eps=1e-5, channels_as_groups=True, groups=1,
                 pool_layers=('avg', 'max', 'std')):
        super().__init__()
        self.eps = eps
        layers = {'avg': nn.AdaptiveAvgPool2d((1, 1)),
                  'max': nn.AdaptiveMaxPool2d((1, 1)),
                  'std': StdPool(),
                  'min': MinPool()}
        self.stats = nn.ModuleList(
            [layers[pool_layer] for pool_layer in pool_layers])
        # Groups arg is a parameter for several outputs
        groups = (len(pool_layers) if channels_as_groups else 1) * groups
        in_channels = in_channels * len(pool_layers)
        hidden_layers = hidden_layers * groups
        out_layers = out_layers * groups

        def get_act():
            return nn.LeakyReLU(negative_slope=NEG_SLOPE,
                                inplace=True)  # alpha because at -1 gradient lokks like leaky relu

        self.mlp = nn.Sequential(conv1x1(in_channels, hidden_layers, groups=groups, bias=False),
                                 nn.BatchNorm2d(hidden_layers, affine=True, track_running_stats=False),
                                 get_act(),
                                 conv1x1(hidden_layers, hidden_layers, groups=groups, bias=False),
                                 nn.BatchNorm2d(hidden_layers, affine=True, track_running_stats=False),
                                 get_act(),
                                 conv1x1(hidden_layers, out_layers, groups=groups, bias=LAST_BIAS),
                                 )

    def forward(self, x):
        x = [stat(x) for stat in self.stats]
        x = torch.cat(x, 1)
        x = self.mlp(x)
        return x


class MinibatchSTD(nn.Module):
    def calculate_std(self, x):
        return torch.std(x, dim=(1), keepdim=True)

    @staticmethod
    def get_num_channels():
        return 1

    def forward(self, x):
        return torch.cat([x, self.calculate_std(x)], dim=1)


# Don't change LeakyRelu
class CartoonGANDiscriminator(nn.Module):
    def __init__(self, use_sigmoid=False,
                 input_channels=3,
                 layers=(32, 64, 128, 128, 256, 256),
                 strides=(1, 2, 1, 2, 1, 1),
                 mlp=(1, 3, 5),
                 mean=(0.0,),
                 std=(1.0,),
                 groups=1,
                 multiplier=1,
                 eps=1e-5,
                 channels_as_groups=True,
                 use_padding=True,
                 apply_instance_norm=False):
        super(CartoonGANDiscriminator, self).__init__()
        self.eps = eps
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.groups = groups

        def r(x):
            return round(x * multiplier) * groups

        def get_act(inplace=True):
            return nn.LeakyReLU(negative_slope=NEG_SLOPE,
                                inplace=inplace)  # alpha because at -1 gradient lokks like leaky relu

        def get_norm(channels, affine=False):
            return nn.InstanceNorm2d(channels, affine=affine) if apply_instance_norm else nn.BatchNorm2d(channels,
                                                                                                         affine=affine,
                                                                                                         eps=self.eps,
                                                                                                         track_running_stats=False)

        def get_reverse_norm(channels, affine=True):
            return nn.Identity()

        layers = [r(layer) for layer in layers]
        in_channels = [input_channels] + layers[:-1]
        out_channels = layers
        strides = list(strides)
        modules = []
        self.mlp = nn.ModuleDict()
        if mlp is None:
            mlp = []
        for i in range(len(strides)):
            if i in mlp:
                self.mlp[str(len(modules))] = ChannelsPool(out_channels[i], out_channels[i], 1, eps=self.eps,
                                                           channels_as_groups=channels_as_groups, groups=groups)
            use_norm = i > 0 and strides[i] == 1
            conv_fn = conv3x3
            modules.extend(
                [conv_fn(in_channels[i], out_channels[i], stride=strides[i], bias=False,
                         use_padding=use_padding, groups=groups),
                 get_norm(out_channels[i]) if use_norm else get_reverse_norm(out_channels[i]),
                 get_act(inplace=use_norm)])

        # get ProGAN std
        if self.groups == 1:
            minibatch_std = MinibatchSTD()
            modules.append(minibatch_std)
            added_channels = minibatch_std.get_num_channels()
        else:
            added_channels = 0
        modules.append(conv3x3(out_channels[-1] + added_channels,
                               out_planes=groups,
                               groups=groups,
                               stride=1,
                               bias=LAST_BIAS,
                               use_padding=use_padding))  # You must have bias as true to good converge
        self.use_sigmoid = use_sigmoid
        self.conv = nn.ModuleList(modules)

    def forward(self, x, apply_sigmoid=True):
        x = (x - self.mean) / self.std
        stats = []
        for i, module in enumerate(self.conv):
            x = module(x)
            if str(i) in self.mlp:
                stats.append(self.mlp[str(i)](x))

        if len(stats) > 0:
            stats = torch.cat(stats, dim=1)
        if self.use_sigmoid and apply_sigmoid:
            x = torch.sigmoid(x)
            if len(stats) > 0:
                stats = torch.sigmoid(stats)

        if len(stats) == 0:
            stats = None
        return x, stats


class StyleCartoonGANDiscriminator(nn.Module):
    def __init__(self, style_layers=("color_shift_with_sobel", "surface", "identity"), weight_mode='uniform', r=5,
                 eps=2e-1, layers=(32, 64, 128, 128, 256, 256), strides=(1, 2, 1, 2, 1, 1), mlp=(1, 3, 5), mean=(0.0,),
                 std=(1.0,), multiplier=1, channels_as_groups=True, use_padding=True, use_sigmoid=False,
                 apply_instance_norm=False):
        super().__init__()
        self.style_representation = StyleRepresentation(layers=style_layers, weight_mode=weight_mode, r=r, eps=eps)
        self.discriminator = CartoonGANDiscriminator(input_channels=self.style_representation.get_num_channels(),
                                                     layers=layers, mlp=mlp,
                                                     strides=strides,
                                                     groups=len(style_layers),
                                                     multiplier=multiplier,
                                                     channels_as_groups=channels_as_groups, use_sigmoid=use_sigmoid,
                                                     use_padding=use_padding,
                                                     apply_instance_norm=apply_instance_norm)
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x, apply_sigmoid=True):
        # Denormalize
        x = x * self.std + self.mean
        features = self.style_representation(x)
        score = self.discriminator(features, apply_sigmoid=apply_sigmoid)
        return score


if __name__ == '__main__':
    x = torch.rand(5, 3, 128, 128)
    discr = StyleCartoonGANDiscriminator()
    print(discr(x))
