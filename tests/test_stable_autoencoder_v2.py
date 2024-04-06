import numpy as np
import torch

from vsk_dl_utils.models.autoencoder.stable_autoencoder import StableAutoencoder
from vsk_dl_utils.models.autoencoder.stable_autoencoder_v2 import StableAutoencoderV2, ImprovedAEV2


def test_stable_autoencoder():
    ae = StableAutoencoderV2(3, 4, norm_name='instance', activation='silu')
    x = torch.rand(10, 3, 32, 32)
    y = ae(x)
    assert y.shape == x.shape
    print(sum([np.prod(p.size()) for p in ae.parameters()]))
    print(ae)

def test_improved_stable_autoencoder():
    ae = ImprovedAEV2(3, 4,)
    x = torch.rand(10, 3, 32, 32)
    y = ae(x)
    assert y.shape == x.shape