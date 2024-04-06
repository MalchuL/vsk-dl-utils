import numpy as np
import torch

from vsk_dl_utils.models.autoencoder.stable_autoencoder import StableAutoencoder


def test_stable_autoencoder():
    ae = StableAutoencoder(3, 4,)
    x = torch.rand(10, 3, 32, 32)
    y = ae(x)
    assert y.shape == x.shape
    print(sum([np.prod(p.size()) for p in ae.parameters()]))

