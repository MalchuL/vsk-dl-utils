from abc import ABC
from typing import List, Union

import torch
from torch import nn


class AbstractAutoEncoder(nn.Module, ABC):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, z_channels: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.z_channels = z_channels

    def forward_encoder(self, x) -> List[torch.Tensor]:
        return self.encoder(x)

    def forward_decoder(self, feats: Union[torch.Tensor, List[torch.Tensor]]):
        """

        Args:
            feats: can be tensor of mean or [mean, std] (where only mean will be taken)

        Returns:

        """
        if isinstance(feats, (tuple, list)):
            feats = feats[0]

        return self.decoder(feats)

    def get_encoded_dim(self) -> int:
        return self.z_channels

    def reset_embedders(self):
        """
        Method to reset embedding layer, which is the last layer of encoder.
        Returns:

        """
        pass

    def forward(self, x):
        feats = self.forward_encoder(x)
        out = self.forward_decoder(feats)
        return out
