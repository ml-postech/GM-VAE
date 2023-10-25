import torch
from torch import nn
from torch.nn import functional as F


class VanillaEncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()

        self.latent_dim = int(args.latent_dim * 2)
        self.feature_dim = feature_dim

        self.variational = nn.Linear(
            self.feature_dim,
            2 * self.latent_dim
        )

    def forward(self, feature):
        feature = self.variational(feature)
        mean, logvar = torch.split(
            feature,
            [self.latent_dim, self.latent_dim],
            dim=-1
        )

        return mean, logvar


class VanillaDecoderLayer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

    def forward(self, z):
        return z

