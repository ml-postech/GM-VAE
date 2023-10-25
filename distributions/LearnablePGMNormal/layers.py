import torch
import geoopt
from torch import nn
# from dgnn.models import DiagonalGaussianLinearLayer
from torch.nn import functional as F
from dgnn.utils import halfplane2disk, disk2lorentz, lorentz2disk, disk2halfplane


class EncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()

        self.latent_dim = args.latent_dim
        self.feature_dim = feature_dim

        self.variational = nn.Linear(
            self.feature_dim,
            4 * self.latent_dim
        )

    def forward(self, feature):
        feature = self.variational(feature)
        alpha, beta, logc, gamma = torch.split(
            feature,
            [
                self.latent_dim, 
                self.latent_dim, 
                self.latent_dim,
                self.latent_dim
            ],
            dim=-1
        )

        return torch.stack([alpha, beta, logc], dim=-1), gamma


class VanillaEncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()
        
        self.encoder = EncoderLayer(args, feature_dim)

    def forward(self, feature):
        return self.encoder(feature)


class ExpEncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()
        
        self.c = torch.tensor([args.c], device=args.device)

        self.manifold = geoopt.manifolds.PoincareBall(-args.c)
        self.encoder = EncoderLayer(args, feature_dim)

    def forward(self, feature):
        mean, gamma = self.encoder(feature)
        mean = disk2halfplane(
            self.manifold.expmap0(mean),
            self.c
        )
        mean = torch.stack([mean[..., 0], mean[..., 1].log() * 2], dim=-1)

        return mean, gamma


class VanillaDecoderLayer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

    def forward(self, z):
        a, b = z[..., 0], (z[..., 1] * 0.5).exp()
        z = torch.concat([a, b], dim=-1)
        return z


class LogDecoderLayer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.c = torch.tensor([args.c], device=args.device)

        self.manifold = geoopt.manifolds.PoincareBall(-args.c)

    def forward(self, z):
        a, b = z[..., 0], (z[..., 1] * 0.5).exp()
        z = torch.stack([a, b], dim=-1)
        z = self.manifold.logmap0(
            halfplane2disk(z, self.c)
        )

        return z.reshape(*z.shape[:-2], -1)

