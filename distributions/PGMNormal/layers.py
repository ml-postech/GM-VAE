import torch
import geoopt
from torch import nn
from torch.nn import functional as F


class EncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()

        self.latent_dim = args.latent_dim
        self.feature_dim = feature_dim

        self.variational = nn.Linear(
            self.feature_dim,
            3 * self.latent_dim
        )

    def forward(self, feature):
        feature = self.variational(feature)
        alpha, beta, gamma = torch.split(
            feature,
            [self.latent_dim, self.latent_dim, self.latent_dim],
            dim=-1
        )

        return torch.stack([alpha, beta], dim=-1), gamma


class VanillaEncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()
        
        self.encoder = EncoderLayer(args, feature_dim)

    def forward(self, feature):
        return self.encoder(feature)


class GeoEncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()
        
        self.c = torch.tensor([args.c], device=args.device)
        self.encoder = EncoderLayer(args, feature_dim)
        self.manifold = geoopt.manifolds.Lorentz(-1 / args.c)

    def forward(self, feature):
        mean, gamma = self.encoder(feature)
        mean = self.manifold.expmap0(F.pad(mean, (1, 0)))
        mean = lorentz2halfplane(mean, self.c, log=torch.Tensor([True]))
        mean = torch.stack([
            mean[..., 0],
            mean[..., 1] * 2
        ], dim=-1)

        return mean, gamma


class VanillaDecoderLayer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

    def forward(self, z):
        z = z.reshape(*z.shape[:-2], -1)
        return z


class GeoDecoderLayer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.c = torch.tensor([args.c], device=args.device)
        self.manifold = geoopt.manifolds.Lorentz(-1 / args.c)

    def forward(self, z):
        a, b = z[..., 0], (z[..., 1] * 0.5).exp()
        z = torch.stack([a, b], dim=-1)
        z = halfplane2lorentz(z, self.c)
        z = self.manifold.logmap0(z)[..., 1:]
        return z.reshape(*z.shape[:-2], -1)

