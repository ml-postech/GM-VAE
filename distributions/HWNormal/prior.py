import torch
import geoopt
from torch.nn import functional as F

from .distribution import Distribution


def get_prior(args):
    m = geoopt.manifolds.Lorentz()

    mean = torch.zeros(
        [1, args.latent_dim, 2], 
        device=args.device
    )
    mean = m.expmap0(F.pad(mean, (1, 0)))

    sigma = torch.ones(
        [1, args.latent_dim, 2], 
        device=args.device
    )

    prior = Distribution(mean, sigma)
    return prior

