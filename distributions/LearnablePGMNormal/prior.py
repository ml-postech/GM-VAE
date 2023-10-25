import torch

from .distribution import Distribution


def get_prior(args):
    mean = torch.zeros(
        [1, args.latent_dim, 2], 
        device=args.device
    )
    covar = torch.zeros(
        [1, args.latent_dim], 
        device=args.device
    )

    prior = Distribution(mean, covar)
    return prior

