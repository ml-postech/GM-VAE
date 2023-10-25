import torch

from .distribution import Distribution


def get_prior(args):
    mean = torch.zeros(
        [1, args.latent_dim, 2], 
        device=args.device
    )
    sigma = torch.ones(
        [1, args.latent_dim, 1], 
        device=args.device
    )

    prior = Distribution(mean, sigma)
    return prior

