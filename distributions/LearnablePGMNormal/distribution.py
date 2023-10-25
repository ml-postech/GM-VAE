import torch
from math import log, sqrt, pi
from torch.distributions import Normal, Gamma

from ..utils import euclidean_kl_div, gamma_kl_div


@torch.jit.script
def _log_prob(kl, log_gamma_square, log_beta_square, c):
    log_prob = -kl / log_gamma_square.exp() / (2 * -c)
    log_prob = log_prob - 1.5 * log_beta_square
    log_prob = log_prob - 0.5 * log_gamma_square

    gamma_factor = (-log_gamma_square).exp() / (4 * -c)
    log_prob = log_prob - gamma_factor
    log_prob = log_prob - torch.lgamma(gamma_factor)
    log_prob = log_prob - gamma_factor * ((4 * -c).log() + log_gamma_square)
    # log_prob = log_prob - 0.5 * log(2 * pi)

    return log_prob


class Distribution():
    def __init__(self, means, log_gamma_square) -> None:
        # self.c = torch.tensor([c], device=means.device)
        self.alpha = means[..., 0]
        self.log_beta_square = means[..., 1]
        self.c = means[..., 2].exp()
        self.log_gamma_square = log_gamma_square

        self.normal_mu = self.alpha
        self.normal_logvar = self.log_beta_square + self.log_gamma_square
        self.base1 = Normal(
            self.normal_mu,
            (0.5 * self.normal_logvar).exp()
        )

        self.gamma_a = (-self.log_gamma_square).exp() / (4 * -self.c) + 1
        self.gamma_b = (-self.normal_logvar).exp() / (4 * -self.c)
        self.base2 = Gamma(
            self.gamma_a,
            self.gamma_b
        )

    def log_prob(self, z):
        target_mean, target_logvar = sqrt(-2 * self.c) * z[..., 0], z[..., 1]
        kl = euclidean_kl_div(
            target_mean, 
            target_logvar, 
            self.alpha * (-2 * self.c).sqrt(), 
            self.log_beta_square
        )
        return _log_prob(kl, self.log_gamma_square, self.log_beta_square, self.c)

    def rsample(self, N):
        sample_mean = self.base1.rsample([N])
        sample_logvar = self.base2.rsample([N]).log()
        return torch.stack([sample_mean, sample_logvar], dim=-1)

    def sample(self, N):
        with torch.no_grad():
            return self.rsample(N)

    def kl_div(self, target_dist):
        kl1 = euclidean_kl_div(
            self.normal_mu,
            self.normal_logvar,
            target_dist.normal_mu,
            target_dist.normal_logvar
        )
        kl2 = gamma_kl_div(
            self.gamma_a,
            self.gamma_b,
            target_dist.gamma_a,
            target_dist.gamma_b
        )
        return kl1 + kl2

