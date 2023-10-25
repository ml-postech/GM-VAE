import torch
from math import log, sqrt, pi
from torch.distributions import Normal, Gamma

from ..utils import euclidean_kl_div, gamma_kl_div


class Distribution():
    def __init__(self, means, log_gamma_square, c=-1) -> None:
        self.c = torch.tensor([c], device=means.device)
        self.alpha = means[..., 0]
        self.log_beta_square = means[..., 1]
        self.log_gamma_square = log_gamma_square

        self.normal_mu = self.alpha
        self.normal_logvar = self.log_beta_square + self.log_gamma_square
        self.base1 = Normal(
            self.normal_mu,
            (0.5 * self.normal_logvar).exp()
        )

        self.gamma_a = (-self.log_gamma_square).exp() / (4 * -self.c) + 1
        self.log_gamma_b = -self.normal_logvar - log(4 * -self.c)
        self.base2 = Gamma(
            self.gamma_a,
            self.log_gamma_b.exp()
        )

    def log_prob(self, z):
        target_mu, target_logvar = z[..., 0], z[..., 1]

        kl = euclidean_kl_div(
            sqrt(2 * -self.c) * target_mu,
            target_logvar,
            sqrt(2 * -self.c) * self.alpha,
            self.log_beta_square
        )

        log_prob = -kl / (2 * -self.c * self.log_gamma_square.exp()) + 1.5 * (target_logvar - self.log_beta_square)
        gamma_factor = (-self.log_gamma_square).exp() / (4 * -self.c)
        log_prob = log_prob - 0.5 * self.log_gamma_square
        log_prob = log_prob - torch.lgamma(gamma_factor)
        log_prob = log_prob - gamma_factor
        log_prob = log_prob - gamma_factor * ((4 * -self.c).log() + self.log_gamma_square)
        return log_prob

    def rsample(self, N):
        sample_mean = self.base1.rsample([N])
        sample_shape = torch.Size([N]) + self.gamma_a.shape
        sample_logvar = torch._standard_gamma(self.gamma_a[None].expand(sample_shape)).log() - self.log_gamma_b[None]
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
            self.log_gamma_b,
            target_dist.gamma_a,
            target_dist.log_gamma_b
        )
        return kl1 + kl2

