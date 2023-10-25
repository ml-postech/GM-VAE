import torch
from torch.distributions import Normal

from ..utils import euclidean_kl_div


class Distribution():
    def __init__(self, mean, logvar) -> None:
        self.mean = mean
        self.logvar = logvar

        self.base = Normal(self.mean, (self.logvar * 0.5).exp())

    def log_prob(self, z):
        return self.base.log_prob(z)

    def rsample(self, N):
        return self.base.rsample([N])

    def sample(self, N):
        with torch.no_grad():
            return self.rsample(N)

    def kl_div(self, target_dist):
        return euclidean_kl_div(self.mean, self.logvar, target_dist.mean, target_dist.logvar)

