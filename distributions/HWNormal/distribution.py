import torch
import geoopt
from torch.nn import functional as F
from torch.distributions import Normal


class Distribution():
    def __init__(self, mean, sigma) -> None:
        self.mean = mean  # (1, *, 3)
        self.sigma = sigma  # (*, 2)

        self.latent_dim = 2
        self.base = Normal(
            torch.zeros([*self.sigma.shape[:-1], 2], device=self.mean.device),
            self.sigma
        )
        self.manifold = geoopt.manifolds.Lorentz()
        self.origin = self.manifold.origin(
            self.mean.size(),
            device=self.mean.device
        )

        self.kl_div = None

    def log_prob(self, z):  # (N, *, 2)
        u = self.manifold.logmap(self.mean, z)  # (N, *, 3)
        v = self.manifold.transp(self.mean, self.origin, u)
        log_prob_v = self.base.log_prob(v[..., 1:]).sum(dim=-1)  # (N, *)

        r = self.manifold.norm(u)  # (N, *)
        log_det = (self.latent_dim - 1) * (torch.sinh(r).log() - r.log())  # (N, *)

        log_prob_z = log_prob_v - log_det  # (N, *)
        return log_prob_z

    def rsample(self, N):
        v = self.base.rsample([N])  # (N, *, 2)
        v = F.pad(v, (1, 0))  # (N, *, 3)

        u = self.manifold.transp0(self.mean, v)  # (N, *, 3)
        z = self.manifold.expmap(self.mean, u)

        return z

    def sample(self, N):
        with torch.no_grad():
            return self.rsample(N)

