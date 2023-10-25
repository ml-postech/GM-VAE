import torch
import geoopt

from .hyperbolic_radius import HyperbolicRadius
from .hyperbolic_uniform import HypersphericalUniform

MIN_NORM = 1e-15


def expmap_polar(c, x, u, r, dim: int = -1):
    m = geoopt.manifolds.PoincareBall(1.0)
    sqrt_c = c.sqrt()
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = (
        (sqrt_c / 2 * r).tanh()
        * u
        / (sqrt_c * u_norm)
    )

    gamma_1 = m.mobius_add(x, second_term)
    return gamma_1


class Distribution():
    def __init__(self, mean, sigma) -> None:
        self.mean = mean
        self.sigma = sigma  # .clamp(min=0.1, max=7.)

        self.manifold = geoopt.manifolds.PoincareBall(1.0)
        self.radius = HyperbolicRadius(2, self.manifold.c, self.sigma.view(-1, 1))
        self.direction = HypersphericalUniform(1, device=mean.device)
        self.kl_div = None

    def log_prob(self, z):
        mean = self.mean[None].expand(z.shape)
        radius_sq = self.manifold.dist(mean, z, keepdim=True).pow(2)
        log_prob_z = - radius_sq / 2 / self.sigma.pow(2)[None] - self.direction._log_normalizer() - self.radius.log_normalizer.view([*self.mean.shape[:-1], -1])
        log_prob_z = log_prob_z.sum(dim=-1)

        return log_prob_z

    def rsample(self, N):
        fixed_shape = self.mean.shape[:-1]
        mean = self.mean.view(-1, 2)
        shape = mean[None].expand([N, *mean.shape]).size()
        alpha = self.direction.sample(torch.Size([*shape[:-1]]))
        radius = self.radius.rsample(torch.Size([N]))
    
        z = expmap_polar(self.manifold.c, mean[None], alpha, radius)
        z = z.reshape([N, *fixed_shape, -1])
        return z

    def sample(self, N):
        with torch.no_grad():
            return self.rsample(N)

