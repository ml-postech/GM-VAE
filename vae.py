import torch
from torch import nn
from math import log
from torch.nn import functional as F

from tasks.utils import discretized_mix_logistic_loss


class VAE(nn.Module):
    def __init__(self, 
        args,
        prior, 
        dist, 
        encoder, 
        encoder_layer, 
        decoder, 
        decoder_layer, 
        loss_type
    ):
        super().__init__()

        self.args = args
        self.prior = prior
        self.dist = dist
        self.encoder = encoder
        self.encoder_layer = encoder_layer
        self.decoder = decoder
        self.decoder_layer = decoder_layer
        self.loss_type = loss_type

    def forward(self, x, n_samples=1, beta=1., iwae=0):
        mean, covar = self.encoder_layer(self.encoder(x))
        if self.args.dist != "PGMNormal":
            variational = self.dist(mean, covar)
        else:
            variational = self.dist(mean, covar, self.args.c)
        
        z = variational.rsample(n_samples)
        x_generated = self.generate(z)
        if self.loss_type == 'BCE':
            recon_loss = F.binary_cross_entropy_with_logits(
                x_generated, 
                x.unsqueeze(0).expand(x_generated.size()), 
                reduction='none'
            )
        elif self.loss_type == 'NLL':
            xmean, xlogvar = x_generated[..., 0], x_generated[..., 1]
            recon_loss = F.gaussian_nll_loss(
                xmean, 
                x.unsqueeze(0).expand(xmean.size()), 
                xlogvar,
                full=True,
                reduction='none'
            )
        elif self.loss_type == 'MSE':
            recon_loss = nn.MSELoss(reduction='none')(
                x_generated,
                x.unsqueeze(0).expand(x_generated.size())
            )
        else:
            recon_loss = discretized_mix_logistic_loss(x, x_generated)

        while len(recon_loss.size()) > 2:
            recon_loss = recon_loss.sum(-1)

        if iwae == 0 or n_samples == 1:
            if variational.kl_div is None:
                kl_loss = variational.log_prob(z) - self.prior.log_prob(z)
                kl_loss = kl_loss.mean(dim=0)
            else:
                kl_loss = variational.kl_div(self.prior)
            kl_loss = kl_loss.sum(dim=-1)
            recon_loss = recon_loss.mean(dim=0)

            total_loss_sum = recon_loss + beta * kl_loss
            loss = total_loss_sum.mean()

            recon_loss = recon_loss.sum()
            kl_loss_ = kl_loss.sum()
            elbo = -(recon_loss + kl_loss_)
        else:
            kl_loss = variational.log_prob(z) - self.prior.log_prob(z)
            kl_loss = kl_loss.sum(dim=-1)
            total_loss_sum = -recon_loss - beta * kl_loss

            loss = total_loss_sum.logsumexp(dim=0) # total_loss_sum.exp().sum(dim=0).log()
            loss = loss - log(n_samples)
            loss = -loss.mean()

            total_elbo_sum = -recon_loss - kl_loss
            elbo = total_elbo_sum.logsumexp(dim=0)
            elbo = elbo - log(n_samples)
            elbo = elbo.sum()

            recon_loss = recon_loss.mean(dim=0).sum()
            kl_loss = kl_loss.mean(dim=0)
            kl_loss_ = kl_loss.sum()

        return loss, elbo, z, mean, recon_loss, kl_loss_, kl_loss

    def generate(self, z):
        return self.decoder(self.decoder_layer(z))

