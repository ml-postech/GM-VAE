import torch


@torch.jit.script
def euclidean_kl_div(mean1, logvar1, mean2, logvar2):
    kl = logvar2 - logvar1
    kl = kl + (logvar1 - logvar2).exp()
    kl = kl + (mean1 - mean2).pow(2) / logvar2.exp()
    kl = kl - 1
    kl = kl * 0.5
    return kl


@torch.jit.script
def gamma_kl_div(a1, logb1, a2, logb2):
    kl = a2 * (logb1 - logb2) 
    kl = kl - (torch.lgamma(a1) - torch.lgamma(a2))
    kl = kl + (a1 - a2) * torch.digamma(a1)
    kl = kl - (1 - (logb2 - logb1).exp()) * a1
    return kl

