import wandb
import torch
import numpy as np
import torchvision.utils as vutils
from torch.nn import functional as F


def evaluation(
    args, 
    vae, 
    dataset, 
    means,
    kls
):
    indices = np.random.choice(means.size(0), 25, replace=False)
    x_true = []
    for idx in indices:
        x_true.append(dataset[idx])
    x_true = torch.stack(x_true, dim=0)
    x_latents = means[indices]
    x_recon = torch.sigmoid(vae.generate(x_latents)).detach()

    img_true = vutils.make_grid(x_true, nrow=5, padding=2)
    img_recon = vutils.make_grid(x_recon, nrow=5, padding=2)
    wandb.log({
        'x_true': wandb.Image(img_true),
        'x_recon': wandb.Image(img_recon)
    })

