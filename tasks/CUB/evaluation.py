import wandb
import torch
import numpy as np
import torchvision.utils as vutils

from ..utils import sample_from_discretized_mix_logistic


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
        x_true.append(torch.Tensor(dataset[idx]))
    x_true = torch.stack(x_true, dim=0)
    # x_true = torch.tensor(x[indices])[:, None] / 255.
    x_latents = means[indices]
    x_recon = vae.generate(x_latents).detach()  # .cpu().numpy()
    # x_recon = sample_from_discretized_mix_logistic(x_recon, 10)

    x_true = x_true / 2 + 0.5
    x_recon = x_recon / 2 + 0.5

    img_true = vutils.make_grid(x_true, nrow=5, padding=2)
    img_recon = vutils.make_grid(x_recon, nrow=5, padding=2)
    wandb.log({
        'x_true': wandb.Image(img_true),
        'x_recon': wandb.Image(img_recon)
    })

