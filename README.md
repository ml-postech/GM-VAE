# Hyperbolic VAE via Latent Gaussian Distributions
This repository is the official implementation of ["Hyperbolic VAE via Latent Gaussian Distributions"](https://arxiv.org/abs/2209.15217) accepted by NeurIPS 2023.

## Abstract
We propose a Gaussian manifold variational auto-encoder (GM-VAE) whose latent space consists of a set of Gaussian distributions. It is known that the set of the univariate Gaussian distributions with the Fisher information metric form a hyperbolic space, which we call a Gaussian manifold. To learn the VAE endowed with the Gaussian manifolds, we propose a pseudo-Gaussian manifold normal distribution based on the Kullback-Leibler divergence, a local approximation of the squared Fisher-Rao distance, to define a density over the latent space. In experiments, we demonstrate the efficacy of GM-VAE on two different tasks: density estimation of image datasets and environment modeling in model-based reinforcement learning. GM-VAE outperforms the other variants of hyperbolic- and Euclidean-VAEs on density estimation tasks and shows competitive performance in model-based reinforcement learning. We observe that our model provides strong numerical stability, addressing a common limitation reported in previous hyperbolic-VAEs.

## Setup
1. Install pytorch and torchvision. The recommended versions are pytorch 2.0.1 and torchvision 0.15.2
2. Run `pip install -r requirements.txt`.
3. Install geoopt py running the following command: `pip install git+https://github.com/geoopt/geoopt.git`.
4. Prepare the datasets by running the script: `sh scripts/download.sh`.

## Usages
You can reproduce the experiments from our paper using the following commands:
```
> python train_vae.py --dist=PGMNormal --exp_name=reproduce --seed 1 --c -1.0 --latent_dim=4 --task=Breakout
> python train_vae.py --dist=PGMNormal --exp_name=reproduce --seed 1 --c -1.0 --latent_dim=35 --task=CUB
> python train_vae.py --dist=PGMNormal --exp_name=reproduce --seed 1 --c -1.0 --latent_dim=35 --task=Food101
> python train_vae.py --dist=PGMNormal --exp_name=reproduce --seed 1 --c -1.0 --latent_dim=35 --task=Oxford102
```

You can also reproduce the entire table by running the wandb sweeps in `scripts/reproduce_breakout.yaml` and `scripts/reproduce_rgb.yaml`.

## Cite
Please cite our paper if you use the model or this code in your own work:
```
@inproceedings{
anonymous2023hyperbolic,
title={Hyperbolic VAE via Latent Gaussian Distributions},
author={Seunghyuk Cho and Juyong Lee and Dongwoo Kim},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=FNn4zibGvw}
}
```
