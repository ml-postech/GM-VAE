import copy
import wandb
import datetime
import importlib
from pathlib import Path

import torch
import argparse
import numpy as np
from torch import nn
from math import sqrt
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, StepLR

from vae import VAE
from arguments import add_train_args, get_initial_parser


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
        # nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


def train(epoch, args, train_loader, vae, optimizer):
    n_data = 0
    train_elbo, train_recon, train_kl = 0., 0., 0.

    for x in train_loader:
        for param in vae.parameters():
            param.grad = None
        x = x.to(args.device)
        loss, elbo, _, _, recon_loss, kl_loss, _ = vae(
            x, 
            args.train_samples, 
            args.beta,
            args.iwae
        )

        loss.backward()
        optimizer.step()
       
        n_data += x.size(0)
        train_elbo += elbo.item()
        train_recon += recon_loss.item()
        train_kl += kl_loss.item()

    if epoch % args.log_interval == 0 or epoch == args.n_epochs:
        train_elbo /= n_data
        train_recon /= n_data
        train_kl /= n_data
        print(f'Epoch: {epoch:6d} | ELBO: {train_elbo:.2f} | Recon Loss: {train_recon:.2f} | KL: {train_kl:.3f}')
        wandb.log({
            'epoch': epoch,
            'train_elbo': train_elbo,
            'train_recon': train_recon,
            'train_kl': train_kl
        })

    return train_elbo


def eval(prefix, epoch, args, test_loader, vae, root_dir, test_data, eval_fn):
    log_dir = root_dir / str(epoch)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    vae.eval()
    with torch.no_grad():
        n_data = 0
        means, kls = None, None
        total_elbo, total_recon, total_kl = 0., 0., 0.
        for x in test_loader:
            x = x.to(args.device)
            _, elbo, _, means_, recon_loss, kl_loss, kls_ = vae(x, (args.test_samples if prefix == 'test' else 1), iwae=args.iwae)

            n_data += x.size(0)
            total_elbo += elbo.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            if means is None:
                means = means_
                kls = kls_
            else:
                means = torch.concat((means, means_), dim=0)
                kls = torch.concat((kls, kls_), dim=0)

        if eval_fn is not None:
            eval_fn(args, vae, test_data, means, kls)

        total_elbo /= n_data
        total_recon /= n_data
        total_kl /= n_data
        print(f'===========> {prefix} ELBO: {total_elbo:.2f} | Recon: {total_recon:.2f} | KL: {total_kl:.2f}')
        wandb.log({
            'epoch': epoch,
            f'{prefix}_elbo': total_elbo,
            f'{prefix}_recon': total_recon,
            f'{prefix}_kl': total_kl
        })

    return total_elbo


if __name__ == "__main__":
    init_parser = get_initial_parser()
    task_name = init_parser.parse_known_args()[0].task
    task_module = importlib.import_module(f'tasks.{task_name}')
    dist_name = init_parser.parse_known_args()[0].dist
    dist_module = importlib.import_module(f'distributions.{dist_name}')

    parser = argparse.ArgumentParser()
    add_train_args(parser)
    getattr(task_module, 'add_task_args')(parser)
    getattr(dist_module, 'add_distribution_args')(parser)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_num_threads(1)

    runId = datetime.datetime.now().isoformat().replace(':', '_')
    root_dir = Path(args.log_dir) / runId

    train_data = getattr(task_module, 'Dataset')(args, split='train')
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=1)
    valid_data = getattr(task_module, 'Dataset')(args, split='valid')
    valid_loader = DataLoader(valid_data, batch_size=args.train_batch_size, shuffle=False, num_workers=1)
    test_data = getattr(task_module, 'Dataset')(args, split='test')
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=1)
    eval_fn = getattr(task_module, 'evaluation')

    variational_fn = getattr(dist_module, 'Distribution')
    prior = getattr(dist_module, 'get_prior')(args)

    encoder = getattr(task_module, 'Encoder')(args)
    encoder_layer = getattr(
        dist_module, 
        f'{args.layer}EncoderLayer'
    )(args, encoder.output_dim)
    decoder = getattr(task_module, 'Decoder')(args)
    decoder_layer = getattr(
        dist_module, 
        f'{args.layer}DecoderLayer'
    )(args)

    recon_loss_type = getattr(task_module, 'recon_loss_type')
    vae = VAE(
        args,
        prior, 
        variational_fn, 
        encoder, 
        encoder_layer, 
        decoder, 
        decoder_layer, 
        recon_loss_type
    )
    vae = vae.to(args.device)

    optimizer = Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(decoder_layer.parameters()) + list(encoder_layer.parameters()), 
        lr=args.lr
    )

    wandb.init(project='GM-VAE')
    wandb.run.name = args.exp_name
    wandb.config.update(args)
    
    best_model = copy.deepcopy(vae)
    best_elbo = -1e9

    print(root_dir)
    for epoch in range(1, args.n_epochs + 1):
        vae.train()
        train_elbo = train(epoch, args, train_loader, vae, optimizer)

        if epoch % args.eval_interval == 0 or epoch == args.n_epochs:
            elbo = eval('valid', epoch, args, valid_loader, vae, root_dir, valid_data, None)
            if best_elbo < elbo:
                best_elbo = elbo
                best_model = copy.deepcopy(vae)
                torch.save(best_model.state_dict(), root_dir / 'model.pt')
        
    _ = eval('test', epoch, args, test_loader, best_model, root_dir, test_data, eval_fn)
    print(root_dir)

