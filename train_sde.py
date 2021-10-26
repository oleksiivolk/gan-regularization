
from __future__ import annotations
import wandb
import argparse

import imageio
import torch
from projector import *
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
from score import ScoreTrainer
import os
import matplotlib.pyplot as plt
import einops

def train_sde_mnist(args):
    device = 'cuda'
    transform = transforms.Compose([transforms.Resize(args.im_size), transforms.ToTensor()])
    dataset = MNIST('.', train=True, transform=transform, download=True)

    exp_name = f'is{args.im_size}-lr{args.lr}-s{args.sigma}-bs{args.batch_size}-n_epochs{args.num_epochs}-max_t{args.max_t}-hidden_dim{args.hidden_dim}-max_t{args.max_t}'
    print(f'Training exp {exp_name}')
    log_dir = f'/home/oleksiiv/logs/{exp_name}'
    os.makedirs(log_dir, exist_ok=True)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    trainer = ScoreTrainer(args, device=device, sigma=args.sigma, im_width=args.im_size, im_height=args.im_size)
    trainer.train(args.data, args.num_epochs, args.batch_size, args.lr, data_loader, max_t=args.max_t, save_path=log_dir)

    num_samples = 10 ** 2
    samples = trainer.pc_sampler(num_steps=1000, batch_size=num_samples)

    ## Sample visualization.
    samples = samples.clamp(0.0, 1.0)   
    sample_grid = make_grid(samples, nrow=int(np.sqrt(num_samples)))

    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    
    plt.savefig(f'{log_dir}/{exp_name}.png')

def train_sde_gan(args):
    device = 'cuda'
    # exp_name = f'is{args.im_size}-lr{args.lr}-s{args.sigma}-bs{args.batch_size}-n_epochs{args.num_epochs}-max_t{args.max_t}-hidden_dim{args.hidden_dim}-max_t{args.max_t}-pkl{args.network_pkl}-latent{args.latent_space}'
    exp_name = wandb.run.name
    print(f'Training exp {exp_name}')
    log_dir = f'/home/oleksiiv/logs/{exp_name}'
    os.makedirs(log_dir, exist_ok=True)

    generator = Generator(args.network_pkl, normalize_w=args.normalize_w, device=device, latent_space=args.latent_space)

    trainer = ScoreTrainer(args, device=device, sigma=args.sigma, im_width=1, im_height=512)
    trainer.train(args.data, args.num_epochs, args.batch_size, args.lr, generator, max_t=args.max_t, ema_beta=args.ema_beta, save_path=log_dir)

    num_samples = 4 ** 2
    samples = trainer.pc_sampler(num_steps=1000, batch_size=num_samples)[:, 0, :, :] # 100, 1, 1, 512 -> 100, 1, 512
    samples = generator.latent_to_image(samples)

    ## Sample visualization.
    samples = samples.clamp(0.0, 1.0)   
    sample_grid = make_grid(samples, nrow=int(np.sqrt(num_samples)))

    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    
    plt.savefig(f'{log_dir}/{exp_name}.png')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-is', '--im_size', type=int, default=28)
    ap.add_argument('-lr', '--lr', type=float, default=1e-3)
    ap.add_argument('-s', '--sigma', type=float, default=25)
    ap.add_argument('-bs', '--batch_size', type=int, default=64)
    ap.add_argument('-n', '--num_epochs', type=int, default=300)
    ap.add_argument('--hidden_dim', type=int, default=32)
    ap.add_argument('--max_t', type=float, default=1)
    ap.add_argument('--data', type=str, default="mnist")
    ap.add_argument('--network_pkl', type=str, default="")
    ap.add_argument('--latent_space', type=str, default="w")
    ap.add_argument('--num_hidden', type=int, default=5)
    ap.add_argument('--normalize', type=bool, default=False)
    ap.add_argument('--normalize_w', type=str, default=None)
    ap.add_argument('--ema_beta', type=float, default=0.99)
    args = ap.parse_args()

    wandb.init(project='gan-reg', entity='oleksiiv')
    wandb.config.update(args)
    wandb.config.model = "wspacemlp"
    wandb.config.type = "sde"

    if args.data == 'mnist':
        train_sde_mnist(args)
    elif args.data == 'gan':
        train_sde_gan(args)