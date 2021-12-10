#@title Defining a time-dependent score-based model (double click to expand or collapse)

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Callable
import copy
import os
from time import perf_counter
import random
import wandb
import argparse
import imageio
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import numpy as np
import functools
from scipy import integrate
import matplotlib.pyplot as plt

from torch.optim import Adam
import tqdm

from metrics.frechet_inception_distance import compute_fid_diffusion_vs_dataset, compute_fid_gaussian_vs_dataset
from metrics.metric_utils import MetricOptions, MetricOptionsDiffusion, MetricOptionsDistribution

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
  
  def forward(self, x, t): 
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t))    
    # Encoding path
    h1 = self.conv1(x)    
    ## Incorporate information from t
    h1 += self.dense1(embed)
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)

    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = self.act(h)
    h = self.tconv3(torch.cat([h, h3], dim=1))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None, None]
    return h

class MLPScoreNet(nn.Module):

  def __init__(self, hidden_dim, num_hidden, normalize, marginal_prob_std, im_width, im_height, embed_dim=256):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    self.im_width = im_width
    self.im_height = im_height
    self.hidden_dim = hidden_dim
    self.num_hidden = num_hidden
    self.normalize = normalize
    self.marginal_prob_std = marginal_prob_std

    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    
    self.linears = nn.ModuleList([nn.Linear(im_width*im_height, self.hidden_dim)])
    self.linears.extend([nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(self.num_hidden-1)])
    self.linears.append(nn.Linear(self.hidden_dim, im_width*im_height))

    self.embeds = nn.ModuleList([])
    self.embeds.extend([nn.Linear(embed_dim, self.hidden_dim) for i in range(self.num_hidden)])
    self.embeds.append(nn.Linear(embed_dim, im_width*im_height))

    if self.normalize:
        group_size = 32
        print(f"Using group norm group size = {group_size}.")
        self.norms = nn.ModuleList([])
        # self.norms.extend([nn.BatchNorm1d(self.hidden_dim) for i in range(self.num_hidden)])
        self.norms.extend([nn.GroupNorm(group_size, num_channels=self.hidden_dim) for i in range(self.num_hidden)])
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    
  def forward(self, x, t):
    s = x.shape
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t))
    h = torch.flatten(x, start_dim = 1)

    for i in range(self.num_hidden):
        h = self.linears[i](h) + self.embeds[i](embed)
        if self.normalize:
            h = self.norms[i](h)
        h = self.act(h)
    h = self.linears[-1](h) + self.embeds[-1](embed)
    
    h = h / self.marginal_prob_std(t)[:, None]
    h = torch.reshape(h, s)
    return h

@dataclass(eq=False)
class ScoreTrainer:
    args: argparse.Namespace
    device: str = "cpu"
    sigma: float = 25.0
    im_width: int = 28
    im_height: int = 28
    def __post_init__(self):
        hidden_dim, num_hidden, normalize = self.args.hidden_dim, self.args.num_hidden, self.args.normalize,
        self.score_model = torch.nn.DataParallel(MLPScoreNet(hidden_dim, num_hidden, normalize, lambda x: self.marginal_prob_std(x), self.im_width, self.im_height))
        # self.score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=lambda x: self.marginal_prob_std(x)))
        self.score_model = self.score_model.to(self.device)
        self.score_model_ema = copy.deepcopy(self.score_model).eval().to(self.device).requires_grad_(False)

    def marginal_prob_std(self, t):
        """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

        Args:    
            t: A vector of time steps.
            sigma: The $\sigma$ in our SDE.  
        
        Returns:
            The standard deviation.
        """
        if type(t) == float or type(t) == int:
            t = torch.tensor(t, device=self.device)
        else:
            t = t.clone().detach()
        return torch.sqrt((self.sigma**(2 * t) - 1.) / 2. / np.log(self.sigma))

    def diffusion_coeff(self, t):
        """Compute the diffusion coefficient of our SDE.

        Args:
            t: A vector of time steps.
            sigma: The $\sigma$ in our SDE.
        
        Returns:
            The vector of diffusion coefficients.
        """
        return (self.sigma**t).clone().detach()

    def loss_fn(self, x, eps=1e-5, max_t=1, ema=False):
        """The loss function for training score-based generative models.

        Args:
            x: A mini-batch of training data.    
            eps: A tolerance value for numerical stability.
        """
        random_t = max_t * (torch.rand(x.shape[0], device=self.device)) * (1. - eps) + eps # 64
        z = torch.randn_like(x)
        std = self.marginal_prob_std(random_t)
        perturbed_x = x + z * std[:, None, None, None]
        if ema:
            self.score_model_ema.eval()
            score = self.score_model_ema(perturbed_x, random_t)
        else:
            self.score_model.train()
            score = self.score_model(perturbed_x, random_t)
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
        return loss
    
    def load_network(self, checkpoint_path):
        # Cannot load whether network is normalized or not
        state_dict = torch.load(checkpoint_path)
        self.score_model.load_state_dict(state_dict)
        self.score_model_ema.load_state_dict(state_dict)
    
    def checkpoint_fid(self, generator, epoch, noise_amount = 1.):
        with torch.no_grad():
            diffusion_opts = MetricOptionsDiffusion(generator=generator, trainer=self, device=self.device)
            dataset_kwargs = {'class_name':'training.dataset.ImageFolderDataset','path':'./lsuncar200k.zip','resolution': 512, 'use_labels':False, 'xflip':False, 'max_size':None}
            data_opts = MetricOptions(G=generator.G, dataset_kwargs=dataset_kwargs, device=self.device)
            fid = compute_fid_diffusion_vs_dataset(diffusion_opts, data_opts, None, 50000, init_t=noise_amount)
        wandb.log({f'{100*noise_amount}percentnoise_fid': fid})

    def checkpoint_ll(self, generator, epoch, batch_size=10000):
        with torch.no_grad():
            x = generator.sample_latent(batch_size)[:, :, None, :]
            z, bpd = self.ode_likelihood(x, batch_size=batch_size)
            # print("bpdshape", bpd.shape)
            wandb.log({'diffusion_ll': torch.mean(bpd)})

    def checkpoint_images(self, generator, epoch, save_path="."):
        num_samples = 4 ** 2
        with torch.no_grad():
            samples = self.pc_sampler(num_steps=1000, batch_size=num_samples)[:, 0, :, :] # 100, 1, 1, 512 -> 100, 1, 512
            samples = generator.latent_to_image(samples)
            samples = (samples*127.5 + 128).clamp(0, 255).to(torch.uint8)
        sample_grid = make_grid(samples, nrow=int(np.sqrt(num_samples)))

        plt.figure(figsize=(6,6))
        plt.axis('off') # FIX
        plt.imshow(sample_grid.permute(1, 2, 0).cpu())
        plt.savefig(f'{save_path}/{epoch}.png')
    
    def checkpoint_video(self, generator, epoch, save_path="."):
        num_rows = 5
        num_samples = num_rows ** 2
        num_steps = 100

        samples = self.pc_sampler_seq(num_steps=num_steps, batch_size=num_samples)[:, :, 0]

        with torch.no_grad():
            video = imageio.get_writer(f'{save_path}/{epoch}video.mp4', mode='I', fps=25, codec='libx264', bitrate='16M')
            print (f'Saving optimization progress video "{save_path}/{epoch}video.mp4"')
            for i in range(num_steps):
                sub_samples = samples[i]
                ims = 127.5 * generator.latent_to_image(sub_samples) + 128
                ims = ims.clamp(0, 255).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
                grid_image = einops.rearrange(ims, "(n1 n2) h w c-> (n1 h) (n2 w) c", n1=num_rows)
                video.append_data(grid_image)
                # video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
            video.close()
    
    def checkpoint_gaussian_model(self, generator, num_samples=10000):
        with torch.no_grad():
            samples = generator.sample_latent(num_samples)[:, :, None, :]
            # print("lol2", samples.shape)
            std, mean = torch.std_mean(samples, 0, unbiased=False)
            # print("lol1", mean.shape)
            d = torch.distributions.normal.Normal(mean, std)
            # https://stats.stackexchange.com/questions/423120/what-is-bits-per-dimension-bits-dim-exactly-in-pixel-cnn-papers
            test_samples = generator.sample_latent(num_samples)[:, :, None, :]
            # print("shape", (-1*d.log_prob(test_samples)).shape) # torch.Size([10000, 1, 1, 512])
            bpd = torch.mean(-d.log_prob(test_samples)) / np.log(2)
            wandb.log({"gaussian_ll":bpd})

            gaussian_opts = MetricOptionsDistribution(generator=generator, distr=d, device=self.device)
            dataset_kwargs = {'class_name':'training.dataset.ImageFolderDataset','path':'./lsuncar200k.zip','resolution': 512, 'use_labels':False, 'xflip':False, 'max_size':None}
            data_opts = MetricOptions(G=generator.G, dataset_kwargs=dataset_kwargs, device=self.device)
            fid = compute_fid_gaussian_vs_dataset(gaussian_opts, data_opts, None, 10000)
            wandb.log({'gaussian_fid': fid})
    
    def train(self, task_type, n_epochs, batch_size, lr, data_source, max_t=1, ema_beta = 0.99, save_path=".", log_freq = 500):
        optimizer = Adam(self.score_model.parameters(), lr=lr)
        tqdm_epoch = tqdm.trange(n_epochs)

        # if task_type == 'gan':
        #     self.checkpoint_gaussian_model(data_source)

        for epoch in tqdm_epoch:
            avg_loss = 0.
            num_items = 0
            if task_type == 'mnist':
                for x, y in data_source:
                    x = x.to(self.device)    
                    loss = self.loss_fn(x, max_t=max_t)
                    optimizer.zero_grad()
                    loss.backward()    
                    optimizer.step()
                    avg_loss += loss.item() * x.shape[0]
                    num_items += x.shape[0]
                
            elif task_type == 'gan':
                num_batches_per_epoch = 500
                for i in range(num_batches_per_epoch):
                    x = data_source.sample_latent(batch_size)[:, :, None, :]
                    loss = self.loss_fn(x, max_t=max_t)
                    optimizer.zero_grad()
                    loss.backward()    
                    optimizer.step()
                    avg_loss += loss.item() * x.shape[0]
                    num_items += x.shape[0]
                
                wandb.log({"num_latent_seen":(epoch+1) * num_batches_per_epoch * batch_size})
                
                if epoch % 50 == 0:
                    self.checkpoint_ll(data_source, epoch, batch_size=100)

                # if True:
                if epoch >= 1000 and epoch % log_freq == 0:
                    # self.checkpoint_fid(data_source, epoch, noise_amount=0.01)
                    self.checkpoint_fid(data_source, epoch, noise_amount=0.05)
                    # self.checkpoint_fid(data_source, epoch, noise_amount=0.25)
                    self.checkpoint_fid(data_source, epoch, noise_amount=1.)
                if epoch % log_freq == 1:
                    self.checkpoint_images(data_source, epoch, save_path=save_path)
                    # self.checkpoint_video(data_source, epoch, save_path=save_path)

            # Update EMA.
            with torch.autograd.profiler.record_function("ema"):
                for p_ema, p in zip(self.score_model_ema.parameters(), self.score_model.parameters()):
                    p_ema.copy_(p.lerp(p_ema, 1 - ema_beta))
                for b_ema, b in zip(self.score_model_ema.buffers(), self.score_model.buffers()):
                    b_ema.copy_(b)

            # Print the averaged training loss so far.
            tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
            wandb.log({'avg_loss': avg_loss / num_items})
            # Update the checkpoint after each epoch of training.
            if epoch % log_freq == 0:
                torch.save(self.score_model_ema.state_dict(), save_path+f'/epoch_{epoch}_ckpt.pth')
                # Compute the ema loss.
                if task_type == 'gan':
                    avg_loss = 0.
                    num_items = 0
                    num_batches_per_epoch = 500
                    for i in range(num_batches_per_epoch):
                        x = data_source.sample_latent(batch_size)[:, :, None, :]
                        loss = self.loss_fn(x, max_t=max_t, ema=True)
                        avg_loss += loss.item() * x.shape[0]
                        num_items += x.shape[0]
                    wandb.log({'ema_avg_loss': avg_loss / num_items})


            torch.save(self.score_model_ema.state_dict(), save_path+'/most_recent_ckpt.pth')
        
    def Euler_Maruyama_sampler(self,
                            batch_size=64, 
                            num_steps=500,
                            eps=1e-3):
        """Generate samples from score-based models with the Euler-Maruyama solver.

        Args:
            batch_size: The number of samplers to generate by calling this function once.
            num_steps: The number of sampling steps. 
            eps: The smallest time step for numerical stability.
        Returns:
            Samples.  
        """
        t = torch.ones(batch_size, device=self.device)
        init_x = torch.randn(batch_size, 1, self.im_width, self.im_height, device=self.device) \
            * self.marginal_prob_std(t)[:, None, None, None]
        time_steps = torch.linspace(1., eps, num_steps, device=self.device)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        with torch.no_grad():
            self.score_model.eval()
            for time_step in tqdm.notebook.tqdm(time_steps):      
                batch_time_step = torch.ones(batch_size, device=self.device) * time_step
                g = self.diffusion_coeff(batch_time_step)
                mean_x = x + (g**2)[:, None, None, None] * self.score_model(x, batch_time_step) * step_size
                x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
        # Do not include any noise in the last sampling step.
        return mean_x
    
    #@title Define the Predictor-Corrector sampler (double click to expand or collapse)


    def pc_sampler(self,
                num_steps =  500,
                batch_size=64,
                snr=0.16,
                eps=1e-3,
                init_x = None,
                init_t = 1.):
        """Generate samples from score-based models with Predictor-Corrector method.
        Args:
            batch_size: The number of samplers to generate by calling this function once.
            num_steps: The number of sampling steps. 
            Equivalent to the number of discretized time steps.    
            eps: The smallest time step for numerical stability.
        
        Returns: 
            Samples.
        """
        if init_x is not None:
            assert batch_size == init_x.shape[0]

        t = torch.ones(batch_size, device=self.device) * init_t
        if init_x is None:
            init_x = torch.randn(batch_size, 1, self.im_width, self.im_height, device=self.device) * self.marginal_prob_std(t)[:, None, None, None]
        time_steps = np.linspace(init_t, eps, num_steps)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        self.score_model_ema.eval()
        with torch.no_grad():
            for time_step in time_steps:      
                batch_time_step = torch.ones(batch_size, device=self.device) * time_step
                # Corrector step (Langevin MCMC)
                grad = self.score_model_ema(x, batch_time_step)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(x.shape[1:]))
                langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
                x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

                # Predictor step (Euler-Maruyama)
                g = self.diffusion_coeff(batch_time_step)
                x_mean = x + (g**2)[:, None, None, None] * self.score_model_ema(x, batch_time_step) * step_size
                x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      
            
            # The last step does not include any noise
            return x_mean

    def pc_sampler_seq(self,
                num_steps =  500,
                batch_size=64,
                snr=0.16,
                eps=1e-3):
        """Generate samples from score-based models with Predictor-Corrector method.
        Args:
            batch_size: The number of samplers to generate by calling this function once.
            num_steps: The number of sampling steps. 
            Equivalent to the number of discretized time steps.    
            eps: The smallest time step for numerical stability.
        
        Returns: 
            Samples.
        """
        out = torch.zeros(num_steps, batch_size, 1, self.im_width, self.im_height, device=self.device)
        t = torch.ones(batch_size, device=self.device)
        init_x = torch.randn(batch_size, 1, self.im_width, self.im_height, device=self.device) * self.marginal_prob_std(t)[:, None, None, None]
        time_steps = np.linspace(1., eps, num_steps)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        i = 0
        self.score_model_ema.eval()
        with torch.no_grad():
            for time_step in time_steps:      
                batch_time_step = torch.ones(batch_size, device=self.device) * time_step
                # Corrector step (Langevin MCMC)
                grad = self.score_model_ema(x, batch_time_step)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(x.shape[1:]))
                langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
                x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

                # Predictor step (Euler-Maruyama)
                g = self.diffusion_coeff(batch_time_step)
                x_mean = x + (g**2)[:, None, None, None] * self.score_model_ema(x, batch_time_step) * step_size
                out[i] = x_mean.clone().detach()
                i += 1
                x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      
            
            # The last step does not include any noise
            return out
    
    def ode_sampler(self,
                    batch_size=64, 
                    atol=1e-5, 
                    rtol=1e-5,
                    z=None,
                    eps=1e-3):
        """Generate samples from score-based models with black-box ODE solvers.

        Args:
            batch_size: The number of samplers to generate by calling this function once.
            atol: Tolerance of absolute errors.
            rtol: Tolerance of relative errors.
            z: The latent code that governs the final sample. If None, we start from p_1;
            otherwise, we start from the given z.
            eps: The smallest time step for numerical stability.
        """
        t = torch.ones(batch_size, device=self.device)
        # Create the latent code
        if z is None:
            init_x = torch.randn(batch_size, 1, self.im_width, self.im_height, device=self.device) * self.marginal_prob_std(t)[:, None, None, None]
        else:
            init_x = z
            
        shape = init_x.shape

        self.score_model_ema.eval()
        def score_eval_wrapper(sample, time_steps):
            """A wrapper of the score-based model for use by the ODE solver."""
            sample = torch.tensor(sample, device=self.device, dtype=torch.float32).reshape(shape)
            time_steps = torch.tensor(time_steps, device=self.device, dtype=torch.float32).reshape((sample.shape[0], ))    
            with torch.no_grad():    
                score = self.score_model_ema(sample, time_steps)
            return score.cpu().numpy().reshape((-1,)).astype(np.float64)
        
        def ode_func(t, x):        
            """The ODE function for use by the ODE solver."""
            time_steps = np.ones((shape[0],)) * t    
            g = self.diffusion_coeff(torch.tensor(t)).cpu().numpy()
            return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
        
        # Run the black-box ODE solver.
        res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
        print(f"Number of function evaluations: {res.nfev}")
        x = torch.tensor(res.y[:, -1], device=self.device).reshape(shape)

        return x

    def prior_likelihood(self, z, sigma):
        """The likelihood of a Gaussian distribution with mean zero and 
            standard deviation sigma."""
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1,2,3)) / (2 * sigma**2)

    def ode_likelihood(self, x, 
                    batch_size=64,
                    eps=1e-5):
        """Compute the likelihood with probability flow ODE.
        
        Args:
            x: Input data.
            batch_size: The batch size. Equals to the leading dimension of `x`.
            eps: A `float` number. The smallest time step for numerical stability.

        Returns:
            z: The latent code for `x`.
            bpd: The log-likelihoods in bits/dim.
        """

        # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
        epsilon = torch.randn_like(x)
        self.score_model_ema.eval()
            
        def divergence_eval(sample, time_steps, epsilon):      
            """Compute the divergence of the score-based model with Skilling-Hutchinson."""
            with torch.enable_grad():
                sample.requires_grad_(True)
                score_e = torch.sum(self.score_model_ema(sample, time_steps) * epsilon)
                grad_score_e = torch.autograd.grad(score_e, sample)[0]
            return torch.sum(grad_score_e * epsilon, dim=(1, 2, 3))    
        
        shape = x.shape

        def score_eval_wrapper(sample, time_steps):
            """A wrapper for evaluating the score-based model for the black-box ODE solver."""
            sample = torch.tensor(sample, device=self.device, dtype=torch.float32).reshape(shape)
            time_steps = torch.tensor(time_steps, device=self.device, dtype=torch.float32).reshape((sample.shape[0], ))    
            with torch.no_grad():    
                score = self.score_model_ema(sample, time_steps)
            return score.cpu().numpy().reshape((-1,)).astype(np.float64)
        
        def divergence_eval_wrapper(sample, time_steps):
            """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
            with torch.no_grad():
                # Obtain x(t) by solving the probability flow ODE.
                sample = torch.tensor(sample, device=self.device, dtype=torch.float32).reshape(shape)
                time_steps = torch.tensor(time_steps, device=self.device, dtype=torch.float32).reshape((sample.shape[0], ))    
                # Compute likelihood.
                div = divergence_eval(sample, time_steps, epsilon)
                return div.cpu().numpy().reshape((-1,)).astype(np.float64)
        
        def ode_func(t, x):
            """The ODE function for the black-box solver."""
            time_steps = np.ones((shape[0],)) * t    
            sample = x[:-shape[0]]
            logp = x[-shape[0]:]
            g = self.diffusion_coeff(torch.tensor(t)).cpu().numpy()
            sample_grad = -0.5 * g**2 * score_eval_wrapper(sample, time_steps)
            logp_grad = -0.5 * g**2 * divergence_eval_wrapper(sample, time_steps)
            return np.concatenate([sample_grad, logp_grad], axis=0)

        init = np.concatenate([x.cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
        # Black-box ODE solver
        res = integrate.solve_ivp(ode_func, (eps, 1.), init, rtol=1e-5, atol=1e-5, method='RK45')  
        zp = torch.tensor(res.y[:, -1], device=self.device)
        z = zp[:-shape[0]].reshape(shape)
        delta_logp = zp[-shape[0]:].reshape(shape[0])
        sigma_max = self.marginal_prob_std(1.)
        prior_logp = self.prior_likelihood(z, sigma_max)
        bpd = -(prior_logp + delta_logp) / np.log(2)
        N = np.prod(shape[1:])
        bpd = bpd / N + 8.
        return z, bpd
