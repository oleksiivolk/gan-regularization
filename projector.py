# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Callable
import copy
import os
from time import perf_counter
import random

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from kmeans_pytorch import kmeans
import einops
import tqdm
import clip


import dnnlib
import legacy

from score import *


def load_target_img(target_fname, img_res):
    target_pil = PIL.Image.open(target_fname).convert("RGB")
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(
        ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
    )
    target_pil = target_pil.resize((img_res, img_res), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)


# TODO Optimize float precision
# G.synthesis.num_fp16_res
# synthesis_kwargs = {num_fp16_res=1}
@dataclass(eq=False)
class Generator:
    network_pkl: str
    normalize_latent: str = None
    device: str = "cpu"
    latent_space: str = "w"
    seed: int = 0
    use_avg_initialization: bool = False

    def __post_init__(self):
        with dnnlib.util.open_url(self.network_pkl) as fp:
            self.G = legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(self.device)  # type: ignore
            self.G = self.G.eval().requires_grad_(False).to(self.device)

        num_samples = 10000
        samples = self.sample_latent_unnormalized(num_samples)

        if self.normalize_latent == "all_dims":
            print("Normalizing across all channels.")
            with torch.no_grad():
                self.mean = torch.mean(samples).detach()  # N, 1, C -> float
                self.std = torch.std(samples).detach()  # N, 1, C -> float
        elif self.normalize_latent == "indep_dims":
            print("Normalizing each channel independently.")
            with torch.no_grad():
                self.mean = torch.mean(samples, axis=0).detach()  # N, 1, C -> 1, C
                self.std = torch.std(samples, axis=0).detach()  # N, 1, C -> 1, C

    def initial_latent(self, batch_size):
        num_samples = 10000
        samples = self.sample_latent(num_samples)

        # Initialize to the mean latent
        if self.use_avg_initialization:
            avg_latent = torch.mean(samples, axis=0)
            initial_latents = einops.repeat(avg_latent, "1 W C -> n W C", n=batch_size)
        else:
            idxs = np.random.permutation(num_samples)[:batch_size]
            initial_latents = samples[idxs]

        if self.normalize_latent:
            return (
                ((initial_latents - self.mean) / self.std)
                .detach()
                .clone()
                .requires_grad_(True)
            )
        else:
            return initial_latents.detach().clone().requires_grad_(True)

    def sample_latent_unnormalized(self, batch_size):
        z_samples = torch.randn(batch_size, self.G.z_dim, device=self.device)
        if self.latent_space == "w":
            sample = self.G.mapping(z_samples, None)[:, :1, :]  # N, 1, C
        elif self.latent_space == "w+":
            sample = self.G.mapping(z_samples, None)  # N, W, C
        elif self.latent_space == "z":
            sample = z_samples
        elif self.latent_space == "z+":
            sample = einops.repeat(z_samples, "N C -> N W C", W=self.G.mapping.num_ws)
        return sample

    def sample_latent(self, batch_size):
        if self.normalize_latent:
            return (self.sample_latent_unnormalized(batch_size) - self.mean) / self.std
        else:
            return self.sample_latent_unnormalized(batch_size)

    # Will re-adjust latent to unnormalized.
    def latent_to_image(self, latent):

        if self.normalize_latent:
            latent = (latent * self.std) + self.mean

        if self.latent_space == "w":
            ws = latent.repeat([1, self.G.mapping.num_ws, 1])  # n,1,c -> n, w, c
            images = self.G.synthesis(ws)
        elif self.latent_space == "w+":
            images = self.G.synthesis(latent)
        elif self.latent_space == "z":
            ws = self.G.mapping(latent, None)
            images = self.G.synthesis(ws)
        elif self.latent_space == "z+":
            N = latent.shape[0]
            latent = einops.rearrange(latent, "N W C -> (N W) C")
            ws = self.G.mapping(latent, None)
            ws = einops.rearrange(ws[:, 0, :], "(N W) C -> N W C", N=N)
            images = self.G.synthesis(ws)
        return images


@dataclass(eq=False)
class DiffusionPrior:
    device: str = "cpu"
    sigma: float = 25.0
    hidden_w: int = 1
    hidden_h: int = 512
    hidden_dim: int = 4000
    num_hidden: int = 7
    normalize: bool = True

    def __post_init__(self):
        self.score_model = torch.nn.DataParallel(
            MLPScoreNet(
                self.hidden_dim,
                self.num_hidden,
                self.normalize,
                lambda x: self.marginal_prob_std(x),
                self.hidden_w,
                self.hidden_h,
            )
        )
        self.score_model = self.score_model.to(self.device)

    def marginal_prob_std(self, t):
        """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$."""
        if type(t) == float or type(t) == int:
            t = torch.tensor(t, device=self.device)
        else:
            t = t.clone().detach()
        return torch.sqrt((self.sigma ** (2 * t) - 1.0) / 2.0 / np.log(self.sigma))

    def diffusion_coeff(self, t):
        """Compute the diffusion coefficient of our SDE."""
        return (self.sigma ** t).clone().detach()

    def step(self, x, t=0.01, snr=0.16, eps=1e-3, step_size=0.01):
        self.score_model.eval()
        batch_size = x.shape[0]
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        time_step = t
        # TODO try varying step size

        batch_time_step = torch.ones(batch_size, device=self.device) * time_step
        # Corrector step (Langevin MCMC)
        grad = self.score_model(x, batch_time_step)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = np.sqrt(np.prod(x.shape[1:]))
        langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
        x = (
            x
            + langevin_step_size * grad
            + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
        )

        # Predictor step (Euler-Maruyama)
        g = self.diffusion_coeff(batch_time_step)
        x_mean = (
            x
            + (g ** 2)[:, None, None, None]
            * self.score_model(x, batch_time_step)
            * step_size
        )
        x = x_mean + torch.sqrt(g ** 2 * step_size)[
            :, None, None, None
        ] * torch.randn_like(x)
        # TODO experiment add back in noise or not
        return x[:, 0, :, :]

    def load_network(self, checkpoint_path):
        # Cannot load whether network is normalized or not
        state_dict = torch.load(checkpoint_path)
        out = self.score_model.load_state_dict(state_dict)
        self.score_model.to(self.device)
        print(out)

    def prior_likelihood(self, z, sigma):
        """The likelihood of a Gaussian distribution with mean zero and 
            standard deviation sigma."""
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1,2,3)) / (2 * sigma**2)

    def ode_likelihood(self, x,
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
        x = x.detach()
        epsilon = torch.randn_like(x)
        self.score_model.eval()
            
        def divergence_eval(sample, time_steps, epsilon):      
            """Compute the divergence of the score-based model with Skilling-Hutchinson."""
            with torch.enable_grad():
                sample.requires_grad_(True)
                score_e = torch.sum(self.score_model(sample, time_steps) * epsilon)
                grad_score_e = torch.autograd.grad(score_e, sample)[0]
            return torch.sum(grad_score_e * epsilon, dim=(1, 2, 3))    
        
        shape = x.shape

        def score_eval_wrapper(sample, time_steps):
            """A wrapper for evaluating the score-based model for the black-box ODE solver."""
            sample = torch.tensor(sample, device=self.device, dtype=torch.float32).reshape(shape)
            time_steps = torch.tensor(time_steps, device=self.device, dtype=torch.float32).reshape((sample.shape[0], ))    
            with torch.no_grad():    
                score = self.score_model(sample, time_steps)
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



@dataclass(eq=False)
class Prior:
    gen: Generator
    device: str = "cpu"
    prior_type: str = "l2"
    regularize_w_l2: float = 0
    regularize_cluster_weight: float = 0
    num_clusters: int = 10
    cluster_samples: int = 100000
    cluster_distance: str = "cosine"
    optimize_cluster_individually: bool = True

    def __post_init__(self):
        if self.prior_type == "l2":
            self.mean_latent = self.calculate_mean_latent()
        elif self.prior_type == "cluster":
            self.clusters = self.sample_clusters(
                distance = self.cluster_distance,
                samples=self.cluster_samples, num_clusters=self.num_clusters
            )
    
    def get_clusters(self):
        return self.clusters

    def forward(self, latent):
        if self.prior_type == "l2":
            # Mean W regularization.
            wdist = (latent - self.mean_latent).square().sum()  # N,1,C - 1,1,C
            return wdist * self.regularize_w_l2
        elif self.prior_type == "cluster":
            # Cluster regularization
            if self.gen.latent_space == "z":
                latent = latent.unsqueeze(1)

            latent_size = latent.shape[1]
            latent_expanded = einops.repeat(
                latent, "N W C -> N num_clusters W C", num_clusters=self.num_clusters
            )
            # Compute l2 to each cluster for each w_opt, take min to find closest cluster, sum over N to compute final loss
            if self.optimize_cluster_individually:
                latents_cluster_l2s, idxs = (
                    (self.clusters - latent_expanded).square().sum(dim=3).min(dim=1)
                )
            else:
                latents_cluster_l2s, idxs = (
                    (self.clusters - latent_expanded)
                    .square()
                    .sum(dim=(2, 3))
                    .min(dim=1)
                )
            loss = latents_cluster_l2s.sum() * self.regularize_cluster_weight / latent_size
            return loss

    def calculate_mean_latent(
        self, samples=10000,
    ):
        latent_samples = self.gen.sample_latent(samples)
        return torch.mean(latent_samples, axis=0).detach()

    def sample_clusters(self, distance="euclidean", samples=10000, num_clusters=10):
        latent_samples = self.gen.sample_latent(samples)
        if self.gen.latent_space == "w+" or self.gen.latent_space == "z+":
            latent_samples = latent_samples[:, :1, :]

        cluster_ids_x, cluster_centers = kmeans(
            X=latent_samples,
            num_clusters=num_clusters,
            distance=distance,
            device=self.device,
        )
        return cluster_centers.to(self.device)  # [num_clusters, C]


@dataclass(eq=False)
class Task:
    device: str = "cpu"
    task_type: str = "perceptual"
    target: torch.Tensor = None
    target_str: str = ""
    mask: torch.Tensor = None

    def __post_init__(self):
        if self.task_type == "perceptual" or self.task_type == "perceptual_inpainting":
            # Load VGG16 feature detector.
            url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
            with dnnlib.util.open_url(url) as f:
                self.vgg16 = torch.jit.load(f).eval().to(self.device)

            # Features for target image.
            target_images = self.target.unsqueeze(0).to(self.device).to(torch.float32)

            if self.task_type == "perceptual_inpainting":
                target_images = target_images * self.mask.to(self.device)

            if target_images.shape[2] > 256:
                target_images = F.interpolate(
                    target_images, size=(256, 256), mode="area"
                )

            self.target_features = self.vgg16(
                target_images, resize_images=False, return_lpips=True
            )
        elif self.task_type == "clip_text":
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            text = clip.tokenize([self.target_str]).to(self.device)
            with torch.no_grad():
                self.target_features = self.clip_model.encode_text(text)
            self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        elif self.task_type == "inpainting":
            self.target = self.target.to(self.device)
            self.mask = self.mask.to(self.device)

    def loss(self, images):
        images = images.to(self.device)
        if self.task_type == "perceptual" or self.task_type == "perceptual_inpainting":
            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            images = (images + 1) * (255 / 2)
            if self.task_type == "perceptual_inpainting":
                images = images * self.mask.to(self.device)
            if images.shape[2] > 256:
                images = F.interpolate(images, size=(256, 256), mode="area")

            # Features for synth images.
            synth_features = self.vgg16(images, resize_images=False, return_lpips=True)
            return (self.target_features - synth_features).square().sum()
        if self.task_type == "inpainting":
            images = (images + 1) * (255 / 2)
            return (self.mask * (self.target - images)).square().sum()
        elif self.task_type == "clip_text":
            # TODO: Maybe preprocess with clip
            if images.shape[2] > 224:
                images = F.interpolate(images, size=(224, 224), mode="area")
            synth_features = self.clip_model.encode_image(images)
            return (1 - self.cos(synth_features, self.target_features)).sum()


@dataclass(eq=False)
class Projector:
    gen: Generator
    task: Task
    prior: Prior = None
    device: str = "cpu"
    seed: int = 0
    outdir: str = "out"
    initial_learning_rate: float = 0.1
    lr_rampdown_length: float = 0.25
    lr_rampup_length: float = 0.05
    stats_samples: int = 10000

    def __post_init__(self):
        pass

    def learning_rate(self, step, num_steps):
        # Learning rate schedule.
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        lr = self.initial_learning_rate * lr_ramp
        return lr

    def project_batched(self, total_num, batch_size, **args):
        opt_latent_shape = list(self.gen.initial_latent(1).shape)[1:]
        latent_out = torch.zeros(
            [total_num] + opt_latent_shape,
            dtype=torch.float32,
            device=self.device,
        )
        for i in tqdm.trange(total_num // batch_size):
            latent_out[i*batch_size:(i+1)*batch_size] = self.project(num_images=batch_size, **args)
        return latent_out

    def project(
        self,
        learning_rate=0.1,
        num_images=1,
        num_steps=1000,
        diffusion_time_schedule="constant",
        mini_end_init_t=0.05,
        num_diffusion_steps_per_step=1,
        prior_loss_weight=1,
        diffusion_step_size=0.01,
        optimizer_step=True,
        diffusion_magnitude_lambda = 2,
        save_video = False,
    ):
        diffusion_total_steps = num_steps * num_diffusion_steps_per_step
        diffusion_steps = 0
        opt_latent = self.gen.initial_latent(num_images)

        if save_video:
            latent_out = torch.zeros(
                [num_steps] + list(opt_latent.shape),
                dtype=torch.float32,
                device=self.device,
            )  # num_steps, *(latent_shape)

        optimizer = torch.optim.Adam(
            [opt_latent], betas=(0.9, 0.999), lr=learning_rate,
        )

        with tqdm.trange(num_steps) as progress_bar:
            for step in progress_bar:
                t = step / num_steps

                lr = self.learning_rate(step, num_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # Synth images from opt_w.
                synth_images = self.gen.latent_to_image(opt_latent)

                if optimizer_step:
                    task_loss = self.task.loss(synth_images)
                else:
                    task_loss = 0
                if isinstance(self.prior, Prior):
                    prior_loss = self.prior.forward(opt_latent) * prior_loss_weight
                else:
                    prior_loss = 0
                loss = prior_loss + task_loss

                before_update_latent = opt_latent.clone().detach()
                # Step
                if loss != 0:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                optimizer_update_norm = torch.norm(
                    opt_latent - before_update_latent, dim=(1, 2)
                )
                # print(2, optimizer_update_norm)

                if isinstance(self.prior, DiffusionPrior):
                    with torch.no_grad():
                        for inner_diffusion_step in range(num_diffusion_steps_per_step):
                            eps = 1e-5
                            if diffusion_time_schedule == "linear":
                                new_latent = self.prior.step(
                                    opt_latent,
                                    t=1 - diffusion_steps / diffusion_total_steps + eps,
                                    step_size=diffusion_step_size,
                                )
                            elif diffusion_time_schedule == "constant":
                                new_latent = self.prior.step(
                                    opt_latent,
                                    t=diffusion_step_size,
                                    step_size=diffusion_step_size,
                                )
                            elif diffusion_time_schedule == "mini":
                                new_latent = self.prior.step(
                                    opt_latent,
                                    t=1
                                    - inner_diffusion_step
                                    / num_diffusion_steps_per_step,
                                    step_size=diffusion_step_size,
                                )
                            elif diffusion_time_schedule == "mini_end":
                                step_size = (
                                    mini_end_init_t / num_diffusion_steps_per_step
                                )
                                new_latent = self.prior.step(
                                    opt_latent,
                                    t=mini_end_init_t
                                    * (
                                        1
                                        - inner_diffusion_step
                                        / num_diffusion_steps_per_step
                                    )
                                    + eps,
                                    step_size=step_size,
                                )

                            diffusion_update_norm = torch.norm(
                                new_latent - opt_latent, dim=(1, 2)
                            )
                            update = (
                                (new_latent - opt_latent)
                                / diffusion_update_norm[:, None, None]
                                * optimizer_update_norm[:, None, None].clamp(max=1)
                                / diffusion_magnitude_lambda
                                * inner_diffusion_step
                                / num_diffusion_steps_per_step
                            )
                            # print("1", diffusion_update_norm)
                            new_latent = opt_latent + update
                            opt_latent.copy_(new_latent)
                            diffusion_steps += 1

                # if isinstance(self.prior, DiffusionPrior):
                #     progress_bar.set_description(f"Optimizer step norm = {optimizer_update_norm}; diffusion norm = {diffusion_update_norm}")

                # print(f"step {step+1:>4d}/{num_steps}: prior_loss {float(prior_loss):<5.2f} task_loss {float(task_loss):<5.2f}")

                # Log magnitudes of task step and diffusion step
                # guided diffusion

                # Save projected W for each optimization step.
                if save_video:
                    latent_out[step] = opt_latent.detach()
        if save_video:
            return latent_out
        else:
            return opt_latent


# ----------------------------------------------------------------------------


# @click.command()
# @click.option("--network", "network_pkl", help="Network pickle filename", required=True)
# @click.option(
#     "--target",
#     "target_fname",
#     help="Target image file to project to",
#     required=True,
#     metavar="FILE",
# )
# @click.option(
#     "--num-steps",
#     help="Number of optimization steps",
#     type=int,
#     default=1000,
#     show_default=True,
# )
# @click.option("--seed", help="Random seed", type=int, default=303, show_default=True)
# @click.option(
#     "--save-video",
#     help="Save an mp4 video of optimization progress",
#     type=bool,
#     default=True,
#     show_default=True,
# )
# @click.option(
#     "--outdir", help="Where to save the output images", required=True, metavar="DIR"
# )
# def run_projection(
#     network_pkl: str,
#     target_fname: str,
#     outdir: str,
#     save_video: bool,
#     seed: int,
#     num_steps: int,
# ):
#     """Project given image to the latent space of pretrained network pickle.

#     Examples:

#     \b
#     python projector.py --outdir=out --target=~/mytargetimg.png \\
#         --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
#     """
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     # Load networks.
#     print('Loading networks from "%s"...' % network_pkl)
#     device = torch.device("cuda")
#     with dnnlib.util.open_url(network_pkl) as fp:
#         G = legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(device)  # type: ignore

#     # Load target image.
#     target_pil = PIL.Image.open(target_fname).convert("RGB")
#     w, h = target_pil.size
#     s = min(w, h)
#     target_pil = target_pil.crop(
#         ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
#     )
#     target_pil = target_pil.resize(
#         (G.img_resolution, G.img_resolution), PIL.Image.LANCZOS
#     )
#     target_uint8 = np.array(target_pil, dtype=np.uint8)

#     # Optimize projection.
#     start_time = perf_counter()
#     projected_w_steps = project(
#         G,
#         target=torch.tensor(
#             target_uint8.transpose([2, 0, 1]), device=device
#         ),  # pylint: disable=not-callable
#         num_steps=num_steps,
#         device=device,
#         verbose=True,
#     )
#     print(f"Elapsed: {(perf_counter()-start_time):.1f} s")

#     # Render debug output: optional video and projected image and W vector.
#     os.makedirs(outdir, exist_ok=True)
#     if save_video:
#         video = imageio.get_writer(
#             f"{outdir}/proj.mp4", mode="I", fps=10, codec="libx264", bitrate="16M"
#         )
#         print(f'Saving optimization progress video "{outdir}/proj.mp4"')
#         for projected_w in projected_w_steps:
#             synth_image = G.synthesis(projected_w.unsqueeze(0))
#             synth_image = (synth_image + 1) * (255 / 2)
#             synth_image = (
#                 synth_image.permute(0, 2, 3, 1)
#                 .clamp(0, 255)
#                 .to(torch.uint8)[0]
#                 .cpu()
#                 .numpy()
#             )
#             video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
#         video.close()

#     # Save final projected frame and W vector.
#     target_pil.save(f"{outdir}/target.png")
#     projected_w = projected_w_steps[-1]
#     synth_image = G.synthesis(projected_w.unsqueeze(0))
#     synth_image = (synth_image + 1) * (255 / 2)
#     synth_image = (
#         synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
#     )
#     PIL.Image.fromarray(synth_image, "RGB").save(f"{outdir}/proj.png")
#     np.savez(f"{outdir}/projected_w.npz", w=projected_w.unsqueeze(0).cpu().numpy())


# # ----------------------------------------------------------------------------

# if __name__ == "__main__":
#     run_projection()  # pylint: disable=no-value-for-parameter

# # ----------------------------------------------------------------------------
