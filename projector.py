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

import dnnlib
import legacy


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
    device: str = "cpu"
    latent_space: str = "w"
    seed: int = 0
    use_avg_initialization: bool = False

    def __post_init__(self):
        with dnnlib.util.open_url(self.network_pkl) as fp:
            self.G = legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(self.device)  # type: ignore
            self.G = self.G.eval().requires_grad_(False).to(self.device)

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
        return initial_latents.detach().clone().requires_grad_(True)

    def sample_latent(self, batch_size):
        z_samples = np.random.RandomState(self.seed).randn(batch_size, self.G.z_dim)
        if self.latent_space == "w":
            sample = self.G.mapping(torch.from_numpy(z_samples).to(self.device), None)[
                :, :1, :
            ]  # N, 1, C
        elif self.latent_space == "w+":
            sample = self.G.mapping(
                torch.from_numpy(z_samples).to(self.device), None
            )  # N, W, C
        elif self.latent_space == "z":
            sample = torch.from_numpy(z_samples).to(self.device)
        return sample

    def latent_to_image(self, latent, w_noise=None):
        if self.latent_space == "w":
            if False:  # TODO
                # if w_noise != None:
                latent += w_noise
            ws = latent.repeat([1, self.G.mapping.num_ws, 1])  # n,1,c -> n, w, c
            images = self.G.synthesis(ws, noise_mode="const")
        elif self.latent_space == "w+":
            if False:  # TODO
                # if w_noise != None:
                latent += w_noise.repeat([1, self.G.mapping.num_ws, 1])
            images = self.G.synthesis(latent, noise_mode="const")
        elif self.latent_space == "z":
            ws = self.G.mapping(latent, None)
            if False:  # TODO
                # if w_noise != None:
                ws += w_noise.repeat([1, self.G.mapping.num_ws, 1])
            images = self.G.synthesis(ws, noise_mode="const")
        return images


@dataclass(eq=False)
class Prior:
    gen: Generator
    device: str = "cpu"
    prior_type: str = "l2"
    regularize_w_l2: float = 0
    regularize_cluster_weight: float = 0
    num_clusters: int = 10

    def __post_init__(self):
        if self.prior_type == "l2":
            self.mean_latent = self.calculate_mean_latent()
        elif self.prior_type == "k_means":
            self.clusters = self.sample_clusters()

    def forward(self, latent):
        if self.prior_type == "l2":
            # Mean W regularization.
            wdist = (latent - self.mean_latent).square().sum()  # N,1,C - 1,1,C
            return wdist * self.regularize_w_l2
        elif self.prior_type == "k_means":
            # Cluster regularization
            w_opt_expanded = einops.repeat(
                latent, "N 1 C -> N num_clusters C", num_clusters=num_clusters
            )
            # Compute l2 to each cluster for each w_opt, take min to find closest cluster, sum over N to compute final loss
            w_opt_min_l2, idxs = (
                (self.clusters - w_opt_expanded).square().sum(dim=2).min(dim=1)
            )
            return w_opt_min_l2.sum() * regularize_cluster_weight

    def calculate_mean_latent(
        self, samples=10000,
    ):
        latent_samples = self.gen.sample_latent(samples)
        return torch.mean(latent_samples, axis=0).detach()

    def sample_clusters(self, distance="euclidean", samples=10000, num_clusters=10):
        latent_samples = self.gen.sample_latent(samples)

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

    def __post_init__(self):
        if self.task_type == "perceptual":
            # Load VGG16 feature detector.
            url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
            with dnnlib.util.open_url(url) as f:
                self.vgg16 = torch.jit.load(f).eval().to(self.device)

            # Features for target image.
            target_images = self.target.unsqueeze(0).to(self.device).to(torch.float32)
            if target_images.shape[2] > 256:
                target_images = F.interpolate(
                    target_images, size=(256, 256), mode="area"
                )
            self.target_features = self.vgg16(
                target_images, resize_images=False, return_lpips=True
            )

    def loss(self, images):
        if self.task_type == "perceptual":
            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            images = (images + 1) * (255 / 2)
            if images.shape[2] > 256:
                images = F.interpolate(images, size=(256, 256), mode="area")

            # Features for synth images.
            synth_features = self.vgg16(images, resize_images=False, return_lpips=True)
            return (self.target_features - synth_features).square().sum()


@dataclass(eq=False)
class Projector:
    gen: Generator
    task: Task
    prior: Prior = None
    device: str = "cpu"
    seed: int = 0
    outdir: str = "out"
    initial_learning_rate: float = 0.1
    initial_noise_factor: float = 0.05
    lr_rampdown_length: float = 0.25
    lr_rampup_length: float = 0.05
    noise_ramp_length: float = 0.75
    regularize_noise_weight: float = 1e5
    stats_samples: int = 10000

    def __post_init__(self):
        pass

    def compute_stats(self):
        print(f"Computing W midpoint and stddev using {self.stats_samples} samples...")
        samples = (
            self.gen.sample_latent(self.stats_samples).cpu().numpy().astype(np.float32)
        )
        avg = np.mean(samples, axis=0, keepdims=True)  # [1, 1, C]
        std = (np.sum((samples - avg) ** 2) / self.stats_samples) ** 0.5
        return samples, avg, std

    def learning_rate(self, step, num_steps):
        # Learning rate schedule.
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        lr = self.initial_learning_rate * lr_ramp
        return lr

    def project(
        self, num_images=1, num_steps=1000,
    ):
        # Compute w stats.
        w_samples, w_avg, w_std = self.compute_stats()

        # Setup noise inputs.
        noise_bufs = {
            name: buf.detach()
            for (name, buf) in copy.deepcopy(self.gen.G.synthesis)
            .eval()
            .requires_grad_(False)
            .to(self.device)
            .named_buffers()
            if "noise_const" in name
        }

        opt_latent = self.gen.initial_latent(num_images)

        w_out = torch.zeros(
            [num_steps] + list(opt_latent.shape),
            dtype=torch.float32,
            device=self.device,
        )  # num_steps, *(latent_shape)

        optimizer = torch.optim.Adam(
            [opt_latent] + list(noise_bufs.values()),
            betas=(0.9, 0.999),
            lr=self.initial_learning_rate,
        )

        # Init noise.
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

        for step in range(num_steps):
            t = step / num_steps
            w_noise_scale = (
                w_std
                * self.initial_noise_factor
                * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
            )

            lr = self.learning_rate(step, num_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Synth images from opt_w.
            w_noise = torch.randn_like(opt_latent) * w_noise_scale
            synth_images = self.gen.latent_to_image(opt_latent, w_noise=w_noise)

            # Noise regularization.
            reg_loss = 0.0
            for v in noise_bufs.values():
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (
                        noise * torch.roll(noise, shifts=1, dims=3)
                    ).mean() ** 2
                    reg_loss += (
                        noise * torch.roll(noise, shifts=1, dims=2)
                    ).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)

            loss = (
                self.prior.forward(synth_images)
                + self.task.loss(synth_images)
                + self.regularize_noise_weight * reg_loss
            )

            # Step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            print(f"step {step+1:>4d}/{num_steps}: loss {float(loss):<5.2f}")

            # Save projected W for each optimization step.
            w_out[step] = opt_latent.detach()

            # Normalize noise.
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

        return w_out


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
#             synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode="const")
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
#     synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode="const")
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
