
from __future__ import annotations
import copy
import os
from time import perf_counter
import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import einops

import dnnlib
import legacy

from projector import *

from metrics.frechet_inception_distance import compute_fid_diffusion_vs_dataset, compute_fid_gaussian_vs_dataset
from metrics.kernel_inception_distance import compute_kid_projector
from metrics.metric_utils import MetricOptions, MetricOptionsDiffusion, MetricOptionsDistribution, MetricOptionsProjector


def create_video(projected_w_steps, projector, num_rows=1, outdir="out", name="test.mp4"):
    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    with torch.no_grad():
        video = imageio.get_writer(f'{outdir}/{name}', mode='I', fps=10, codec='libx264', bitrate='8M')
        print (f'Saving optimization progress video {outdir}/{name}')
        frame_count = 0
        

        for projected_w in projected_w_steps:
            frame_count += 1
            if projected_w_steps.shape[0] < 500 or frame_count % 10 == 0:
                synth_image = projector.gen.latent_to_image(projected_w)
                synth_image = (synth_image + 1) * (255/2)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
                grid_image = einops.rearrange(synth_image, "(n1 n2) h w c-> (n1 h) (n2 w) c", n1=num_rows)
                video.append_data(grid_image)
                # video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()
def save_image(w, projector, num_rows=1, outdir="out", name="test.png"):
    os.makedirs(outdir, exist_ok=True)
    print (f'Saving image {outdir}/{name}')
    with torch.no_grad():
        synth_image = projector.gen.latent_to_image(w)
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
        grid_image = einops.rearrange(synth_image, "(n1 n2) h w c-> (n1 h) (n2 w) c", n1=num_rows)
        plt.imsave(f'{outdir}/{name}.png', grid_image)


def project_latent(args):
    network_pkl = args.network_pkl
    latent_space = args.latent_space
    target_fname = args.target_fname
    mask_fname = args.mask_fname
    prior_type = args.prior_type
    lr = args.lr
    num_img = args.num_img
    num_steps = args.num_steps 
    prior_weight = args.prior_weight
    out_dir = args.out_dir
    out_im = f"{latent_space}_{prior_type}_lr{lr}_steps{num_steps}_pw{prior_weight}"

    seed = 203
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda')

    # generator = Generator(network_pkl, normalize_latent="all_dims", device=device, latent_space=latent_space)
    generator = Generator(network_pkl, device=device, latent_space=latent_space)

    prior = Prior(generator, device=device, prior_type=prior_type, regularize_w_l2=1)
    
    target = torch.tensor(plt.imread(target_fname)[:, :, :3]*255).to(torch.uint8).permute(2, 0, 1).to(device)
    mask = torch.tensor(plt.imread(mask_fname)[:, :, :3]).to(torch.uint8).permute(2, 0, 1).to(device)
    task = Task(device=device, target=target, task_type = "perceptual_inpainting", mask = mask)
    # task = Task(device=device, target=target, task_type = "inpainting", mask = mask)

    projector = Projector(generator, task, prior=prior, device=device)

    # later TODO mem optimization -> mixed precision, gradient checkpointing, multiGPU 
    projected_w_steps = projector.project_batched(
        num_img,
        6, # batch size -> depends on gpu
        learning_rate=lr,
        num_steps=num_steps,
        prior_loss_weight=prior_weight,
    )
    
    save_image(projected_w_steps, projector, num_rows=6, outdir=out_dir, name=out_im)

    # z, bpd = prior.ode_likelihood(projected_w_steps.detach()[:, :, None, :], eps=1e-6)
    # idxs = np.argsort(bpd.cpu().numpy())
    # ll_w_steps = projected_w_steps[idxs]
    # save_image(ll_w_steps, projector, num_rows=5, outdir="out10", name="cluster_prior1")

    # create_video(projected_w_steps, projector, num_rows=2, outdir="out6", name=f"diff11-initt{init_t}-lr{lr}-mode{mode}-innersteps{d}.mp4")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--network_pkl', type=str, default="https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-car-config-f.pkl")
    ap.add_argument('--latent_space', type=str, default="style")
    ap.add_argument('--target_fname', type=str, default="")
    ap.add_argument('--out_dir', type=str, default="out")
    ap.add_argument('--mask_fname', type=str, default="")
    ap.add_argument('--prior_type', type=str, default="l2")
    ap.add_argument('--prior_weight', type=float, default=0)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--num_img', type=int, default=6)
    ap.add_argument('--num_steps', type=int, default=100)
    args = ap.parse_args()
    project_latent(args)