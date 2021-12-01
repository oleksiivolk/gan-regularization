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
from metrics.frechet_inception_distance import compute_fid_diffusion_vs_dataset, compute_fid_gaussian_vs_dataset, compute_fid_projector_vs_dataset
from metrics.kernel_inception_distance import compute_kid_projector
from metrics.metric_utils import MetricOptions, MetricOptionsDiffusion, MetricOptionsDistribution, MetricOptionsProjector


# network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-car-config-f.pkl"
target_fname = "efros.jpg"
outdir = "out"
save_video = True
seed = 202
np.random.seed(seed)
torch.manual_seed(seed)
num_steps = 200
device = torch.device('cuda')


generator = Generator(network_pkl, normalize_latent="all_dims", device=device, latent_space="w")
# generator = Generator(network_pkl, device=device, latent_space="w")

target = torch.tensor(plt.imread("target.png")[:, :, :3]*255).to(torch.uint8).permute(2, 0, 1).to(device)
# target.min()
# plt.imsave("target.png", target.permute(1, 2, 0).cpu().numpy())

# prior = Prior(generator, device=device, prior_type='cluster', regularize_cluster_weight=1, num_clusters=18)


# prior = DiffusionPrior(device)
# prior.load_network("/home/oleksiiv/logs/ominous-wraith-125/epoch_2000_ckpt.pth")


prior = Prior(generator, device=device, prior_type='l2', regularize_w_l2=1)

# target = torch.cat((torch.zeros(2, 512, 512), 255*torch.ones(1, 512, 512)), 0)
task = Task(device=device, target=target)
# task = Task(device=device, task_type = 'clip_text', target_str = 'racecar')

# mask = torch.tensor(np.concatenate((np.zeros((3, 256, 512)), np.ones((3, 256, 512))), axis =1))
# task = Task(device=device, target=target, task_type = "perceptual_inpainting", mask = mask) # lr = 10
# task = Task(device=device, target=target, task_type = "inpainting", mask = mask)
projector = Projector(generator, task, prior=prior, device=device)


projector_kwargs = {
    "learning_rate":0.01,
    "num_steps":100,
    "prior_loss_weight": 0.01,
}

proj_opts = MetricOptionsProjector(projector=projector, device=device, kwargs=projector_kwargs)
dataset_kwargs = {'class_name':'training.dataset.ImageFolderDataset','path':'/home/oleksiiv/gan-regularization/lsuncar200k.zip','resolution': 512, 'use_labels':False, 'xflip':False, 'max_size':None}
data_opts = MetricOptions(G=generator.G, dataset_kwargs=dataset_kwargs, device=device)
fid = compute_fid_projector_vs_dataset(proj_opts, data_opts, None, 1000)

print(projector_kwargs)
print(f"FID = {fid}")