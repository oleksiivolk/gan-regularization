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

def create_video(projected_w_steps, projector, num_rows=1, outdir="out", name="test.mp4"):
    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    with torch.no_grad():
        video = imageio.get_writer(f'{outdir}/{name}', mode='I', fps=10, codec='libx264', bitrate='1M')
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
prior = DiffusionPrior(device)
prior.load_network("/home/oleksiiv/logs/ominous-wraith-125/epoch_2000_ckpt.pth")

blue_target = torch.cat((torch.zeros(2, 512, 512), 255*torch.ones(1, 512, 512)), 0)
task = Task(device=device, target=blue_target)
projector = Projector(generator, task, prior=prior, device=device)



start_time = perf_counter()

# learning_rates = [
#     # 0.1, 0.03, 0.01, 
# 0.003, 0.001, 0.0003]
# time_schedules = ["constant"]
# num_d_steps = [
#     # 1, 10, 
#     100, 1000]
# optimizers = [True, False]


# stepsizes = {
#     1:[0.05, 0.01, 0.005],
# 10:[0.005, 0.001, 0.0005],
# 100:[1e-5, 5e-5, 1e-5, 5e-6, 1e-6],
# 1000:[1e-5, 5e-5,1e-6, 5e-7, 1e-7]
# }

# for lr in learning_rates:
#     for d in num_d_steps:
#         for s in stepsizes[d]:
#             for mode in time_schedules:
                
#                     # later TODO mem optimization -> mixed precision, gradient checkpointing, multiGPU 
#                     projected_w_steps = projector.project(
#                         learning_rate=lr,
#                         num_images=12,
#                         num_steps=100,
#                         diffusion_time_schedule=mode,
#                         num_diffusion_steps_per_step = d,
#                         diffusion_step_size = s
#                     )

#                     print (f'Elapsed: {(perf_counter()-start_time):.1f} s')
#                     create_video(projected_w_steps, projector, num_rows=3, outdir="out2", name=f"diff-lr{lr}-dfsteps{s}-mode{mode}-innersteps{d}.mp4")
#                     del(projected_w_steps)

learning_rates = [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
time_schedules = ["mini_end"]
init_ts = [0.2, 0.1, 0.05, 0.01, 0.005]
num_d_steps = [10, 100, 1000]
optimizers = [True, False]

for init_t in init_ts:
    for lr in learning_rates:
        for d in num_d_steps:
            for mode in time_schedules:
                
                    # later TODO mem optimization -> mixed precision, gradient checkpointing, multiGPU 
                    projected_w_steps = projector.project(
                        learning_rate=lr,
                        num_images=12,
                        num_steps=100,
                        diffusion_time_schedule=mode,
                        num_diffusion_steps_per_step = d,
                        # diffusion_step_size = s,
                        mini_end_init_t = init_t,
                    )

                    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')
                    create_video(projected_w_steps, projector, num_rows=3, outdir="out4", name=f"diff-initt{init_t}-lr{lr}-mode{mode}-innersteps{d}.mp4")
                    del(projected_w_steps)

print (f'Elapsed: {(perf_counter()-start_time):.1f} s')
