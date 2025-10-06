import torch
import torch.nn.functional as F

import genesis as gs

def _sample_latents(num_latents, latent_dim, device, mean=0.0, std=1.0):
    """Sample latents from a normal distribution."""
    z = mean + std*torch.randn((num_latents, latent_dim), dtype=gs.tc_float).to(device)
    z = F.normalize(z, p=2, dim=1)
    return z