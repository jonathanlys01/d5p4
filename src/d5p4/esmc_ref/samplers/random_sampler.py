"""
Random sampling for MDLM.
Simply generates a sample using the model's default sampling procedure.
"""

import numpy as np
import torch
from typing import Tuple


def set_seeds(seed_value: int):
    """Set seeds for reproducible sampling."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)


def random_sampling(
    model,
    num_steps: int,
    seed: int,
    eps: float = 1e-5
) -> Tuple[torch.Tensor, dict]:
    """
    Random sampling - standard MDLM sampling procedure.
    
    Args:
        model: MDLM model
        num_steps: number of diffusion steps
        seed: random seed for reproducibility
        eps: minimum time value
    
    Returns:
        sample: generated sample tensor
        logs: dictionary containing sampling logs
    """
    set_seeds(seed)
    
    device = model.device
    seq_len = model.config.model.length
    
    # Initialize from prior
    sample = model._sample_prior(1, seq_len).to(device)
    
    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
    dt = (1 - eps) / num_steps
    
    entropy_trajectory = []
    p_x0_cache = None
    
    # Standard sampling loop
    for step in range(num_steps):
        t_val = timesteps[step]
        t = t_val * torch.ones(1, 1, device=device)
        
        current_entropy = model._compute_mask_conditional_entropy(sample, t)
        entropy_trajectory.append(current_entropy.item())
        
        if model.sampler == 'ddpm':
            sample = model._ddpm_update(sample, t, dt)
        elif model.sampler == 'ddpm_cache':
            p_x0_cache, sample_next = model._ddpm_caching_update(
                sample, t, dt, p_x0=p_x0_cache)
            if (not torch.allclose(sample_next, sample) or model.time_conditioning):
                p_x0_cache = None
            sample = sample_next
        else:
            sample = model._analytic_update(sample, t, dt)
    
    # Final denoising
    if model.config.sampling.noise_removal:
        t_final = timesteps[-1] * torch.ones(sample.shape[0], 1, device=device)
        final_entropy = model._compute_mask_conditional_entropy(sample, t_final)
        entropy_trajectory.append(final_entropy.item())
        
        if model.sampler == 'analytic':
            sample = model._denoiser_update(sample, t_final)
        else:
            unet_conditioning = model.noise(t_final)[0]
            sample = model.forward(sample, unet_conditioning).argmax(dim=-1)
    
    time_points = timesteps[:len(entropy_trajectory)].cpu().numpy()
    u_denoise = abs(np.trapz(entropy_trajectory, x=time_points))
    
    logs = {
        'entropy_trajectory': entropy_trajectory,
        'u_denoise': u_denoise,
        'seed': seed
    }
    
    return sample, logs

