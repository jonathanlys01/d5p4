"""
Best-of-N sampling for MDLM.
Generates N samples and selects the one with minimum U_denoise.
"""

import numpy as np
import torch
from typing import List, Tuple


def set_seeds(seed_value: int):
    """Set seeds for reproducible sampling."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)


def bon_sampling(
    model,
    num_particles: int,
    num_steps: int,
    particle_seeds: List[int],
    eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Best-of-N sampling - generate N samples and select best based on U_denoise.
    
    Args:
        model: MDLM model
        num_particles: number of samples to generate (N)
        num_steps: number of diffusion steps
        particle_seeds: list of seeds for each particle
        eps: minimum time value
    
    Returns:
        best_sample: sample with minimum U_denoise
        all_samples: all N generated samples
        logs: dictionary containing detailed logs
    """
    device = model.device
    seq_len = model.config.model.length
    
    all_samples = []
    all_u_denoise = []
    
    # Generate N samples
    for i, seed in enumerate(particle_seeds):
        set_seeds(seed)
        particle = model._sample_prior(1, seq_len).to(device)
        
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        
        entropy_trajectory = []
        p_x0_cache = None
        
        for step in range(num_steps):
            t_val = timesteps[step]
            t = t_val * torch.ones(1, 1, device=device)
            
            current_entropy = model._compute_mask_conditional_entropy(particle, t)
            entropy_trajectory.append(current_entropy.item())
            
            if model.sampler == 'ddpm':
                particle = model._ddpm_update(particle, t, dt)
            elif model.sampler == 'ddpm_cache':
                p_x0_cache, particle_next = model._ddpm_caching_update(
                    particle, t, dt, p_x0=p_x0_cache)
                if (not torch.allclose(particle_next, particle) or model.time_conditioning):
                    p_x0_cache = None
                particle = particle_next
            else:
                particle = model._analytic_update(particle, t, dt)
        
        if model.config.sampling.noise_removal:
            t_final = timesteps[-1] * torch.ones(particle.shape[0], 1, device=device)
            final_entropy = model._compute_mask_conditional_entropy(particle, t_final)
            entropy_trajectory.append(final_entropy.item())
            
            if model.sampler == 'analytic':
                particle = model._denoiser_update(particle, t_final)
            else:
                unet_conditioning = model.noise(t_final)[0]
                particle = model.forward(particle, unet_conditioning).argmax(dim=-1)
        
        time_points = timesteps[:len(entropy_trajectory)].cpu().numpy()
        u_denoise = abs(np.trapz(entropy_trajectory, x=time_points))
        
        all_samples.append(particle)
        all_u_denoise.append(u_denoise)
    
    # Select best sample
    best_idx = np.argmin(all_u_denoise)
    best_sample = all_samples[best_idx]
    all_samples_tensor = torch.cat(all_samples, dim=0)
    
    logs = {
        'best_idx': best_idx,
        'all_u_denoise': all_u_denoise,
        'best_u_denoise': all_u_denoise[best_idx],
        'particle_seeds': particle_seeds
    }
    
    return best_sample, all_samples_tensor, logs

