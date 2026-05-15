"""
SMC (Sequential Monte Carlo) sampling for MDLM.
Also known as FK steering - uses entropy-based rewards and resampling.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple


def set_seeds(seed_value: int):
    """Set seeds for reproducible sampling."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)


def compute_entropy_reward(model, particles: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy-based reward for particles.
    Lower entropy = higher reward (better quality).
    """
    with torch.no_grad():
        entropies = model._compute_mask_conditional_entropy(particles, t)
        max_entropy = torch.log2(torch.tensor(model.vocab_size, dtype=torch.float32, device=particles.device))
        rewards = max_entropy - entropies
        rewards = torch.clamp(rewards / max_entropy, 0.0, 1.0)
    return rewards


def compute_entropies_and_rewards(
    model,
    particles: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute entropies and normalized rewards for a batch of particles."""
    with torch.no_grad():
        entropies = model._compute_mask_conditional_entropy(particles, t)
        max_entropy = torch.log2(
            torch.tensor(model.vocab_size, dtype=torch.float32, device=particles.device)
        )
        rewards = max_entropy - entropies
        rewards = torch.clamp(rewards / max_entropy, 0.0, 1.0)
    return entropies, rewards


def compute_potential_scores(
    rewards: torch.Tensor, 
    lambda_weight: float, 
    potential_type: str = 'max'
) -> torch.Tensor:
    """Compute potential function scores for resampling."""
    if potential_type == 'max':
        scores = torch.exp(lambda_weight * rewards)
    elif potential_type == 'mean':
        scores = lambda_weight * rewards + 1.0
    else:
        raise ValueError(f"Unknown potential type: {potential_type}")
    return scores


def smc_sampling(
    model,
    num_particles: int,
    num_steps: int,
    resample_interval: int,
    lambda_weight: float,
    potential_type: str,
    particle_seeds: List[int],
    eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    SMC sampling - Sequential Monte Carlo with entropy-based rewards.
    
    Args:
        model: MDLM model
        num_particles: number of particles to maintain
        num_steps: total number of diffusion steps
        resample_interval: how often to resample (every N steps)
        lambda_weight: temperature parameter for potential function
        potential_type: type of potential function ('max' or 'mean')
        particle_seeds: list of seeds for initializing particles
        eps: minimum time value
    
    Returns:
        best_sample: best generated sample
        all_particles: final particles
        logs: dictionary containing detailed tracking information
    """
    device = model.device
    seq_len = model.config.model.length
    
    # Initialize particles using seeds
    particles = []
    for seed in particle_seeds:
        set_seeds(seed)
        particle = model._sample_prior(1, seq_len).to(device)
        particles.append(particle)
    particles = torch.cat(particles, dim=0)
    
    # Initialize tracking
    logs = {
        'particle_entropies': np.zeros((num_steps + 1, num_particles)),
        'particle_rewards': np.zeros((num_steps + 1, num_particles)),
        'resampling_steps': [],
        'resampling_info': [],
        'timesteps': [],
        'particle_lineage': [list(range(num_particles))],
    }
    
    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
    dt = (1 - eps) / num_steps
    
    # SMC main loop
    for step in range(num_steps):
        t_val = timesteps[step]
        t = t_val * torch.ones(num_particles, 1, device=device)
        logs['timesteps'].append(t_val.item())
        
        # Compute entropy and rewards
        current_entropies, current_rewards = compute_entropies_and_rewards(model, particles, t)
        logs['particle_entropies'][step] = current_entropies.detach().cpu().numpy()
        logs['particle_rewards'][step] = current_rewards.detach().cpu().numpy()
        
        # Generate next state for all particles
        if model.sampler == 'ddpm':
            particles = model._ddpm_update(particles, t, dt)
        elif model.sampler == 'ddpm_cache':
            particles = model._ddpm_update(particles, t, dt)
        else:
            particles = model._analytic_update(particles, t, dt)
        
        # Resampling
        if (step + 1) % resample_interval == 0 and step < num_steps - 1:
            t_resample = timesteps[step + 1] * torch.ones(num_particles, 1, device=device)
            
            resample_entropies, resample_rewards = compute_entropies_and_rewards(
                model, particles, t_resample
            )
            
            potential_scores = compute_potential_scores(
                resample_rewards, lambda_weight, potential_type
            )
            
            probabilities = F.softmax(potential_scores, dim=0)
            selected_indices = torch.multinomial(probabilities, num_particles, replacement=True)
            particles = particles[selected_indices]
            
            current_lineage = logs['particle_lineage'][-1]
            new_lineage = [current_lineage[idx.item()] for idx in selected_indices]
            logs['particle_lineage'].append(new_lineage)
            
            logs['resampling_steps'].append(step + 1)
            logs['resampling_info'].append({
                'selected_indices': selected_indices.cpu().numpy(),
                'probabilities': probabilities.detach().cpu().numpy(),
                'pre_resample_entropies': resample_entropies.detach().cpu().numpy(),
                'pre_resample_rewards': resample_rewards.detach().cpu().numpy(),
                'particle_lineage': new_lineage.copy()
            })
        else:
            if step == 0 or (step + 1) % resample_interval != 0:
                logs['particle_lineage'].append(logs['particle_lineage'][-1].copy())
    
    logs['timesteps'].append(eps)
    
    # Final denoising
    if model.config.sampling.noise_removal:
        t_final = timesteps[-1] * torch.ones(particles.shape[0], 1, device=device)
        
        final_entropies, final_rewards = compute_entropies_and_rewards(model, particles, t_final)
        logs['particle_entropies'][num_steps] = final_entropies.detach().cpu().numpy()
        logs['particle_rewards'][num_steps] = final_rewards.detach().cpu().numpy()
        
        if model.sampler == 'analytic':
            particles = model._denoiser_update(particles, t_final)
        else:
            unet_conditioning = model.noise(t_final)[0]
            particles = model.forward(particles, unet_conditioning).argmax(dim=-1)
    else:
        t_final = eps * torch.ones(particles.shape[0], 1, device=device)
        
        final_entropies, final_rewards = compute_entropies_and_rewards(model, particles, t_final)
        logs['particle_entropies'][num_steps] = final_entropies.detach().cpu().numpy()
        logs['particle_rewards'][num_steps] = final_rewards.detach().cpu().numpy()
    
    # Select best sample
    t_final = eps * torch.ones(particles.shape[0], 1, device=device)
    final_rewards = compute_entropy_reward(model, particles, t_final)
    best_idx = torch.argmax(final_rewards)
    best_sample = particles[best_idx:best_idx+1]
    
    logs['final_rewards'] = final_rewards.detach().cpu().numpy()
    logs['best_particle_idx'] = best_idx.item()
    logs['num_particles'] = num_particles
    logs['num_steps'] = num_steps
    logs['resample_interval'] = resample_interval
    logs['lambda_weight'] = lambda_weight
    
    # Compute U_denoise for best particle
    best_particle_idx = logs['best_particle_idx']
    entropy_trajectory = logs['particle_entropies'][:, best_particle_idx]
    timesteps_array = np.array(logs['timesteps'])
    u_denoise = abs(np.trapz(entropy_trajectory, x=timesteps_array))
    logs['best_u_denoise'] = u_denoise
    
    return best_sample, particles, logs
