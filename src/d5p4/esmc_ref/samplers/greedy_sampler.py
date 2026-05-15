"""
Greedy entropy minimization sampling for MDLM.
Uses beam search to select candidates with lowest entropy at each step.
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


def greedy_sampling(
    model,
    num_steps: int,
    num_candidates: int,
    beam_size: int,
    seed: int,
    eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Greedy entropy minimization with beam search.
    
    At each step, generates multiple candidates for each beam and selects
    the top beam_size candidates with the lowest cumulative entropy.
    
    Args:
        model: MDLM model
        num_steps: number of diffusion steps
        num_candidates: number of candidates to generate at each step for each beam
        beam_size: number of beams to maintain (1 = greedy, >1 = beam search)
        seed: random seed for reproducibility
        eps: minimum time value
    
    Returns:
        best_sample: best generated sample
        all_beams: all final beam samples
        logs: dictionary containing detailed tracking information
    """
    device = model.device
    seq_len = model.config.model.length
    
    # Validate parameters
    if beam_size < 1:
        raise ValueError(f"beam_size must be >= 1, got {beam_size}")
    if beam_size > num_candidates:
        raise ValueError(f"beam_size ({beam_size}) cannot be larger than num_candidates ({num_candidates})")
    
    # Initialize beams from prior
    beams = []
    for beam_idx in range(beam_size):
        set_seeds(seed + beam_idx)
        initial_sample = model._sample_prior(1, seq_len).to(device)
        beams.append({
            'sample': initial_sample,
            'cumulative_entropy': 0.0,
            'entropy_trajectory': []
        })
    
    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
    dt = (1 - eps) / num_steps
    
    # Detailed logs for analysis
    logs = {
        'candidate_entropies': [],
        'selected_indices': [],
        'beam_entropies': [],
        'timesteps': []
    }
    
    # Beam search generation loop
    for step in range(num_steps):
        t_val = timesteps[step]
        t = t_val * torch.ones(1, 1, device=device)
        logs['timesteps'].append(t_val.item())
        
        # Store current beam entropies
        current_beam_entropies = []
        for beam in beams:
            current_entropy = model._compute_mask_conditional_entropy(beam['sample'], t)
            beam['entropy_trajectory'].append(current_entropy.item())
            current_beam_entropies.append(current_entropy.item())
        logs['beam_entropies'].append(current_beam_entropies)
        
        # Generate candidates for all beams
        all_candidates = []
        all_candidate_entropies = []
        all_candidate_info = []
        
        for beam_idx, beam in enumerate(beams):
            current_sample = beam['sample']
            
            # Generate multiple candidate next states for this beam
            for cand_idx in range(num_candidates):
                # Set different seed for each candidate
                candidate_seed = seed + step * (beam_size * num_candidates) + beam_idx * num_candidates + cand_idx
                set_seeds(candidate_seed)
                
                # Generate next state from current beam sample
                if model.sampler == 'ddpm':
                    candidate = model._ddpm_update(current_sample, t, dt)
                elif model.sampler == 'ddpm_cache':
                    candidate = model._ddpm_update(current_sample, t, dt)
                else:
                    candidate = model._analytic_update(current_sample, t, dt)
                
                # Compute entropy for this candidate at next timestep
                t_next = timesteps[step + 1] * torch.ones(1, 1, device=device)
                with torch.no_grad():
                    cand_entropy = model._compute_mask_conditional_entropy(candidate, t_next)
                
                # Compute cumulative entropy for this path
                cumulative_entropy = beam['cumulative_entropy'] + cand_entropy.item()
                
                all_candidates.append(candidate)
                all_candidate_entropies.append(cumulative_entropy)
                all_candidate_info.append({
                    'beam_idx': beam_idx,
                    'cand_idx': cand_idx,
                    'step_entropy': cand_entropy.item(),
                    'cumulative_entropy': cumulative_entropy,
                    'parent_trajectory': beam['entropy_trajectory'].copy()
                })
        
        # Select top beam_size candidates with lowest cumulative entropy
        sorted_indices = np.argsort(all_candidate_entropies)[:beam_size]
        
        # Create new beams from selected candidates
        new_beams = []
        selected_info = []
        for idx in sorted_indices:
            info = all_candidate_info[idx]
            new_beam = {
                'sample': all_candidates[idx],
                'cumulative_entropy': info['cumulative_entropy'],
                'entropy_trajectory': info['parent_trajectory']
            }
            new_beams.append(new_beam)
            selected_info.append({
                'beam_idx': info['beam_idx'],
                'cand_idx': info['cand_idx'],
                'entropy': info['step_entropy'],
                'cumulative_entropy': info['cumulative_entropy']
            })
        
        beams = new_beams
        
        # Log the selection
        logs['candidate_entropies'].append(all_candidate_entropies)
        logs['selected_indices'].append(selected_info)
    
    # Select the best beam (lowest cumulative entropy)
    best_beam_idx = np.argmin([beam['cumulative_entropy'] for beam in beams])
    best_beam = beams[best_beam_idx]
    current_sample = best_beam['sample']
    entropy_trajectory = best_beam['entropy_trajectory']
    
    # Final denoising step
    if model.config.sampling.noise_removal:
        t_final = timesteps[-1] * torch.ones(current_sample.shape[0], 1, device=device)
        final_entropy = model._compute_mask_conditional_entropy(current_sample, t_final)
        entropy_trajectory.append(final_entropy.item())
        
        if model.sampler == 'analytic':
            current_sample = model._denoiser_update(current_sample, t_final)
        else:
            unet_conditioning = model.noise(t_final)[0]
            current_sample = model.forward(current_sample, unet_conditioning).argmax(dim=-1)
    
    logs['timesteps'].append(eps)
    logs['best_beam_idx'] = best_beam_idx
    logs['final_beam_entropies'] = [beam['cumulative_entropy'] for beam in beams]
    
    # Compute U_denoise for best beam
    time_points = timesteps[:len(entropy_trajectory)].cpu().numpy()
    u_denoise = abs(np.trapz(entropy_trajectory, x=time_points))
    logs['best_u_denoise'] = u_denoise
    
    # Collect all final beam samples
    all_beam_samples = torch.cat([beam['sample'] for beam in beams], dim=0)
    
    logs['num_candidates'] = num_candidates
    logs['beam_size'] = beam_size
    logs['num_steps'] = num_steps
    logs['entropy_trajectory'] = entropy_trajectory
    
    return current_sample, all_beam_samples, logs

