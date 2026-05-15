# Sweep Experiments

This directory contains scripts for running hyperparameter sweeps using Optuna. Most sweeps optimize for **Perplexity** and **Average Cosine Similarity (ACS)**.

## Available Sweeps

### `map_plus.py` (MAP+)
Optimizes the interaction weight for the **Additive DPP Kernel**.
- **Kernel**: `K = w_interaction * S + diag(scores)`
- **Parameters**:
  - `w_interaction`: [0.0, 10.0]

### `map_times.py` (MAP*)
Optimizes the interaction weight for the **Multiplicative DPP Kernel** (Quality-Diversity decomposition).
- **Kernel**: `L_ij = q_i * S_ij * q_j` where `q_i = exp(score_i * w_interaction)`
- **Parameters**:
  - `w_interaction`: [0.0, 50.0]

### `cat.py` (CAT)
Optimizes the Softmax temperature for token sampling across all diffusion steps.
- **Parameters**:
  - `cat_temperature`: [0.1, 2.0]

### `divbs.py` (DivBS)
Optimizes the diversity penalty for Diverse Beam Search.
- **Note**: Requires setting `method="diverse_beam"` in the global configuration.
- **Parameters**:
  - `_diversity_alpha`: [0.0, 10.0]

## Running a Sweep

Sweeps are designed to be run in distributed environments. The `run_sweep` utility handles Optuna study creation and multi-worker synchronization.

```bash
# Example: Run MAP* sweep with 4 groups
python -m d5p4.exps.sweeps.map_times --n_groups 4 --group_size 2
```
