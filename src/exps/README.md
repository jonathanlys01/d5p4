# Experiments Module

This directory contains scripts for running various experiments, sweeps, and baseline evaluations for the D5P4 project.

## Main Experiment Scripts

- **`cfg_exp.py`**: Runs sweeps over Classifier-Free Guidance (CFG) scales to evaluate the trade-off between quality and diversity.
- **`interaction_exp.py`**: Evaluates the effect of the interaction weight (`_w_interaction`) on DPP-based subsampling.
- **`baseline_llada.py` / `baseline_mdlm.py`**: Simple scripts to run baseline generations for LLaDA and MDLM models without advanced subsampling.
- **`llada.py` / `main.py`**: Entry points for standard generation runs with full configuration support.

## Correlation Analysis

The `correlation/` subdirectory contains scripts to analyze the relationship between different metrics:
- **`embeddings_*.py`**: Extract and compare embeddings from different models (MDLM, LLaDA).
- **`likelihood_*.py`**: Compute token-level likelihoods for correlation study.

## Running Experiments

Most scripts use the centralized configuration system in `src/config.py`. You can override parameters from the command line:

```bash
python src/exps/cfg_exp.py n_runs=10 batch_size=4
```

Results are typically saved in the `results/` directory as JSON files containing both generated text and computed metrics.
