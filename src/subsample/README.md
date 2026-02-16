# Subsample Module

This module implements various subsampling methods used to select high-quality and diverse subsets from a pool of generated text candidates.

## Subsampling Methods

### Baseline
Does not apply any subsampling. It simply passes through the candidates. Only compatible with `group_size=1`.

### Random
Implements random subsampling. Useful as a baseline for diversity metrics.

### Beam Search
- **Naive Beam**: Selects the top-$k$ candidates based on their log-probability/score.
- **Diverse Beam**: Uses Maximal Marginal Relevance (MMR) to select a diverse set of candidates. Controlled by `_diversity_alpha`.

### DPP Selector
Uses the `dppy` library to implement Determinantal Point Process (DPP) based subsampling. 

### Greedy MAP
Implements a fast greedy MAP inference algorithm for DPPs. This is an efficient approximation of the optimal DPP selection, based on the paper ["Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity"](https://arxiv.org/abs/1709.05135).

### Exhaustive
Performs an exhaustive search over all possible subsets to find the one that maximizes the subdeterminant. Only practical for very small pool sizes because of its combinatorial complexity.

## Partitioned (Transversal) Sampling

The project supports **Transversal Selection** (controlled by `transversal=True` in `Config`). In this mode, the pool is partitioned into groups, and the selector must choose exactly one item from each group. This is particularly useful for maintaining samples from different "beams" or prompts while ensuring diversity across them.

## Key Configuration Parameters

- `method`: The subsampling algorithm to use (e.g., `greedy_map`, `dpp`, `random`).
- `transversal`: Boolean flag to enable/disable partitioned selection.
- `_w_interaction`: Weight for the diversity term in DPP methods (higher means more diversity).
- `_kernel_type`: Similarity kernel to use (default: `cosine`, fallback: `rbf`).
- `_score_method`: Quality score metric (`entropy` or `self-certainty`).

## Benchmarking

Use `src/subsample/test.py` to compare methods:
```bash
python src/subsample/test.py
```
This script evaluates algorithms based on:
1. **Log-Determinant Quality**: A measure of group diversity.
2. **MAE (Oracle)**: Mean Absolute Error compared to exhaustive search.
3. **Validity**: Percentage of selections that satisfy transversal constraints.
4. **Time**: Execution speed.
