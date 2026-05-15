"""Subsample benchmark: rank selector quality against random valid partitions."""

from time import perf_counter

import numpy as np
import torch

from d5p4.config import Config
from d5p4.subsample import get_subsample_selector
from d5p4.subsample.benchmark_utils import (
    DEVICE,
    compute_log_det_numpy,
    indices_to_list,
    is_valid_transversal,
    make_random_cache,
)
from d5p4.utils import tqdm


# Configuration
REFERENCE_POOL_SIZE = 100_000
N_TRIALS = 100
N_GROUPS = 8
GROUP_SIZE = 8
TOTAL_ITEMS = N_GROUPS * GROUP_SIZE

KWARGS = {
    "_w_interaction": 10.0,
    "_temperature": 1e-4,
    "_diversity_alpha": 10.0,
    "_kernel_power": 3,
    "_kernel_type": "cosine",
}

IMPLEMENTED_METHODS = [
    ("exhaustive", True),
    ("greedy_map", True),
    ("greedy_beam", True),
    ("diverse_beam", True),
    ("random", True),
]


def generate_reference_scores(kernel: torch.Tensor, n_samples: int, n_groups: int, group_size: int) -> torch.Tensor:
    """Generate log-determinant scores for random valid partitions."""
    group_offsets = torch.arange(0, n_groups * group_size, group_size, device=DEVICE)
    random_shifts = torch.randint(0, group_size, (n_samples, n_groups), device=DEVICE)
    indices = random_shifts + group_offsets.unsqueeze(0)

    row_idx = indices.unsqueeze(2).expand(-1, -1, n_groups)
    col_idx = indices.unsqueeze(1).expand(-1, n_groups, -1)
    sub_matrices = kernel[row_idx, col_idx]

    sign, logdet = torch.linalg.slogdet(sub_matrices)
    return torch.where(sign > 0, logdet, torch.tensor(float("-inf"), device=DEVICE))


def compute_ranks(method_scores: list, reference_scores: torch.Tensor) -> list:
    """Compute percentile ranks for method scores against reference distribution."""
    scores_t = torch.tensor(method_scores, device=DEVICE, dtype=reference_scores.dtype)
    comparisons = reference_scores.unsqueeze(1) < scores_t.unsqueeze(0)
    counts = comparisons.sum(dim=0).float()
    return ((counts / reference_scores.size(0)) * 100.0).tolist()


def main():
    """Run a fixed-size percentile-rank benchmark across transversal selectors."""
    print("Rank-Based Partition Sampler Benchmark")
    print(f"Parameters: k={N_GROUPS}, n={GROUP_SIZE}")
    print(f"Reference Pool: {REFERENCE_POOL_SIZE} samples")
    print("-" * 60)

    results = {
        f"{m} (Transv: {t})": {"log_dets": [], "valid": [], "times": [], "ranks": []} for m, t in IMPLEMENTED_METHODS
    }

    base_config = Config(method="dpp", transversal=False, group_size=GROUP_SIZE, n_groups=N_GROUPS, **KWARGS)
    base_selector = get_subsample_selector(config=base_config)
    all_selectors = {
        (m, t): get_subsample_selector(
            Config(method=m, transversal=t, group_size=GROUP_SIZE, n_groups=N_GROUPS, **KWARGS),
        )
        for m, t in IMPLEMENTED_METHODS
    }

    for _ in tqdm(range(N_TRIALS), desc="Trials"):
        cache = make_random_cache(TOTAL_ITEMS)

        kernel = base_selector.compute_kernel(cache)
        assert kernel is not None
        kernel_np = kernel.detach().cpu().numpy()
        ref_scores = generate_reference_scores(kernel, REFERENCE_POOL_SIZE, N_GROUPS, GROUP_SIZE)

        trial_scores, trial_keys = [], []

        for method, transversal in IMPLEMENTED_METHODS:
            name = f"{method} (Transv: {transversal})"
            selector = all_selectors[(method, transversal)]

            start = perf_counter()
            indices = indices_to_list(selector.subsample(cache))
            elapsed = perf_counter() - start

            results[name]["times"].append(elapsed)
            is_valid = is_valid_transversal(indices, N_GROUPS, GROUP_SIZE)
            results[name]["valid"].append(is_valid)

            log_det = compute_log_det_numpy(kernel_np, indices) if is_valid else -np.inf
            results[name]["log_dets"].append(log_det)

            trial_keys.append(name)
            trial_scores.append(log_det)

        ranks = compute_ranks(trial_scores, ref_scores)
        for name, rank in zip(trial_keys, ranks):
            results[name]["ranks"].append(rank)

    # Report
    print("\n" + "=" * 100)
    print(" --- Comparison Results ---")
    print("=" * 100)
    print(f"{'Method':<30} | {'Avg. Rank (%)':<15} | {'Avg. Log-Det':<15} | {'Validity':<10} | {'Time (s)':<10}")
    print("-" * 100)

    for name, res in results.items():
        avg_rank = np.mean(res["ranks"])
        valid_log_dets = [x for x in res["log_dets"] if x > -1e9]
        avg_log_det = np.mean(valid_log_dets) if valid_log_dets else -np.inf
        valid_pct = np.mean(res["valid"]) * 100
        avg_time = np.mean(res["times"])
        print(f"{name:<30} | {avg_rank:>14.4f}% | {avg_log_det:>15.4f} | {valid_pct:>9.1f}% | {avg_time:>10.5f}")


if __name__ == "__main__":
    main()
