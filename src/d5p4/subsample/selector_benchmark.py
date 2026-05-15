"""Subsample benchmark: compare selector quality, validity, and latency."""

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
N_TRIALS = 1000
N_GROUPS = 8
GROUP_SIZE = 8
TOTAL_ITEMS = N_GROUPS * GROUP_SIZE

KWARGS = {
    "_w_interaction": 10.0,
    "_temperature": 1e-4,
    "_diversity_alpha": 10.0,
    "_kernel_type": "cosine",
}

IMPLEMENTED_METHODS = [
    ("dpp", False),
    ("exhaustive", True),
    ("greedy_map", False),
    ("greedy_map", True),
    ("greedy_beam", False),
    ("greedy_beam", True),
    ("diverse_beam", False),
    ("diverse_beam", True),
    ("random", False),
    ("random", True),
]


def main():  # noqa: PLR0915
    """Run a fixed-size benchmark across implemented selector methods."""
    print("Partition Sampler Benchmark")
    print(f"Parameters: k={N_GROUPS} groups, n={GROUP_SIZE} items/group, B={TOTAL_ITEMS} total")
    print(f"Running {N_TRIALS} trials on device: {DEVICE}")
    print("-" * 60)

    results = {
        f"{m} (Transv: {t})": {"log_dets": [], "valid": [], "times": [], "maes": []} for m, t in IMPLEMENTED_METHODS
    }

    base_config = Config(method="dpp", transversal=False, group_size=GROUP_SIZE, n_groups=N_GROUPS, **KWARGS)
    base_selector = get_subsample_selector(config=base_config)
    all_selectors = {}
    for method, transversal in IMPLEMENTED_METHODS:
        config = Config(method=method, transversal=transversal, group_size=GROUP_SIZE, n_groups=N_GROUPS, **KWARGS)
        selector = get_subsample_selector(config)
        selector.forward = torch.compile(selector.forward)
        all_selectors[(method, transversal)] = selector

    for _ in tqdm(range(N_TRIALS), desc="Trials"):
        cache = make_random_cache(TOTAL_ITEMS)

        kernel = base_selector.compute_kernel(cache)
        assert kernel is not None
        kernel_np = kernel.detach().cpu().numpy()

        trial_log_dets = {}

        for method, transversal in IMPLEMENTED_METHODS:
            name = f"{method} (Transv: {transversal})"
            selector = all_selectors[(method, transversal)]

            start = perf_counter()
            indices = indices_to_list(selector.subsample(cache))
            elapsed = perf_counter() - start

            log_det = compute_log_det_numpy(kernel_np, indices)
            trial_log_dets[name] = log_det

            results[name]["times"].append(elapsed)
            results[name]["log_dets"].append(log_det)
            results[name]["valid"].append(is_valid_transversal(indices, N_GROUPS, GROUP_SIZE))

        oracle_key = "exhaustive (Transv: True)"
        oracle_val = trial_log_dets.get(oracle_key, -np.inf)

        for name, val in trial_log_dets.items():
            if oracle_val > -np.inf and val > -np.inf:
                results[name]["maes"].append(abs(val - oracle_val))
            else:
                results[name]["maes"].append(np.nan)

    # Report
    print("\n" + "=" * 115)
    print("           --- Comparison Results ---")
    print("=" * 115)
    print(
        f"{'Method':<35} | {'Avg. Log-Det':>15} | {'Std. Log-Det':>15} | "
        f"{'MAE (Oracle)':>15} | {'Validity (%)':>13} | {'Avg. Time (s)':>15}",
    )
    print("-" * 130)

    for name, res in results.items():
        avg = np.mean(res["log_dets"])
        std = np.std(res["log_dets"])
        try:
            mae = np.nanmean(res["maes"])
        except Exception:
            mae = np.nan
        valid = np.mean(res["valid"]) * 100
        time = np.mean(res["times"])
        print(f"{name:<35} | {avg:>15.4f} | {std:>15.4f} | {mae:>15.6f} | {valid:>12.1f}% | {time:>15.6f}")

    print("-" * 130)
    print("\n'Avg. Log-Det': Higher is better (excludes invalid/singular results). 'MAE': Lower is better.")


if __name__ == "__main__":
    main()
