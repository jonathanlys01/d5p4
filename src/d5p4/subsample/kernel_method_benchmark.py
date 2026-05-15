"""Subsample benchmark: compare additive and multiplicative DPP kernels."""

from itertools import product

import numpy as np
import torch

from d5p4.config import Cache, Config
from d5p4.subsample import get_subsample_selector
from d5p4.subsample.benchmark_utils import DEVICE, compute_cosine_similarity, compute_log_det_torch, indices_to_list


# Configuration
N_TRIALS = 200
N_GROUPS = 4
GROUP_SIZE = 4
TOTAL_ITEMS = N_GROUPS * GROUP_SIZE


def get_all_transversals(n_groups: int, group_size: int) -> list[tuple]:
    """Generate all valid transversal selections (one item per group)."""
    groups = [list(range(g * group_size, (g + 1) * group_size)) for g in range(n_groups)]
    return list(product(*groups))


def compute_rank(ref_kernel: torch.Tensor, selected_indices: list, all_transversals: list) -> int:
    """Compute rank of selected indices on REFERENCE kernel (1 = best)."""
    selected_log_det = compute_log_det_torch(ref_kernel, selected_indices)
    # Count how many transversals have higher log-det on reference kernel
    better_count = sum(
        1 for t in all_transversals if compute_log_det_torch(ref_kernel, list(t)) > selected_log_det + 1e-9
    )
    return better_count + 1


def main():
    """Compare kernel methods by rank on a pure cosine-similarity oracle."""
    print("Kernel Method Comparison: Selection Quality on Reference Similarity")
    print(f"k={N_GROUPS} groups, n={GROUP_SIZE} items/group, {N_TRIALS} trials")
    print(f"Total transversals: {GROUP_SIZE**N_GROUPS}")
    print("Evaluation: Rank on pure cosine similarity kernel (no quality weighting)\n")

    all_transversals = get_all_transversals(N_GROUPS, GROUP_SIZE)
    w_values = [0.1, 0.25, 0.5, 0.75, 1.0]

    print(f"{'w_interaction':>12} | {'Additive Rank':>15} | {'Multiplicative Rank':>20} | {'Winner':>12}")
    print("-" * 70)

    for w in w_values:
        ranks: dict[str, list[int]] = {"additive": [], "multiplicative": []}

        for trial in range(N_TRIALS):
            # Generate same data for both methods in this trial
            torch.manual_seed(trial)
            embeddings = torch.randn(TOTAL_ITEMS, 8, 32, device=DEVICE)
            lpx = torch.randn(TOTAL_ITEMS, 8, 50, device=DEVICE)
            cache = Cache(embeddings=embeddings, log_p_x0=lpx)

            # Reference kernel: pure similarity (what we actually want to maximize)
            ref_kernel = compute_cosine_similarity(embeddings)

            for method in ["additive", "multiplicative"]:
                config = Config(
                    method="greedy_map",
                    transversal=True,
                    group_size=GROUP_SIZE,
                    n_groups=N_GROUPS,
                    _kernel_method=method,
                    _w_interaction=w,
                    _kernel_type="cosine",
                )
                selector = get_subsample_selector(config)

                # Selection uses method's kernel
                indices = indices_to_list(selector.subsample(cache))

                # Rank on REFERENCE kernel (pure similarity)
                rank = compute_rank(ref_kernel, indices, all_transversals)
                ranks[method].append(rank)

        add_avg = np.mean(ranks["additive"])
        mul_avg = np.mean(ranks["multiplicative"])
        winner = "additive" if add_avg < mul_avg else "multiplicative" if mul_avg < add_avg else "tie"

        print(f"{w:>12.2f} | {add_avg:>15.2f} | {mul_avg:>20.2f} | {winner:>12}")

    print("-" * 70)
    print("\nLower rank = selection is more diverse on pure similarity kernel.")


if __name__ == "__main__":
    main()
