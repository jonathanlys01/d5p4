"""Subsample benchmark: scale selector quality and latency over larger grids."""

import csv
import time
from collections import defaultdict
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from d5p4.config import HIDDEN_SIZE_MDLM, Cache, Config
from d5p4.subsample import get_subsample_selector
from d5p4.subsample.benchmark_utils import (
    DEVICE,
    compute_cosine_similarity,
    compute_log_det_torch,
    indices_to_list,
    sync_if_cuda,
)


if TYPE_CHECKING:
    from d5p4.subsample import BaseSelector


# Configuration
N_TRIALS = 500
WARMUP_TRIALS = 10
SEQ_LEN = 8
HIDDEN_SIZE = HIDDEN_SIZE_MDLM
VOCAB_SIZE = 50

N_GROUPS_LIST = [4, 8, 16, 32, 64]
GROUP_SIZE_LIST = [4, 8, 16, 32, 64]
W_VALUES = np.logspace(np.log10(3), np.log10(99.9), num=15).tolist()  # 3 to 100 log scale

IMPLEMENTED_METHODS = [
    ("_greedy_map", True),  # non-triton implementation
    ("greedy_map", True),  # triton implementation
    ("greedy_beam", True),
    ("diverse_beam", True),
    ("dpp", False),
    ("random", True),
]

# ANSI colors
C_GM = "\033[96m"  # Cyan          — _greedy_map (non-triton)
C_GMt = "\033[36m"  # Dark Cyan     — greedy_map  (triton)
C_GB = "\033[93m"  # Yellow        — greedy_beam
C_DB = "\033[92m"  # Green         — diverse_beam
C_DPP = "\033[91m"  # Red           — dpp
C_R = "\033[95m"  # Magenta       — random
C_RST = "\033[0m"  # Reset


# Method metadata for dynamic reporting
METHOD_META = {
    "_greedy_map": {"label": "GM", "color": C_GM},
    "greedy_map": {"label": "GMt", "color": C_GMt},
    "greedy_beam": {"label": "GB", "color": C_GB},
    "diverse_beam": {"label": "DB", "color": C_DB},
    "dpp": {"label": "DPP", "color": C_DPP},
    "random": {"label": "R", "color": C_R},
}


def make_synthetic_cache(total_items: int, seed: int) -> Cache:
    """
    Generate realistic synthetic model outputs calibrated to match MDLM statistics:

    Embeddings (target):
      Mean ≈ -0.80, Std ≈ 41.0, range [-665, +608]
      Off-diagonal cosine similarity: mean ≈ 0.55, std ≈ 0.08

    Logits / quality (50258-vocab model adapted to VOCAB_SIZE=50):
      Entropy mean ≈ 2.54 (max for V=50: ln(50)≈3.91)
      Top-K mean sorted logits: Top1≈19.75, Top2≈17.0, Bottom≈-17.6

    Design:
      - Embeddings: shared prototype direction + small noise → high cosine sim.
        Heavy-tail via squaring, then rescale to hit target mean/std.
      - Logits: Zipfian rank bias with spacing ~2.5 + Gaussian noise, giving the
        observed sharpness profile and low entropy even for V=50.
    """
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)

    # ------------------------------------------------------------------
    # Embeddings: correlated heavy-tailed vectors
    # ------------------------------------------------------------------
    # A shared prototype per sequence position gives high inter-token cosine sim.
    # Mixing ratio ALPHA ≈ 0.58 yields off-diagonal cosine sim mean ≈ 0.55.
    ALPHA = 0.58
    prototype = torch.randn(1, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, generator=g)
    noise = torch.randn(total_items, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, generator=g)
    e_mixed = ALPHA * prototype + (1.0 - ALPHA) * noise

    # Heavy-tail transformation: squaring inflates tails while preserving sign structure.
    embeddings = torch.sign(e_mixed) * (e_mixed**2)

    # Rescale to target mean=-0.80, std=41.0 (observed MDLM embedding statistics).
    emb_flat = embeddings.reshape(-1, HIDDEN_SIZE).float()
    emb_mean = emb_flat.mean()
    emb_std = emb_flat.std()
    TARGET_EMB_MEAN = -0.80
    TARGET_EMB_STD = 41.0
    embeddings = (embeddings - emb_mean) / (emb_std + 1e-8) * TARGET_EMB_STD + TARGET_EMB_MEAN

    # ------------------------------------------------------------------
    # Logits: sharp Zipfian distribution calibrated to observed top-K profile
    # ------------------------------------------------------------------
    # Observed profile: Top1≈19.75, spacing ≈2.5/rank at top, Bottom≈-17.6.
    # Linear Zipf bias -k * ZIPF_SPACING + offset reproduces this slope.
    ZIPF_SPACING = 2.5  # logit drop per rank
    LOGIT_NOISE = 1.0  # Gaussian noise scale; keeps entropy from collapsing to 0

    logits = torch.randn(total_items, SEQ_LEN, VOCAB_SIZE, device=DEVICE, generator=g)
    zipf_bias = -torch.arange(VOCAB_SIZE, device=DEVICE).float() * ZIPF_SPACING + 19.75
    logits = logits * LOGIT_NOISE + zipf_bias

    log_p_x0 = F.log_softmax(logits, dim=-1)

    seq = torch.arange(total_items, device=DEVICE)
    return Cache(embeddings=embeddings, log_p_x0=log_p_x0, x=seq)


def compute_ranks_1_to_N(scores: list[float]) -> list[float]:
    """Compute average-tied ranks where 1 is best (highest score)."""

    score_to_indices = defaultdict(list)
    for i, s in enumerate(scores):
        score_to_indices[s].append(i)

    ranks = [0.0] * len(scores)
    current_rank = 1.0
    for s in sorted(score_to_indices.keys(), reverse=True):
        indices = score_to_indices[s]
        avg_rank = current_rank + (len(indices) - 1) / 2.0
        for idx in indices:
            ranks[idx] = avg_rank
        current_rank += len(indices)
    return ranks


def main():  # noqa: C901, PLR0912, PLR0915
    """Run a scaling benchmark across sizes, interaction weights, and methods."""
    print("Subsampling Methods Scaling Benchmark")
    print(f"Trials per setting: {N_TRIALS}")
    print(f"Warmup trials per setting: {WARMUP_TRIALS}")
    print("Diverse beam uses diversity_alpha = 10 * w_int")
    print(f"Synthetic shapes: embeddings=[B,{SEQ_LEN},{HIDDEN_SIZE}], log_p_x0=[B,{SEQ_LEN},{VOCAB_SIZE}]")
    print("Metrics: Raw average log-det on reference kernel, and average rank (1=best)\n")

    # Dynamic header generation
    headers = [f"{'N_G':>4}", f"{'N_I':>4}", f"{'w_int':>5}"]

    raw_h: list[str] = []
    rnk_h: list[str] = []
    t50_h: list[str] = []

    for method, _ in IMPLEMENTED_METHODS:
        meta = METHOD_META[method]
        lbl = meta["label"]
        clr = meta["color"]
        raw_h.append(f"{clr}{f'Raw {lbl}':>8}{C_RST}")
        rnk_h.append(f"{clr}{f'Rnk {lbl}':>7}{C_RST}")
        t50_h.append(f"{clr}{f'T50 {lbl}':>8}{C_RST}")

    print(f"{' | '.join(headers)} | {' | '.join(raw_h)} | {' | '.join(rnk_h)} | {' | '.join(t50_h)}")
    print("-" * (15 + 11 * len(IMPLEMENTED_METHODS) + 10 * len(IMPLEMENTED_METHODS) + 11 * len(IMPLEMENTED_METHODS)))

    all_results = []

    for n_groups in N_GROUPS_LIST:
        for group_size in GROUP_SIZE_LIST:
            total_items = n_groups * group_size

            for w in W_VALUES:
                raw_scores = {m[0]: [] for m in IMPLEMENTED_METHODS}
                ranks = {m[0]: [] for m in IMPLEMENTED_METHODS}
                times = {m[0]: [] for m in IMPLEMENTED_METHODS}
                selectors: dict[str, BaseSelector] = {}

                for method, transversal in IMPLEMENTED_METHODS:
                    config = Config(
                        method=method,
                        transversal=transversal,
                        group_size=group_size,
                        n_groups=n_groups,
                        _w_interaction=w,
                        _diversity_alpha=w,
                        _temperature=0.0,
                    )
                    selector_ = get_subsample_selector(config)
                    selectors[method] = selector_

                # Warmup: exclude first-call/setup effects from timed measurements.
                for warmup_idx in range(WARMUP_TRIALS):
                    cache = make_synthetic_cache(total_items, seed=100_000 + warmup_idx)

                    for method, _ in IMPLEMENTED_METHODS:
                        sync_if_cuda()
                        _ = selectors[method].subsample(cache)
                        sync_if_cuda()

                # Post-warmup timing probe: skip methods that exceed 1 s.
                SKIP_THRESHOLD = 1.0
                skipped: set[str] = set()
                probe_cache = make_synthetic_cache(total_items, seed=200_000)
                for method, _ in IMPLEMENTED_METHODS:
                    sync_if_cuda()
                    t0 = perf_counter()
                    _ = selectors[method].subsample(probe_cache)
                    sync_if_cuda()
                    probe_elapsed = perf_counter() - t0
                    if probe_elapsed > SKIP_THRESHOLD:
                        skipped.add(method)
                        clr = METHOD_META[method]["color"]
                        lbl = METHOD_META[method]["label"]
                        print(
                            f"  {clr}[SKIP]{C_RST} {lbl} ({method}) took "
                            f"{probe_elapsed:.2f}s after warmup — skipping for this setting.",
                        )

                for trial in range(N_TRIALS):
                    cache = make_synthetic_cache(total_items, seed=trial)
                    assert cache.embeddings is not None
                    ref_kernel = compute_cosine_similarity(cache.embeddings)

                    trial_raw_scores = []

                    for method, _ in IMPLEMENTED_METHODS:
                        if method in skipped:
                            trial_raw_scores.append(float("-inf"))
                            raw_scores[method].append(float("-inf"))
                            times[method].append(float("nan"))
                            continue

                        selector = selectors[method]

                        sync_if_cuda()
                        start_time = perf_counter()
                        indices = indices_to_list(selector.subsample(cache))
                        sync_if_cuda()
                        elapsed = perf_counter() - start_time

                        score = compute_log_det_torch(ref_kernel, indices)
                        trial_raw_scores.append(score)

                        raw_scores[method].append(score)
                        times[method].append(elapsed)

                    # Compute rank for this trial (higher raw score is better)
                    trial_r = compute_ranks_1_to_N(trial_raw_scores)
                    for (method, _), r in zip(IMPLEMENTED_METHODS, trial_r):
                        ranks[method].append(r)

                avg_raw = []
                avg_rnk = []
                med_times = []
                p95_times = []
                std_times = []

                for method, _ in IMPLEMENTED_METHODS:
                    if method in skipped:
                        avg_raw.append(-np.inf)
                        avg_rnk.append(float("nan"))
                        med_times.append(float("nan"))
                        p95_times.append(float("nan"))
                        std_times.append(float("nan"))
                        continue
                    scores = [s for s in raw_scores[method] if s > -1e9]
                    avg_raw.append(float(np.mean(scores)) if scores else -np.inf)
                    avg_rnk.append(float(np.mean(ranks[method])))
                    valid_times = [t for t in times[method] if not np.isnan(t)]
                    med_times.append(float(np.median(valid_times)) if valid_times else float("nan"))
                    p95_times.append(float(np.percentile(valid_times, 95)) if valid_times else float("nan"))
                    std_times.append(float(np.std(valid_times)) if valid_times else float("nan"))

                # Dynamic row printing
                row_cols = [f"{n_groups:>4}", f"{group_size:>4}", f"{w:>5.2f}"]

                raw_cells: list[str] = []
                rnk_cells: list[str] = []
                t50_cells: list[str] = []

                for i, (method, _) in enumerate(IMPLEMENTED_METHODS):
                    clr = METHOD_META[method]["color"]
                    raw_cells.append(f"{clr}{avg_raw[i]:>8.2f}{C_RST}")
                    rnk_cells.append(f"{clr}{avg_rnk[i]:>7.2f}{C_RST}")
                    t50_cells.append(f"{clr}{med_times[i]:>8.4f}{C_RST}")

                print(
                    f"{' | '.join(row_cols)} | "
                    f"{' | '.join(raw_cells)} | "
                    f"{' | '.join(rnk_cells)} | "
                    f"{' | '.join(t50_cells)}",
                )

                results_row = {
                    "n_groups": n_groups,
                    "group_size": group_size,
                    "w_int": w,
                }
                for i, (method, _) in enumerate(IMPLEMENTED_METHODS):
                    lbl = METHOD_META[method]["label"].lower()
                    results_row[f"raw_{lbl}"] = avg_raw[i]
                    results_row[f"rnk_{lbl}"] = avg_rnk[i]
                    results_row[f"time50_{lbl}"] = med_times[i]
                    results_row[f"time95_{lbl}"] = p95_times[i]
                    results_row[f"time_std_{lbl}"] = std_times[i]

                all_results.append(results_row)

    print("\nMethod Key:")
    for method, _ in IMPLEMENTED_METHODS:
        meta = METHOD_META[method]
        print(f"{meta['color']}{meta['label']}{C_RST}: {method}")
    print("\nTiming columns in table use median latency (T50, seconds). CSV also includes T95 and std.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_file = f"subsample_benchmark_{timestamp}.csv"
    with open(csv_file, mode="w", newline="") as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    print(f"\nResults saved to {csv_file}")


if __name__ == "__main__":
    main()
