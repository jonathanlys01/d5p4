import math
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from d5p4.config import RESULTS_DIR


def compute_cka(ref_embeddings: torch.Tensor, model_outputs: torch.Tensor) -> float:
    """Compute CKA between reference embeddings and model outputs."""
    with torch.no_grad():
        ref_embeddings = ref_embeddings.to(torch.float32)
    model_outputs = model_outputs.to(torch.float32)

    # Center embeddings
    ref_embeddings = ref_embeddings - ref_embeddings.mean(0, keepdim=True)
    model_outputs = model_outputs - model_outputs.mean(0, keepdim=True)

    # Compute Gram matrices
    ref_gram = ref_embeddings @ ref_embeddings.t()
    model_gram = model_outputs @ model_outputs.t()

    ref_norm = torch.norm(ref_gram, p="fro")
    model_norm = torch.norm(model_gram, p="fro")

    if ref_norm == 0 or model_norm == 0:
        return 0.0

    cka = (ref_gram * model_gram).sum() / (ref_norm * model_norm)
    return cka.item()


def compute_cosine_similarity_stats(embeddings: torch.Tensor) -> dict[str, float]:
    """Compute the mean and standard deviation of pairwise cosine similarities."""
    sim_matrix = compute_cosine_similarity_matrix(embeddings)
    return compute_cosine_similarity_stats_from_matrix(sim_matrix)


def compute_cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise cosine similarity matrix after row-wise normalization."""
    embeddings_norm = F.normalize(embeddings.to(torch.float32), p=2, dim=1)
    return embeddings_norm @ embeddings_norm.t()


def compute_cosine_similarity_stats_from_matrix(sim_matrix: torch.Tensor) -> dict[str, float]:
    """Compute pairwise cosine summary statistics from a precomputed similarity matrix."""
    batch_size = sim_matrix.shape[0]
    if batch_size <= 1:
        return {"mean": 0.0, "std": 0.0}

    mask = ~torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)
    pairwise_similarities = sim_matrix[mask]

    return {
        "mean": pairwise_similarities.mean().item(),
        "std": pairwise_similarities.std().item(),
    }


def compute_avg_cosine_similarity(embeddings: torch.Tensor) -> float:
    """Compute the average pairwise cosine similarity (excluding self-similarity)."""
    return compute_cosine_similarity_stats(embeddings)["mean"]


def estimate_rho_from_cosine_matrices(
    plain_sim_matrix: torch.Tensor,
    residual_sim_matrix: torch.Tensor,
    clamp_eps: float = 1e-6,
) -> torch.Tensor:
    """
    Estimate rho from the off-diagonal mean of cos(flatten) - cos(flatten_no_special).

    The estimate is clamped to [0, 1) because the closed-form H construction assumes that range.
    """
    if plain_sim_matrix.shape != residual_sim_matrix.shape:
        raise ValueError(
            "plain_sim_matrix and residual_sim_matrix must have the same shape, "
            f"got {plain_sim_matrix.shape} and {residual_sim_matrix.shape}",
        )

    batch_size = plain_sim_matrix.shape[0]
    if batch_size <= 1:
        return torch.zeros((), device=plain_sim_matrix.device, dtype=plain_sim_matrix.dtype)

    mask = ~torch.eye(batch_size, dtype=torch.bool, device=plain_sim_matrix.device)
    rho_estimate = (plain_sim_matrix - residual_sim_matrix)[mask].mean()
    return rho_estimate.clamp(min=0.0, max=1.0 - clamp_eps)


def compute_H_from_Z(Z: torch.Tensor, rho: float = 0.4) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the BOS/EOS-adjusted interaction matrix H from residual embeddings Z.

    Args:
        Z: Residual embeddings [k, d], typically flattened token embeddings without BOS/EOS.
        rho: Shared BOS/EOS contribution parameter in [0, 1).

    Returns:
        H: [k, k]
        spec_norm: scalar tensor containing ||H||_2
    """
    residual_sim_matrix = compute_cosine_similarity_matrix(Z)
    return compute_H_from_residual_cosine_matrix(residual_sim_matrix, rho=rho)


def compute_H_from_residual_cosine_matrix(
    residual_sim_matrix: torch.Tensor,
    rho: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute H from a cached cosine similarity matrix of residual embeddings.

    Args:
        residual_sim_matrix: Pairwise cosine similarity matrix of normalized residual embeddings [k, k].
        rho: Shared BOS/EOS contribution parameter in [0, 1).

    Returns:
        H: [k, k]
        spec_norm: scalar tensor containing ||H||_2
    """
    if not 0 <= rho < 1:
        raise ValueError(f"rho must be in [0, 1), got {rho}")

    residual_sim_matrix = residual_sim_matrix.to(torch.float32)

    k = residual_sim_matrix.shape[0]
    if k <= 1:
        H = torch.zeros((k, k), device=residual_sim_matrix.device, dtype=residual_sim_matrix.dtype)
        spec_norm = torch.zeros((), device=residual_sim_matrix.device, dtype=residual_sim_matrix.dtype)
        return H, spec_norm

    device, dtype = residual_sim_matrix.device, residual_sim_matrix.dtype

    identity = torch.eye(k, device=device, dtype=dtype)
    ones = torch.ones((k, 1), device=device, dtype=dtype)

    # Residual cosine matrix: off-diag = z_i^T z_j, diag = 0.
    C_tilde = residual_sim_matrix - identity

    # Projectors onto span(1) and its orthogonal complement.
    P1 = (ones @ ones.T) / k
    Pperp = identity - P1

    B_inv_half = P1 / math.sqrt(1 + (k - 1) * rho) + Pperp / math.sqrt(1 - rho)

    H = (1 - rho) * (B_inv_half @ C_tilde @ B_inv_half)
    H = 0.5 * (H + H.T)
    spec_norm = torch.linalg.eigvalsh(H).abs().max()

    return H, spec_norm


def get_pooled_output(
    outputs: torch.Tensor,
    strategy: str,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply a pooling strategy to token-level outputs.

    Args:
        outputs: Token-level hidden states [B, L, D]
        strategy: "mean", "flatten", "flatten_no_special", "pool_masked", or "pool_non_masked"
        mask: Boolean mask [B, L] where True means masked token
    """
    if strategy == "flatten":
        return outputs.reshape(outputs.size(0), -1)
    elif strategy == "flatten_no_special":
        return outputs[:, 1:-1, :].reshape(outputs.size(0), -1)
    elif strategy == "mean":
        return torch.mean(outputs, dim=1)
    elif strategy == "pool_masked":
        if mask is None:
            raise ValueError("Mask is required for 'pool_masked' strategy")
        mask_expanded = mask.unsqueeze(-1).to(outputs.dtype)
        masked_outputs = outputs * mask_expanded
        num_masked = torch.sum(mask, dim=1, keepdim=True).clamp(min=1)
        return torch.sum(masked_outputs, dim=1) / num_masked
    elif strategy == "pool_non_masked":
        if mask is None:
            raise ValueError("Mask is required for 'pool_non_masked' strategy")
        non_mask = ~mask
        non_mask_expanded = non_mask.unsqueeze(-1).to(outputs.dtype)
        non_masked_outputs = outputs * non_mask_expanded
        num_non_masked = torch.sum(non_mask, dim=1, keepdim=True).clamp(min=1)
        return torch.sum(non_masked_outputs, dim=1) / num_non_masked
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")


def save_results_csv(
    results: dict[str, dict[str, Any]],
    x_values: list[Any],
    x_name: str,
    filename: str,
    ref_acs_baseline: float | None = None,
) -> pd.DataFrame:
    """Save results to a CSV file using pandas."""
    n = len(x_values)
    data: dict[str, Any] = {x_name: x_values}

    if ref_acs_baseline is not None:
        # Explicitly broadcast scalar to list to satisfy certain linters/pandas edge cases
        data["ref_acs_baseline"] = [ref_acs_baseline] * n

    for strategy, metrics in results.items():
        for metric_name, values in metrics.items():
            if len(values) != n:
                # This should not happen based on script logic, but good for robustness
                print(f"Warning: strategy {strategy} metric {metric_name} has length {len(values)}, expected {n}")
            data[f"{strategy}_{metric_name}"] = values

    df = pd.DataFrame(data)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")
    return df


def plot_cka_acs(
    df: pd.DataFrame,
    x_name: str,
    title_suffix: str,
    n_samples: int,
    ref_acs_baseline: float | None = None,
    plot_filename: str | None = None,
):
    """Plot CKA and ACS results."""
    strategies = [col[:-4] for col in df.columns if col.endswith("_cka")]

    ax = plt.subplots(2, 1, figsize=(14, 16), sharex=True)[1]

    for strategy in strategies:
        cka_col = f"{strategy}_cka"
        mask = ~df[cka_col].isna()
        ax[0].plot(df.loc[mask, x_name], df.loc[mask, cka_col], marker="o", linestyle="-", label=strategy)

    ax[0].set_ylabel("CKA Score")
    ax[0].set_title(f"{title_suffix} (CKA) vs. {x_name} (Avg. over {n_samples} samples)")
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_ylim(bottom=0)

    for strategy in strategies:
        acs_col = f"{strategy}_acs"
        mask = ~df[acs_col].isna()
        ax[1].plot(df.loc[mask, x_name], df.loc[mask, acs_col], marker="o", linestyle="-", label=strategy)

    if ref_acs_baseline is not None:
        ax[1].axhline(
            y=ref_acs_baseline,
            color="r",
            linestyle="--",
            label=f"Reference ACS ({ref_acs_baseline:.3f})",
        )

    ax[1].set_xlabel(x_name)
    ax[1].set_ylabel("Avg. Cosine Similarity (ACS)")
    ax[1].set_title(f"Average Cosine Similarity (ACS) vs. {x_name} (Avg. over {n_samples} samples)")
    ax[1].legend()
    ax[1].grid(True)

    # Dynamically set y-axis limits
    acs_cols = [f"{s}_acs" for s in strategies]
    all_acs = df[acs_cols].to_numpy().flatten()  # type: ignore
    all_acs = all_acs[~np.isnan(all_acs)]

    if len(all_acs) > 0:
        min_acs = np.min(all_acs)
        max_acs = np.max(all_acs)
        y_margin = (max_acs - min_acs) * 0.1 if max_acs > min_acs else 0.1
        ax[1].set_ylim((max(0.0, min_acs - y_margin), min(1.0, max_acs + y_margin)))

    plt.tight_layout()
    if plot_filename:
        plot_path = os.path.join(RESULTS_DIR, plot_filename)
        plt.savefig(plot_path)
        print(f"Plots saved to {plot_path}")
    else:
        plt.show()
