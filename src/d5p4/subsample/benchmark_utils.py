"""Shared utilities for subsample benchmark and diagnostic scripts."""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from d5p4.config import Cache


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sync_if_cuda() -> None:
    """Synchronize CUDA work before or after timing measurements."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def indices_to_list(indices: torch.Tensor | Sequence[int] | None) -> list[int]:
    """Convert selector output to a plain Python list."""
    if indices is None:
        return []
    if isinstance(indices, torch.Tensor):
        return [int(i) for i in indices.detach().cpu().tolist()]
    return [int(i) for i in indices]


def is_valid_transversal(indices: Sequence[int], num_groups: int, group_size: int) -> bool:
    """Check whether indices contain exactly one item from each group."""
    if len(indices) != num_groups:
        return False
    groups = {int(i) // group_size for i in indices}
    return len(groups) == num_groups


def compute_log_det_numpy(kernel: np.ndarray, indices: Sequence[int]) -> float:
    """Compute log-determinant of a NumPy kernel submatrix."""
    if not indices or len(set(indices)) != len(indices):
        return float("-inf")
    try:
        unique = sorted(int(i) for i in indices)
        sub = kernel[np.ix_(unique, unique)]
        sign, logdet = np.linalg.slogdet(sub)
        return float(logdet) if sign > 0 else float("-inf")
    except np.linalg.LinAlgError:
        return float("-inf")


def compute_log_det_torch(kernel: torch.Tensor, indices: Sequence[int]) -> float:
    """Compute log-determinant of a Torch kernel submatrix."""
    if not indices or len(set(indices)) != len(indices):
        return float("-inf")
    idx = [int(i) for i in indices]
    sub = kernel[idx][:, idx]
    sign, logdet = torch.linalg.slogdet(sub)
    return float(logdet.item()) if sign > 0 else float("-inf")


def compute_cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute a pure cosine-similarity matrix from flattened embeddings."""
    flat = embeddings.float().reshape(embeddings.size(0), -1)
    flat = F.normalize(flat, dim=-1, eps=1e-12)
    return flat @ flat.T


def make_random_cache(
    total_items: int,
    *,
    seq_len: int = 16,
    hidden_size: int = 64,
    vocab_size: int = 50,
    seed: int | None = None,
    device: str = DEVICE,
) -> Cache:
    """Generate simple random embeddings and logits for selector diagnostics."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    embeddings = torch.randn(total_items, seq_len, hidden_size, device=device, generator=generator)
    log_p_x0 = torch.randn(total_items, seq_len, vocab_size, device=device, generator=generator)
    seq = torch.arange(total_items, device=device)
    return Cache(embeddings=embeddings, log_p_x0=log_p_x0, x=seq)
