"""Determinantal Point Process (DPP) subset selector using exact k-DPP sampling."""

import numpy as np
import torch
from dppy.finite_dpps import FiniteDPP

from config import Cache
from subsample.base import BaseSelector


class DPP(BaseSelector):
    """DPP selector using exact k-DPP sampling from dppy library."""

    def _transversal(self, cache: Cache):
        raise NotImplementedError("DPP sampling does not support exact transversal mode.")

    def _non_transversal(self, cache: Cache):
        """Unconstrained DPP Sampling."""
        if (L := self.compute_kernel(cache)) is None:
            return None

        L = L.cpu().numpy()
        k = self.config.n_groups * self.distributed_mul

        try:
            selected_indices = _sample_dpp(L, k)
        except Exception as e:
            print(f"DPP sampling failed with error: {e}. Falling back to greedy selection.")
            selected_indices = _fallback_greedy(L, k)

        return torch.from_numpy(selected_indices).to(self.device)


def _sample_dpp(L: np.ndarray, k: int) -> np.ndarray:
    """Sample k items from a DPP defined by kernel L. Sometimes fails due to numerical issues."""
    dpp = FiniteDPP("likelihood", L=L)
    return np.array(dpp.sample_exact_k_dpp(size=k))


def _fallback_greedy(L: np.ndarray, k: int) -> np.ndarray:
    """Greedy fallback based on the diagonal of L."""
    diag = np.diag(L)
    selected_indices = np.argsort(-diag)[:k]
    return selected_indices
