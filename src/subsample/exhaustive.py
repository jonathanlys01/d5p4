"""Exhaustive subset selector that evaluates all transversal combinations."""

import torch

from config import Cache, Config
from subsample.base import BaseSelector


class Exhaustive(BaseSelector):
    """Exhaustive selector that evaluates all transversal combinations to find optimal."""

    def __init__(self, config: Config):
        super().__init__(config)

        self.cached_group_cartesian = None
        if self.config.transversal:
            self.cached_group_cartesian = _group_cartesian(
                group_size=self.config.group_size,
                n_groups=self.config.n_groups * self.distributed_mul,
            ).to(self.device)

    def _transversal(self, cache: Cache):
        """Transversal Exhaustive Search: argmax/sampling over all transversal combinations."""
        if (L := self.compute_kernel(cache)) is None:
            return None

        assert self.cached_group_cartesian is not None

        L_sub = L[self.cached_group_cartesian[:, :, None], self.cached_group_cartesian[:, None, :]]
        sign, logdet = torch.linalg.slogdet(L_sub)
        if self.config._temperature == 0:  # argmax for temperature 0
            sampled_index = torch.argmax(logdet)
            return self.cached_group_cartesian[sampled_index]

        scaled_logits = logdet / self.config._temperature
        # max trick for numerical stability (low determinant values can lead to NaNs)
        max_logit = torch.max(scaled_logits)
        scaled_logits = scaled_logits - max_logit
        scaled_logits[sign <= 0] = -torch.inf  # invalidate non-positive definite
        det = torch.exp(scaled_logits)
        sampled_index = torch.multinomial(det, num_samples=1).squeeze(-1)
        return self.cached_group_cartesian[sampled_index]

    def _non_transversal(self, cache: Cache):
        raise NotImplementedError("Non-transversal mode is not implemented as it is too computationally expensive.")


def _group_cartesian(group_size: int, n_groups: int) -> torch.Tensor:
    """Generate Cartesian product of group indices."""
    grids = torch.meshgrid(*[torch.arange(group_size) + i * group_size for i in range(n_groups)], indexing="ij")
    stacked: torch.Tensor = torch.stack(grids, dim=-1)
    reshaped = stacked.reshape(-1, n_groups)
    return reshaped
