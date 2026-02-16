"""Baseline selector that returns sequential indices (no actual selection)."""

import torch

from config import Cache, Config
from subsample.base import BaseSelector


class BaselineSelection(BaseSelector):
    """Baseline selector: identity selection returning first n_groups indices."""

    def __init__(self, config: Config):
        super().__init__(config)
        assert config.group_size == 1

    def _sample(self):
        """Independent sampling: return global indices on rank 0, None otherwise."""
        if self.distributed_utils and self.distributed_utils.rank != 0:
            return None

        n_global = self.config.n_groups * self.distributed_mul
        return torch.arange(n_global, device=self.device)

    def _transversal(self, cache: Cache):  # noqa: ARG002
        return self._sample()

    def _non_transversal(self, cache: Cache):  # noqa: ARG002
        return self._sample()
