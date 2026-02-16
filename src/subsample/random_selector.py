"""Random subset selector for baseline comparison."""

import numpy as np
import torch

from config import Cache
from subsample.base import BaseSelector


class RandomSelection(BaseSelector):
    """Random selector: uniform random selection without quality/diversity consideration."""

    def _transversal(self, cache: Cache) -> torch.Tensor | None:  # noqa: ARG002
        """Randomly select one item per group."""
        if self.distributed_utils and self.distributed_utils.rank != 0:
            return None
        gr_size = self.config.group_size
        n_groups = self.config.n_groups * self.distributed_mul
        indices = torch.randint(0, gr_size, size=(n_groups,)) + torch.arange(n_groups) * gr_size
        return indices.to(self.device)

    def _non_transversal(self, cache: Cache) -> torch.Tensor | None:  # noqa: ARG002
        """Randomly select n_groups items from all items."""
        if self.distributed_utils and self.distributed_utils.rank != 0:
            return None
        B = self.config.n_groups * self.config.group_size * self.distributed_mul
        n_groups = self.config.n_groups * self.distributed_mul
        indices = np.random.choice(B, n_groups, replace=False)
        return torch.from_numpy(indices).to(self.device)
