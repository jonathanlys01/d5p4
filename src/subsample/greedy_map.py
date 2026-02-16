"""Greedy MAP-DPP subset selector with full exploration."""

import torch

from config import Cache
from subsample.base import BaseSelector, fallback_greedy, fallback_greedy_block


class GreedyMAP(BaseSelector):
    """Greedy MAP-DPP selector maximizing log-determinant of the kernel submatrix."""

    def _guard_unique(self, ret: torch.Tensor) -> bool:
        """Check that all selected indices are unique."""
        return torch.unique(ret).size(0) >= self.config.n_groups * self.distributed_mul

    def _transversal(self, cache: Cache) -> torch.Tensor | None:
        """Transversal selection: one sample per group, maximizing log-determinant."""
        if (L := self.compute_kernel(cache)) is None:
            return None

        n_groups = self.config.n_groups * self.distributed_mul
        item_to_group = torch.arange(n_groups, device=L.device).repeat_interleave(L.size(0) // n_groups)
        ret = _greedy_map_full_explore(L, n_groups, item_to_group)

        if not self._guard_unique(ret):
            ret = fallback_greedy_block(L, self.config.group_size, n_groups)
        return ret

    def _non_transversal(self, cache: Cache) -> torch.Tensor | None:
        """Global selection without group constraints, maximizing log-determinant."""
        if (L := self.compute_kernel(cache)) is None:
            return None

        n_groups = self.config.n_groups * self.distributed_mul
        item_to_group = torch.arange(L.size(0), device=L.device)
        ret = _greedy_map_full_explore(L, n_groups, item_to_group)

        if not self._guard_unique(ret):
            ret = fallback_greedy(L, n_groups)
        return ret


@torch.jit.script
def _greedy_map_full_explore(
    kernel: torch.Tensor,
    num_groups: int,
    item_to_group: torch.Tensor,
) -> torch.Tensor:
    """
    Run N parallel greedy DPP selections, each starting from a different item.

    Uses Cholesky-like orthogonalization to incrementally compute log-determinant.
    Returns the trajectory with highest log-determinant.
    """
    device = kernel.device
    dtype = kernel.dtype
    n_items = kernel.size(0)

    # State tensors
    di2s = kernel.diag().unsqueeze(0).expand(n_items, -1).clone()  # (N, N)
    selected = torch.empty((n_items, num_groups), dtype=torch.long, device=device)
    log_dets = torch.zeros(n_items, dtype=dtype, device=device)

    epsilon = 1e-10  # Local constant for JIT compatibility

    # Step 0: each trajectory b starts with item b
    start_items = torch.arange(n_items, device=device)
    selected[:, 0] = start_items
    di_sq = kernel.diag().clamp(min=epsilon)
    log_dets = torch.log(di_sq)

    # First orthogonal vectors: e0[b, j] = kernel[b, j] / sqrt(kernel[b, b])
    e_prev = kernel / torch.sqrt(di_sq).unsqueeze(1)  # (N, N)
    di2s = di2s - e_prev**2

    # Mask starting groups
    start_groups = item_to_group[start_items]
    group_mask = item_to_group.unsqueeze(0) == start_groups.unsqueeze(1)
    di2s[group_mask] = -float("inf")

    # Stack of orthogonal vectors: e_all[b, k, j]
    e_all = e_prev.unsqueeze(1)  # (N, 1, N)

    for k in range(num_groups - 1):
        # Select next item for each trajectory
        next_items = torch.argmax(di2s, dim=1)  # (N,)
        selected[:, k + 1] = next_items

        # Get di_sq for next items
        di_sq = torch.gather(di2s, 1, next_items.unsqueeze(1)).squeeze(1).clamp(min=epsilon)
        log_dets = log_dets + torch.log(di_sq)

        # Mask selected groups
        next_groups = item_to_group[next_items]
        group_mask = item_to_group.unsqueeze(0) == next_groups.unsqueeze(1)
        di2s[group_mask] = -float("inf")

        if k < num_groups - 2:  # No need to compute for last iteration
            # Compute new orthogonal vector
            elements = kernel[next_items, :]  # (N, N)

            # Get coefficients at selected items
            idx = next_items.view(n_items, 1, 1).expand(-1, k + 1, 1)
            coeffs = torch.gather(e_all, 2, idx).squeeze(2)  # (N, k+1)

            # Compute dot product
            dot_prod = torch.bmm(coeffs.unsqueeze(1), e_all).squeeze(1)  # (N, N)

            e_new = (elements - dot_prod) / torch.sqrt(di_sq).unsqueeze(1)
            e_all = torch.cat([e_all, e_new.unsqueeze(1)], dim=1)  # (N, k+2, N)

            di2s = di2s - e_new**2

    return selected[torch.argmax(log_dets), :]


def fast_greedy_map(
    kernel: torch.Tensor,
    num_groups: int,
    item_to_group: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation: single-trajectory greedy MAP-DPP selection.

    This is the original O(N*K) version without full exploration. It selects greedily
    from a single starting point. The _greedy_map_full_explore function above runs N
    parallel trajectories and picks the best one, which yields better results at the
    cost of O(N^2*K) complexity.

    Adapted from: https://github.com/laming-chen/fast-map-dpp/blob/master/dpp.py
    """
    device, dtype = kernel.device, kernel.dtype
    n_items = kernel.size(0)
    cis = torch.zeros((num_groups, n_items), dtype=dtype, device=device)
    di2s = kernel.diag().clone()
    selected = torch.empty((num_groups,), dtype=torch.long, device=device)

    # First selection
    selected_item = torch.argmax(di2s)
    selected[0] = selected_item
    di2s[item_to_group == item_to_group[selected_item]] = -torch.inf

    # Remaining selections
    for k in range(1, num_groups):
        ci_optimal = cis[:k, selected_item]
        di_optimal = torch.sqrt(di2s[selected_item])
        elements = kernel[selected_item, :]
        eis = (elements - torch.matmul(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= eis**2

        selected_item = torch.argmax(di2s)
        di2s[item_to_group == item_to_group[selected_item]] = -torch.inf
        selected[k] = selected_item

    return selected


if __name__ == "__main__":
    import timeit

    dummy_kernel = torch.randn(8, 8)
    dummy_item_to_group = torch.arange(8)
    print(timeit.timeit(lambda: _greedy_map_full_explore(dummy_kernel, 8, dummy_item_to_group), number=1000))
    print(timeit.timeit(lambda: fast_greedy_map(dummy_kernel, 8, dummy_item_to_group), number=1000))
