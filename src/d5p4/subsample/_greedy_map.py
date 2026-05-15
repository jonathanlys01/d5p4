"""Greedy MAP-DPP subset selector with full exploration. Plain torch code."""

import torch

from d5p4.config import Cache
from d5p4.subsample.base import BaseSelector, fallback_greedy, fallback_greedy_block


class _GreedyMAP(BaseSelector):
    """Greedy MAP-DPP selector maximizing log-determinant of the kernel submatrix."""

    def _guard_unique(self, ret: torch.Tensor) -> bool:
        """Check that all selected indices are unique."""
        return torch.unique(ret).size(0) >= self.config.n_groups * self.distributed_mul

    def _transversal(self, cache: Cache) -> torch.Tensor | None:
        """Transversal selection: one sample per group, maximizing log-determinant."""
        if (L := self.compute_kernel(cache)) is None:
            return None

        n_groups = self.config.n_groups * self.distributed_mul
        group_size_each = L.size(0) // n_groups

        item_to_group = torch.arange(n_groups, device=L.device).repeat_interleave(group_size_each)
        group_member_table = torch.arange(L.size(0), device=L.device).view(n_groups, group_size_each)

        ret = _greedy_map_full_explore(L, n_groups, item_to_group, group_member_table)

        if not self._guard_unique(ret):
            ret = fallback_greedy_block(L, self.config.group_size, n_groups)
        return ret

    def _non_transversal(self, cache: Cache) -> torch.Tensor | None:
        """Global selection without group constraints, maximizing log-determinant."""
        if (L := self.compute_kernel(cache)) is None:
            return None

        n_groups = self.config.n_groups * self.distributed_mul

        item_to_group = torch.arange(L.size(0), device=L.device)
        group_member_table = torch.arange(L.size(0), device=L.device).view(L.size(0), 1)

        ret = _greedy_map_full_explore(L, n_groups, item_to_group, group_member_table)

        if not self._guard_unique(ret):
            ret = fallback_greedy(L, n_groups)
        return ret


@torch.jit.script
def _greedy_map_full_explore(
    kernel: torch.Tensor,
    num_groups: int,
    item_to_group: torch.Tensor,
    group_member_table: torch.Tensor,
) -> torch.Tensor:
    """
    Run N parallel greedy DPP selections, each starting from a different item.
    Uses Cholesky-like orthogonalization to incrementally compute log-determinant.
    Returns the trajectory with highest log-determinant.
    """
    device = kernel.device
    dtype = kernel.dtype
    n_items = kernel.size(0)
    epsilon = 1e-10

    selected = torch.empty((n_items, num_groups), dtype=torch.long, device=device)
    log_dets = torch.zeros(n_items, dtype=dtype, device=device)

    start_items = torch.arange(n_items, device=device)
    selected[:, 0] = start_items

    diag = torch.diagonal(kernel).clamp(min=epsilon)
    log_dets = log_dets + torch.log(diag)

    di2s = diag.unsqueeze(0).expand(n_items, -1).clone()

    start_groups = item_to_group[start_items]
    start_members = group_member_table[start_groups]
    di2s.scatter_(1, start_members, -float("inf"))

    e_all = torch.zeros((n_items, num_groups, n_items), dtype=dtype, device=device)

    e_0 = kernel[start_items, :] / torch.sqrt(diag).unsqueeze(1)
    e_all[:, 0, :] = e_0
    di2s = di2s - e_0**2

    for k in range(1, num_groups):
        next_items = torch.argmax(di2s, dim=1)  # (N,)
        selected[:, k] = next_items

        di_sq = torch.gather(di2s, 1, next_items.unsqueeze(1)).squeeze(1).clamp(min=epsilon)
        log_dets = log_dets + torch.log(di_sq)

        next_groups = item_to_group[next_items]
        next_members = group_member_table[next_groups]
        di2s.scatter_(1, next_members, -float("inf"))

        if k < num_groups - 1:
            e_new = kernel[next_items, :]  # (N, N)
            e_active = e_all[:, :k, :]  # (N, k, N)

            idx = next_items.view(n_items, 1, 1).expand(-1, k, 1)
            coeffs = torch.gather(e_active, 2, idx).squeeze(2)  # (N, k)

            dot_prod = torch.bmm(coeffs.unsqueeze(1), e_active).squeeze(1)  # (N, N)

            e_new = (e_new - dot_prod) / torch.sqrt(di_sq).unsqueeze(1)
            e_all[:, k, :] = e_new

            di2s = di2s - e_new**2

    return selected[torch.argmax(log_dets), :]


def fast_greedy_map(
    kernel: torch.Tensor,
    num_groups: int,
    item_to_group: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation: single-trajectory greedy MAP-DPP selection.

    This is the original O(N*K) version without full exploration.
    It selects greedily from a single starting point.
    The _greedy_map_full_explore function above runs N parallel trajectories
    and picks the best one, which yields better results at the cost of O(N^2*K) complexity.

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

    # Provide a symmetric semi-positive definite dummy block to ensure sqrt yields real answers
    dummy_kernel = torch.randn(8, 8)
    dummy_kernel = dummy_kernel @ dummy_kernel.T

    dummy_item_to_group = torch.arange(8)
    dummy_group_member_table = torch.arange(8).view(-1, 1)

    print(
        timeit.timeit(
            lambda: _greedy_map_full_explore(dummy_kernel, 8, dummy_item_to_group, dummy_group_member_table),
            number=1000,
        ),
    )
    print(timeit.timeit(lambda: fast_greedy_map(dummy_kernel, 8, dummy_item_to_group), number=1000))
