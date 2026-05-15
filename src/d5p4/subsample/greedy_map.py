"""Greedy MAP-DPP subset selector with full exploration."""

import torch


try:
    import triton  # type: ignore[import]
    import triton.language as tl  # type: ignore[import]

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None

from d5p4.config import Cache, Config
from d5p4.subsample.base import BaseSelector, fallback_greedy, fallback_greedy_block


if HAS_TRITON:

    @triton.autotune(  # type: ignore[misc]
        configs=[
            triton.Config({}, num_warps=4, num_stages=2),  # type: ignore[misc]
            triton.Config({}, num_warps=8, num_stages=2),  # type: ignore[misc]
            triton.Config({}, num_warps=16, num_stages=2),  # type: ignore[misc]
            triton.Config({}, num_warps=4, num_stages=3),  # type: ignore[misc]
            triton.Config({}, num_warps=8, num_stages=3),  # type: ignore[misc]
            triton.Config({}, num_warps=16, num_stages=3),  # type: ignore[misc]
            triton.Config({}, num_warps=4, num_stages=4),  # type: ignore[misc]
            triton.Config({}, num_warps=8, num_stages=4),  # type: ignore[misc]
            triton.Config({}, num_warps=16, num_stages=4),  # type: ignore[misc]
            triton.Config({}, num_warps=32, num_stages=4),  # type: ignore[misc]
        ],
        key=["n_items", "num_groups"],
    )
    @triton.jit  # type: ignore[misc]
    def _triton_greedy_map_kernel(
        kernel_ptr,
        stride_k_row,
        stride_k_col,
        diag_ptr,
        item_to_group_ptr,
        e_all_ptr,
        stride_e_traj,
        stride_e_vec,
        stride_e_item,
        selected_ptr,
        stride_sel_traj,
        stride_sel_step,
        log_dets_ptr,
        n_items: tl.constexpr,
        num_groups: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Triton kernel simulating N full trajectories in parallel.
        Each program/block evaluates one start_item trajectory.
        """
        pid = tl.program_id(0)

        # Setup SRAM offsets and mask
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_items

        # Load static mappings and initial diagonal
        item_groups = tl.load(item_to_group_ptr + offsets, mask=mask, other=-1)
        di2s = tl.load(diag_ptr + offsets, mask=mask, other=-float("inf"))

        log_det_total = 0.0
        current_item = pid  # Step 0: start item is the trajectory ID
        current_di_sq = tl.load(diag_ptr + current_item)

        # Main greedy selection loop
        for k in range(num_groups):
            # Record selection
            tl.store(selected_ptr + pid * stride_sel_traj + k * stride_sel_step, current_item)

            # Selected score from previous argmax (or initial diagonal at k=0)
            di_sq = current_di_sq
            di_sq = tl.maximum(di_sq, 1e-10)

            # Accumulate log determinant
            log_det_total += tl.math.log(di_sq)

            # Mask out the group of the selected item
            current_group = tl.load(item_to_group_ptr + current_item)
            is_same_group = item_groups == current_group
            di2s = tl.where(is_same_group, -float("inf"), di2s)

            if k < num_groups - 1:
                # Load the full kernel row for the newly selected item
                k_row = tl.load(kernel_ptr + current_item * stride_k_row + offsets * stride_k_col, mask=mask, other=0.0)
                e_new = k_row

                # Gram-Schmidt style orthogonalization against previously selected vectors
                for j in range(k):
                    # Load e_j from HBM
                    e_j = tl.load(
                        e_all_ptr + pid * stride_e_traj + j * stride_e_vec + offsets * stride_e_item,
                        mask=mask,
                        other=0.0,
                    )
                    # Get scalar coefficient directly: c_j = e_j[current_item]
                    c_j = tl.load(
                        e_all_ptr + pid * stride_e_traj + j * stride_e_vec + current_item * stride_e_item,
                    )
                    # Project out
                    e_new = e_new - c_j * e_j

                # Normalize the new vector
                inv_sqrt = 1.0 / tl.math.sqrt(di_sq)
                e_new = e_new * inv_sqrt

                # Store back to HBM memory
                tl.store(e_all_ptr + pid * stride_e_traj + k * stride_e_vec + offsets * stride_e_item, e_new, mask=mask)

                # Update di2s
                di2s = di2s - (e_new * e_new)

                # Find the next best item
                current_item = tl.argmax(di2s, axis=0)
                current_di_sq = tl.max(di2s, axis=0)

        # Store the final score for this trajectory
        tl.store(log_dets_ptr + pid, log_det_total)


def _run_triton_greedy_map(
    kernel: torch.Tensor,
    num_groups: int,
    item_to_group: torch.Tensor,
    bufs: list[torch.Tensor],
) -> torch.Tensor:
    """Dispatches the pre-allocated buffers to the Triton kernel."""
    n_items = kernel.size(0)

    # Unpack pre-allocated Triton buffers
    diag_buf = bufs[0]
    selected = bufs[1]
    log_dets = bufs[2]
    e_all = bufs[3]

    # Prepare contiguous diagonal
    diag_buf.copy_(torch.diagonal(kernel))

    BLOCK_SIZE = triton.next_power_of_2(n_items)  # type: ignore[union-attr]

    _triton_greedy_map_kernel[(n_items,)](
        kernel,
        kernel.stride(0),
        kernel.stride(1),
        diag_buf,
        item_to_group,
        e_all,
        e_all.stride(0),
        e_all.stride(1),
        e_all.stride(2),
        selected,
        selected.stride(0),
        selected.stride(1),
        log_dets,
        n_items=n_items,
        num_groups=num_groups,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # CPU/GPU sync point: pick the best trajectory
    best_traj = torch.argmax(log_dets)
    return selected[best_traj, :]


class GreedyMAP(BaseSelector):
    """Greedy MAP-DPP selector maximizing log-determinant of the kernel submatrix."""

    def __init__(self, config: Config):
        super().__init__(config)
        self._buf_signature: tuple[int, str, str, bool] | None = None
        self._precomputed: list[torch.Tensor] = []
        self._bufs: list[torch.Tensor] = []

    def _guard_unique(self, ret: torch.Tensor) -> bool:
        """Check that all selected indices are unique."""
        return torch.unique(ret).size(0) >= self.config.n_groups * self.distributed_mul

    def _build_precomputed(
        self,
        n_items: int,
        device: torch.device,
        transversal: bool,
    ) -> tuple[list[torch.Tensor], int]:
        """Build static indexing tensors used by the buffered greedy kernel."""
        n_groups = self.config.n_groups * self.distributed_mul
        if transversal:
            expected = n_groups * self.config.group_size
            assert n_items == expected, (
                "Kernel size must match config in transversal mode: "
                f"got {n_items}, expected {expected} ({n_groups} groups x {self.config.group_size} items)."
            )
            group_size_each = self.config.group_size
            item_to_group = torch.arange(n_groups, device=device).repeat_interleave(group_size_each)
        else:
            group_size_each = 1
            item_to_group = torch.arange(n_items, device=device)

        assert n_items % group_size_each == 0, "Invalid group partition for greedy MAP buffer setup."
        group_member_table = torch.arange(n_items, device=device).view(-1, group_size_each)
        start_items = torch.arange(n_items, device=device)
        return [item_to_group, group_member_table, start_items], group_size_each

    def _ensure_buffers(self, n_items: int, device: torch.device, dtype: torch.dtype, transversal: bool):
        """Lazily allocate work buffers and precomputed indices for the current mode/shape."""
        n_groups = self.config.n_groups * self.distributed_mul
        signature = (n_items, str(device), str(dtype), transversal)
        if self._buf_signature == signature:
            return

        precomputed, group_size_each = self._build_precomputed(n_items, device, transversal)
        self._precomputed = precomputed

        if HAS_TRITON and device.type == "cuda":
            n_traj = n_items
            max_vecs = n_groups - 1
            # Triton only needs a few buffers, handled completely in VRAM
            self._bufs = [
                torch.empty(n_items, dtype=dtype, device=device),  # diag_buf
                torch.empty((n_traj, n_groups), dtype=torch.long, device=device),  # selected
                torch.empty(n_traj, dtype=dtype, device=device),  # log_dets
                torch.empty((n_traj, max(1, max_vecs), n_items), dtype=dtype, device=device),  # e_all
            ]
        else:
            # Fallback CPU buffers are allocated dynamically in a simpler loop
            self._bufs = []

        self._buf_signature = signature

    def _transversal(self, cache: Cache) -> torch.Tensor | None:
        """Transversal selection: one sample per group, maximizing log-determinant."""
        if (L := self.compute_kernel(cache)) is None:
            return None

        n_groups = self.config.n_groups * self.distributed_mul
        assert n_groups <= L.size(0), "Number of requested selections cannot exceed kernel size."
        self._ensure_buffers(L.size(0), L.device, L.dtype, transversal=True)

        if HAS_TRITON and L.device.type == "cuda":
            ret = _run_triton_greedy_map(L, n_groups, self._precomputed[0], self._bufs)
        else:
            ret = _greedy_map_cpu(L, n_groups, self._precomputed[0], self._precomputed[1], self._precomputed[2])

        if not self._guard_unique(ret):
            ret = fallback_greedy_block(L, self.config.group_size, n_groups)
        return ret

    def _non_transversal(self, cache: Cache) -> torch.Tensor | None:
        """Global selection without group constraints, maximizing log-determinant."""
        if (L := self.compute_kernel(cache)) is None:
            return None

        n_groups = self.config.n_groups * self.distributed_mul
        assert n_groups <= L.size(0), "Number of requested selections cannot exceed kernel size."
        self._ensure_buffers(L.size(0), L.device, L.dtype, transversal=False)

        if HAS_TRITON and L.device.type == "cuda":
            ret = _run_triton_greedy_map(L, n_groups, self._precomputed[0], self._bufs)
        else:
            ret = _greedy_map_cpu(L, n_groups, self._precomputed[0], self._precomputed[1], self._precomputed[2])

        if not self._guard_unique(ret):
            ret = fallback_greedy(L, n_groups)
        return ret


def _greedy_map_cpu(
    kernel: torch.Tensor,
    num_groups: int,
    item_to_group: torch.Tensor,
    group_member_table: torch.Tensor,
    start_items: torch.Tensor,
) -> torch.Tensor:
    """CPU fallback for greedy MAP-DPP."""
    n_items = kernel.size(0)
    epsilon = 1e-10

    selected = torch.empty((n_items, num_groups), dtype=torch.long, device=kernel.device)
    log_dets = torch.zeros(n_items, dtype=kernel.dtype, device=kernel.device)

    selected[:, 0] = start_items

    diag = torch.diagonal(kernel).clamp(min=epsilon)
    log_dets += torch.log(diag)

    di2s = diag.unsqueeze(0).expand(n_items, -1).clone()

    start_groups = item_to_group[start_items]
    start_members = group_member_table[start_groups]
    di2s.scatter_(1, start_members, -float("inf"))

    # Store Gram-Schmidt basis vectors
    e_all = torch.zeros((n_items, num_groups, n_items), dtype=kernel.dtype, device=kernel.device)

    e_0 = kernel[start_items] / torch.sqrt(diag).unsqueeze(1)
    e_all[:, 0, :] = e_0
    di2s -= e_0**2

    for k in range(1, num_groups):
        next_items = torch.argmax(di2s, dim=1)
        selected[:, k] = next_items

        di_sq = di2s[torch.arange(n_items), next_items].clamp(min=epsilon)
        log_dets += torch.log(di_sq)

        next_groups = item_to_group[next_items]
        next_members = group_member_table[next_groups]
        di2s.scatter_(1, next_members, -float("inf"))

        if k < num_groups - 1:
            e_new = kernel[next_items].clone()
            e_active = e_all[:, :k, :]

            # Project out active basis using batched gather for coeffs
            coeffs = e_active[torch.arange(n_items).unsqueeze(1), torch.arange(k).unsqueeze(0), next_items.unsqueeze(1)]

            e_new -= torch.bmm(coeffs.unsqueeze(1), e_active).squeeze(1)
            e_new /= torch.sqrt(di_sq).unsqueeze(1)
            e_all[:, k, :] = e_new

            di2s -= e_new**2

    return selected[torch.argmax(log_dets), :]


if __name__ == "__main__":
    import timeit

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on: {device}")

    # Simulated Kernel Data
    n_items, n_groups = 128, 8
    dummy_kernel = torch.randn(n_items, n_items, device=device)
    dummy_kernel = dummy_kernel @ dummy_kernel.T  # make it PSD to prevent complex sqrts

    # Simulate Transversal group structure
    group_size_each = n_items // n_groups
    dummy_item_to_group = torch.arange(n_groups, device=device).repeat_interleave(group_size_each)

    if device == "cuda" and HAS_TRITON:
        # Pre-allocate Triton buffers
        max_vecs = n_groups - 1
        triton_bufs = [
            torch.empty(n_items, dtype=dummy_kernel.dtype, device=device),
            torch.empty((n_items, n_groups), dtype=torch.long, device=device),
            torch.empty(n_items, dtype=dummy_kernel.dtype, device=device),
            torch.empty((n_items, max_vecs, n_items), dtype=dummy_kernel.dtype, device=device),
        ]

        # Warmup and compile JIT
        _run_triton_greedy_map(dummy_kernel, n_groups, dummy_item_to_group, triton_bufs)

        print("\nTriton Kernel Benchmark (CUDA):")
        print(
            timeit.timeit(
                lambda: _run_triton_greedy_map(dummy_kernel, n_groups, dummy_item_to_group, triton_bufs),
                number=1000,
            ),
        )
