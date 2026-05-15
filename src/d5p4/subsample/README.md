# Subsample Module

This module implements various subsampling methods used to select high-quality and diverse subsets from a pool of generated text candidates.

## Subsampling Methods

### Baseline
Does not apply any subsampling. It simply passes through the candidates. Only compatible with `group_size=1`.

### Random
Implements random subsampling. Useful as a baseline for diversity metrics.

### Beam Search
- **Naive Beam**: Selects the top-$k$ candidates based on their log-probability/score.
- **Diverse Beam**: Uses Maximal Marginal Relevance (MMR) to select a diverse set of candidates. Controlled by `_diversity_alpha`.

### DPP Selector
Uses the `dppy` library to implement Determinantal Point Process (DPP) based subsampling.

### Greedy MAP
Implements a fast greedy MAP inference algorithm for DPPs. This is an efficient approximation of the optimal DPP selection, based on the paper ["Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity"](https://arxiv.org/abs/1709.05135).

Two implementations are provided:
- `greedy_map`: Fast Triton-based GPU kernel implementation.
- `_greedy_map`: Reference plain PyTorch implementation.
Both versions are equivalent in output; the Triton version is recommended for production use on CUDA devices.

#### Rough GPU flow for transversal MAP-DPP

For `method=greedy_map` with `transversal=True`, the selector treats each beam/group as a partition and must keep exactly one candidate from each partition. In distributed inference, `n_groups` is local to each rank, and MAP-DPP runs on the gathered global pool.

A small two-GPU example with `world_size=2`, `n_groups=3`, and `group_size=2` looks like this:

```text
Before the distributed selector call

  rank 0 local batch                  rank 1 local batch
  +----------+----------+----------+  +----------+----------+----------+
  | g0       | g1       | g2       |  | g3       | g4       | g5       |
  | id 0 1   | id 2 3   | id 4 5   |  | id 6 7   | id 8 9   | id 10 11 |
  | x0  x1   | x2  x3   | x4  x5   |  | x6  x7   | x8  x9   | x10 x11  |
  +----------+----------+----------+  +----------+----------+----------+

  local batch_size = n_groups * group_size = 6
  global items     = world_size * local batch_size = 12
  global groups    = world_size * n_groups = 6

  item_to_group = [0,0, 1,1, 2,2, 3,3, 4,4, 5,5]
  selector must return 6 global ids, one per global group
```

The synchronization-heavy part is the handoff from local model outputs to a global rank-0 MAP decision:

```text
rank 0 GPU                                  rank 1 GPU
---------                                   ---------
model forward                              model forward
  local log_p_x0 [6,L,V]                     local log_p_x0 [6,L,V]
  local embeds   [6,L,H]                     local embeds   [6,L,H]
      |                                           |
      | scores0, flat0                            | scores1, flat1
      +--------------------+----------------------+
                           |
                           v
                 SYNC 1: all_gather(flat, scores)
                           |
                           v
rank 0 receives global tensors:
  flat   [12, L*H]
  scores [12]

rank 1 receives no selector work after gather:
  compute_kernel returns None on nonzero ranks
```

The tensors used by MAP-DPP stay on the GPU in the normal CUDA path:

```text
model forward
    |
    | produces cache.log_p_x0:  [B, L, V]
    | produces cache.embeddings:[B, L, H]
    v
BaseSelector.compute_kernel(cache)
    |
    | 1. scores = quality(cache.log_p_x0)
    |      - entropy or self-certainty
    |      - normalized to [0, 1]
    |
    | 2. flat = normalize(reshape(cache.embeddings, [B, L*H]))
    |
    | 3. distributed sync:
    |      all_gather(flat, scores)
    |      - every rank contributes its local candidates
    |      - rank 0 receives the global candidate pool
    |      - nonzero ranks return None until selected ids are dispatched back
    |
    | 4. S = similarity(flat)
    |      - cosine: flat @ flat.T
    |      - rbf: exp(-gamma * cdist(flat)^2)
    |
    | 5. K = DPP kernel
    |      additive:       K = w_interaction * S + diag(scores)
    |      multiplicative: K_ij = q_i * S_ij * q_j
    v
GreedyMAP._transversal(cache)
```

For the two-rank example, rank 0 builds a `12 x 12` kernel. The diagonal carries the quality term; the off-diagonal entries carry similarity/diversity information. At the rank level, the matrix is:

```text
                 columns 0..5       columns 6..11
              +------------------+------------------+
  rows 0..5   | rank0 x rank0   | rank0 x rank1   |
              +------------------+------------------+
  rows 6..11  | rank1 x rank0   | rank1 x rank1   |
              +------------------+------------------+
```

Inside that block matrix, the item-level structure is still one row and one column per candidate:

```text
            c0   c1   c2   c3   c4   c5   c6   c7   c8   c9   c10  c11
          +----+----+----+----+----+----+----+----+----+----+----+----+
  g0 c0   | q  | s  | s  | s  | s  | s  | s  | s  | s  | s  | s  | s  |
     c1   | s  | q  | s  | s  | s  | s  | s  | s  | s  | s  | s  | s  |
  g1 c2   | s  | s  | q  | s  | s  | s  | s  | s  | s  | s  | s  | s  |
     c3   | s  | s  | s  | q  | s  | s  | s  | s  | s  | s  | s  | s  |
  ...     | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. |
  g5 c10  | s  | s  | s  | s  | s  | s  | s  | s  | s  | s  | q  | s  |
     c11  | s  | s  | s  | s  | s  | s  | s  | s  | s  | s  | s  | q  |
          +----+----+----+----+----+----+----+----+----+----+----+----+
```

The Triton implementation then explores every possible starting item in parallel on rank 0. Each Triton program owns one trajectory, so with twelve global candidates it evaluates twelve greedy completions:

```text
program 0 starts at c0: [c0, ?, ?, ?, ?, ?]
program 1 starts at c1: [c1, ?, ?, ?, ?, ?]
...
program 10 starts at c10: [c10, ?, ?, ?, ?, ?]
program 11 starts at c11: [c11, ?, ?, ?, ?, ?]

Each trajectory selects 6 total items, one from each global group.
```

Inside each trajectory, the selected item's whole group is masked out, so the next argmax cannot pick another candidate from the same partition:

```text
Example global trajectory starting at c1

  step 0:
    selected = [c1]
    mask group 0 -> c0 and c1 are no longer selectable

  step 1:
    pick best remaining marginal determinant item, say c8
    selected = [c1, c8]
    mask group 4 -> c8 and c9 are no longer selectable

  ...

  valid transversal output:
    one from each of groups 0, 1, 2, 3, 4, 5
```

At each step the kernel row for the newly selected item is loaded, orthogonalized against the trajectory's previous basis vectors, normalized, and used to update the remaining marginal gains:

```text
selected item -> load K[selected, :]
              -> remove projections onto previous e vectors
              -> normalize by current marginal gain
              -> store e vector in GPU work buffer
              -> update di2s = di2s - e_new^2
              -> argmax(di2s) gives the next item
```

After rank 0 picks the best trajectory, the selected global ids are sent back through `dispatch_batch_indices`:

```text
rank 0 best global ids:
  [1, 2, 5, 6, 9, 10]

             global ids owned by each rank
             rank 0: [0, 1, 2, 3, 4, 5]
             rank 1: [6, 7, 8, 9, 10, 11]

SYNC 2: dispatch_batch_indices(selected_global_ids)

rank 0 receives local slice_idx:
  [1, 2, 5]

rank 1 receives local slice_idx:
  [0, 3, 4]
  because global [6, 9, 10] - rank_offset 6 = local [0, 3, 4]
```

The main synchronization points are:

```text
SYNC 1: all_gather(flat, scores)
  - all ranks wait here
  - rank 0 receives the global score/embedding pool
  - nonzero ranks have no global kernel to build

rank 0 only:
  - builds K on GPU
  - launches Triton MAP-DPP
  - runs GPU argmax(log_dets) to choose the best trajectory
  - validation may read small scalar values back to Python

SYNC 2: dispatch_batch_indices(selected_global_ids)
  - rank 0 contributes selected ids
  - other ranks contribute empty padded buffers
  - every rank receives the local indices it owns

local continuation on every rank:
  - slice local rows with local slice_idx
  - repeat survivors with repeat_interleave(group_size)
  - resample to refill the local expanded batch
```

So, for transversal MAP-DPP, scores and embeddings do not move to CPU for the actual kernel construction or greedy MAP search. They are reshaped, gathered when distributed, and transformed into the DPP kernel on GPU; the selected indices are then used to slice the candidate batch for the next diffusion step.

#### Graphic view of slice, repeat, and resample

In LLaDA, the selector is called before the next block sample is drawn. The current batch already contains expanded candidates from the previous subsampling step. MAP-DPP chooses the survivors, and each survivor is then repeated to refill the local beam/group structure.

Continuing the same two-GPU example, each rank repeats and resamples only its local survivors. For rank 0:

```text
Before selector.subsample(cache) on rank 0

  local row   0      1      2      3      4      5
  global id   0      1      2      3      4      5
            +------+------+------+------+------+------+
  x         | x0   | x1   | x2   | x3   | x4   | x5   |
  logits    | l0   | l1   | l2   | l3   | l4   | l5   |
  embeds    | e0   | e1   | e2   | e3   | e4   | e5   |
            +------+------+------+------+------+------+
  group      g0     g0     g1     g1     g2     g2

  cache passed to MAP-DPP contains:
    log_p_x0[:, block_start:block_end]
    embeddings[:, block_start:block_end]
    x[:, block_start:block_end]
```

After global dispatch, rank 0 receives local `slice_idx`:

```text
slice_idx = [1, 2, 5]

  keep row 1 from group 0
  keep row 2 from group 1
  keep row 5 from group 2
```

Rank 1 does the same thing independently with its own local `slice_idx`:

```text
rank 1 global survivors = [6, 9, 10]
rank 1 local slice_idx  = [0, 3, 4]
```

On each rank, the logits used to sample the next candidate tokens are sliced before expansion:

```text
logits_to_sample = index_select(log_p_x0, dim=0, index=slice_idx)

            +------+------+------+
  rows      |  1   |  2   |  5   |
            +------+------+------+
  logits    | l1   | l2   | l5   |
            +------+------+------+
```

Then each rank repeats its selected rows to rebuild `group_size=2` candidates per local survivor:

```text
expanded_idx = repeat_interleave(slice_idx, group_size)
             = [1, 1, 2, 2, 5, 5]

            +------+------+------+------+------+------+
  new rows  |  1   |  1   |  2   |  2   |  5   |  5   |
            +------+------+------+------+------+------+
  x         | x1   | x1   | x2   | x2   | x5   | x5   |
  log_p_x0  | l1   | l1   | l2   | l2   | l5   | l5   |
  masks     | m1   | m1   | m2   | m2   | m5   | m5   |
            +------+------+------+------+------+------+
  new group   g0     g0     g1     g1     g2     g2
```

Finally `_block_sample(logits_to_sample, subsample_step=True)` produces `group_size` samples per kept row:

```text
input logits_to_sample rows:

            +------+------+------+
  rows      |  1   |  2   |  5   |
            +------+------+------+

sample_categorical(..., expand=2) or argmax + repeat:

            +------+------+------+------+------+------+
  samples   | s1a  | s1b  | s2a  | s2b  | s5a  | s5b  |
            +------+------+------+------+------+------+
  paired x  | x1   | x1   | x2   | x2   | x5   | x5   |
            +------+------+------+------+------+------+

candidate_x0 = where(mask_index, samples, repeated_x)
```

So the high-level multi-GPU loop is:

```text
rank-local expanded candidates
          |
          | compute local scores + local embeddings
          v
SYNC 1: all_gather scores/embeddings
          |
          v
rank 0 builds global DPP kernel and runs MAP-DPP
          |
          v
SYNC 2: dispatch selected global ids back to owner ranks
          |
          v
rank-local slice_idx
          |
          | slice logits for sampling
          | expanded_idx = slice_idx repeated group_size times
          v
repeat local state rows to refill beams
          |
          | draw group_size token samples per local survivor
          v
next rank-local expanded candidate batch
```

MDLM follows the same conceptual shape, but the repeat happens around the DDPM transition tensors: it slices `p_x0`, `move_chance_*`, `copy_flag`, and `original_x` by `slice_idx`; sampling expands by `group_size`; then `copy_flag` and `original_x` are repeated so preserved tokens line up with the expanded samples.

### Exhaustive
Performs an exhaustive search over all possible subsets to find the one that maximizes the subdeterminant. Only practical for very small pool sizes because of its combinatorial complexity.

## Partitioned (Transversal) Sampling

The project supports **Transversal Selection** (controlled by `transversal=True` in `Config`). In this mode, the pool is partitioned into groups, and the selector must choose exactly one item from each group. This is particularly useful to avoid *ancestral collapse*, as described in the Diverse Beam Search paper.

## Key Configuration Parameters

- `method`: The subsampling algorithm to use (e.g., `greedy_map`, `_greedy_map`, `dpp`, `random`).
- `transversal`: Boolean flag to enable/disable partitioned selection.
- `_w_interaction`: Weight for the diversity term in DPP methods (higher means more diversity).
- `_kernel_type`: Similarity kernel to use (default: `cosine`, fallback: `rbf`).
- `_score_method`: Quality score metric (`entropy` or `self-certainty`).

## Benchmarking

Use `selector_benchmark.py` to compare methods on a fixed synthetic setup:

```bash
uv run python -m d5p4.subsample.selector_benchmark
```

This script evaluates algorithms based on:
1. **Log-Determinant Quality**: A measure of group diversity.
2. **MAE (Oracle)**: Mean Absolute Error compared to exhaustive search.
3. **Validity**: Percentage of selections that satisfy transversal constraints.
4. **Time**: Execution speed.

Other benchmark scripts:
- `reference_rank_benchmark.py`: percentile rank against random valid partitions.
- `scaling_benchmark.py`: quality and latency over larger group/item grids.
- `kernel_method_benchmark.py`: additive vs multiplicative DPP kernel comparison.
