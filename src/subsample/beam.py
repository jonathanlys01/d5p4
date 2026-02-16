"""Beam search subset selectors with optional diversity penalties."""

import torch
import torch.nn.functional as F

from config import Cache
from subsample.base import BaseSelector, _compute_scores


def _sample_from_logits(scores: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
    """Sample k indices from scores, using argmax if temperature=0, else multinomial sampling."""
    if temperature == 0:
        return torch.topk(scores, k=k).indices
    scaled = (scores / temperature) - (scores / temperature).max()
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=k, replacement=False)


def _sample_per_group(
    scores: torch.Tensor,
    n_groups: int,
    group_size: int,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample one index per group (transversal), using argmax if temperature=0, else multinomial."""
    scores_grouped = scores.view(n_groups, group_size)
    if temperature == 0:
        local_indices = torch.argmax(scores_grouped, dim=1)
    else:
        scaled = (scores_grouped / temperature) - (scores_grouped / temperature).max(dim=1, keepdim=True).values
        probs = F.softmax(scaled, dim=-1)
        local_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return local_indices + torch.arange(n_groups, device=device) * group_size


class GreedyBeamSearch(BaseSelector):
    """Greedy beam search selector, quality-only (no diversity penalty)."""

    def _transversal(self, cache: Cache) -> torch.Tensor | None:
        """Transversal selection: one sample per group."""
        if (scores := self.compute_scores(cache)) is None:
            return None
        return _sample_per_group(
            scores,
            self.config.n_groups,
            self.config.group_size,
            self.config._temperature,
            scores.device,
        )

    def _non_transversal(self, cache: Cache) -> torch.Tensor | None:
        """Global top-k selection without group constraints."""
        if (scores := self.compute_scores(cache)) is None:
            return None
        return _sample_from_logits(scores, self.config.n_groups, self.config._temperature)


class DiverseBeamSearch(BaseSelector):
    """Diverse beam search selector using MMR-style diversity penalty."""

    def _transversal(self, cache: Cache) -> torch.Tensor | None:
        """Transversal selection with diversity penalty, one sample per group."""
        assert cache.embeddings is not None
        assert cache.log_p_x0 is not None

        flat = F.normalize(cache.embeddings.float().reshape(cache.embeddings.size(0), -1), dim=-1, eps=1e-12)
        scores = _compute_scores(cache)

        if self.distributed_utils:
            # Must gather BOTH tensors together to avoid rank synchronization issues
            flat, scores = self.distributed_utils.all_gather(flat, scores)
            if flat is None or scores is None:
                return None

        total_groups = self.config.n_groups * self.distributed_mul
        item_to_group = torch.arange(
            total_groups,
            device=scores.device,
        ).repeat_interleave(scores.size(0) // total_groups)
        return _diverse_beam_full_explore(scores, flat, total_groups, self.config._diversity_alpha, item_to_group)

    def _non_transversal(self, cache: Cache) -> torch.Tensor | None:
        """Global MMR selection without group constraints."""
        assert cache.embeddings is not None
        assert cache.log_p_x0 is not None

        flat = F.normalize(cache.embeddings.float().reshape(cache.embeddings.size(0), -1), dim=-1, eps=1e-12)
        scores = _compute_scores(cache)

        if self.distributed_utils:
            # Must gather BOTH tensors together to avoid rank synchronization issues
            flat, scores = self.distributed_utils.all_gather(flat, scores)
            if flat is None or scores is None:
                return None

        item_size = scores.size(0)
        item_to_group = torch.arange(item_size, device=scores.device)
        total_groups = self.config.n_groups * self.distributed_mul
        return _diverse_beam_full_explore(scores, flat, total_groups, self.config._diversity_alpha, item_to_group)


def _diverse_beam_full_explore(
    scores: torch.Tensor,
    embeddings: torch.Tensor,
    num_groups: int,
    alpha: float,
    item_to_group: torch.Tensor,
) -> torch.Tensor:
    """
    Run N parallel diverse beam search trajectories (one starting from each item).

    Uses MMR-style selection: adjusted_score = quality - alpha * diversity_penalty.
    Returns the trajectory with highest cumulative adjusted score.
    """
    device, dtype = scores.device, scores.dtype
    n_items = scores.size(0)
    emb_dim = embeddings.size(1)

    # State: selected indices, cumulative score, sum of selected embeddings, exclusion mask
    selected = torch.zeros((n_items, num_groups), dtype=torch.long, device=device)
    cumulative = torch.zeros(n_items, dtype=dtype, device=device)
    emb_sum = torch.zeros((n_items, emb_dim), dtype=dtype, device=device)
    mask = torch.zeros((n_items, n_items), dtype=dtype, device=device)

    # Step 0: Initialize each trajectory with its own starting item
    start_items = torch.arange(n_items, device=device)
    selected[:, 0] = start_items
    cumulative += scores[start_items]
    emb_sum += embeddings[start_items]

    # Mask out starting groups
    start_groups = item_to_group[start_items]
    group_mask = start_groups.unsqueeze(1) == item_to_group.unsqueeze(0)
    mask.masked_fill_(group_mask, -torch.inf)

    # Steps 1 to K-1: greedy selection with diversity penalty
    for k in range(1, num_groups):
        mean_emb = emb_sum / k
        diversity_penalty = torch.matmul(mean_emb, embeddings.T)
        adjusted = scores.unsqueeze(0) - alpha * diversity_penalty + mask

        next_items = torch.argmax(adjusted, dim=1)
        selected[:, k] = next_items
        cumulative += torch.gather(adjusted, 1, next_items.unsqueeze(1)).squeeze(1)
        emb_sum += embeddings[next_items]

        # Update mask
        new_groups = item_to_group[next_items]
        new_mask = new_groups.unsqueeze(1) == item_to_group.unsqueeze(0)
        mask.masked_fill_(new_mask, -torch.inf)

    return selected[torch.argmax(cumulative), :]
