"""Correlation experiment comparing entropy and self-certainty scoring methods against GPT-2 PPL."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

from config import Cache, Config
from eval_core import Perplexity
from mdlm_ref.modeling_mdlm import MDLM
from subsample.base import _compute_scores
from utils import get_tokenizer, seed_all, tqdm


@torch.no_grad()
def forward_process(batch, mask_id):
    """
    Randomly mask tokens in the batch using a range of mask ratios.
    Based on the LLADA reference implementation.
    """
    b, L = batch.shape
    device = batch.device

    # Sample a starting number of tokens to mask
    k = torch.randint(1, L + 1, (), device=device)

    # Create a range of masking ratios across the batch for better MC coverage
    # This matches the logic from the LLADA reference _get_log_likelihood.py
    x = torch.round(torch.linspace(float(k), k + (b - 1) * (L / b), steps=b, device=device)).long()
    x = ((x - 1) % L) + 1

    indices = torch.arange(L, device=device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(L)]

    noisy_batch = torch.where(is_mask, mask_id, batch)
    p_mask = (x / L).unsqueeze(1).repeat(1, L)
    return noisy_batch, p_mask, is_mask


@torch.no_grad()
def get_mdlm_log_likelihood(model, sequence, mc_num=128, batch_size=16, mask_id=None) -> float:
    """
    Estimate the log-likelihood of a sequence using Monte Carlo samples.
    Adaptation of the LLADA likelihood calculation logic for Discrete Diffusion.
    """
    device = model.device
    # Repeat sequence for batch processing of MC samples
    seq_batch = sequence[None, :].repeat(batch_size, 1).to(device)

    likelihood_sum = 0.0
    num_batches = mc_num // batch_size

    for _ in range(num_batches):
        perturbed_seq, p_mask, mask_index = forward_process(seq_batch, mask_id)

        # Use the mask ratio as the timestep for MDLM
        timesteps = p_mask[:, 0]

        output = model(perturbed_seq, timesteps=timesteps, return_dict=True)
        logits = output.logits

        # Compute Log-Likelihood estimate: log P(x) ~= E [ -CE(logits, target) / ratio ]
        # We only consider loss on masked tokens as per reference
        flat_logits = logits[mask_index]
        flat_targets = seq_batch[mask_index]
        flat_ratios = p_mask[mask_index]

        ce_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        weighted_log_p = -ce_loss / flat_ratios

        # Aggregate log-probs per sample in the batch
        sample_indices = torch.where(mask_index)[0]
        sample_log_probs = torch.zeros(batch_size, device=device)
        sample_log_probs.index_add_(0, sample_indices, weighted_log_p)

        likelihood_sum += sample_log_probs.mean().item()

    return likelihood_sum / num_batches


@torch.no_grad()
def compute_internal_scores(model, sequence, mask_id, mc_num=64, batch_size=16) -> tuple[float, float]:
    """
    Compute both entropy and self-certainty scores for a sequence.

    Returns:
        tuple: (entropy_score, self_certainty_score) both in [0, 1]
    """
    device = model.device
    seq_batch = sequence[None, :].repeat(batch_size, 1).to(device)

    entropy_scores_accum: list[float] = []
    self_certainty_scores_accum: list[float] = []

    num_batches = mc_num // batch_size

    for _ in range(num_batches):
        perturbed_seq, p_mask, _ = forward_process(seq_batch, mask_id)
        timesteps = p_mask[:, 0]

        output = model(perturbed_seq, timesteps=timesteps, return_dict=True)
        log_probs = F.log_softmax(output.logits, dim=-1)  # [B, L, V]

        # Create cache with log probabilities
        cache = Cache(log_p_x0=log_probs)

        # Compute both scores (unnormalized within batch)
        entropy_score = _compute_scores(cache, score_method="entropy")  # [B]
        self_certainty_score = _compute_scores(cache, score_method="self-certainty")  # [B]

        entropy_scores_accum.append(float(entropy_score.mean().item()))
        self_certainty_scores_accum.append(float(self_certainty_score.mean().item()))

    return np.mean(entropy_scores_accum).item(), np.mean(self_certainty_scores_accum).item()


def main():  # noqa: PLR0915
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Score Method Correlation Experiment on {device}")

    seed_all(config.seed)

    # 1. Load MDLM
    print(f"Loading MDLM from {config.mdlm_model_path}...")
    mdlm_model = (
        MDLM.from_pretrained(
            config.mdlm_model_path,
            cache_dir=config.cache_dir,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    mdlm_tokenizer = get_tokenizer(config, "mdlm")
    mask_id = mdlm_model.config.vocab_size - 1

    # 2. Load GPT-2 for Reference Quality (Perplexity)
    print("Loading GPT-2 reference model...")
    gpt2_model = AutoModel.from_pretrained("gpt2", cache_dir=config.cache_dir).to(device).eval()
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=config.cache_dir)
    ppl_evaluator = Perplexity(gpt2_model, gpt2_tokenizer)

    # 3. Data Setup
    data_path = config.data_path
    if not os.path.exists(data_path):
        print(f"Warning: Data path {data_path} not found. Using dummy data for demonstration if needed.")
        if "path_to.bin" in data_path:
            print("Please provide a valid data_path in Config or via CLI.")
            return

    print(f"Loading sequences from {data_path}...")
    data = np.memmap(data_path, dtype=np.uint16, mode="r")

    # Experiment parameters
    N_SAMPLES = 10_000
    SEQ_LENGTH = 1024
    MC_SAMPLES = 128
    MC_BATCH_SIZE = 64

    # Storage for all methods
    entropy_scores: list[float] = []
    self_certainty_scores: list[float] = []
    mdlm_likelihoods: list[float] = []
    gpt2_perplexities: list[float] = []

    print(f"Processing {N_SAMPLES} samples...")
    for i in tqdm(range(N_SAMPLES), desc="Score Estimation"):
        start_idx = np.random.randint(0, len(data) - SEQ_LENGTH - 1)
        sample_ids = data[start_idx : start_idx + SEQ_LENGTH]

        # 1. GPT-2 Perplexity
        text: str = mdlm_tokenizer.decode(sample_ids, skip_special_tokens=True)
        ppl_ = ppl_evaluator._forward([text])
        assert ppl_ is not None
        ppl = float(ppl_[0])

        # 2. MDLM Log-Likelihood
        seq_tensor = torch.from_numpy(sample_ids.astype(np.int64)).to(device)

        ll = get_mdlm_log_likelihood(
            mdlm_model,
            seq_tensor,
            mc_num=MC_SAMPLES,
            batch_size=MC_BATCH_SIZE,
            mask_id=mask_id,
        )

        # 3. Internal scores (entropy and self-certainty)
        entropy_s, self_cert_s = compute_internal_scores(
            mdlm_model,
            seq_tensor,
            mask_id,
            mc_num=MC_SAMPLES,
            batch_size=MC_BATCH_SIZE,
        )

        entropy_scores.append(entropy_s)
        self_certainty_scores.append(self_cert_s)
        mdlm_likelihoods.append(ll)
        gpt2_perplexities.append(ppl)

    if len(mdlm_likelihoods) < 2:
        print("Insufficient data collected. Check errors above.")
        return

    # Convert to arrays
    entropy_scores = np.array(entropy_scores)  # type: ignore
    self_certainty_scores = np.array(self_certainty_scores)  # type: ignore
    mdlm_likelihoods = np.array(mdlm_likelihoods)  # type: ignore
    gpt2_perplexities = np.array(gpt2_perplexities)  # type: ignore

    # Reference: GPT-2 log-likelihood (approx -log PPL)
    ref_log_likelihoods = -np.log(gpt2_perplexities)

    # Compute correlations for all methods
    corr_ll, p_ll = spearmanr(mdlm_likelihoods, ref_log_likelihoods)
    corr_entropy, p_entropy = spearmanr(entropy_scores, ref_log_likelihoods)
    corr_self_cert, p_self_cert = spearmanr(self_certainty_scores, ref_log_likelihoods)

    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS: Score Method Correlation with GPT-2 PPL")
    print("=" * 60)
    print(f"\nSamples processed: {len(mdlm_likelihoods)}")
    print(f"\n{'Method':<25} {'Spearman Corr':<15} {'p-value':<15}")
    print("-" * 55)
    print(f"{'MDLM Log-Likelihood':<25} {corr_ll:>12.4f}   {p_ll:>12.2e}")
    print(f"{'Entropy Score':<25} {corr_entropy:>12.4f}   {p_entropy:>12.2e}")
    print(f"{'Self-Certainty Score':<25} {corr_self_cert:>12.4f}   {p_self_cert:>12.2e}")
    print("=" * 60)

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: MDLM Log-Likelihood
    axes[0].scatter(ref_log_likelihoods, mdlm_likelihoods, alpha=0.4, c="darkblue", s=10)
    axes[0].set_xlabel("GPT-2 Log-Likelihood (-log PPL)")
    axes[0].set_ylabel("MDLM Log-Likelihood")
    axes[0].set_title(f"MDLM LL\nSpearman: {corr_ll:.4f}")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Entropy Score
    axes[1].scatter(ref_log_likelihoods, entropy_scores, alpha=0.4, c="darkgreen", s=10)
    axes[1].set_xlabel("GPT-2 Log-Likelihood (-log PPL)")
    axes[1].set_ylabel("Entropy Score")
    axes[1].set_title(f"Entropy\nSpearman: {corr_entropy:.4f}")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Plot 3: Self-Certainty Score
    axes[2].scatter(ref_log_likelihoods, self_certainty_scores, alpha=0.4, c="darkred", s=10)
    axes[2].set_xlabel("GPT-2 Log-Likelihood (-log PPL)")
    axes[2].set_ylabel("Self-Certainty Score")
    axes[2].set_title(f"Self-Certainty\nSpearman: {corr_self_cert:.4f}")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Score Method Correlation with External Quality (GPT-2)", fontsize=14)
    plt.tight_layout()

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "score_method_correlation.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot saved to {plot_path}")

    # Save results
    np.savez(
        os.path.join(results_dir, "score_method_results.npz"),
        entropy_scores=entropy_scores,
        self_certainty_scores=self_certainty_scores,
        mdlm_ll=mdlm_likelihoods,
        gpt2_ppl=gpt2_perplexities,
        ref_ll=ref_log_likelihoods,
    )
    print(f"Raw results saved to {results_dir}/score_method_results.npz")


if __name__ == "__main__":
    main()
