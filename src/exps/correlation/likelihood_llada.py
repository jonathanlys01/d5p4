"""Correlation experiment comparing LLaDA entropy and self-certainty scores against Llama-3 Log-Likelihood."""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Cache, Config
from data.qa import get_qa_dataset
from llada_ref.modeling_llada import LLaDAModelLM
from subsample.base import _compute_scores
from utils import seed_all, tqdm


# Set non-interactive backend for matplotlib to avoid issues on clusters
matplotlib.use("Agg")


def forward_process(batch, prompt_index, mask_id):
    """
    Randomly mask tokens in the answer part of the batch using a range of mask ratios.
    Adapted from src/llada_ref/_get_log_likelohood.py
    """
    b, L = batch.shape

    # Calculate length of the target (answer) part
    target_len = (L - prompt_index.sum()).item()
    if target_len <= 0:
        # Should not happen with proper prompts, but safety check
        return None

    # Sample a starting number of tokens to mask
    k = torch.randint(1, target_len + 1, (), device=batch.device)

    # Create a range of masking ratios across the batch
    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    # Concatenate unmasked prompt (False) with masked answer (is_mask)
    # prompt_index.sum() should be equal for all if rectangular, but here we assume single sequence logic repeated
    # In batch processing, we need to be careful. The original code assumed single sequence repeated.
    # Here, we will assume the caller handles repetition.

    # Reconstruct full mask: prompt is NEVER masked.
    prompt_len = int(prompt_index.sum().item())
    full_mask = torch.cat(
        (torch.zeros(b, prompt_len, dtype=torch.bool, device=batch.device), is_mask),
        dim=1,
    )

    noisy_batch = torch.where(full_mask, mask_id, batch)

    # Return noisy batch and mask ratio (for entire sequence length, though only target is masked)
    p_mask: torch.Tensor = (x / target_len).unsqueeze(1).repeat(1, L)
    return noisy_batch, p_mask, full_mask


@torch.no_grad()
def get_llada_log_likelihood(model, prompt, answer, mc_num=128, batch_size=16, mask_id=126336) -> float:  # noqa: PLR0913
    """
    Estimate the log-likelihood of (answer | prompt) using LLaDA.
    """
    device = model.device

    # Concatenate prompt and answer
    seq = torch.cat([prompt, answer]).unsqueeze(0)  # [1, L]
    seq_batch = seq.repeat(batch_size, 1).to(device)

    # Create boolean index for prompt (True for prompt tokens)
    prompt_index = torch.arange(seq.shape[1], device=device) < len(prompt)

    likelihood_sum = 0.0
    num_batches = mc_num // batch_size
    if num_batches == 0:
        num_batches = 1
        batch_size = mc_num  # Adjust if mc_num < default batch_size

    for _ in range(num_batches):
        output = forward_process(seq_batch, prompt_index, mask_id)
        assert output is not None
        perturbed_seq, p_mask, mask_index = output

        # Forward pass
        output = model(perturbed_seq)
        logits = output.logits

        # Compute Log-Likelihood estimate
        # We only consider loss on masked tokens (which are part of the answer)
        # The reference implementation divides by p_mask (ratio) to debias

        flat_logits = logits[mask_index]
        flat_targets = seq_batch[mask_index]
        flat_ratios = p_mask[mask_index]

        ce_loss = F.cross_entropy(flat_logits, flat_targets, reduction="mean")

        # Weighted log prob: -CE / ratio
        weighted_log_p = -ce_loss / flat_ratios

        # Aggregate per sample. Since we flattened, we need to be careful.
        # But wait, original code:
        # loss = F.cross_entropy(logits[mask_index], seq[mask_index], reduction="none") / p_mask[mask_index]
        # loss = loss.sum() / batch_size
        # return -sum(loss_) / len(loss_)
        # It computes AVG loss per batch.
        # Let's match the logic: Sum of (CE/ratio) over all masked tokens, averaged over batch size.
        # The return value is Log Likelihood (negative of loss).

        batch_loss = weighted_log_p.sum()  # Sum of all weighted losses in batch

        # We need the average log-likelihood per sample.
        # The original code sums up loss for the *entire batch* then divides by batch_size.
        # That means it's calculating E[Loss] per sample.

        likelihood_sum += batch_loss.item() / batch_size

    return likelihood_sum / num_batches


@torch.no_grad()
def compute_internal_scores(model, prompt, answer, mc_num=64, batch_size=16, mask_id=126336) -> tuple[float, float]:  # noqa: PLR0913
    """
    Compute entropy and self-certainty scores for LLaDA on the answer part.
    """
    device = model.device
    seq = torch.cat([prompt, answer]).unsqueeze(0)
    seq_batch = seq.repeat(batch_size, 1).to(device)
    prompt_index = torch.arange(seq.shape[1], device=device) < len(prompt)

    entropy_scores_accum: list[float] = []
    self_certainty_scores_accum: list[float] = []

    num_batches = mc_num // batch_size
    if num_batches == 0:
        num_batches = 1
        batch_size = mc_num

    for _ in range(num_batches):
        out = forward_process(seq_batch, prompt_index, mask_id)
        assert out is not None

        perturbed_seq, _, _ = out

        output = model(perturbed_seq)
        log_probs = F.log_softmax(output.logits, dim=-1)  # [B, L, V]

        # Compute scores
        # Note: _compute_scores calculates scores for the WHOLE sequence.
        # We strictly only care about the answer part for correlation?
        # Typically, for generation quality estimation, we look at the whole sequence or just the generated part.
        # Given we are correlating with P(Answer|Prompt), we should probably focus on Answer tokens.
        # However, `_compute_scores` returns [B] (mean over sequence).
        # Let's slice the log_probs to only include answer part before passing to cache?
        # Or just compute on full. The prompt is unmasked, so confidence should be high/fixed?
        # Actually in LLaDA forward_process, prompt is NOT masked.
        # So the model sees the prompt perfectly. It should be very confident about it (it's input).
        # Including prompt might dilute the score signal of the answer.
        # Let's slice to answer part only.

        answer_start = len(prompt)
        log_probs_answer = log_probs[:, answer_start:, :]

        cache_answer = Cache(log_p_x0=log_probs_answer)

        entropy_score = _compute_scores(cache_answer, score_method="entropy")  # [B]
        self_certainty_score = _compute_scores(cache_answer, score_method="self-certainty")  # [B]

        entropy_scores_accum.append(float(entropy_score.mean().item()))
        self_certainty_scores_accum.append(float(self_certainty_score.mean().item()))

    return np.mean(entropy_scores_accum).item(), np.mean(self_certainty_scores_accum).item()


@torch.no_grad()
def compute_ar_likelihood(model, tokenizer, prompt_text, answer_text) -> float:
    """
    Compute log P(answer | prompt) using an Autoregressive model.
    """
    # Tokenize
    # We want Input = Prompt + Answer
    # Label = Prompt (ignore) + Answer (loss)

    # Mask prompt in labels (-100 is default ignore_index in CrossEntropyLoss)
    # Note: Llama tokenizer might add beginning of sentence token?
    # We should verify. Typically AutoTokenizer handles it.
    # If we simply rely on length, we might be off by 1 if merge happens.
    # For correlation experiments, slight misalignment is usually acceptable, but let's try to be precise.
    # Better: encode prompt, encode answer, concat.

    prompt_enc = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt").input_ids.to(model.device)
    # For answer, we don't want special tokens if they are BOS
    answer_enc = tokenizer(answer_text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)

    # Concatenate
    input_ids = torch.cat([prompt_enc, answer_enc], dim=1)
    labels = input_ids.clone()

    # Mask prompt
    labels[:, : prompt_enc.shape[1]] = -100

    # Forward
    outputs = model(input_ids, labels=labels)

    # Loss is avg neg log likelihood per token
    # We want total log likelihood? Or avg log likelihood?
    # Usually LL per token or Perplexity.
    # Function returns likelihood (log capability).
    # metrics usually are -loss.

    return -outputs.loss.item()


def main():  # noqa: PLR0915
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting LLaDA Score Correlation Experiment on {device}")

    seed_all(config.seed)

    # 1. Load LLaDA
    print(f"Loading LLaDA from {config.llada_model_path}...")
    llada_model = (
        LLaDAModelLM.from_pretrained(
            config.llada_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=config.cache_dir,
        )
        .to(device)
        .eval()
    )
    llada_tokenizer = AutoTokenizer.from_pretrained(
        config.llada_model_path,
        trust_remote_code=True,
        cache_dir=config.cache_dir,
    )
    mask_id = 126336  # Standard MASK ID for LLaDA

    # 2. Load AR Reference (Llama-3)
    print(f"Loading AR Reference from {config.ar_model_path}...")
    ar_model = (
        AutoModelForCausalLM.from_pretrained(
            config.ar_model_path,
            cache_dir=config.cache_dir,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    ar_tokenizer = AutoTokenizer.from_pretrained(config.ar_model_path, cache_dir=config.cache_dir)

    # 3. Load QA Dataset
    print(f"Loading QA Dataset: {config.qa_dataset}...")
    qa_df = get_qa_dataset(config)
    if config.qa_dataset_len > 0:
        qa_df = qa_df.head(config.qa_dataset_len)

    print(f"Processing {len(qa_df)} samples...")

    # Experiment parameters
    MC_SAMPLES = 64
    MC_BATCH_SIZE = 16

    # Storage
    entropy_scores = []
    self_certainty_scores = []
    llada_likelihoods = []
    ar_likelihoods = []

    for idx, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Score Estimation"):
        question = row["question"]
        # Use the first correct answer
        if len(row["correct_answers"]) == 0:
            continue
        answer = row["correct_answers"][0]

        # Prepare Tensors for LLaDA
        # LLaDA expects explicit tensors for prompt and answer
        p_ids = llada_tokenizer(question, add_special_tokens=False)["input_ids"]
        a_ids = llada_tokenizer(answer, add_special_tokens=False)["input_ids"]

        prompt_tensor = torch.tensor(p_ids, device=device)
        answer_tensor = torch.tensor(a_ids, device=device)

        # 1. LLaDA Log-Likelihood
        ll_llada = get_llada_log_likelihood(
            llada_model,
            prompt_tensor,
            answer_tensor,
            mc_num=MC_SAMPLES,
            batch_size=MC_BATCH_SIZE,
            mask_id=mask_id,
        )

        # 2. Internal scores
        entropy_s, self_cert_s = compute_internal_scores(
            llada_model,
            prompt_tensor,
            answer_tensor,
            mc_num=MC_SAMPLES,
            batch_size=MC_BATCH_SIZE,
            mask_id=mask_id,
        )

        # 3. AR Log-Likelihood
        ll_ar = compute_ar_likelihood(ar_model, ar_tokenizer, question, answer)

        entropy_scores.append(entropy_s)
        self_certainty_scores.append(self_cert_s)
        llada_likelihoods.append(ll_llada)
        ar_likelihoods.append(ll_ar)

    if len(llada_likelihoods) < 2:
        print("Insufficient data collected.")
        return

    # Convert to arrays
    entropy_scores = np.array(entropy_scores)
    self_certainty_scores = np.array(self_certainty_scores)
    llada_likelihoods = np.array(llada_likelihoods)
    ar_likelihoods = np.array(ar_likelihoods)

    # Compute correlations
    # We correlate with AR Likelihood (Higher is better)
    corr_ll, p_ll = spearmanr(llada_likelihoods, ar_likelihoods)
    corr_entropy, p_entropy = spearmanr(entropy_scores, ar_likelihoods)
    corr_self_cert, p_self_cert = spearmanr(self_certainty_scores, ar_likelihoods)

    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS: LLaDA Score Correlation with Llama-3 LL")
    print("=" * 60)
    print(f"\nSamples processed: {len(llada_likelihoods)}")
    print(f"\n{'Method':<25} {'Spearman Corr':<15} {'p-value':<15}")
    print("-" * 55)
    print(f"{'LLaDA Log-Likelihood':<25} {corr_ll:>12.4f}   {p_ll:>12.2e}")
    print(f"{'Entropy Score':<25} {corr_entropy:>12.4f}   {p_entropy:>12.2e}")
    print(f"{'Self-Certainty Score':<25} {corr_self_cert:>12.4f}   {p_self_cert:>12.2e}")
    print("=" * 60)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: LLaDA LL vs AR LL
    axes[0].scatter(ar_likelihoods, llada_likelihoods, alpha=0.4, c="darkblue", s=10)
    axes[0].set_xlabel("Llama-3 Log-Likelihood")
    axes[0].set_ylabel("LLaDA Log-Likelihood")
    axes[0].set_title(f"LLaDA LL\nSpearman: {corr_ll:.4f}")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Entropy vs AR LL
    axes[1].scatter(ar_likelihoods, entropy_scores, alpha=0.4, c="darkgreen", s=10)
    axes[1].set_xlabel("Llama-3 Log-Likelihood")
    axes[1].set_ylabel("Entropy Score")
    axes[1].set_title(f"Entropy\nSpearman: {corr_entropy:.4f}")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Plot 3: Self-Certainty vs AR LL
    axes[2].scatter(ar_likelihoods, self_certainty_scores, alpha=0.4, c="darkred", s=10)
    axes[2].set_xlabel("Llama-3 Log-Likelihood")
    axes[2].set_ylabel("Self-Certainty Score")
    axes[2].set_title(f"Self-Certainty\nSpearman: {corr_self_cert:.4f}")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("LLaDA Score Correlation with Reference Quality (Llama-3)", fontsize=14)
    plt.tight_layout()

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "score_method_correlation_llada.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot saved to {plot_path}")

    # Save raw results
    np.savez(
        os.path.join(results_dir, "score_method_results_llada.npz"),
        entropy_scores=entropy_scores,
        self_certainty_scores=self_certainty_scores,
        llada_ll=llada_likelihoods,
        ar_ll=ar_likelihoods,
    )
    print(f"Raw results saved to {results_dir}/score_method_results_llada.npz")


if __name__ == "__main__":
    main()
