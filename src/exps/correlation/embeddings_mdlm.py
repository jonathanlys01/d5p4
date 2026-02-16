import numpy as np
import torch
from transformers import AutoModel

from config import Config
from exps.correlation.common import (
    compute_avg_cosine_similarity,
    compute_cka,
    get_pooled_output,
    plot_cka_acs,
    save_results_csv,
)
from mdlm_ref.modeling_mdlm import MDLM
from utils import get_tokenizer, tqdm


def main():  # noqa: PLR0915
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ref_model_id = config.cos_model_id
    mdlm_model_id = config.mdlm_model_path
    path_to_bin = config.data_path

    N_TOTAL_SAMPLES = 2048  # total samples to process for a stable estimate
    BATCH_SIZE = 64  # max samples per chunk (limited by CKA/ACS O(n^2))
    N_BATCHES = N_TOTAL_SAMPLES // BATCH_SIZE
    print(f"Running experiment with {N_BATCHES} batches of {BATCH_SIZE} samples each (Total: {N_TOTAL_SAMPLES})")

    ref_model = AutoModel.from_pretrained(ref_model_id, cache_dir=config.cache_dir, trust_remote_code=True)
    ref_model.eval()
    ref_model.to(device)

    mdlm_embedder = MDLM.from_pretrained(mdlm_model_id, cache_dir=config.cache_dir, trust_remote_code=True)
    mask_index = mdlm_embedder.config.vocab_size - 1
    mdlm_embedder.to(device)
    mdlm_tokenizer = get_tokenizer(config, "mdlm")
    mdlm_embedder.eval()

    data = np.memmap(path_to_bin, dtype=np.uint16, mode="r")
    seq_length = 1024 - 2  # account for bos/eos tokens

    # seed for reproducibility of data sampling
    np.random.seed(42)
    torch.manual_seed(42)

    mask_ratios = list(np.linspace(0.0, 0.99, num=50))  # 0.0 to 0.99 inclusive
    pooling_strategies = ["mean", "pool_non_masked", "pool_masked", "flatten"]

    results = {strategy: {"cka": [], "acs": []} for strategy in pooling_strategies}
    all_ref_acs_scores: list[float] = []

    print("\nStarting experiment sweep...")
    for mask_ratio in mask_ratios:
        print(f"--- Testing Mask Ratio: {mask_ratio:.2f} ---")

        batch_scores_per_strategy: dict[str, dict[str, list[float]]] = {
            strategy: {"cka": [], "acs": []} for strategy in pooling_strategies
        }

        for i in tqdm(range(N_BATCHES), desc="Batches"):
            sample_texts = []
            for _ in range(BATCH_SIZE):
                start_idx = np.random.randint(0, len(data) - seq_length - 1)
                sample_ids = data[start_idx : start_idx + seq_length]
                sample_text = mdlm_tokenizer.decode(sample_ids, skip_special_tokens=True)
                sample_texts.append(sample_text)

            with torch.inference_mode():
                ref_embeddings = ref_model.encode(
                    sample_texts,
                    convert_to_tensor=True,
                    device=device,
                )

            # Only compute ref_acs_baseline if mask_ratio is 0.0 (it's constant)
            if mask_ratio == 0.0:
                all_ref_acs_scores.append(compute_avg_cosine_similarity(ref_embeddings))

            inputs = mdlm_tokenizer(
                sample_texts,
                return_tensors="pt",
                padding="max_length",
                max_length=seq_length,
                truncation=True,
            )
            bos_tensor = torch.full((inputs["input_ids"].shape[0], 1), mdlm_tokenizer.bos_token_id)
            eos_tensor = torch.full((inputs["input_ids"].shape[0], 1), mdlm_tokenizer.eos_token_id)
            base_input_ids = torch.cat([bos_tensor, inputs["input_ids"], eos_tensor], dim=1)
            base_input_ids = base_input_ids.to(device)

            masked_input_ids = base_input_ids.clone()
            rand_tensor = torch.rand(masked_input_ids.shape, device=device)
            full_token_mask = rand_tensor < mask_ratio
            masked_input_ids[full_token_mask] = mask_index

            with torch.inference_mode():
                mdlm_all_states = mdlm_embedder.forward(
                    masked_input_ids,
                    return_dict=True,
                    output_hidden_states=True,
                )
                mdlm_outputs = mdlm_all_states.hidden_states[-1]

            for strategy in pooling_strategies:
                # edge cases
                if (strategy == "pool_masked" and mask_ratio == 0.0) or (
                    strategy == "pool_non_masked" and mask_ratio == 1.0
                ):
                    batch_scores_per_strategy[strategy]["cka"].append(float("nan"))
                    batch_scores_per_strategy[strategy]["acs"].append(float("nan"))
                    continue

                with torch.inference_mode():
                    mdlm_pooled = get_pooled_output(mdlm_outputs, strategy, full_token_mask)

                cka_score = compute_cka(ref_embeddings, mdlm_pooled)
                acs_score = compute_avg_cosine_similarity(mdlm_pooled)

                batch_scores_per_strategy[strategy]["cka"].append(cka_score)
                batch_scores_per_strategy[strategy]["acs"].append(acs_score)

        print(f"    Aggregating results for mask ratio {mask_ratio:.2f}...")
        for strategy in pooling_strategies:
            avg_cka = np.mean(batch_scores_per_strategy[strategy]["cka"])
            avg_acs = np.mean(batch_scores_per_strategy[strategy]["acs"])

            results[strategy]["cka"].append(avg_cka)
            results[strategy]["acs"].append(avg_acs)
            print(f"    Strategy: {strategy:<17} | Avg CKA: {avg_cka:7.4f}, Avg ACS: {avg_acs:7.4f}")

    ref_model.to("cpu")
    mdlm_embedder.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
    print("Models offloaded to CPU.")

    final_ref_acs_baseline = float(np.mean(all_ref_acs_scores))
    print(f"Final averaged Reference ACS baseline: {final_ref_acs_baseline:.4f}")

    # Save results to CSV
    df = save_results_csv(
        results=results,
        x_values=mask_ratios,
        x_name="mask_ratio",
        filename="embeddings_mdlm_results.csv",
        ref_acs_baseline=final_ref_acs_baseline,
    )

    # Plot results
    plot_cka_acs(
        df=df,
        x_name="mask_ratio",
        title_suffix="MDLM Representation Quality",
        n_samples=N_TOTAL_SAMPLES,
        ref_acs_baseline=final_ref_acs_baseline,
        plot_filename=f"cka_acs_results_mdlm_{N_TOTAL_SAMPLES}_samples.png",
    )


if __name__ == "__main__":
    main()
