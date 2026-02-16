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
from llada_ref.modeling_llada import LLaDAModelLM
from utils import get_tokenizer, tqdm


def main():  # noqa: C901, PLR0915
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ref_model_id = config.cos_model_id
    llada_model_id = config.llada_model_path
    path_to_bin = config.data_path

    N_TOTAL_SAMPLES = 2048  # total samples to process for a stable estimate
    BATCH_SIZE = 64  # max samples per chunk (limited by CKA/ACS O(n^2))
    N_BATCHES = N_TOTAL_SAMPLES // BATCH_SIZE
    print(f"Running experiment with {N_BATCHES} batches of {BATCH_SIZE} samples each (Total: {N_TOTAL_SAMPLES})")

    ref_model = AutoModel.from_pretrained(ref_model_id, cache_dir=config.cache_dir, trust_remote_code=True)
    ref_model.eval()
    ref_model.to(device)

    llada_embedder = LLaDAModelLM.from_pretrained(llada_model_id, cache_dir=config.cache_dir, trust_remote_code=True)
    mask_index = llada_embedder.config.mask_token_id
    llada_embedder.to(device)
    llada_tokenizer = get_tokenizer(config, "llada")
    llada_embedder.eval()

    data = np.memmap(path_to_bin, dtype=np.uint16, mode="r")
    seq_length = config.block_length  # use block_length from config (no bos/eos tokens)

    # seed for reproducibility of data sampling
    np.random.seed(42)
    torch.manual_seed(42)

    mask_ratios = list(np.linspace(0.0, 0.99, num=100))  # 0.0 to 0.99 inclusive
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
                sample_text = llada_tokenizer.decode(sample_ids, skip_special_tokens=True)
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

            # Apply chat template if using instruct model, matching diffusion_llada.py
            if "instruct" in llada_model_id.lower():
                formatted_texts = []
                for text in sample_texts:
                    message = [{"role": "user", "content": text}]
                    formatted_text = llada_tokenizer.apply_chat_template(
                        message,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    formatted_texts.append(formatted_text)
            else:
                formatted_texts = sample_texts

            inputs = llada_tokenizer(
                formatted_texts,
                return_tensors="pt",
                padding="max_length",
                max_length=seq_length,
                truncation=True,
                add_special_tokens=False,  # no bos/eos tokens, matching diffusion_llada.py
            )
            base_input_ids = inputs["input_ids"].to(device)

            masked_input_ids = base_input_ids.clone()
            rand_tensor = torch.rand(masked_input_ids.shape, device=device)
            full_token_mask = rand_tensor < mask_ratio
            masked_input_ids[full_token_mask] = mask_index

            with torch.inference_mode():
                llada_all_states = llada_embedder.forward(
                    masked_input_ids,
                    return_dict=True,
                    output_hidden_states=True,
                )
                llada_outputs = llada_all_states.hidden_states[-1]

            for strategy in pooling_strategies:
                # edge cases
                if (strategy == "pool_masked" and mask_ratio == 0.0) or (
                    strategy == "pool_non_masked" and mask_ratio == 1.0
                ):
                    batch_scores_per_strategy[strategy]["cka"].append(float("nan"))
                    batch_scores_per_strategy[strategy]["acs"].append(float("nan"))
                    continue

                with torch.inference_mode():
                    llada_pooled = get_pooled_output(llada_outputs, strategy, full_token_mask)

                cka_score = compute_cka(ref_embeddings, llada_pooled)
                acs_score = compute_avg_cosine_similarity(llada_pooled)

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
    llada_embedder.to("cpu")
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
        filename="embeddings_llada_results.csv",
        ref_acs_baseline=final_ref_acs_baseline,
    )

    # Plot results
    plot_cka_acs(
        df=df,
        x_name="mask_ratio",
        title_suffix="LLaDA Representation Quality",
        n_samples=N_TOTAL_SAMPLES,
        ref_acs_baseline=final_ref_acs_baseline,
        plot_filename=f"cka_acs_results_llada_{N_TOTAL_SAMPLES}_samples.png",
    )


if __name__ == "__main__":
    main()
