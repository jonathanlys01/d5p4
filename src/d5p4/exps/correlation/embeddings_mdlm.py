import numpy as np
import torch
from transformers import AutoModel

from d5p4.config import Config
from d5p4.exps.correlation.common import (
    compute_cka,
    compute_cosine_similarity_matrix,
    compute_cosine_similarity_stats_from_matrix,
    compute_H_from_residual_cosine_matrix,
    estimate_rho_from_cosine_matrices,
    get_pooled_output,
    plot_cka_acs,
    save_results_csv,
)
from d5p4.mdlm_ref.modeling_mdlm import MDLM
from d5p4.utils import get_tokenizer, tqdm


def main():
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
    pooling_strategies = ["mean", "pool_non_masked", "pool_masked", "flatten", "flatten_no_special"]

    results = {strategy: {"cka": [], "acs": [], "acs_std": []} for strategy in pooling_strategies}
    results["flatten_no_special"]["h_spec_norm"] = []
    results["flatten_no_special"]["rho_estimate"] = []
    all_ref_acs_scores: list[dict[str, float]] = []

    print("\nStarting experiment sweep...")
    for mask_ratio in mask_ratios:
        print(f"--- Testing Mask Ratio: {mask_ratio:.2f} ---")

        batch_scores_per_strategy: dict[str, dict[str, list[float]]] = {
            strategy: {"cka": [], "acs": [], "acs_std": []} for strategy in pooling_strategies
        }
        batch_scores_per_strategy["flatten_no_special"]["h_spec_norm"] = []
        batch_scores_per_strategy["flatten_no_special"]["rho_estimate"] = []

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
                ref_sim_matrix = compute_cosine_similarity_matrix(ref_embeddings)
                all_ref_acs_scores.append(compute_cosine_similarity_stats_from_matrix(ref_sim_matrix))

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

            cosine_matrices: dict[str, torch.Tensor] = {}

            for strategy in pooling_strategies:
                # edge cases
                if (strategy == "pool_masked" and mask_ratio == 0.0) or (
                    strategy == "pool_non_masked" and mask_ratio == 1.0
                ):
                    batch_scores_per_strategy[strategy]["cka"].append(float("nan"))
                    batch_scores_per_strategy[strategy]["acs"].append(float("nan"))
                    batch_scores_per_strategy[strategy]["acs_std"].append(float("nan"))
                    continue

                with torch.inference_mode():
                    mdlm_pooled = get_pooled_output(mdlm_outputs, strategy, full_token_mask)
                    sim_matrix = compute_cosine_similarity_matrix(mdlm_pooled)

                cosine_matrices[strategy] = sim_matrix
                cka_score = compute_cka(ref_embeddings, mdlm_pooled)
                acs_stats = compute_cosine_similarity_stats_from_matrix(sim_matrix)

                batch_scores_per_strategy[strategy]["cka"].append(cka_score)
                batch_scores_per_strategy[strategy]["acs"].append(acs_stats["mean"])
                batch_scores_per_strategy[strategy]["acs_std"].append(acs_stats["std"])

            residual_sim_matrix = cosine_matrices["flatten_no_special"]
            plain_sim_matrix = cosine_matrices["flatten"]
            rho_estimate = estimate_rho_from_cosine_matrices(plain_sim_matrix, residual_sim_matrix)
            _, h_spec_norm = compute_H_from_residual_cosine_matrix(
                residual_sim_matrix,
                rho=float(rho_estimate),
            )
            batch_scores_per_strategy["flatten_no_special"]["rho_estimate"].append(rho_estimate.item())
            batch_scores_per_strategy["flatten_no_special"]["h_spec_norm"].append(h_spec_norm.item())

        print(f"    Aggregating results for mask ratio {mask_ratio:.2f}...")
        for strategy in pooling_strategies:
            avg_cka = np.mean(batch_scores_per_strategy[strategy]["cka"])
            avg_acs = np.mean(batch_scores_per_strategy[strategy]["acs"])
            avg_acs_std = np.mean(batch_scores_per_strategy[strategy]["acs_std"])

            results[strategy]["cka"].append(avg_cka)
            results[strategy]["acs"].append(avg_acs)
            results[strategy]["acs_std"].append(avg_acs_std)

            summary = (
                f"    Strategy: {strategy:<17} | "
                f"Avg CKA: {avg_cka:7.4f}, Avg ACS: {avg_acs:7.4f}, ACS Std: {avg_acs_std:7.4f}"
            )
            if strategy == "flatten_no_special":
                avg_rho_estimate = np.mean(batch_scores_per_strategy[strategy]["rho_estimate"])
                avg_h_spec_norm = np.mean(batch_scores_per_strategy[strategy]["h_spec_norm"])
                results[strategy]["rho_estimate"].append(avg_rho_estimate)
                results[strategy]["h_spec_norm"].append(avg_h_spec_norm)
                summary += (
                    f", Rho Est: {avg_rho_estimate:7.4f} "
                    f"(cos(flatten)-cos(flatten_no_special)), H SpecNorm: {avg_h_spec_norm:7.4f}"
                )

            print(summary)

    ref_model.to("cpu")
    mdlm_embedder.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
    print("Models offloaded to CPU.")

    final_ref_acs_mean = float(np.mean([s["mean"] for s in all_ref_acs_scores]))
    final_ref_acs_std = float(np.mean([s["std"] for s in all_ref_acs_scores]))
    print(f"Final averaged Reference ACS baseline: {final_ref_acs_mean:.4f} (std: {final_ref_acs_std:.4f})")

    flatten_no_special_rho = np.asarray(results["flatten_no_special"]["rho_estimate"], dtype=float)
    rho_max_idx = int(np.argmax(flatten_no_special_rho))
    rho_min_idx = int(np.argmin(flatten_no_special_rho))
    print(
        "Summary: flatten_no_special rho estimate from "
        "cos(flatten)-cos(flatten_no_special) "
        f"mean={flatten_no_special_rho.mean():.4f}, "
        f"min={flatten_no_special_rho[rho_min_idx]:.4f} at mask_ratio={mask_ratios[rho_min_idx]:.2f}, "
        f"max={flatten_no_special_rho[rho_max_idx]:.4f} at mask_ratio={mask_ratios[rho_max_idx]:.2f}",
    )

    flatten_no_special_h_spec_norm = np.asarray(results["flatten_no_special"]["h_spec_norm"], dtype=float)
    max_idx = int(np.argmax(flatten_no_special_h_spec_norm))
    min_idx = int(np.argmin(flatten_no_special_h_spec_norm))
    print(
        "Summary: flatten_no_special H spectral norm with dynamic rho "
        f"mean={flatten_no_special_h_spec_norm.mean():.4f}, "
        f"min={flatten_no_special_h_spec_norm[min_idx]:.4f} at mask_ratio={mask_ratios[min_idx]:.2f}, "
        f"max={flatten_no_special_h_spec_norm[max_idx]:.4f} at mask_ratio={mask_ratios[max_idx]:.2f}",
    )

    # Save results to CSV
    df = save_results_csv(
        results=results,
        x_values=mask_ratios,
        x_name="mask_ratio",
        filename="embeddings_mdlm_results.csv",
        ref_acs_baseline=final_ref_acs_mean,
    )

    # Plot results
    plot_cka_acs(
        df=df,
        x_name="mask_ratio",
        title_suffix="MDLM Representation Quality",
        n_samples=N_TOTAL_SAMPLES,
        ref_acs_baseline=final_ref_acs_mean,
        plot_filename=f"cka_acs_results_mdlm_{N_TOTAL_SAMPLES}_samples.png",
    )


if __name__ == "__main__":
    main()
