"""
Interaction Parameter Experiment for LLaDA.

This script sweeps _w_interaction values and measures all available metrics:
- perplexity: language model perplexity
- cosine_similarity: average pairwise cosine similarity
- distinct_2: ratio of unique bigrams (lower = more repetition)
- self_bleu: BLEU score between generations (higher = more repetition)
- f1_at_k, bleu_at_k, cos_at_k: best-of-k metrics against references
"""

import json
import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch

import utils
from config import RESULTS_DIR, Config
from data.qa import get_qa_dataset
from diffusion_llada import LLADASampler
from eval_core import Evaluator
from utils import compile_model, seed_all
from utils import print as u_print


def run_interaction_experiment(cfg: Config, interaction_values: list[float] | None = None) -> dict:  # noqa: C901, PLR0912, PLR0915
    """Run the interaction experiment across multiple _w_interaction values, measuring all metrics."""

    if interaction_values is None:
        if cfg._w_interaction != 0.0:
            interaction_values = [cfg._w_interaction]
        else:
            # Default sweep: logarithmic scale from 0.1 to 10.0
            interaction_values = np.logspace(np.log10(0.1), np.log10(10.0), num=8).tolist()

    utils.INTERACTIVE = cfg.interactive
    seed_all(cfg.seed)

    # Initialize evaluator for all metrics
    evaluator = Evaluator(
        batch_size=cfg.eval_batch_size,
        ppl_model_id=cfg.ppl_model_id,
        cos_model_id=cfg.cos_model_id,
    )

    # Load dataset once
    dataset = get_qa_dataset(cfg)
    if cfg.qa_dataset_len > 0:
        dataset = dataset.head(cfg.qa_dataset_len)

    # Extract references for metric@k computation
    references: list[list[str]] = [row.correct_answers for row in dataset.itertuples()]  # type: ignore

    u_print(f"Running interaction experiment with {len(dataset)} samples")
    u_print(f"Interaction values to test: {interaction_values}")
    u_print(f"Method: {cfg.method}")

    all_results: dict = {
        "interaction_values": interaction_values,
        "metrics_by_interaction": {},
        "samples_by_interaction": {},
    }

    # Create sampler once and reuse across all interaction values
    sampler = LLADASampler(cfg)
    sampler.model = compile_model(sampler.model, cfg, dynamic=True)

    for idx, interaction_value in enumerate(interaction_values):
        u_print(f"\n{'=' * 60}")
        u_print(f"Testing _w_interaction: {interaction_value}")
        u_print(f"{'=' * 60}")

        iter_dict = asdict(cfg)
        iter_dict["_w_interaction"] = interaction_value
        iter_dict["disable_sys_args"] = True
        iter_cfg = Config(**iter_dict)
        sampler.update_config(iter_cfg)

        all_generations: list[list[str]] = []

        # Sampling loop
        for i, row in enumerate(dataset.itertuples()):
            prompt: str = row.question  # type: ignore

            u_print(f"[{i + 1}/{len(dataset)}] Prompt: {prompt[:50]}...", verbose=True)

            with torch.no_grad():
                sample_ids = sampler.sample(prompt=prompt)

            # Decode
            batch_gen = []
            for sample in sample_ids:
                prompt_tokens = sampler._preprocess_prompt(prompt)
                prompt_len = prompt_tokens.shape[1]
                completion_tokens = sample[prompt_len:]
                gen_text = sampler.tokenizer.decode(completion_tokens.tolist(), skip_special_tokens=True).strip()
                batch_gen.append(gen_text)

            all_generations.append(batch_gen)

        # Compute all metrics for this interaction value
        metrics = evaluator.evaluate(all_generations, references=references)

        # Compute additional metric@k variants
        string_metrics = evaluator.compute_string_metrics(all_generations, references)
        metrics.update(string_metrics)

        # Extract core metrics (filter out CI and summary stats for display)
        core_metrics = {
            k: v
            for k, v in metrics.items()
            if not any(
                suffix in k for suffix in ["_ci95", "_std", "_lower", "_upper", "_median", "_min", "_max", "_summary"]
            )
            and k != "metrics_summary"
        }

        print(f"\nResults for _w_interaction={interaction_value}:")
        for k, v in core_metrics.items():
            if isinstance(v, float):
                print(f"  {k:25}: {v:.4f}")
            else:
                print(f"  {k:25}: {v}")
        if "metrics_summary" in metrics:
            print(f"  Summary: {metrics['metrics_summary']}")

        all_results["metrics_by_interaction"][str(interaction_value)] = metrics
        all_results["samples_by_interaction"][str(interaction_value)] = all_generations

    # Cleanup after all iterations
    if sampler.distributed_utils:
        sampler.distributed_utils.cleanup()
    del sampler
    torch.cuda.empty_cache()

    # Summary table
    if len(interaction_values) > 1:
        u_print(f"\n{'=' * 100}")
        u_print("SUMMARY: Interaction vs All Metrics")
        u_print(f"{'=' * 100}")
        u_print(
            f"{'W_INT':>10} | {'PPL':>10} | {'Cos-Sim':>10} | {'Dist-2':>10} | {'S-BLEU':>10} | "
            f"{'F1@k':>10} | {'BLEU@k':>10} | {'Cos@k':>10}",
        )
        u_print("-" * 100)

        for int_val in interaction_values:
            m = all_results["metrics_by_interaction"][str(int_val)]
            u_print(
                f"{int_val:>10.3f} | {m.get('perplexity', 0):>10.2f} | {m.get('cosine_similarity', 0):>10.4f} | "
                f"{m.get('distinct_2', 0):>10.4f} | {m.get('self_bleu', 0):>10.4f} | "
                f"{m.get('f1_at_k', 0):>10.4f} | {m.get('bleu_at_k', 0):>10.4f} | {m.get('cos_at_k', 0):>10.4f}",
            )

        u_print("-" * 100)

    return all_results


def main():
    cfg = Config()
    results = run_interaction_experiment(cfg)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    int_suffix = f"_wint{cfg._w_interaction:.2f}" if cfg._w_interaction != 0.0 else "_sweep"
    save_path = f"{RESULTS_DIR}/interaction_exp_{timestamp}{int_suffix}.json"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "results": results["metrics_by_interaction"],
                "interaction_values": results["interaction_values"],
                "text_samples": results["samples_by_interaction"],
            },
            f,
            indent=4,
        )
    u_print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
