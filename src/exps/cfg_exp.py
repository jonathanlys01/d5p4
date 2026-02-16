"""
CFG Experiment for LLaDA.

This script sweeps CFG values and measures all available metrics:
- perplexity: language model perplexity
- cosine_similarity: average pairwise cosine similarity
- distinct_2: ratio of unique bigrams (lower = more repetition)
- self_bleu: BLEU score between generations (higher = more repetition)
- f1, bleu: reference-based metrics (when references available)
- f1_at_k, bleu_at_k: best-of-k metrics
"""

import json
import os
from dataclasses import asdict
from datetime import datetime

import idr_torch
import numpy as np
import torch

import utils
from config import RESULTS_DIR, Config
from data.qa import get_qa_dataset
from diffusion_llada import LLADASampler
from eval_core import Evaluator
from utils import compile_model, seed_all
from utils import print as u_print


def run_cfg_experiment(cfg: Config, cfg_values: list[float] | None = None) -> dict:  # noqa: C901, PLR0912, PLR0915
    """Run the CFG experiment across multiple CFG values, measuring all metrics."""

    if cfg_values is None:
        cfg_values = [cfg.cfg_scale] if cfg.cfg_scale != 0.0 else np.logspace(np.log10(1), np.log10(3), num=6).tolist()

    utils.INTERACTIVE = cfg.interactive
    seed_all(cfg.seed)

    # Create sampler first to initialize DistributedUtils (and thus the process group/device)
    sampler = LLADASampler(cfg)
    sampler.model = compile_model(sampler.model, cfg, dynamic=True)

    # Initialize evaluator for all metrics (now safe to load extra models on the correct device)
    evaluator = Evaluator(
        batch_size=cfg.eval_batch_size,
        ppl_model_id=cfg.ppl_model_id,
        cos_model_id=cfg.cos_model_id,
    )

    # Load dataset once
    dataset = get_qa_dataset(cfg)
    if cfg.qa_dataset_len > 0:
        dataset = dataset.head(cfg.qa_dataset_len)

    u_print(f"Running CFG experiment with {len(dataset)} samples")
    u_print(f"CFG values to test: {cfg_values}")

    all_results: dict = {
        "cfg_values": cfg_values,
        "metrics_by_cfg": {},
        "samples_by_cfg": {},
    }

    for idx, cfg_value in enumerate(cfg_values):
        u_print(f"\n{'=' * 60}")
        u_print(f"Testing CFG scale: {cfg_value}")
        u_print(f"{'=' * 60}")

        iter_dict = asdict(cfg)
        iter_dict["cfg_scale"] = cfg_value
        iter_dict["disable_sys_args"] = True
        iter_cfg = Config(**iter_dict)
        sampler.update_config(iter_cfg)

        all_generations: list[list[str]] = []
        all_good_refs: list[list[str]] = []
        all_bad_refs: list[list[str]] = []
        wd_good_scores: list[float] = []
        wd_bad_scores: list[float] = []

        # Sampling loop
        for i, row in enumerate(dataset.itertuples()):
            prompt: str = row.question  # type: ignore
            correct_answers: list[str] = row.correct_answers  # type: ignore
            incorrect_answers: list[str] = row.incorrect_answers  # type: ignore

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
            all_good_refs.append(correct_answers)
            all_bad_refs.append(incorrect_answers)

            # Wasserstein Distance for this sample
            wd_good, wd_bad = evaluator.compute_wasserstein_distance(
                batch_gen,
                correct_answers,
                incorrect_answers,
            )
            wd_good_scores.append(wd_good)
            wd_bad_scores.append(wd_bad)

        # Only Rank 0 computes and prints metrics
        if idr_torch.rank == 0:
            # Compute all metrics for this CFG value
            metrics = evaluator.evaluate(all_generations, references=all_good_refs)

            # Add string metrics (like cos_at_k)
            string_metrics = evaluator.compute_string_metrics(all_generations, all_good_refs)
            metrics.update(string_metrics)

            # Wasserstein Distance metrics
            metrics.update(
                {
                    "avg_wd_good": sum(wd_good_scores) / len(wd_good_scores),
                    "avg_wd_bad": sum(wd_bad_scores) / len(wd_bad_scores),
                },
            )

            # Extract core metrics (filter out CI and summary stats for display)
            core_metrics = {
                k: v
                for k, v in metrics.items()
                if not any(
                    suffix in k
                    for suffix in ["_ci95", "_std", "_lower", "_upper", "_median", "_min", "_max", "_summary"]
                )
                and k != "metrics_summary"
            }

            u_print(f"\nResults for CFG={cfg_value}:")
            for k, v in core_metrics.items():
                if isinstance(v, float):
                    u_print(f"  {k:25}: {v:.4f}")
                else:
                    u_print(f"  {k:25}: {v}")
            if "metrics_summary" in metrics:
                u_print(f"  Summary: {metrics['metrics_summary']}")

            all_results["metrics_by_cfg"][str(cfg_value)] = metrics
            all_results["samples_by_cfg"][str(cfg_value)] = all_generations

    # Synchronize all ranks before cleanup
    if sampler.distributed_utils:
        torch.distributed.barrier()

    # Cleanup after all iterations
    if sampler.distributed_utils:
        sampler.distributed_utils.cleanup()
    del sampler
    torch.cuda.empty_cache()

    # Summary table (Rank 0 only)
    if idr_torch.rank == 0 and len(cfg_values) > 1:
        u_print(f"\n{'=' * 105}")
        u_print("SUMMARY: CFG vs All Metrics")
        u_print(f"{'=' * 105}")
        u_print(
            f"{'CFG':>8} | {'PPL':>10} | {'F1':>10} | {'BLEU':>10} | {'Cos-Sim':>10} | {'Dist-2':>10} | {'S-BLEU':>10}",
        )
        u_print("-" * 105)

        for cfg_val in cfg_values:
            m = all_results["metrics_by_cfg"][str(cfg_val)]
            u_print(
                f"{cfg_val:>8.2f} | {m.get('perplexity', 0):>10.2f} | {m.get('f1', 0):>10.4f} |"
                f" {m.get('bleu', 0):>10.2f} | {m.get('cosine_similarity', 0):>10.4f} |"
                f" {m.get('distinct_2', 0):>10.4f} | {m.get('self_bleu', 0):>10.4f}",
            )

        u_print("-" * 105)

    return all_results


def main():
    cfg = Config()
    results = run_cfg_experiment(cfg)

    # Save results (Rank 0 only)
    if idr_torch.rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg_suffix = f"_cfg{cfg.cfg_scale:.2f}" if cfg.cfg_scale != 0.0 else "_sweep"
        save_path = f"{RESULTS_DIR}/cfg_exp_{timestamp}{cfg_suffix}.json"
        os.makedirs(RESULTS_DIR, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(
                {
                    "config": asdict(cfg),
                    "results": results["metrics_by_cfg"],
                    "cfg_values": results["cfg_values"],
                    "text_samples": results["samples_by_cfg"],
                },
                f,
                indent=4,
            )
        u_print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
