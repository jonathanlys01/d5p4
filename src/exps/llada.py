import json
import os
from dataclasses import asdict
from datetime import datetime

import utils
from config import RESULTS_DIR, Config
from data.qa import get_qa_dataset
from diffusion_llada import LLADASampler
from eval_core import Evaluator
from utils import compile_model, seed_all
from utils import print as u_print


def main():  # noqa: PLR0915
    cfg = Config()
    utils.INTERACTIVE = cfg.interactive
    seed_all(cfg.seed)

    # 1. Setup
    sampler = LLADASampler(cfg)
    sampler.model = compile_model(sampler.model, cfg, dynamic=True)
    evaluator = Evaluator(
        batch_size=cfg.eval_batch_size,
        ppl_model_id=cfg.ppl_model_id,
        cos_model_id=cfg.cos_model_id,
    )

    # 2. Load Dataset
    dataset = get_qa_dataset(cfg)
    if cfg.qa_dataset_len > 0:
        dataset = dataset.head(cfg.qa_dataset_len)

    u_print(f"Evaluating {len(dataset)} samples from {cfg.qa_dataset}...")

    all_generations, all_good_refs, all_bad_refs = [], [], []

    wd_good_scores: list[float] = []
    wd_bad_scores: list[float] = []

    # 3. Sampling loop
    for i, row in enumerate(dataset.itertuples()):
        prompt: str = row.question  # type: ignore
        correct_answers: list[str] = row.correct_answers  # type: ignore
        incorrect_answers: list[str] = row.incorrect_answers  # type: ignore

        if cfg.interactive:
            u_print(f"[{i + 1}/{len(dataset)}] Prompt: {prompt[:50]}...", verbose=True)
        else:
            u_print(f"[{i + 1}/{len(dataset)}]")

        # Sample
        sample_ids = sampler.sample(prompt=prompt)

        # Decode
        batch_gen = []
        for sample in sample_ids:
            # Extract completion by slicing off the prompt
            prompt_tokens = sampler._preprocess_prompt(prompt)
            prompt_len = prompt_tokens.shape[1]
            completion_tokens = sample[prompt_len:]
            gen_text = sampler.tokenizer.decode(completion_tokens.tolist(), skip_special_tokens=True).strip()
            batch_gen.append(gen_text)
            u_print(f"  Generated: {gen_text}", verbose=True)

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

    # 4. Global Metrics
    # PPL and Average Cosine expect list[list[str]] (batches)
    global_metrics = evaluator.evaluate(all_generations)

    string_metrics = evaluator.compute_string_metrics(all_generations, all_good_refs)
    global_metrics.update(string_metrics)  # add bleu and f1

    # Wasserstein Distance metrics
    global_metrics.update(
        {
            "avg_wd_good": sum(wd_good_scores) / len(wd_good_scores),
            "avg_wd_bad": sum(wd_bad_scores) / len(wd_bad_scores),
        },
    )

    # 5. Report Results
    print("\n" + "=" * 40)
    print("Evaluation Results:")
    for k, v in global_metrics.items():
        if k != "metrics_summary" and isinstance(v, (int, float)):
            print(f"{k:25}: {v:.4f}")
    print("-" * 40)
    print(f"Summary: {global_metrics.get('metrics_summary', 'N/A')}")
    print("=" * 40)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{RESULTS_DIR}/llada_eval_{timestamp}.json"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "results": global_metrics,
                "text_samples": all_generations,
            },
            f,
            indent=4,
        )
    print(f"Results saved to {save_path}")

    if sampler.distributed_utils:
        sampler.distributed_utils.cleanup()


if __name__ == "__main__":
    main()
