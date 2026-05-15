"""
Single-run script for LLaDA on GSM8K math evaluation.

Feeds each problem through LLaDA, parses the numeric answer from each
generation with MathEvaluator, and reports per-question accuracy together
with the overall accuracy across the dataset.
"""

import json
import os
import uuid
from datetime import datetime

from d5p4.config import Config
from d5p4.data.math_ds import gsm8k
from d5p4.diffusion_llada import LLADASampler
from d5p4.eval_core import MathEvaluator
from d5p4.result_schema import build_generation_result_payload
from d5p4.utils import compile_model, print, seed_all
from d5p4.utils import print as uprint


LLADA_INTERNAL_SCORE_METADATA = {
    "name": "confidence",
    "method": "final_step_mean_token_logprob",
    "scope": "generated_tokens",
    "higher_is_better": True,
}


def _text_samples_from_results(results: list[dict]) -> list[list[str]]:
    return [row["generations"] for row in results]


def _references_from_results(results: list[dict]) -> list[list[str]]:
    return [[row["answer_str"] if row["answer_str"] else row["gold_answer"]] for row in results]


def save(
    results: list[dict],
    config: Config,
    uid: uuid.UUID,
    rank: int = 0,
    internal_scores: list[list[float]] | None = None,
) -> None:
    """Write an intermediate checkpoint to a temp JSON file."""
    payload = build_generation_result_payload(
        text_samples=_text_samples_from_results(results),
        config=config,
        references=_references_from_results(results),
        internal_scores=internal_scores,
        internal_score_metadata=LLADA_INTERNAL_SCORE_METADATA if internal_scores is not None else None,
        experiment_id=str(uid),
        extra={"results": results},
    )

    name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}_{uid}"
    os.makedirs(config.results_dir, exist_ok=True)
    with open(os.path.join(config.results_dir, f"{name}.json"), "w") as f:
        json.dump(payload, f, indent=4)


def main() -> None:
    config = Config()
    assert config.qa_dataset == "gsm8k"

    model = LLADASampler(config)
    model.model = compile_model(model.model, config, dynamic=True)

    offset = 0
    if model.distributed_utils:
        offset = model.distributed_utils.rank

    seed_all(config.seed + offset)

    unique_id = uuid.uuid4()
    print(f"Experiment ID: {unique_id}")

    # ── dataset ──────────────────────────────────────────────────────────────
    dataset = gsm8k(config)
    limit = config.qa_dataset_len if config.qa_dataset_len > 0 else len(dataset)
    rows = list(dataset.itertuples())[:limit]
    prompts: list[str] = [row.question for row in rows]  # type: ignore[union-attr]
    answer_strings: list[str] = [row.answer_str for row in rows]  # type: ignore[union-attr]
    answer_numbers: list[str] = [row.answer_number for row in rows]  # type: ignore[union-attr]

    # ── evaluator ────────────────────────────────────────────────────────────
    evaluator = MathEvaluator()

    # ── generation + evaluation loop ─────────────────────────────────────────
    results: list[dict] = []  # one entry per question
    internal_scores_all: list[list[float]] = []

    for i, (prompt, gold, answer_str) in enumerate(zip(prompts, answer_numbers, answer_strings)):
        uprint(f"Sampling {i + 1}/{len(prompts)}  (gold={gold!r})...", progress=True)

        raw_samples, internal_scores = model.sample(prompt=prompt, return_internal_scores=True)

        generations: list[str] = []
        for sample in raw_samples:
            prompt_tokens = model._preprocess_prompt(prompt)
            prompt_len = prompt_tokens.shape[1]
            completion_tokens = sample[prompt_len:]
            gen_text = model.tokenizer.decode(completion_tokens.tolist(), skip_special_tokens=True).strip()
            generations.append(gen_text)

        internal_score_group = [float(score) for score in internal_scores.detach().cpu().tolist()]
        internal_scores_all.append(internal_score_group)

        scores = evaluator.score_group(generations, gold)
        acc = evaluator.accuracy(generations, gold)

        results.append(
            {
                "question": prompt,
                "gold_answer": gold,
                "answer_str": answer_str,
                "generations": generations,
                "scores": scores,
                "accuracy": acc,
            },
        )

        uprint(f"  → accuracy for this question: {acc:.2%}  ({sum(scores)}/{len(scores)} correct)", progress=True)
        save(results, config, unique_id, rank=offset, internal_scores=internal_scores_all)

    # ── final aggregation ────────────────────────────────────────────────────
    overall_acc = sum(r["accuracy"] for r in results) / len(results) if results else 0.0
    print(f"\n acc: {overall_acc:.4%}  ({sum(r['accuracy'] > 0 for r in results)}/{len(results)} qs with ≥1 correct)")

    all_generations: list[list[str]] = [r["generations"] for r in results]
    num_workers = min(8, os.cpu_count() or 1)
    print(f"Computing aggregate math metrics with {num_workers} CPU worker(s)...")
    string_references = [[answer_str] for answer_str in answer_strings]
    math_metrics = (
        evaluator.evaluate(
            all_generations,
            answer_numbers,
            string_references=string_references,
            num_workers=num_workers,
        )
        if results
        else {}
    )
    math_metrics_summary = math_metrics.get("math_metrics_summary")
    if math_metrics_summary:
        print(f"math metrics: {math_metrics_summary}")

    payload = build_generation_result_payload(
        text_samples=all_generations,
        config=config,
        references=string_references,
        internal_scores=internal_scores_all,
        internal_score_metadata=LLADA_INTERNAL_SCORE_METADATA,
        metrics=math_metrics,
        experiment_id=str(unique_id),
        extra={
            "results": results,
            "overall_accuracy": overall_acc,
            "math_metrics": math_metrics,
        },
    )

    if model.distributed_utils is None or model.distributed_utils.rank == 0:
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}"
        os.makedirs(config.results_dir, exist_ok=True)
        out_path = os.path.join(config.results_dir, f"math-{name}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=4)
        print(f"Saved results to {out_path}")

    # Clean up temp files
    for file in os.listdir(config.results_dir):
        if file.startswith("temp_") and file.endswith(f"_rank{offset}_{unique_id}.json"):
            os.remove(os.path.join(config.results_dir, file))

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()
