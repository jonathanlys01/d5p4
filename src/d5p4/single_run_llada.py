"""
Single run script for LLaDA text generation.
"""

import json
import os
import uuid
from datetime import datetime

from d5p4.config import Config
from d5p4.data import get_qa_dataset
from d5p4.diffusion_llada import LLADASampler
from d5p4.eval_core import Evaluator
from d5p4.result_schema import build_generation_result_payload
from d5p4.utils import compile_model, print, seed_all
from d5p4.utils import print as uprint


LLADA_INTERNAL_SCORE_METADATA = {
    "name": "confidence",
    "method": "final_step_mean_token_logprob",
    "scope": "generated_tokens",
    "higher_is_better": True,
}


def save(
    text,
    config,
    uid,
    rank=0,
    references=None,
    eval_text=None,
    internal_scores=None,
    eval_selected_indices=None,
    eval_selection=None,
):
    samples = build_generation_result_payload(
        text_samples=text,
        config=config,
        references=references,
        eval_text_samples=eval_text,
        internal_scores=internal_scores,
        eval_selected_indices=eval_selected_indices,
        eval_selection=eval_selection,
        internal_score_metadata=LLADA_INTERNAL_SCORE_METADATA if internal_scores is not None else None,
    )

    name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}_{str(uid)}"
    os.makedirs(config.results_dir, exist_ok=True)
    with open(os.path.join(config.results_dir, f"{name}.json"), "w") as f:
        json.dump(samples, f, indent=4)


def _select_group_representatives(
    texts: list[str],
    scores: list[float],
    group_size: int,
) -> tuple[list[str], list[int]]:
    if len(texts) != len(scores):
        raise ValueError(f"Expected one score per generated sequence, got {len(scores)} scores for {len(texts)} texts.")
    if group_size <= 1:
        return texts.copy(), list(range(len(texts)))
    if len(texts) % group_size != 0:
        raise ValueError(
            f"Expected generated sequence count divisible by group_size, got {len(texts)} and {group_size}.",
        )

    selected_texts = []
    selected_indices = []
    for start in range(0, len(texts), group_size):
        group_scores = scores[start : start + group_size]
        best_local_idx = max(range(group_size), key=lambda idx: group_scores[idx])
        best_idx = start + best_local_idx
        selected_texts.append(texts[best_idx])
        selected_indices.append(best_idx)

    return selected_texts, selected_indices


def main():
    config = Config()

    model = LLADASampler(config)
    model.model = compile_model(model.model, config, dynamic=True)

    offset = 0
    if model.distributed_utils:
        offset = model.distributed_utils.rank

    seed_all(config.seed + offset)
    texts = []
    eval_texts = []
    internal_scores_all = []
    eval_selected_indices = []
    use_internal_representatives = config.group_size > 1
    master = model.distributed_utils is None or model.distributed_utils.rank == 0

    unique_id = uuid.uuid4()
    print(f"Experiment ID: {unique_id}")
    if master:
        print("Dumping final-step internal confidence scores for every generated sequence.")
    if use_internal_representatives and master:
        print("Using final-step internal scores to select one evaluation representative per group.")

    dataset = get_qa_dataset(config)
    limit = config.qa_dataset_len if config.qa_dataset_len > 0 else len(dataset)
    rows = list(dataset.itertuples())[:limit]
    prompts: list[str] = [row.question for row in rows]  # type: ignore
    references_all: list[list[str]] = [row.correct_answers for row in rows]  # type: ignore

    for i, prompt in enumerate(prompts):
        uprint(f"Sampling batch {i + 1}/{len(prompts)}...", progress=True)
        samples, internal_scores = model.sample(prompt=prompt, return_internal_scores=True)

        prompt_tokens = model._preprocess_prompt(prompt)
        prompt_len = prompt_tokens.shape[1]
        texts_ = []
        for sample in samples:
            completion_tokens = sample[prompt_len:]
            gen_text = model.tokenizer.decode(completion_tokens.tolist(), skip_special_tokens=True).strip()
            texts_.append(gen_text)

        texts.append(texts_)
        scores = [float(score) for score in internal_scores.detach().cpu().tolist()]
        internal_scores_all.append(scores)
        if use_internal_representatives and master:
            eval_texts_, selected_indices = _select_group_representatives(texts_, scores, config.group_size)
            eval_texts.append(eval_texts_)
            eval_selected_indices.append(selected_indices)

        save(
            texts,
            config,
            unique_id,
            rank=offset,
            references=references_all[: i + 1],
            eval_text=eval_texts if use_internal_representatives and master else None,
            internal_scores=internal_scores_all,
            eval_selected_indices=eval_selected_indices if use_internal_representatives and master else None,
            eval_selection={
                "method": "final_internal_signal",
                "score_key": "internal_scores",
                "score_method": LLADA_INTERNAL_SCORE_METADATA["method"],
                "group_size": config.group_size,
            }
            if use_internal_representatives and master
            else None,
        )

    metrics = None
    if master:
        if config.skip_eval:
            print("Skipping evaluation because skip_eval=True.")
        else:
            print("Running evaluation...")
            evaluator = Evaluator(
                batch_size=config.eval_batch_size,
                force=True,
                ppl_model_id=config.ppl_model_id,
                cos_model_id=config.cos_model_id,
            )
            metric_texts = eval_texts if use_internal_representatives else texts
            metrics = evaluator.evaluate(metric_texts, references=references_all)
            assert metrics["metrics_summary"] is not None
            print(f"Evaluation complete: {metrics['metrics_summary']}")

    eval_selection = None
    if use_internal_representatives and master:
        eval_selection = {
            "method": "final_internal_signal",
            "score_key": "internal_scores",
            "score_method": LLADA_INTERNAL_SCORE_METADATA["method"],
            "group_size": config.group_size,
        }

    samples = build_generation_result_payload(
        text_samples=texts,
        config=config,
        references=references_all,
        eval_text_samples=eval_texts if use_internal_representatives and master else None,
        internal_scores=internal_scores_all,
        eval_selected_indices=eval_selected_indices if use_internal_representatives and master else None,
        eval_selection=eval_selection,
        internal_score_metadata=LLADA_INTERNAL_SCORE_METADATA,
        metrics=metrics,
        experiment_id=str(unique_id),
    )

    if master:  # save on master only (or non-distributed)
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(unique_id)}"
        os.makedirs(config.results_dir, exist_ok=True)
        output_path = os.path.join(config.results_dir, f"exp-{name}.json")
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=4)
        print(f"Saved in {output_path}")

    for file in os.listdir(config.results_dir):
        if file.startswith("temp_") and file.endswith(f"_rank{offset}_{unique_id}.json"):
            os.remove(os.path.join(config.results_dir, file))

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()
