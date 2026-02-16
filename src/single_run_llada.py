"""
Single run script for MDLM text generation.
"""

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime

from common_exps import eval_samples
from config import RESULTS_DIR, Config
from data import get_qa_dataset
from diffusion_llada import LLADASampler
from utils import compile_model, print, seed_all


def save(text, config, uid, rank=0):
    samples = {
        "text_samples": text,  # list of lists of strings
        "config": asdict(config),
    }

    name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}_{str(uid)}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/{name}.json", "w") as f:
        json.dump(samples, f, indent=4)


def main():
    config = Config()

    model = LLADASampler(config)
    model.model = compile_model(model.model, config, dynamic=True)

    offset = 0
    if model.distributed_utils:
        offset = model.distributed_utils.rank

    seed_all(config.seed + offset)
    texts = []

    unique_id = uuid.uuid4()
    print(f"Experiment ID: {unique_id}")

    dataset = get_qa_dataset(config)
    limit = config.qa_dataset_len if config.qa_dataset_len > 0 else len(dataset)
    prompts: list[str] = [row.question for row in dataset.itertuples()][:limit]  # type: ignore

    for i in range(len(prompts)):
        print(f"Sampling batch {i + 1}/{len(prompts)}...")
        samples = model.sample(prompt=prompts[i])
        texts_ = []
        for sample in samples:
            prompt_tokens = model._preprocess_prompt(prompts[i])
            prompt_len = prompt_tokens.shape[1]
            completion_tokens = sample[prompt_len:]
            gen_text = model.tokenizer.decode(completion_tokens.tolist(), skip_special_tokens=True).strip()
            texts_.append(gen_text)

        # Deduplicate to handle potential duplicates from group expansion
        if config.group_size > 1 and len(texts_) != config.n_groups:
            texts_ = texts_[:: config.group_size]
        texts.append(texts_)
        save(texts, config, unique_id, rank=offset)

    samples = {
        "text_samples": texts,  # list of lists of strings
        "config": asdict(config),
        "experiment_id": str(unique_id),
    }

    if model.distributed_utils is None or model.distributed_utils.rank == 0:
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(unique_id)}"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(f"{RESULTS_DIR}/exp-{name}.json", "w") as f:
            json.dump(samples, f, indent=4)
        print(f"Saved in {RESULTS_DIR}/exp-{name}.json")

    for file in os.listdir(RESULTS_DIR):
        if file.startswith("temp_") and file.endswith(f"_rank{offset}_{unique_id}.json"):
            os.remove(os.path.join(RESULTS_DIR, file))

    if model.distributed_utils is None or model.distributed_utils.rank == 0:
        print("Running evaluation...")
        references: list[list[str]] = [row.correct_answers for row in dataset.itertuples()][:limit]  # type: ignore
        metrics = eval_samples(str(unique_id), config, references=references)
        assert metrics is not None and metrics["metrics_summary"] is not None
        print(f"Evaluation complete: {metrics['metrics_summary']}")

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()
