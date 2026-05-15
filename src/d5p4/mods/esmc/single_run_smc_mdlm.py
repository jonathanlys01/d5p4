"""
Single run script for ESMC MDLM text generation.
Using entropy-based final selection
"""

import json
import os
import uuid
from datetime import datetime

from d5p4.common_exps import eval_samples
from d5p4.config import Config
from d5p4.mods.esmc.diffusion_smc_mdlm import SMC_MDLMSampler
from d5p4.result_schema import build_generation_result_payload
from d5p4.utils import compile_model, print, seed_all


################################################################################


def save(text, config, uid, rank=0):
    samples = build_generation_result_payload(text_samples=text, config=config)

    name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}_{str(uid)}"
    os.makedirs(config.results_dir, exist_ok=True)
    with open(os.path.join(config.results_dir, f"{name}.json"), "w") as f:
        json.dump(samples, f, indent=4)


def main():
    config = Config()

    model = SMC_MDLMSampler(config)
    model.model = compile_model(model.model, config)

    offset = 0
    if model.distributed_utils:
        offset = model.distributed_utils.rank

    seed_all(config.seed + offset)
    texts = []

    unique_id = uuid.uuid4()
    print(f"Experiment ID: {unique_id}")

    for i in range(config.n_runs):
        print(f"Sampling batch {i + 1}/{config.n_runs}...", progress=True)
        samples = model.sample(select_best_final=True)
        texts.append(model.tokenizer.batch_decode(samples, skip_special_tokens=True))
        save(texts, config, unique_id, rank=offset)

    samples = build_generation_result_payload(text_samples=texts, config=config, experiment_id=str(unique_id))

    if model.distributed_utils is None or model.distributed_utils.rank == 0:  # save on master only (or non-distributed)
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(unique_id)}"
        os.makedirs(config.results_dir, exist_ok=True)
        output_path = os.path.join(config.results_dir, f"exp-{name}.json")
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=4)
        print(f"OUTPUT_PATH:{output_path}")

    for file in os.listdir(config.results_dir):
        if file.startswith("temp_") and file.endswith(f"_rank{offset}_{unique_id}.json"):
            os.remove(os.path.join(config.results_dir, file))

    # Evaluate samples on master only
    if model.distributed_utils is None or model.distributed_utils.rank == 0:
        if config.skip_eval:
            print("Skipping evaluation because skip_eval=True.")
        else:
            print("Running evaluation...")
            metrics = eval_samples(str(unique_id), config)
            assert metrics is not None and metrics["metrics_summary"] is not None
            print(f"Evaluation complete: {metrics['metrics_summary']}")

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()
