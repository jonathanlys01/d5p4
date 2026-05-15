"""
DPP Sweep: Additive Kernel (MAP+)
K = w_interaction * S + diag(scores)
"""

from dataclasses import asdict

import numpy as np

from d5p4.common_exps import _bcast, print, run_experiment, run_sweep
from d5p4.config import Config


SWEEP_NAME = "d5p4_map_times_optuna"


def _objective(trial, og_config: Config, model, evaluator):
    w_interaction = trial.suggest_float("w_interaction", 0.0, 50.0)

    dict_config = asdict(og_config)
    dict_config["_w_interaction"] = w_interaction
    dict_config["disable_sys_args"] = True
    config = Config(**dict_config)

    _bcast(True)  # sync before starting -> proceed
    _bcast(config)  # broadcast config to all workers

    print(f"Trial {trial.number}: w_inter={w_interaction:.4f}")

    metrics = run_experiment(config, model, evaluator)
    assert metrics is not None

    perplexity = metrics["perplexity"]
    cos_sim = metrics["cosine_similarity"]
    trial.set_user_attr("metrics", metrics)

    print(f"Trial {trial.number} completed: Perplexity={perplexity:.4f}, Cosine Similarity={cos_sim:.4f}")

    return perplexity, cos_sim


if __name__ == "__main__":
    og_config = Config()
    assert og_config.method == "greedy_map" and og_config._kernel_method == "additive"
    # Initial trials for a quick look at the range
    init_trials = [{"w_interaction": w} for w in np.linspace(0.0, 50.0, 5)]
    run_sweep(SWEEP_NAME, og_config, _objective, init_trials=init_trials)
