"""
Sampling Sweep: Categorical Temperature (CAT)
Sweeps over the Softmax temperature for token sampling.
"""

from dataclasses import asdict

import numpy as np

from d5p4.common_exps import _bcast, print, run_experiment, run_sweep
from d5p4.config import Config


SWEEP_NAME = "d5p4_cat_optuna"


def _objective(trial, og_config: Config, model, evaluator):
    cat_temperature = trial.suggest_float("cat_temperature", 0.1, 2.0)

    dict_config = asdict(og_config)
    dict_config["cat_temperature"] = cat_temperature
    dict_config["disable_sys_args"] = True
    config = Config(**dict_config)

    _bcast(True)  # sync before starting -> proceed
    _bcast(config)  # broadcast config to all workers

    print(f"Trial {trial.number}: cat_temp={cat_temperature:.4f}")

    metrics = run_experiment(config, model, evaluator)
    assert metrics is not None

    perplexity = metrics["perplexity"]
    cos_sim = metrics["cosine_similarity"]
    trial.set_user_attr("metrics", metrics)

    print(f"Trial {trial.number} completed: Perplexity={perplexity:.4f}, Cosine Similarity={cos_sim:.4f}")

    return perplexity, cos_sim


if __name__ == "__main__":
    og_config = Config()
    init_trials = [{"cat_temperature": t} for t in np.linspace(1.0, 1.5, 5)]
    run_sweep(SWEEP_NAME, og_config, _objective, init_trials=init_trials)
