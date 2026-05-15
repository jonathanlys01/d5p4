"""
Search Sweep: Diverse Beam Search (DivBS)
Sweeps over the diversity penalty coefficient.
"""

from dataclasses import asdict

import numpy as np

from d5p4.common_exps import _bcast, print, run_experiment, run_sweep
from d5p4.config import Config


SWEEP_NAME = "d5p4_divbs_optuna"


def _objective(trial, og_config: Config, model, evaluator):
    div_alpha = trial.suggest_float("_diversity_alpha", 0.0, 10.0)

    dict_config = asdict(og_config)
    dict_config["_diversity_alpha"] = div_alpha
    dict_config["disable_sys_args"] = True
    config = Config(**dict_config)

    _bcast(True)  # sync before starting -> proceed
    _bcast(config)  # broadcast config to all workers

    print(f"Trial {trial.number}: div_alpha={div_alpha:.4f}")

    metrics = run_experiment(config, model, evaluator)
    assert metrics is not None

    perplexity = metrics["perplexity"]
    cos_sim = metrics["cosine_similarity"]
    trial.set_user_attr("metrics", metrics)

    print(f"Trial {trial.number} completed: Perplexity={perplexity:.4f}, Cosine Similarity={cos_sim:.4f}")

    return perplexity, cos_sim


if __name__ == "__main__":
    og_config = Config()
    # Note: Ensure the method is set to diverse_beam in config or via CLI
    init_trials = [{"_diversity_alpha": alpha} for alpha in np.linspace(0.0, 5.0, 5)]
    run_sweep(SWEEP_NAME, og_config, _objective, init_trials=init_trials)
