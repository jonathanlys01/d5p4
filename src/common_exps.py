"""
Shared experiment logic for distributed runs and Optuna sweeps.
"""

import json
import os
import signal
import uuid
from dataclasses import asdict
from datetime import datetime

import idr_torch
import optuna
import torch
import torch.distributed as dist
from optuna import Study
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from config import RESULTS_DIR, Config
from diffusion_mdlm import MDLMSampler
from eval_core import Evaluator
from utils import compile_model, print, seed_all


# Graceful shutdown handling for SLURM pre-termination signal (--signal=B:SIGTERM@<timeout>)
_shutdown_requested = False


def _handle_shutdown_signal(signum, _frame):
    """Signal handler that sets the shutdown flag without interrupting current work."""
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = True
    print(f"Received signal {signum}, will stop after current trial completes.")


def _bcast(obj):
    """Broadcast a single Python object from rank 0; return it on all ranks."""
    if not dist.is_available() or not dist.is_initialized():
        return obj
    is_master: bool = idr_torch.is_master  # type: ignore
    obj_list = [obj] if is_master else [None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def _save(text, config, uid, rank=0):
    samples = {
        "text_samples": text,  # list of lists of strings
        "config": asdict(config),
    }

    name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}_{str(uid)}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/{name}.json", "w") as f:
        json.dump(samples, f, indent=4)


def generate_samples_with_model(config: Config, model: MDLMSampler, evaluator: Evaluator | None = None):
    """Generate samples using a pre-initialized model."""
    model.update_config(config)
    offset = 0
    if model.distributed_utils:
        offset = model.distributed_utils.rank

    seed_all(config.seed + offset)
    texts = []

    unique_id = uuid.uuid4()
    print(f"Experiment ID: {unique_id}, n_runs: {config.n_runs}")

    # Check if we need to do K-subsampling (only on master rank)
    is_master = model.distributed_utils is None or model.distributed_utils.rank == 0
    should_subsample = config.subsample_k > 0 and is_master
    if should_subsample and evaluator is None:
        raise ValueError("K-subsampling requires an evaluator to be provided")

    for _ in range(config.n_runs):
        samples = model.sample()  # dispatch_sequences gathers all seqs to all ranks
        decoded = model.tokenizer.batch_decode(samples, skip_special_tokens=True)

        if should_subsample:
            assert evaluator is not None
            print(f"Selecting {config.subsample_k} best sequences from {len(decoded)} candidates (metric: ppl)...")
            selected_groups = evaluator.evaluate_baseline([decoded], metric="ppl", k=config.subsample_k)
            selected = selected_groups[0]
            texts.append(selected)
        elif is_master or config.subsample_k == 0:
            texts.append(decoded)

        if is_master or config.subsample_k == 0:
            _save(texts, config, unique_id, rank=offset)

    samples = {
        "text_samples": texts,
        "config": asdict(config),
        "experiment_id": str(unique_id),
    }
    master = model.distributed_utils is None or model.distributed_utils.rank == 0
    if master:
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(unique_id)}"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(f"{RESULTS_DIR}/exp-{name}.json", "w") as f:
            json.dump(samples, f, indent=4)

    for file in os.listdir(RESULTS_DIR):
        if file.startswith("temp_") and file.endswith(f"_rank{offset}_{unique_id}.json"):
            os.remove(os.path.join(RESULTS_DIR, file))

    return unique_id, master


def generate_samples(config: Config):
    """Generate samples by creating a new model instance."""
    model = MDLMSampler(config)
    model.model = compile_model(model.model, config)
    return generate_samples_with_model(config, model)


def eval_samples(
    unique_id: str,
    config: Config,
    evaluator: Evaluator | None = None,
    references: list[list[str]] | None = None,
):
    if evaluator is None:
        evaluator = Evaluator(
            batch_size=config.eval_batch_size,
            force=True,
            ppl_model_id=config.ppl_model_id,
            cos_model_id=config.cos_model_id,
        )

    metrics = {}
    # Evaluation expects the result file to exist
    for file in os.listdir(RESULTS_DIR):
        if file.endswith(f"{unique_id}.json"):
            file_path = os.path.join(RESULTS_DIR, file)
            metrics = evaluator.eval_from_file(file_path, references=references)

    return metrics


def run_experiment(
    config: Config,
    model: MDLMSampler | None = None,
    evaluator: Evaluator | None = None,
    references: list[list[str]] | None = None,
):
    """Run experiment with optional pre-initialized model."""
    torch.cuda.empty_cache()
    if model is None:
        unique_id, master = generate_samples(config)
    else:
        unique_id, master = generate_samples_with_model(config, model, evaluator)
    torch.cuda.empty_cache()  # clear GPU memory before evaluation
    if not master:
        return None
    metrics = eval_samples(str(unique_id), config, evaluator, references=references)
    return metrics


class _GracefulShutdownCallback:
    """Optuna callback that stops optimization when shutdown is requested."""

    def __call__(self, study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        if _shutdown_requested:
            print("Graceful shutdown: stopping optimization after trial completion.")
            study.stop()


def run_sweep(sweep_name, og_config, objective_fn, init_trials=None, study_to_restart: Study | None = None):
    """
    Unified Optuna sweep loop handling both master and worker ranks.
    Model is initialized once and reused across all trials.

    Handles SIGTERM gracefully by stopping after the current trial completes.
    """
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = False  # Reset in case of prior runs

    n_trials = og_config.n_trials

    # Register signal handler for graceful shutdown (SLURM --signal=B:SIGTERM@<timeout>)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=idr_torch.world_size,  # type: ignore
        rank=idr_torch.rank,  # type: ignore
    )

    device = f"cuda:{idr_torch.local_rank}"
    torch.cuda.set_device(device)

    is_master: bool = idr_torch.is_master  # type: ignore

    # Initialize model once before the sweep
    model = MDLMSampler(og_config)
    model.model = compile_model(model.model, og_config)

    if is_master:
        # Initialize evaluator once before the sweep
        evaluator = Evaluator(
            batch_size=og_config.eval_batch_size,
            force=True,
            ppl_model_id=og_config.ppl_model_id,
            cos_model_id=og_config.cos_model_id,
        )

        if not study_to_restart:
            storage = JournalStorage(JournalFileBackend(f"optuna_{sweep_name}.log"))
            study = optuna.create_study(
                directions=["minimize", "minimize"],
                study_name=sweep_name,
                storage=storage,
                load_if_exists=True,
            )
        else:
            study = study_to_restart

        if len(study.trials) == 0:  # enqueue initial points
            study.set_user_attr("og_config", asdict(og_config))
            if init_trials:
                for trial_params in init_trials:
                    study.enqueue_trial(trial_params)

        study.optimize(
            lambda trial: objective_fn(trial, og_config, model, evaluator),
            n_trials=n_trials,
            callbacks=[_GracefulShutdownCallback()],
        )
        _bcast(False)  # signal workers to stop

    else:
        # Workers don't need evaluator - K-subsampling only happens on master
        # (all ranks get same sequences via dispatch_sequences anyway)
        while True:
            proceed = _bcast(None)
            if not proceed:
                break

            cfg = _bcast(None)
            assert cfg is not None
            run_experiment(cfg, model)

            # Workers also check for shutdown signal
            if _shutdown_requested:
                print("Worker: graceful shutdown requested, waiting for master.")

    dist.destroy_process_group()
