"""
Main entry point for sampling experiments with MDLM.
Supports three sampling methods: random, best-of-n, and smc.
"""

import os
import csv
import json
import pickle
import uuid
import math
import numpy as np
from datetime import datetime
import hydra
import lightning as L
import omegaconf
import torch

import dataloader
import utils
from samplers import random_sampling, bon_sampling, smc_sampling, greedy_sampling
from samplers.utils import (
    load_model,
    restore_model,
    generate_particle_seeds,
    compute_sentence_entropy,
    compute_generative_perplexity,
)
import warnings
from time import perf_counter

from d5p4.eval_core import Evaluator

warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_float32_matmul_precision("high")


def _register_resolvers():
    """Register OmegaConf resolvers used by the upstream Hydra configs."""

    def eval_resolver(expr):
        allowed_builtins = {
            "len": len,
            "min": min,
            "max": max,
            "int": int,
            "float": float,
            "round": round,
            "getattr": getattr,
            "range": range,
            "__import__": __import__,
        }
        return eval(expr, {"__builtins__": allowed_builtins}, {})  # noqa: S307

    def div_up_resolver(a, b):
        a = int(a)
        b = int(b)
        if b == 0:
            raise ValueError("div_up resolver received b=0")
        return math.ceil(a / b)

    def device_count_resolver():
        count = torch.cuda.device_count()
        return count if count > 0 else 1

    def cwd_resolver():
        return os.getcwd()

    def env_path_or_resolver(env_name, suffix, fallback):
        env_value = os.getenv(str(env_name))
        return os.path.join(env_value, str(suffix)) if env_value else str(fallback)

    omegaconf.OmegaConf.register_new_resolver("eval", eval_resolver, replace=True)
    omegaconf.OmegaConf.register_new_resolver("div_up", div_up_resolver, replace=True)
    omegaconf.OmegaConf.register_new_resolver("device_count", device_count_resolver, replace=True)
    omegaconf.OmegaConf.register_new_resolver("cwd", cwd_resolver, replace=True)
    omegaconf.OmegaConf.register_new_resolver("env_path_or", env_path_or_resolver, replace=True)


_register_resolvers()


def _resolve_d5p4_eval_defaults():
    """Load evaluator model defaults from the main project's config."""
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_default.yaml"))
    cfg = omegaconf.OmegaConf.load(config_path)
    resolved = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    return {
        "ppl_model_id": resolved["ppl_model_id"],
        "cos_model_id": resolved["cos_model_id"],
        "eval_batch_size": resolved["eval_batch_size"],
    }


def _maybe_compile_backbone(model, config):
    """Compile the generation backbone when requested."""
    if not getattr(config, "compile_model", True):
        return model

    if not hasattr(model, "backbone"):
        return model

    print("Compiling generation backbone...")
    try:
        model.backbone = torch.compile(model.backbone)
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"Backbone compilation failed, continuing without compile: {exc}")

    return model


def _prepare_cache_paths(config):
    """Resolve cache paths against the original Hydra launch directory."""
    original_cwd = hydra.utils.get_original_cwd()
    cache_dir = config.data.cache_dir
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.abspath(os.path.join(original_cwd, cache_dir))
        config.data.cache_dir = cache_dir

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        hf_home = os.path.join(cache_dir, "huggingface")
        os.environ["HF_HOME"] = hf_home

    transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    if not transformers_cache:
        os.environ["TRANSFORMERS_CACHE"] = hf_home

    return cache_dir


def _build_esmc_metrics(results):
    """Store only average summary metrics in the JSON payload."""
    if not results:
        return {}

    u_denoise_values = [r["u_denoise"] for r in results]
    sent_entropy_values = [r["sentence_entropy"] for r in results]
    ppl_values = [r["perplexity"] for r in results]

    return {
        "avg_u_denoise": float(np.mean(u_denoise_values)),
        "avg_sentence_entropy": float(np.mean(sent_entropy_values)),
        "avg_perplexity": float(np.mean(ppl_values)),
    }


def save_json(text_samples, results, config, output_path, experiment_id):
    """Save generated text batches using the project's JSON schema."""
    payload = {
        "text_samples": text_samples,
        "config": omegaconf.OmegaConf.to_container(config, resolve=True),
        "experiment_id": str(experiment_id),
        "esmc_metrics": _build_esmc_metrics(results),
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def run_random_sampling(model, config, num_runs, base_seed, json_path, experiment_id):
    """Run random sampling experiments."""
    print(f"\n{'=' * 60}")
    print(f"Running Random Sampling")
    print(f"  Runs: {num_runs}")
    print(f"  Steps: {config.sampling.steps}")
    print(f"{'=' * 60}\n")

    results = []
    text_samples = []
    for run_id in range(num_runs):
        run_start = perf_counter()
        seed = base_seed + run_id * 10000

        sample, logs = random_sampling(model=model, num_steps=config.sampling.steps, seed=seed)

        batch_texts = model.tokenizer.batch_decode(sample, skip_special_tokens=True)
        text_samples.append(batch_texts)
        text = model.tokenizer.decode(sample[0])
        sent_entropy = compute_sentence_entropy(sample[0])
        ppl = compute_generative_perplexity(model, text, config.eval.gen_ppl_eval_model_name_or_path)

        result = {
            "run_id": run_id,
            "sample": text,
            "u_denoise": logs["u_denoise"],
            "sentence_entropy": sent_entropy,
            "perplexity": ppl,
            "seed": seed,
        }
        results.append(result)
        save_json(text_samples, results, config, json_path, experiment_id)
        elapsed = perf_counter() - run_start

        print(
            f"Run {run_id + 1}/{num_runs}: U_denoise={logs['u_denoise']:.4f}, "
            f"Sent_Entropy={sent_entropy:.4f}, PPL={ppl:.4f}, Time={elapsed:.2f}s"
        )

    return results


def run_bon_sampling(model, config, num_runs, base_seed, json_path, experiment_id):
    """Run best-of-n sampling experiments."""
    num_particles = config.sampling.num_particles

    print(f"\n{'=' * 60}")
    print(f"Running Best-of-N Sampling")
    print(f"  Runs: {num_runs}")
    print(f"  Particles (N): {num_particles}")
    print(f"  Steps: {config.sampling.steps}")
    print(f"{'=' * 60}\n")

    results = []
    text_samples = []
    for run_id in range(num_runs):
        run_start = perf_counter()
        particle_seeds = generate_particle_seeds(base_seed, num_particles, run_id)

        best_sample, all_samples, logs = bon_sampling(
            model=model, num_particles=num_particles, num_steps=config.sampling.steps, particle_seeds=particle_seeds
        )

        batch_texts = model.tokenizer.batch_decode(all_samples, skip_special_tokens=True)
        text_samples.append(batch_texts)
        text = model.tokenizer.decode(best_sample[0])
        sent_entropy = compute_sentence_entropy(best_sample[0])
        ppl = compute_generative_perplexity(model, text, config.eval.gen_ppl_eval_model_name_or_path)

        result = {
            "run_id": run_id,
            "sample": text,
            "u_denoise": logs["best_u_denoise"],
            "sentence_entropy": sent_entropy,
            "perplexity": ppl,
            "best_idx": logs["best_idx"],
            "all_u_denoise": logs["all_u_denoise"],
            "particle_seeds": particle_seeds,
        }
        results.append(result)
        save_json(text_samples, results, config, json_path, experiment_id)
        elapsed = perf_counter() - run_start

        print(
            f"Run {run_id + 1}/{num_runs}: U_denoise={logs['best_u_denoise']:.4f}, "
            f"Sent_Entropy={sent_entropy:.4f}, PPL={ppl:.4f}, Time={elapsed:.2f}s"
        )

    return results


def run_smc_sampling(model, config, num_runs, base_seed, json_path, experiment_id):
    """Run SMC sampling experiments."""
    num_particles = config.smc.num_particles
    resample_interval = config.smc.resample_interval
    lambda_weight = config.smc.lambda_weight
    potential_type = config.smc.potential_type

    print(f"\n{'=' * 60}")
    print(f"Running SMC Sampling")
    print(f"  Runs: {num_runs}")
    print(f"  Particles: {num_particles}")
    print(f"  Steps: {config.sampling.steps}")
    print(f"  Resample Interval: {resample_interval}")
    print(f"  Lambda: {lambda_weight}")
    print(f"  Potential Type: {potential_type}")
    print(f"{'=' * 60}\n")

    results = []
    text_samples = []
    for run_id in range(num_runs):
        run_start = perf_counter()
        particle_seeds = generate_particle_seeds(base_seed, num_particles, run_id)

        best_sample, all_particles, logs = smc_sampling(
            model=model,
            num_particles=num_particles,
            num_steps=config.sampling.steps,
            resample_interval=resample_interval,
            lambda_weight=lambda_weight,
            potential_type=potential_type,
            particle_seeds=particle_seeds,
        )

        batch_texts = model.tokenizer.batch_decode(all_particles, skip_special_tokens=True)
        text_samples.append(batch_texts)
        text = model.tokenizer.decode(best_sample[0])
        sent_entropy = compute_sentence_entropy(best_sample[0])
        ppl = compute_generative_perplexity(model, text, config.eval.gen_ppl_eval_model_name_or_path)

        result = {
            "run_id": run_id,
            "sample": text,
            "u_denoise": logs["best_u_denoise"],
            "sentence_entropy": sent_entropy,
            "perplexity": ppl,
            "best_particle_idx": logs["best_particle_idx"],
            "num_resampling_events": len(logs["resampling_steps"]),
            "particle_seeds": particle_seeds,
            "detailed_logs": logs,
        }
        results.append(result)
        save_json(text_samples, results, config, json_path, experiment_id)
        elapsed = perf_counter() - run_start

        print(
            f"Run {run_id + 1}/{num_runs}: U_denoise={logs['best_u_denoise']:.4f}, "
            f"Sent_Entropy={sent_entropy:.4f}, PPL={ppl:.4f}, "
            f"Resamples={len(logs['resampling_steps'])}, Time={elapsed:.2f}s"
        )

    return results


def run_greedy_sampling(model, config, num_runs, base_seed, json_path, experiment_id):
    """Run greedy entropy minimization sampling experiments."""
    num_candidates = config.greedy.num_candidates
    beam_size = config.greedy.beam_size

    print(f"\n{'=' * 60}")
    print(f"Running Greedy Entropy Minimization")
    print(f"  Runs: {num_runs}")
    print(f"  Candidates: {num_candidates}")
    print(f"  Beam Size: {beam_size}")
    print(f"  Steps: {config.sampling.steps}")
    print(f"{'=' * 60}\n")

    results = []
    text_samples = []
    for run_id in range(num_runs):
        run_start = perf_counter()
        seed = base_seed + run_id * 10000

        best_sample, all_beams, logs = greedy_sampling(
            model=model, num_steps=config.sampling.steps, num_candidates=num_candidates, beam_size=beam_size, seed=seed
        )

        batch_texts = model.tokenizer.batch_decode(all_beams, skip_special_tokens=True)
        text_samples.append(batch_texts)
        text = model.tokenizer.decode(best_sample[0])
        sent_entropy = compute_sentence_entropy(best_sample[0])
        ppl = compute_generative_perplexity(model, text, config.eval.gen_ppl_eval_model_name_or_path)

        result = {
            "run_id": run_id,
            "sample": text,
            "u_denoise": logs["best_u_denoise"],
            "sentence_entropy": sent_entropy,
            "perplexity": ppl,
            "best_beam_idx": logs["best_beam_idx"],
            "num_candidates": num_candidates,
            "beam_size": beam_size,
            "seed": seed,
            "detailed_logs": logs,
        }
        results.append(result)
        save_json(text_samples, results, config, json_path, experiment_id)
        elapsed = perf_counter() - run_start

        print(
            f"Run {run_id + 1}/{num_runs}: U_denoise={logs['best_u_denoise']:.4f}, "
            f"Sent_Entropy={sent_entropy:.4f}, PPL={ppl:.4f}, Time={elapsed:.2f}s"
        )

    return results


def save_results(results, method, config, output_dir="results"):
    """Save results to CSV and pickle files."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save CSV
    csv_file = os.path.join(output_dir, f"{method}_{timestamp}.csv")
    if results:
        fieldnames = list(results[0].keys())
        # Skip detailed_logs and other complex fields for CSV
        fieldnames = [f for f in fieldnames if f not in ["detailed_logs", "all_u_denoise", "particle_seeds"]]

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                row = {k: v for k, v in result.items() if k in fieldnames}
                writer.writerow(row)

        print(f"\nResults saved to: {csv_file}")

    # Save pickle with full data
    pkl_file = os.path.join(output_dir, f"{method}_{timestamp}.pkl")
    with open(pkl_file, "wb") as f:
        pickle.dump(results, f)
    print(f"Full results saved to: {pkl_file}")


def print_summary(results, method):
    """Print summary statistics."""
    print(f"\n{'=' * 60}")
    print(f"{method.upper()} Summary Statistics (N={len(results)})")
    print(f"{'=' * 60}")

    u_denoise_values = [r["u_denoise"] for r in results]
    sent_entropy_values = [r["sentence_entropy"] for r in results]
    ppl_values = [r["perplexity"] for r in results]

    print(f"U_denoise:        {np.mean(u_denoise_values):.4f} ± {np.std(u_denoise_values):.4f}")
    print(f"Sent Entropy:     {np.mean(sent_entropy_values):.4f} ± {np.std(sent_entropy_values):.4f}")
    print(f"Perplexity:       {np.mean(ppl_values):.4f} ± {np.std(ppl_values):.4f}")
    print(f"{'=' * 60}\n")


@hydra.main(version_base=None, config_path="configs", config_name="config_sampling")
def main(config):
    """Main entry point for sampling experiments."""

    # Validate method
    method = config.sampling.method.lower()
    if method not in ["random", "bon", "smc", "greedy"]:
        raise ValueError(f"Invalid method: {method}. Choose from: random, bon, smc, greedy")

    # Validate SMC config
    if method == "smc":
        if not hasattr(config, "smc") or config.smc is None:
            raise ValueError(
                "SMC method requires 'smc' configuration section. "
                "Please ensure config_sampling.yaml includes the 'smc' section with: "
                "num_particles, resample_interval, lambda_weight, potential_type"
            )
        # Validate required SMC parameters
        required_smc_params = ["num_particles", "resample_interval", "lambda_weight", "potential_type"]
        for param in required_smc_params:
            if not hasattr(config.smc, param):
                raise ValueError(f"SMC configuration missing required parameter: {param}")

    # Validate Best-of-N config
    if method == "bon":
        if not hasattr(config.sampling, "num_particles"):
            raise ValueError(
                "Best-of-N method requires 'num_particles' in sampling configuration. "
                "Please set sampling.num_particles in config or via command line."
            )

    # Validate Greedy config
    if method == "greedy":
        if not hasattr(config, "greedy") or config.greedy is None:
            raise ValueError(
                "Greedy method requires 'greedy' configuration section. "
                "Please ensure config includes the 'greedy' section with: "
                "num_candidates, beam_size"
            )
        # Validate required Greedy parameters
        required_greedy_params = ["num_candidates", "beam_size"]
        for param in required_greedy_params:
            if not hasattr(config.greedy, param):
                raise ValueError(f"Greedy configuration missing required parameter: {param}")

        # Validate parameter values
        if config.greedy.num_candidates < 2:
            raise ValueError(f"num_candidates must be >= 2, got {config.greedy.num_candidates}")
        if config.greedy.beam_size < 1:
            raise ValueError(f"beam_size must be >= 1, got {config.greedy.beam_size}")
        if config.greedy.beam_size > config.greedy.num_candidates:
            raise ValueError(
                f"beam_size ({config.greedy.beam_size}) cannot be larger than "
                f"num_candidates ({config.greedy.num_candidates})"
            )

    # Setup
    L.seed_everything(config.seed)
    logger = utils.get_logger(__name__)
    _prepare_cache_paths(config)
    tokenizer = dataloader.get_tokenizer(config)

    # Load model
    model = load_model(config, tokenizer)
    model = _maybe_compile_backbone(model, config)

    num_runs = config.sampling.num_sample_batches
    base_seed = config.seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = uuid.uuid4()
    output_dir = os.path.join(hydra.utils.get_original_cwd(), "results")
    json_path = os.path.join(output_dir, f"exp-{method}_{timestamp}_{experiment_id}.json")
    print(f"Experiment ID: {experiment_id}")

    # Run sampling
    if method == "random":
        results = run_random_sampling(model, config, num_runs, base_seed, json_path, experiment_id)
    elif method == "bon":
        results = run_bon_sampling(model, config, num_runs, base_seed, json_path, experiment_id)
    elif method == "smc":
        results = run_smc_sampling(model, config, num_runs, base_seed, json_path, experiment_id)
    elif method == "greedy":
        results = run_greedy_sampling(model, config, num_runs, base_seed, json_path, experiment_id)

    # Restore model and save results
    restore_model(model)
    print(f"OUTPUT_PATH:{json_path}")
    print(f"Text samples saved to: {json_path}")
    save_results(results, method, config, output_dir=output_dir)
    print("Running evaluation...")
    eval_defaults = _resolve_d5p4_eval_defaults()
    evaluator = Evaluator(
        batch_size=config.eval.perplexity_batch_size or eval_defaults["eval_batch_size"],
        force=True,
        ppl_model_id=config.eval.gen_ppl_eval_model_name_or_path or eval_defaults["ppl_model_id"],
        cos_model_id=eval_defaults["cos_model_id"],
    )
    metrics = evaluator.eval_from_file(json_path)
    assert metrics is not None and metrics["metrics_summary"] is not None
    print(f"Evaluation complete: {metrics['metrics_summary']}")
    print_summary(results, method)


if __name__ == "__main__":
    main()
