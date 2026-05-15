"""
Post-hoc best-of-N selection for oversampled generation result files.

Input files
- Scans OVERSAMPLE_BASELINE_PATH, or config.results_dir when unset.
- Reads JSON files with text_samples shaped as list[list[str]], where each outer
  group is a prompt/question and each inner group is that prompt's N candidates.
- Also supports aggregate files with results_by_cfg[*].samples[*].independent_pool_n.
- Skips temp files indirectly by only reading final .json files and ignores outputs
  already named with "-bon-".

Selection behavior
- Selects subsample_k candidates per prompt group; this is not a global top-k
  across prompts.
- Uses Evaluator.evaluate_baseline with transversal=False, so it does not enforce
  one item per transversal subgroup.
- Supported selectors:
  - f1: enabled only when references can be loaded.
  - ppl: always enabled.
  - int: enabled only when internal_scores or legacy eval_internal_scores exists.
  - random: always enabled; equivalent to k IID candidates from the N-sample pool.

Environment flags
- OVERSAMPLE_BASELINE_PATH: input directory or single input JSON file. Default:
  config.results_dir. Outputs are written next to the input file(s).
- OVERSAMPLE_BASELINE_SAVE_SAMPLES: include selected/raw samples in outputs.
  Default: true. When false, write metrics/metadata only.
- OVERSAMPLE_BASELINE_METHOD: optional source config.method filter. Default: unset
  (process every compatible source method).
- OVERSAMPLE_BASELINE_METRICS: optional comma-separated metric filter, e.g.
  "f1,ppl,int,random". Default: all available selectors for each source file.
"""

import json
import os
from dataclasses import fields
from typing import Any

from d5p4.config import Config
from d5p4.data import get_qa_dataset
from d5p4.eval_core import Evaluator


def _stable_variant_seed(base_seed: int, variant_name: str) -> int:
    return base_seed + sum(ord(ch) for ch in variant_name)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> list[str] | None:
    value = os.getenv(name)
    if value is None:
        return None
    items = [item.strip().lower() for item in value.split(",")]
    return [item for item in items if item]


def _select_and_evaluate_baseline(
    evaluator: Evaluator,
    texts: list[list[str]],
    metric: str,
    subsample_k: int,
    references: list[list[str]] | None = None,
    internal_scores: list[list[float]] | None = None,
    random_seed: int | None = None,
) -> tuple[list[list[str]], dict[str, float | str]]:
    baseline_kwargs: dict[str, Any] = {"references": references}
    if internal_scores is not None:
        baseline_kwargs["internal_scores"] = internal_scores
    if random_seed is not None:
        baseline_kwargs["random_seed"] = random_seed

    selected = evaluator.evaluate_baseline(
        texts,
        metric,
        subsample_k,
        **baseline_kwargs,
    )
    metrics = evaluator.evaluate(selected, references=references)
    return selected, metrics


def _load_references(current_config: Config, expected_groups: int) -> list[list[str]] | None:
    if not current_config.qa_dataset:
        return None

    try:
        dataset = get_qa_dataset(current_config)
        limit = current_config.qa_dataset_len if current_config.qa_dataset_len > 0 else len(dataset)
        references = [row.correct_answers for row in dataset.itertuples()][:limit]  # type: ignore
        print(f"Loaded {len(references)} references for {current_config.qa_dataset}")
    except Exception as e:
        print(f"Warning: Could not load references for {current_config.qa_dataset}: {e}")
        return None

    if len(references) != expected_groups:
        print(
            "Warning: Reference count does not match text groups "
            f"({len(references)} refs vs {expected_groups} groups); skipping reference-based metrics.",
        )
        return None

    return references


def _get_internal_scores(data: dict, expected_groups: int) -> list[list[float]] | None:
    scores = data.get("internal_scores")
    if scores is None:
        scores = data.get("eval_internal_scores")

    if not isinstance(scores, list):
        return None
    if len(scores) != expected_groups:
        print(
            "Warning: Internal score count does not match text groups "
            f"({len(scores)} score groups vs {expected_groups} text groups); skipping int selection.",
        )
        return None

    return scores


def _iter_text_groups(file_name: str, data: dict) -> list[tuple[str, dict, list[list[str]], list[list[float]] | None]]:
    file_config_dict = data.get("config", {})
    texts = data.get("text_samples")
    if isinstance(texts, list):
        return [(file_name.removesuffix(".json"), file_config_dict, texts, _get_internal_scores(data, len(texts)))]

    results_by_cfg = data.get("results_by_cfg")
    if not isinstance(results_by_cfg, dict):
        return []

    extracted_groups: list[tuple[str, dict, list[list[str]], list[list[float]] | None]] = []
    for cfg_key, cfg_result in results_by_cfg.items():
        samples = cfg_result.get("samples")
        if not isinstance(samples, list):
            continue

        cfg_texts: list[list[str]] = []
        for sample in samples:
            if not isinstance(sample, dict):
                cfg_texts = []
                break
            pool = sample.get("independent_pool_n")
            if not isinstance(pool, list):
                cfg_texts = []
                break
            cfg_texts.append(pool)

        if not cfg_texts:
            continue

        cfg_config_dict = dict(file_config_dict)
        cfg_config_dict["cfg_scale"] = float(cfg_key)
        extracted_groups.append((f"{file_name.removesuffix('.json')}-cfg-{cfg_key}", cfg_config_dict, cfg_texts, None))

    return extracted_groups


def _resolve_input_files(path: str) -> tuple[str, list[str]]:
    if os.path.isfile(path):
        if not path.endswith(".json"):
            raise ValueError(f"OVERSAMPLE_BASELINE_PATH file must be a JSON file: {path}")
        directory = os.path.dirname(path) or "."
        return directory, [os.path.basename(path)]

    if os.path.isdir(path):
        files = sorted([f for f in os.listdir(path) if f.endswith(".json") and "-bon-" not in f])
        return path, files

    raise FileNotFoundError(f"OVERSAMPLE_BASELINE_PATH does not exist: {path}")


if __name__ == "__main__":
    config = Config()
    evaluator = Evaluator(
        batch_size=config.eval_batch_size,
        ppl_model_id=config.ppl_model_id,
        cos_model_id=config.cos_model_id,
    )
    save_samples = _env_flag("OVERSAMPLE_BASELINE_SAVE_SAMPLES", default=True)
    method_filter = os.getenv("OVERSAMPLE_BASELINE_METHOD")
    requested_metrics = _env_list("OVERSAMPLE_BASELINE_METRICS")

    path = os.path.abspath(os.path.expanduser(os.getenv("OVERSAMPLE_BASELINE_PATH", config.results_dir)))
    output_dir, files = _resolve_input_files(path)

    if os.path.isfile(path):
        print(f"Processing oversample baseline file: {path}")
    else:
        print(f"Scanning oversample baseline files in: {path}")

    subsample_k = config.subsample_k
    assert subsample_k != 0

    print("Using per-group subsample_k: ", subsample_k)

    for file in files:
        file_path = os.path.join(output_dir, file)
        with open(file_path, "r") as f:
            data = json.load(f)

        grouped_sources = _iter_text_groups(file, data)
        if not grouped_sources:
            print(f"Skipping {file}: no compatible oversample candidate groups found")
            continue

        for output_stem, file_config_dict, texts, internal_scores in grouped_sources:
            current_config = config
            if file_config_dict:
                valid_fields = {f.name for f in fields(Config)}
                filtered_config = {k: v for k, v in file_config_dict.items() if k in valid_fields}
                filtered_config.pop("disable_sys_args", None)
                current_config = Config(disable_sys_args=True, **filtered_config)

            if method_filter is not None and current_config.method != method_filter:
                print(f"Skipping {output_stem}.json: method={current_config.method!r}, expected {method_filter!r}")
                continue

            references = _load_references(current_config, len(texts))
            available_metrics = ["f1", "ppl"] if references is not None else ["ppl"]
            if internal_scores is not None:
                available_metrics.append("int")
            available_metrics.append("random")

            if requested_metrics is None:
                metrics_to_run = available_metrics
            else:
                metrics_to_run = [metric for metric in requested_metrics if metric in available_metrics]
                skipped_metrics = [metric for metric in requested_metrics if metric not in available_metrics]
                if skipped_metrics:
                    print(
                        f"Skipping unavailable metrics for {output_stem}.json: {', '.join(skipped_metrics)} "
                        f"(available: {', '.join(available_metrics)})",
                    )

            for metric in metrics_to_run:
                print(f"File: {output_stem}.json | Metric: {metric}")
                selected, metrics = _select_and_evaluate_baseline(
                    evaluator,
                    texts,
                    metric,
                    subsample_k,
                    references=references,
                    internal_scores=internal_scores if metric == "int" else None,
                    random_seed=_stable_variant_seed(current_config.seed, output_stem) if metric == "random" else None,
                )

                save_data = {
                    "config": file_config_dict,
                    "metrics": metrics,
                    "experiment_id": data.get("experiment_id", ""),
                    "source_file": file,
                    "selection_metric": metric,
                    "subsample_k": subsample_k,
                }
                if save_samples:
                    save_data["text_samples"] = selected
                    save_data["raw_text_samples"] = texts
                    if metric == "int" and internal_scores is not None:
                        save_data["raw_internal_scores"] = internal_scores
                out_name = f"{output_stem}-bon-{metric}.json"
                with open(os.path.join(output_dir, out_name), "w") as f_out:
                    json.dump(save_data, f_out, indent=4)

                print("-" * 80)
                for key, value in metrics.items():
                    print(f"{metric}_{key}: {value}")
                print("-" * 80)
