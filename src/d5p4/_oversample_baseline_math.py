"""
Post-hoc best-of-N selection for oversampled math result files.

Input files
- Scans OVERSAMPLE_MATH_BASELINE_PATH, or config.results_dir when unset.
- Reads JSON files in the llada_math.py format:
  {"results": [{question, gold_answer, answer_str, generations, ...}], ...}
- Also supports temp checkpoint shape {"results": {"results": [...]}, ...}.
- Skips outputs already named with "-math-bon-".

Selection behavior
- Selects candidates per question, never globally across questions.
- Uses Evaluator.evaluate_baseline for selector scores:
  - acc: oracle exact math correctness via MathEvaluator.
  - f1: uses answer_str/gold_answer as string references.
  - ppl: lower perplexity is better.
  - int: uses internal_scores or eval_internal_scores when present.
  - random: uniform random subset from the N-candidate pool.
- Recomputes MathEvaluator scores, per-question accuracy, overall_accuracy, and
  math_metrics on the selected candidates.

Environment flags
- OVERSAMPLE_MATH_BASELINE_PATH: input directory or single input JSON file.
  Default: config.results_dir. Outputs are written next to the input file(s).
- OVERSAMPLE_MATH_BASELINE_SAVE_RAW: include raw_results in outputs. Default:
  true. Selected results are always written as "results" to preserve math format.
- OVERSAMPLE_MATH_BASELINE_METHOD: optional source config.method filter. Default:
  unset (process every compatible source method).
- OVERSAMPLE_MATH_BASELINE_METRICS: optional comma-separated metric filter, e.g.
  "acc,f1,ppl,int,random". Default: all available selectors for each source file.
- OVERSAMPLE_MATH_BASELINE_TRANSVERSAL: when true, pick one representative from
  each contiguous lineage subgroup of size group_size. This uses k=1 within each
  subgroup and is useful when candidates are sibling leaves from search.
- OVERSAMPLE_MATH_BASELINE_GROUP_SIZE: override source config.group_size for
  transversal selection.
- OVERSAMPLE_MATH_BASELINE_EXPECTED_SELECTED_K: optional guard requiring every
  selected question to have exactly this many generations before evaluation.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import fields
from typing import Any

from d5p4.config import Config
from d5p4.eval_core import Evaluator, MathEvaluator


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


def _resolve_input_files(path: str) -> tuple[str, list[str]]:
    if os.path.isfile(path):
        if not path.endswith(".json"):
            raise ValueError(f"OVERSAMPLE_MATH_BASELINE_PATH file must be a JSON file: {path}")
        directory = os.path.dirname(path) or "."
        return directory, [os.path.basename(path)]

    if os.path.isdir(path):
        files = sorted([f for f in os.listdir(path) if f.endswith(".json") and "-math-bon-" not in f])
        return path, files

    raise FileNotFoundError(f"OVERSAMPLE_MATH_BASELINE_PATH does not exist: {path}")


def _extract_results(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]

    if not isinstance(data, dict):
        return []

    results = data.get("results")
    if isinstance(results, dict):
        results = results.get("results", [])
    if not isinstance(results, list):
        return []

    return [row for row in results if isinstance(row, dict)]


def _iter_math_sources(file_name: str, data: Any) -> list[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
    results = _extract_results(data)
    if not results:
        return []

    compatible_results = [row for row in results if isinstance(row.get("generations"), list) and "gold_answer" in row]
    if not compatible_results:
        return []

    config_dict = data.get("config", {}) if isinstance(data, dict) else {}
    if not isinstance(config_dict, dict):
        config_dict = {}

    return [(file_name.removesuffix(".json"), config_dict, compatible_results)]


def _get_internal_scores(data: Any, results: list[dict[str, Any]]) -> list[list[float]] | None:
    expected_groups = len(results)

    if isinstance(data, dict):
        scores = data.get("internal_scores")
        if scores is None:
            scores = data.get("eval_internal_scores")
        if isinstance(scores, list):
            if len(scores) == expected_groups:
                return scores
            print(
                "Warning: Internal score count does not match math result groups "
                f"({len(scores)} score groups vs {expected_groups} result groups); skipping int selection.",
            )
            return None

    per_row_scores = []
    for row in results:
        scores = row.get("internal_scores")
        if scores is None:
            scores = row.get("eval_internal_scores")
        if not isinstance(scores, list):
            return None
        per_row_scores.append(scores)
    return per_row_scores


def _math_groups(results: list[dict[str, Any]]) -> tuple[list[list[str]], list[str], list[list[str]]]:
    texts: list[list[str]] = []
    gold_answers: list[str] = []
    references: list[list[str]] = []

    for row in results:
        generations = row.get("generations")
        if not isinstance(generations, list):
            continue
        texts.append([str(generation) for generation in generations])
        gold_answer = str(row.get("gold_answer", ""))
        gold_answers.append(gold_answer)
        answer_str = row.get("answer_str")
        references.append([answer_str] if isinstance(answer_str, str) and answer_str else [gold_answer])

    return texts, gold_answers, references


def _math_accuracy_scores(
    math_evaluator: MathEvaluator,
    texts: list[list[str]],
    gold_answers: list[str],
) -> list[list[float]]:
    return [
        [float(score) for score in math_evaluator.score_group(generations, gold_answer)]
        for generations, gold_answer in zip(texts, gold_answers)
    ]


def _build_selected_results(
    source_results: list[dict[str, Any]],
    selected: list[list[str]],
    math_evaluator: MathEvaluator,
) -> list[dict[str, Any]]:
    selected_results: list[dict[str, Any]] = []

    for source_row, selected_generations in zip(source_results, selected):
        gold_answer = str(source_row.get("gold_answer", ""))
        scores = math_evaluator.score_group(selected_generations, gold_answer)
        accuracy = sum(scores) / len(scores) if scores else 0.0

        row = deepcopy(source_row)
        row["generations"] = selected_generations
        row["scores"] = scores
        row["accuracy"] = accuracy
        row["source_generation_count"] = len(source_row.get("generations", []))
        row["selected_generation_count"] = len(selected_generations)
        selected_results.append(row)

    return selected_results


def _expected_selected_counts(
    texts: list[list[str]],
    subsample_k: int,
    transversal: bool,
    group_size: int,
) -> list[int]:
    if transversal:
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}.")
        counts: list[int] = []
        for idx, group_texts in enumerate(texts):
            if len(group_texts) % group_size != 0:
                raise ValueError(
                    f"Question {idx} has {len(group_texts)} candidates, not divisible by group_size={group_size}.",
                )
            counts.append(len(group_texts) // group_size)
        return counts
    return [subsample_k for _ in texts]


def _validate_selected_cardinality(
    selected: list[list[str]],
    expected_counts: list[int],
    expected_selected_k: int | None = None,
) -> None:
    if len(selected) != len(expected_counts):
        raise ValueError(f"Selected {len(selected)} question groups, expected {len(expected_counts)}.")

    for idx, (selected_generations, expected_count) in enumerate(zip(selected, expected_counts)):
        selected_count = len(selected_generations)
        if selected_count != expected_count:
            raise ValueError(
                f"Question {idx} selected {selected_count} generations, expected {expected_count}.",
            )
        if expected_selected_k is not None and selected_count != expected_selected_k:
            raise ValueError(
                f"Question {idx} selected {selected_count} generations, "
                f"but expected selected cardinality is {expected_selected_k}.",
            )


def _select_and_evaluate_math_baseline(
    selection_evaluator: Evaluator,
    math_evaluator: MathEvaluator,
    results: list[dict[str, Any]],
    metric: str,
    subsample_k: int,
    internal_scores: list[list[float]] | None = None,
    random_seed: int | None = None,
    transversal: bool = False,
    group_size: int = 1,
    expected_selected_k: int | None = None,
    num_workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, float | str]]:
    texts, gold_answers, references = _math_groups(results)
    selection_metric = metric
    selection_internal_scores = internal_scores if metric == "int" else None
    if metric == "acc":
        selection_metric = "int"
        selection_internal_scores = _math_accuracy_scores(math_evaluator, texts, gold_answers)

    selected = selection_evaluator.evaluate_baseline(
        texts,
        selection_metric,
        1 if transversal else subsample_k,
        references=references,
        transversal=transversal,
        group_size=group_size,
        internal_scores=selection_internal_scores,
        random_seed=random_seed,
    )
    _validate_selected_cardinality(
        selected,
        _expected_selected_counts(texts, subsample_k, transversal, group_size),
        expected_selected_k=expected_selected_k,
    )
    selected_results = _build_selected_results(results, selected, math_evaluator)
    metrics = math_evaluator.evaluate(
        selected,
        gold_answers,
        string_references=references,
        num_workers=num_workers,
    )
    return selected_results, metrics


def _overall_accuracy(results: list[dict[str, Any]]) -> float:
    accuracies = [float(row.get("accuracy", 0.0)) for row in results]
    return sum(accuracies) / len(accuracies) if accuracies else 0.0


def _config_from_dict(base_config: Config, config_dict: dict[str, Any]) -> Config:
    if not config_dict:
        return base_config

    valid_fields = {f.name for f in fields(Config)}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
    filtered_config.pop("disable_sys_args", None)
    return Config(disable_sys_args=True, **filtered_config)


if __name__ == "__main__":
    config = Config()
    selection_evaluator = Evaluator(
        batch_size=config.eval_batch_size,
        ppl_model_id=config.ppl_model_id,
        cos_model_id=config.cos_model_id,
    )
    math_evaluator = MathEvaluator()

    save_raw = _env_flag("OVERSAMPLE_MATH_BASELINE_SAVE_RAW", default=True)
    method_filter = os.getenv("OVERSAMPLE_MATH_BASELINE_METHOD")
    requested_metrics = _env_list("OVERSAMPLE_MATH_BASELINE_METRICS")
    transversal = _env_flag("OVERSAMPLE_MATH_BASELINE_TRANSVERSAL", default=False)
    group_size_override = os.getenv("OVERSAMPLE_MATH_BASELINE_GROUP_SIZE")
    expected_selected_k_env = os.getenv("OVERSAMPLE_MATH_BASELINE_EXPECTED_SELECTED_K")
    expected_selected_k = int(expected_selected_k_env) if expected_selected_k_env is not None else None

    path = os.path.abspath(os.path.expanduser(os.getenv("OVERSAMPLE_MATH_BASELINE_PATH", config.results_dir)))
    output_dir, files = _resolve_input_files(path)

    if os.path.isfile(path):
        print(f"Processing oversample math baseline file: {path}")
    else:
        print(f"Scanning oversample math baseline files in: {path}")

    subsample_k = config.subsample_k
    assert subsample_k != 0 or transversal
    selection_k = 1 if transversal else subsample_k

    print("Using per-question selection k: ", selection_k)
    if transversal:
        print("Using transversal lineage collapse: one selected candidate per lineage subgroup")

    num_workers = min(8, os.cpu_count() or 1)

    for file in files:
        file_path = os.path.join(output_dir, file)
        with open(file_path, "r") as f:
            data = json.load(f)

        grouped_sources = _iter_math_sources(file, data)
        if not grouped_sources:
            print(f"Skipping {file}: no compatible math result groups found")
            continue

        for output_stem, file_config_dict, results in grouped_sources:
            current_config = _config_from_dict(config, file_config_dict)

            if method_filter is not None and current_config.method != method_filter:
                print(f"Skipping {output_stem}.json: method={current_config.method!r}, expected {method_filter!r}")
                continue

            group_size = (
                int(group_size_override)
                if group_size_override is not None
                else int(file_config_dict.get("group_size", current_config.group_size))
            )
            internal_scores = _get_internal_scores(data, results)
            available_metrics = ["acc", "f1", "ppl"]
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
                selected_results, math_metrics = _select_and_evaluate_math_baseline(
                    selection_evaluator,
                    math_evaluator,
                    results,
                    metric,
                    selection_k,
                    internal_scores=internal_scores,
                    random_seed=_stable_variant_seed(current_config.seed, output_stem) if metric == "random" else None,
                    transversal=transversal,
                    group_size=group_size,
                    expected_selected_k=expected_selected_k,
                    num_workers=num_workers,
                )

                save_data: dict[str, Any] = {
                    "results": selected_results,
                    "overall_accuracy": _overall_accuracy(selected_results),
                    "math_metrics": math_metrics,
                    "config": file_config_dict,
                    "experiment_id": data.get("experiment_id", "") if isinstance(data, dict) else "",
                    "source_file": file,
                    "selection_metric": metric,
                    "subsample_k": 1 if transversal else subsample_k,
                    "expected_selected_k": expected_selected_k,
                    "transversal": transversal,
                    "group_size": group_size if transversal else 1,
                }
                if save_raw:
                    save_data["raw_results"] = results
                    if metric == "int" and internal_scores is not None:
                        save_data["raw_internal_scores"] = internal_scores

                out_name = f"{output_stem}-math-bon-{metric}.json"
                with open(os.path.join(output_dir, out_name), "w") as f_out:
                    json.dump(save_data, f_out, indent=4)

                print("-" * 80)
                for key, value in math_metrics.items():
                    print(f"{metric}_{key}: {value}")
                print("-" * 80)
