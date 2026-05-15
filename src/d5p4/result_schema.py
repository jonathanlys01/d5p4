"""Shared JSON structure for generation result files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, ValidationInfo, field_validator, model_validator


TEXT_SAMPLES = "text_samples"
EVAL_TEXT_SAMPLES = "eval_text_samples"
REFERENCES = "references"
CONFIG = "config"
METRICS = "metrics"
EXPERIMENT_ID = "experiment_id"
INTERNAL_SCORES = "internal_scores"
LEGACY_INTERNAL_SCORES = "eval_internal_scores"
EVAL_SELECTED_INDICES = "eval_selected_indices"
EVAL_SELECTION = "eval_selection"
INTERNAL_SCORE_METADATA = "internal_score_metadata"


def _config_to_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    raise TypeError(f"config must be a dict or dataclass instance, got {type(config).__name__}.")


class InternalScoreMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    method: str | None = None
    scope: str | None = None
    higher_is_better: bool | None = None


def _check_group_count(field_name: str, groups: list[Any] | None, n_groups: int) -> None:
    if groups is not None and len(groups) != n_groups:
        raise ValueError(
            f"{field_name} must contain one group per text_samples group, got {len(groups)} for {n_groups}.",
        )


def _check_internal_scores_alignment(text_samples: list[list[str]], score_groups: list[list[float]]) -> None:
    _check_group_count(INTERNAL_SCORES, score_groups, len(text_samples))
    for group_idx, (score_group, text_group) in enumerate(zip(score_groups, text_samples)):
        if len(score_group) != len(text_group):
            raise ValueError(
                f"{INTERNAL_SCORES}[{group_idx}] must contain one score per sequence, got "
                f"{len(score_group)} for {len(text_group)}.",
            )


def _check_selected_indices_alignment(
    text_samples: list[list[str]],
    eval_text_samples: list[list[str]] | None,
    selected_indices: list[list[int]],
) -> None:
    _check_group_count(EVAL_SELECTED_INDICES, selected_indices, len(text_samples))
    if eval_text_samples is not None:
        _check_group_count(EVAL_TEXT_SAMPLES, eval_text_samples, len(selected_indices))

    for group_idx, (index_group, text_group) in enumerate(zip(selected_indices, text_samples)):
        if eval_text_samples is not None and len(index_group) != len(eval_text_samples[group_idx]):
            raise ValueError(
                f"{EVAL_SELECTED_INDICES}[{group_idx}] must contain one index per eval sequence, got "
                f"{len(index_group)} for {len(eval_text_samples[group_idx])}.",
            )
        for item_idx, index in enumerate(index_group):
            if index < 0 or index >= len(text_group):
                raise ValueError(
                    f"{EVAL_SELECTED_INDICES}[{group_idx}][{item_idx}]={index} is outside "
                    f"text group length {len(text_group)}.",
                )


class GenerationResult(BaseModel):
    """Canonical result-file schema.

    ``internal_scores`` is the canonical key. ``eval_internal_scores`` remains
    accepted as a compatibility alias for older result files.
    """

    model_config = ConfigDict(extra="allow")

    text_samples: list[list[str]] | None = None
    config: dict[str, Any] | None = Field(default_factory=dict)
    eval_text_samples: list[list[str]] | None = None
    references: list[list[str]] | None = None
    internal_scores: list[list[FiniteFloat]] | None = None
    eval_internal_scores: list[list[FiniteFloat]] | None = None
    eval_selected_indices: list[list[int]] | None = None
    eval_selection: dict[str, Any] | None = None
    internal_score_metadata: InternalScoreMetadata | dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    experiment_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_score_aliases(_cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if INTERNAL_SCORES not in normalized and LEGACY_INTERNAL_SCORES in normalized:
            normalized[INTERNAL_SCORES] = normalized[LEGACY_INTERNAL_SCORES]
        return normalized

    @field_validator("internal_scores", "eval_internal_scores", mode="before")
    @classmethod
    def _validate_score_values(_cls, value: Any, info: ValidationInfo) -> Any:
        if value is None:
            return value
        if not isinstance(value, list):
            raise ValueError(f"{info.field_name} must be a list of score groups.")
        for group_idx, score_group in enumerate(value):
            if not isinstance(score_group, list):
                raise ValueError(f"{info.field_name}[{group_idx}] must be a list of numbers.")
            for score_idx, score in enumerate(score_group):
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise ValueError(
                        f"{info.field_name}[{group_idx}][{score_idx}] must be a finite number.",
                    )
        return value

    @field_validator("eval_selected_indices", mode="before")
    @classmethod
    def _validate_index_values(_cls, value: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, list):
            raise ValueError(f"{EVAL_SELECTED_INDICES} must be a list of index groups.")
        for group_idx, index_group in enumerate(value):
            if not isinstance(index_group, list):
                raise ValueError(f"{EVAL_SELECTED_INDICES}[{group_idx}] must be a list of integers.")
            for item_idx, index in enumerate(index_group):
                if isinstance(index, bool) or not isinstance(index, int):
                    raise ValueError(
                        f"{EVAL_SELECTED_INDICES}[{group_idx}][{item_idx}] must be an integer.",
                    )
        return value

    @model_validator(mode="after")
    def _validate_alignment(self) -> GenerationResult:
        if self.internal_scores is None and self.eval_internal_scores is not None:
            self.internal_scores = self.eval_internal_scores

        if self.text_samples is None:
            return self

        n_groups = len(self.text_samples)
        _check_group_count(EVAL_TEXT_SAMPLES, self.eval_text_samples, n_groups)
        _check_group_count(REFERENCES, self.references, n_groups)
        if self.internal_scores is not None:
            _check_internal_scores_alignment(self.text_samples, self.internal_scores)
        if self.eval_selected_indices is not None:
            _check_selected_indices_alignment(self.text_samples, self.eval_text_samples, self.eval_selected_indices)

        return self


def normalize_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy aliases in-place and return *payload*."""
    if INTERNAL_SCORES not in payload and LEGACY_INTERNAL_SCORES in payload:
        payload[INTERNAL_SCORES] = payload[LEGACY_INTERNAL_SCORES]
    return payload


def validate_generation_result_payload(
    payload: dict[str, Any],
    *,
    require_text_samples: bool = True,
) -> None:
    normalize_result_payload(payload)
    result = GenerationResult.model_validate(payload)
    if require_text_samples and result.text_samples is None:
        raise ValueError(f"Result payload is missing required field {TEXT_SAMPLES!r}.")
    normalize_result_payload(payload)


def build_generation_result_payload(
    *,
    text_samples: list[list[str]],
    config: Any,
    eval_text_samples: list[list[str]] | None = None,
    references: list[list[str]] | None = None,
    internal_scores: list[list[float]] | None = None,
    eval_selected_indices: list[list[int]] | None = None,
    eval_selection: dict[str, Any] | None = None,
    internal_score_metadata: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    experiment_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        TEXT_SAMPLES: text_samples,
        CONFIG: _config_to_dict(config),
        EVAL_TEXT_SAMPLES: eval_text_samples,
        REFERENCES: references,
        INTERNAL_SCORES: internal_scores,
        EVAL_SELECTED_INDICES: eval_selected_indices,
        EVAL_SELECTION: eval_selection,
        INTERNAL_SCORE_METADATA: internal_score_metadata,
        METRICS: metrics,
        EXPERIMENT_ID: experiment_id if experiment_id is not None else None,
    }
    if extra:
        payload.update(extra)

    result = GenerationResult.model_validate(payload)
    return result.model_dump(exclude_none=True)


def get_eval_text_groups(payload: dict[str, Any]) -> list[list[str]] | None:
    result = GenerationResult.model_validate(normalize_result_payload(payload))
    return result.eval_text_samples if result.eval_text_samples is not None else result.text_samples


def _value_summary(value: Any) -> str:
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, str):
        return f"str[{len(value)}]"
    if value is None:
        return "null"
    return type(value).__name__


def payload_tree_lines(
    payload: Any,
    *,
    name: str = "payload",
    max_items: int = 3,
) -> list[str]:
    lines: list[str] = []

    def _visit_children(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            items = list(value.items())
            visible_items = items[:max_items]
            for idx, (key, child) in enumerate(visible_items):
                branch = "└── " if idx == len(visible_items) - 1 and len(items) <= max_items else "├── "
                child_prefix = prefix + ("    " if branch == "└── " else "│   ")
                lines.append(f"{prefix}{branch}{key}: {_value_summary(child)}")
                if isinstance(child, (dict, list)):
                    _visit_children(child, child_prefix)
            if len(items) > max_items:
                lines.append(f"{prefix}└── ... {len(items) - max_items} more")
            return

        if isinstance(value, list):
            visible_items = value[:max_items]
            for idx, child in enumerate(visible_items):
                branch = "└── " if idx == len(visible_items) - 1 and len(value) <= max_items else "├── "
                child_prefix = prefix + ("    " if branch == "└── " else "│   ")
                lines.append(f"{prefix}{branch}[{idx}]: {_value_summary(child)}")
                if isinstance(child, (dict, list)):
                    _visit_children(child, child_prefix)
            if len(value) > max_items:
                lines.append(f"{prefix}└── ... {len(value) - max_items} more")

    lines.append(f"{name}: {_value_summary(payload)}")
    _visit_children(payload, "")
    return lines


def print_payload_tree(payload: Any, *, name: str = "payload", max_items: int = 3) -> None:
    for line in payload_tree_lines(payload, name=name, max_items=max_items):
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and print a tree view of a generation result payload.")
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to a generation result JSON file. If omitted, a dummy structure is printed.",
    )
    parser.add_argument("--max-items", type=int, default=3, help="Maximum children to print per object/list node.")
    parser.add_argument("--no-validate", action="store_true", help="Print the raw JSON tree without schema validation.")
    args = parser.parse_args()

    if args.input_path:
        with open(args.input_path) as f:
            payload = json.load(f)
    else:
        payload = {
            TEXT_SAMPLES: [["Sample text 1", "Sample text 2"], ["Sample text 3"]],
            CONFIG: {"model": "test-model", "temp": 0.7},
            METRICS: {"accuracy": 0.99, "latency": 150},
            INTERNAL_SCORES: [[0.5, 0.6], [0.7]],
        }

    if not args.no_validate:
        if not isinstance(payload, dict):
            raise ValueError("Generation result payload must be a JSON object.")
        result = GenerationResult.model_validate(normalize_result_payload(payload))
        payload = result.model_dump(exclude_none=True)

    print_payload_tree(payload, name=args.input_path or "dummy_payload", max_items=args.max_items)


if __name__ == "__main__":
    main()
