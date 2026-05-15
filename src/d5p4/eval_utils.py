"""
Utility helpers for eval_core.py.

Covers:
- Statistics computation and formatting
- Parallel / sequential task dispatch
- Timing utilities
- String-metric internals (tokenizer caches, distinct, self-BLEU, F1)
- Pickable worker task functions for ProcessPoolExecutor
- File-type detection helpers
"""

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from time import perf_counter

import numpy as np
import sacrebleu
from scipy.stats._continuous_distns import t
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from d5p4.utils import get_hpc_hf_model_path, is_hpc_cluster, process_model_args
from d5p4.utils import print as u_print


# Module-level singleton state


_STRING_METRICS_TOKENIZER: AutoTokenizer | None = None
_STRING_METRICS_SENTENCE_BLEU: sacrebleu.metrics.BLEU | None = None
_STRING_METRICS_CORPUS_BLEU: sacrebleu.metrics.BLEU | None = None

_BATCH_SELF_BLEU_REFERENCE_CAP = 128
_BATCH_SELF_BLEU_EXACT_THRESHOLD = 256


# Statistics helpers


def compute_statistics(values: list[float], prefix: str) -> dict[str, float]:
    """Compute comprehensive statistics for a list of values.

    Returns a dictionary with keys formatted as ``{prefix}_{stat}``.
    The mean is also returned as ``{prefix}`` for backward compatibility.
    """
    valid_values = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]

    stats: dict[str, float] = {}
    n = len(valid_values)

    if n == 0:
        for k in ("mean", "median", "min", "max", "std", "mad", "stderr", "ci95"):
            stats[f"{prefix}_{k}"] = float("nan")
        stats[prefix] = float("nan")
        stats[f"{prefix}_count"] = 0.0
        return stats

    data = np.array(valid_values)
    mean_val = np.mean(data).item()
    median_val = np.median(data).item()
    min_val = np.min(data).item()
    max_val = np.max(data).item()
    std_val = np.std(data, ddof=1).item() if n > 1 else 0.0
    mad_val = np.mean(np.abs(data - median_val)).item()
    stderr_val = std_val / np.sqrt(n) if n > 0 else 0.0

    if 1 < n < 30:
        critical_value = float(t.ppf(0.975, df=n - 1))
    elif n >= 30:
        critical_value = 1.96
    else:
        critical_value = 0.0

    ci95_val = critical_value * stderr_val

    stats[prefix] = mean_val
    stats[f"{prefix}_mean"] = mean_val
    stats[f"{prefix}_median"] = median_val
    stats[f"{prefix}_min"] = min_val
    stats[f"{prefix}_max"] = max_val
    stats[f"{prefix}_std"] = std_val
    stats[f"{prefix}_mad"] = mad_val
    stats[f"{prefix}_stderr"] = stderr_val
    stats[f"{prefix}_ci95"] = ci95_val
    stats[f"{prefix}_count"] = float(n)

    return stats


# Formatting helpers


def _format_num(x: float | str, sig_figs: int = 4) -> str:
    """Format a number with specified significant figures."""
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    if np.isnan(x):
        return "NaN"
    return f"{x:.{sig_figs}g}"


def _format_summary_value(mean: float | str, ci95: float | str, sig_figs: int = 4) -> str:
    """Format a mean ± symmetric CI as a string."""
    return f"{_format_num(mean, sig_figs)} pm {_format_num(ci95, sig_figs)}"


def _format_asymmetric_ci(mean: float | str, lower: float | str, upper: float | str, sig_figs: int = 4) -> str:
    """Format a mean with asymmetric CI bounds."""
    return f"{_format_num(mean, sig_figs)} [{_format_num(lower, sig_figs)}, {_format_num(upper, sig_figs)}]"


# Parallelism / timing helpers


def _resolve_num_workers(num_items: int, num_workers: int) -> int:
    if num_items <= 1 or num_workers <= 1:
        return 1
    return min(num_workers, num_items)


def _map_tasks(fn, tasks: list, num_workers: int):
    """Run *fn* over *tasks*, in parallel when ``num_workers > 1``, else sequentially.

    Yields results in input order (same guarantee as ``executor.map``).
    """
    worker_count = _resolve_num_workers(len(tasks), num_workers)
    if worker_count > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            yield from executor.map(fn, tasks)
    else:
        for task in tasks:
            yield fn(task)


def _time_call(fn, *args, **kwargs):
    """Call *fn* and return ``(result, elapsed_seconds)``."""
    start = perf_counter()
    result = fn(*args, **kwargs)
    return result, perf_counter() - start


def _emit_timing_summary(scope: str, timings: list[tuple[str, float]]) -> None:
    if not timings:
        return
    formatted = " | ".join(f"{name}: {seconds:.3f}s" for name, seconds in timings)
    u_print(f"[timing] {scope} | {formatted}", progress=True)


# String-metric singletons


_BERT_BASE_MODEL_ID = "bert-base-uncased"


def _get_string_metrics_tokenizer() -> PreTrainedTokenizerBase:
    global _STRING_METRICS_TOKENIZER  # noqa: PLW0603
    if _STRING_METRICS_TOKENIZER is None:
        path = get_hpc_hf_model_path(_BERT_BASE_MODEL_ID) if is_hpc_cluster() else _BERT_BASE_MODEL_ID
        args = process_model_args(path)
        _STRING_METRICS_TOKENIZER = AutoTokenizer.from_pretrained(**args)
    return _STRING_METRICS_TOKENIZER  # type: ignore


def _get_sentence_bleu_metric() -> sacrebleu.metrics.BLEU:
    global _STRING_METRICS_SENTENCE_BLEU  # noqa: PLW0603
    if _STRING_METRICS_SENTENCE_BLEU is None:
        _STRING_METRICS_SENTENCE_BLEU = sacrebleu.metrics.BLEU(effective_order=True)
    return _STRING_METRICS_SENTENCE_BLEU


def _get_corpus_bleu_metric() -> sacrebleu.metrics.BLEU:
    global _STRING_METRICS_CORPUS_BLEU  # noqa: PLW0603
    if _STRING_METRICS_CORPUS_BLEU is None:
        _STRING_METRICS_CORPUS_BLEU = sacrebleu.metrics.BLEU()
    return _STRING_METRICS_CORPUS_BLEU


# Cached token helpers


@lru_cache(maxsize=65536)
def _cached_metric_tokenize(text: str) -> tuple[str, ...]:
    return tuple(_get_string_metrics_tokenizer().tokenize(text))


@lru_cache(maxsize=65536)
def _cached_lower_split_tokens(text: str) -> tuple[str, ...]:
    return tuple(text.lower().split())


@lru_cache(maxsize=65536)
def _cached_lower_split_counter(text: str) -> Counter[str]:
    return Counter(_cached_lower_split_tokens(text))


# String metric implementations (stateless, used by StringMetrics & tasks)


def _compute_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = _cached_lower_split_tokens(prediction)
    ground_truth_tokens = _cached_lower_split_tokens(ground_truth)
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    common = _cached_lower_split_counter(prediction) & _cached_lower_split_counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def _vocab_size_from_refs(references_for_vocab: list[str] | None) -> int | None:
    """Return the number of distinct tokens across all reference strings, or ``None``."""
    if references_for_vocab is None:
        return None
    vocab: set[str] = set()
    for sentence in references_for_vocab:
        vocab.update(_cached_metric_tokenize(sentence))
    return len(vocab)


def _empirical_entropy_from_counts(token_counts: Counter[str], total_tokens: int) -> float:
    if total_tokens == 0:
        return 0.0

    return float(-sum((count / total_tokens) * np.log(count / total_tokens) for count in token_counts.values()))


def _compute_sequence_empirical_entropy_impl(texts: list[str]) -> float:
    if not texts:
        return 0.0

    entropies = [
        _empirical_entropy_from_counts(Counter(tokens), len(tokens))
        for tokens in (_cached_metric_tokenize(text) for text in texts)
    ]
    return float(np.mean(entropies)) if entropies else 0.0


def _compute_distinct_metrics_impl(
    texts: list[str],
    vocab_size: int | None = None,
    references_for_vocab: list[str] | None = None,
) -> dict[str, float]:
    if not texts:
        return {}

    if vocab_size is None and references_for_vocab is not None:
        vocab: set[str] = set()
        for sentence in references_for_vocab:
            vocab.update(_cached_metric_tokenize(sentence))
        vocab_size = len(vocab)

    distinct_tokens: set[str] = set()
    distinct_tokens_2grams: set[tuple[str, str]] = set()
    distinct_tokens_3grams: set[tuple[str, str, str]] = set()
    token_counts: Counter[str] = Counter()
    total_tokens = 0

    for prediction in texts:
        tokens = _cached_metric_tokenize(prediction)
        distinct_tokens.update(tokens)
        token_counts.update(tokens)
        total_tokens += len(tokens)

        prev_1 = "<s>"
        prev_2 = "<s>"
        for token in tokens:
            distinct_tokens_2grams.add((prev_1, token))
            distinct_tokens_3grams.add((prev_2, prev_1, token))
            prev_2, prev_1 = prev_1, token

    metrics: dict[str, float] = {}
    metrics["distinct_1"] = len(distinct_tokens) / total_tokens if total_tokens else 0.0
    metrics["distinct_2"] = len(distinct_tokens_2grams) / total_tokens if total_tokens else 0.0
    metrics["distinct_3"] = len(distinct_tokens_3grams) / total_tokens if total_tokens else 0.0
    metrics["empirical_entropy"] = _empirical_entropy_from_counts(token_counts, total_tokens)

    if vocab_size is not None and total_tokens > 0:
        try:
            ead = len(distinct_tokens) / (vocab_size * (1 - ((vocab_size - 1) / vocab_size) ** total_tokens))
            metrics["expectation_adjusted_distinct"] = ead
        except ZeroDivisionError:
            metrics["expectation_adjusted_distinct"] = 0.0

    return metrics


def _compute_self_bleu_impl(texts: list[str]) -> float:
    if len(texts) <= 1:
        return 0.0

    bleu_metric = _get_sentence_bleu_metric()
    bleu_scores = [
        bleu_metric.sentence_score(hyp, [texts[j] for j in range(len(texts)) if j != i]).score
        for i, hyp in enumerate(texts)
    ]
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0


def _select_self_bleu_references(texts: list[str], hypothesis_index: int, max_refs: int) -> list[str]:
    n = len(texts)
    available = n - 1
    if available <= max_refs:
        return [texts[j] for j in range(n) if j != hypothesis_index]

    stride = max(1, available // max_refs)
    refs: list[str] = []
    seen: set[int] = set()
    cursor = (hypothesis_index + 1) % n
    while len(refs) < max_refs:
        if cursor != hypothesis_index and cursor not in seen:
            refs.append(texts[cursor])
            seen.add(cursor)
        cursor = (cursor + stride) % n
    return refs


def _compute_self_bleu_bounded_impl(
    texts: list[str],
    max_refs_per_hypothesis: int = _BATCH_SELF_BLEU_REFERENCE_CAP,
) -> float:
    if len(texts) <= 1:
        return 0.0
    if len(texts) - 1 <= max_refs_per_hypothesis:
        return _compute_self_bleu_impl(texts)

    bleu_metric = _get_sentence_bleu_metric()
    bleu_scores = [
        bleu_metric.sentence_score(hyp, _select_self_bleu_references(texts, i, max_refs_per_hypothesis)).score
        for i, hyp in enumerate(texts)
    ]
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0


# Pickable worker task functions (used with ProcessPoolExecutor)


def _group_diversity_task(args: tuple[list[str], int | None]) -> tuple[dict[str, float], float]:
    group, vocab_size = args
    distinct_metrics = _compute_distinct_metrics_impl(group, vocab_size=vocab_size)
    distinct_metrics["empirical_entropy"] = _compute_sequence_empirical_entropy_impl(group)
    self_bleu = _compute_self_bleu_impl(group)
    return distinct_metrics, self_bleu


def _group_reference_alignment_task(args: tuple[list[str], list[str]]) -> tuple[list[float], float, float]:
    preds, refs = args
    if not refs:
        return [0.0 for _ in preds], 0.0, 0.0

    bleu_metric = _get_sentence_bleu_metric()
    f1_scores = [max((_compute_f1_score(pred, ref) for ref in refs), default=0.0) for pred in preds]
    best_f1 = max(f1_scores) if f1_scores else 0.0
    best_bleu = max(bleu_metric.sentence_score(pred, refs).score for pred in preds)
    return f1_scores, best_f1, best_bleu


def _math_group_task(args: tuple[list[str], str, list[int], bool]) -> tuple[float, dict[int, float]]:
    # Import locally to avoid circular imports when used in worker processes.
    from d5p4.eval_core import MathEvaluator  # noqa: PLC0415

    generations, gold_answer, effective_ks, use_math_parser = args
    evaluator = MathEvaluator(use_math_parser=use_math_parser)
    scores = evaluator.score_group(generations, gold_answer)
    n = len(scores)
    c = sum(scores)
    per_question_acc = c / n
    pass_at_k = {k: MathEvaluator._pass_at_k_estimator(n, c, k) for k in effective_ks}
    return per_question_acc, pass_at_k


# File-type detection


def _is_math_results_file(file_path: str) -> bool:
    """Heuristically detect math-eval result files written by ``llada_math.py``."""
    try:
        with open(file_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    raw = data
    if isinstance(raw, dict):
        raw = raw.get("results", raw)
        if isinstance(raw, dict):
            raw = raw.get("results", [])

    if not isinstance(raw, list) or not raw:
        return False

    first = raw[0]
    if not isinstance(first, dict):
        return False

    return isinstance(first.get("generations"), list) and "gold_answer" in first


# Expose constants so callers can use eval_utils directly without reaching into internals.
__all__ = [
    # stats
    "compute_statistics",
    # formatting
    "_format_num",
    "_format_summary_value",
    "_format_asymmetric_ci",
    # parallel / timing
    "_map_tasks",
    "_time_call",
    "_emit_timing_summary",
    # string-metric building blocks
    "_get_string_metrics_tokenizer",
    "_get_sentence_bleu_metric",
    "_get_corpus_bleu_metric",
    "_cached_metric_tokenize",
    "_cached_lower_split_tokens",
    "_cached_lower_split_counter",
    "_compute_f1_score",
    "_vocab_size_from_refs",
    "_compute_sequence_empirical_entropy_impl",
    "_compute_distinct_metrics_impl",
    "_compute_self_bleu_impl",
    "_compute_self_bleu_bounded_impl",
    "_select_self_bleu_references",
    # worker tasks
    "_group_diversity_task",
    "_group_reference_alignment_task",
    "_math_group_task",
    # file detection
    "_is_math_results_file",
    # constants
    "_BATCH_SELF_BLEU_REFERENCE_CAP",
    "_BATCH_SELF_BLEU_EXACT_THRESHOLD",
]
