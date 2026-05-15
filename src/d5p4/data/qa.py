import pandas as pd
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

from d5p4.config import Config
from d5p4.data.math_ds import gsm8k


QA_DATASET_SPLITS = {
    "truthful_qa": "validation",
    "commonsense_qa": "validation",
    "ai2_arc": "test",
}

_DATASET_PATH_FIELDS = {
    "truthful_qa": "truthful_qa_path",
    "commonsense_qa": "commonsense_qa_path",
    "ai2_arc": "ai2_arc_path",
}


def _get_choice_lookup(item: dict) -> dict[str, str]:
    choices = item["choices"]["text"]
    labels = item["choices"].get("label")

    if labels is not None:
        return dict(zip(labels, choices, strict=True))

    return {chr(ord("A") + i): choice for i, choice in enumerate(choices)}


def _extract_choice_answer_sets(item: dict) -> tuple[list[str], list[str]]:
    answer_key = item["answerKey"]
    choice_lookup = _get_choice_lookup(item)

    if answer_key not in choice_lookup:
        raise ValueError(f"Answer key {answer_key!r} missing from choices for question {item['question']!r}")

    correct = choice_lookup[answer_key]
    incorrect = [choice for label, choice in choice_lookup.items() if label != answer_key]
    return [correct], incorrect


def _load_dataset_with_default_fallback(
    cfg: Config,
    dataset_name: str,
    dataset_path: str,
    subset: str | None = None,
) -> DatasetDict:
    """Load a QA dataset, falling back from stale absolute paths to default HF IDs."""

    def _load(path: str, cache_dir: str, download_mode: str | None = None) -> DatasetDict:
        kwargs = {"cache_dir": cache_dir}
        if download_mode is not None:
            kwargs["download_mode"] = download_mode
        ret = load_dataset(path, **kwargs) if subset is None else load_dataset(path, subset, **kwargs)
        assert isinstance(ret, DatasetDict)
        return ret

    try:
        return _load(dataset_path, cfg.cache_dir)
    except Exception:
        default_cfg = Config(disable_sys_args=True)
        default_path = str(getattr(default_cfg, _DATASET_PATH_FIELDS[dataset_name]))
        if dataset_path == default_path:
            raise

        print(
            f"Could not load {dataset_name} from {dataset_path!r}; "
            f"retrying default dataset id {default_path!r} with cache_dir={default_cfg.cache_dir!r}.",
        )
        try:
            return _load(default_path, default_cfg.cache_dir, download_mode="reuse_dataset_if_exists")
        except Exception as exc:
            print(
                f"Could not load cached/default {dataset_name} ({exc}); "
                "retrying with download_mode='force_redownload'.",
            )
            return _load(default_path, default_cfg.cache_dir, download_mode="force_redownload")


def _format_few_shot_prefix(examples: list[dict]) -> str:
    """
    Format a list of multiple-choice QA examples into a few-shot prefix string.

    Expected example format:
    {
        "question": "...",
        "answerKey": "A",
        "choices": {
            "text": ["choice1", "choice2", ...],
            "label": ["A", "B", ...],  # optional
        }
    }
    """
    prefix = ""
    for item in examples:
        q = item["question"]
        correct_answers, _ = _extract_choice_answer_sets(item)
        a = correct_answers[0]

        prefix += f"Question: {q}\nAnswer: {a}\n\n"

    return prefix


def truthful_qa(cfg: Config) -> pd.DataFrame:
    assert cfg.qa_n_shots == 0, "TruthfulQA does not support n_shots"
    dataset = _load_dataset_with_default_fallback(cfg, "truthful_qa", cfg.truthful_qa_path, "generation")[
        QA_DATASET_SPLITS["truthful_qa"]
    ]
    dataset = dataset.shuffle(seed=cfg.seed)  # type: ignore
    questions = [item["question"] for item in dataset]  # type: ignore
    good = [item["correct_answers"] for item in dataset]  # type: ignore
    bad = [item["incorrect_answers"] for item in dataset]  # type: ignore

    df = pd.DataFrame({"question": questions, "correct_answers": good, "incorrect_answers": bad})
    return df


def _multiple_choice_qa(cfg: Config, dataset_path: str, dataset_name: str, subset: str | None = None) -> pd.DataFrame:
    dataset_splits = _load_dataset_with_default_fallback(cfg, dataset_name, dataset_path, subset)
    dataset = dataset_splits[QA_DATASET_SPLITS[dataset_name]].shuffle(seed=cfg.seed)  # type: ignore

    questions = []
    good = []
    bad = []

    train_dataset = dataset_splits["train"] if cfg.qa_n_shots > 0 else None

    for i, item in enumerate(dataset):
        assert isinstance(item, dict)
        correct_answers, incorrect_answers = _extract_choice_answer_sets(item)
        good.append(correct_answers)
        bad.append(incorrect_answers)

        q = item["question"]
        if cfg.qa_n_shots > 0:
            assert train_dataset is not None
            # Get n_shots unique examples for this question
            start_idx = i * cfg.qa_n_shots
            end_idx = (i + 1) * cfg.qa_n_shots
            # Wrap around if we exceed train set size (unlikely for reasonable n_shots)
            examples: list[dict] = [train_dataset[idx % len(train_dataset)] for idx in range(start_idx, end_idx)]  # type: ignore
            prefix = _format_few_shot_prefix(examples)
            questions.append(f"{prefix}Question: {q}\nAnswer:")
        else:
            questions.append(q)

    df = pd.DataFrame({"question": questions, "correct_answers": good, "incorrect_answers": bad})

    return df


def commonsense_qa(cfg: Config) -> pd.DataFrame:
    return _multiple_choice_qa(cfg, cfg.commonsense_qa_path, "commonsense_qa")


def ai2_arc(cfg: Config) -> pd.DataFrame:
    return _multiple_choice_qa(cfg, cfg.ai2_arc_path, "ai2_arc", cfg.ai2_arc_subset)


def get_qa_dataset(cfg: Config) -> pd.DataFrame:
    """Get the QA dataset based on the config's qa_dataset field."""
    if cfg.qa_dataset == "truthful_qa":
        return truthful_qa(cfg)
    elif cfg.qa_dataset == "commonsense_qa":
        return commonsense_qa(cfg)
    elif cfg.qa_dataset == "ai2_arc":
        return ai2_arc(cfg)
    elif cfg.qa_dataset == "gsm8k":
        df = gsm8k(cfg)
        # Ensure compatibility with standard QA evaluator by providing correct_answers
        if "correct_answers" not in df.columns and "answer_str" in df.columns:
            df["correct_answers"] = df["answer_str"].apply(lambda x: [x])
        return df
    else:
        raise ValueError(
            f"Unknown qa_dataset: {cfg.qa_dataset}. Available: 'truthful_qa', 'commonsense_qa', 'ai2_arc', 'gsm8k'",
        )


if __name__ == "__main__":
    cfg = Config()
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.expand_frame_repr", False)

    for name, func in [("TruthfulQA", truthful_qa), ("CommonsenseQA", commonsense_qa), ("AI2 ARC", ai2_arc)]:
        print("\n" + "=" * 50)
        print(f" DATASET: {name}")
        print("=" * 50)

        df = func(cfg)
        print("\nSample Data (Top 3):")
        print(df.head(3))

        print("\n--- Statistics ---")
        print(f"{'Total samples:':<35} {len(df)}")

        # Helper to compute stats
        def get_stats(col):
            lengths = df[col].apply(len)
            avg_count = lengths.mean()

            all_answers = [str(ans) for labels in df[col] for ans in labels]
            avg_chars = sum(len(a) for a in all_answers) / len(all_answers) if all_answers else 0
            return avg_count, avg_chars

        avg_correct_n, avg_correct_chars = get_stats("correct_answers")
        avg_incorrect_n, avg_incorrect_chars = get_stats("incorrect_answers")

        print(f"{'Avg # correct answers:':<35} {avg_correct_n:.2f}")
        print(f"{'Avg chars per correct answer:':<35} {avg_correct_chars:.2f}")
        print(f"{'Avg # incorrect answers:':<35} {avg_incorrect_n:.2f}")
        print(f"{'Avg chars per incorrect answer:':<35} {avg_incorrect_chars:.2f}")
        print("=" * 50)
