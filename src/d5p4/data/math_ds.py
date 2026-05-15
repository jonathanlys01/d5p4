import re

import pandas as pd
from datasets import load_dataset

from d5p4.config import Config


# GSM8K stores the numeric answer after "####" in the answer string
_GSM8K_ANSWER_RE = re.compile(r"####\s*([\-\d,]+)")


def _parse_gsm8k_answer(answer_str: str) -> str:
    """Extract the numeric answer from a GSM8K answer string.

    GSM8K answers end with '#### <number>' (with optional commas).
    Returns the number as a plain string (commas removed), or 'NULL'
    if the sentinel is not found.
    """
    match = _GSM8K_ANSWER_RE.search(answer_str)
    if match:
        return match.group(1).replace(",", "")
    return "NULL"


def _format_gsm8k_few_shot_prefix(examples: list[dict]) -> str:
    """Format a list of GSM8K examples into a few-shot prefix string.

    Expected example format:
    {
        "question": "...",
        "answer": "... #### <number>"
    }
    """
    prefix = ""
    for item in examples:
        q = item["question"]
        # Match benchmark-style GSM8K prompting: few-shot examples include
        # the full worked solution, not only the final numeric answer.
        a = item["answer"].strip()
        prefix += f"Question: {q}\nAnswer:{a}\n\n"
    return prefix


def _format_gsm8k_query(question: str) -> str:
    """Format a GSM8K question using benchmark-style QA scaffolding."""
    return f"Question: {question}\nAnswer:"


def gsm8k(cfg: Config) -> pd.DataFrame:
    """Load the GSM8K test split as a DataFrame.

    Columns
    -------
    question : str
        The question text (possibly prefixed with few-shot examples).
    answer_str : str
        The raw GSM8K answer string (chain-of-thought + '#### <number>').
    answer_number : str
        The numeric answer extracted from ``answer_str`` (commas removed).
    """
    dataset = load_dataset(cfg.gsm8k_path, "main", cache_dir=cfg.cache_dir)["test"]
    dataset = dataset.shuffle(seed=cfg.seed)  # type: ignore

    few_shot_prefix = ""
    if cfg.qa_n_shots > 0:
        train_dataset = load_dataset(cfg.gsm8k_path, "main", cache_dir=cfg.cache_dir)["train"]
        train_dataset = train_dataset.shuffle(seed=cfg.seed)  # type: ignore
        few_shot_examples: list[dict] = [train_dataset[idx] for idx in range(cfg.qa_n_shots)]  # type: ignore
        few_shot_prefix = _format_gsm8k_few_shot_prefix(few_shot_examples)

    questions = []
    answer_strs = []
    answer_numbers = []

    for item in dataset:
        q = item["question"]
        raw_answer = item["answer"]

        if cfg.qa_n_shots > 0:
            questions.append(f"{few_shot_prefix}{_format_gsm8k_query(q)}")
        else:
            questions.append(_format_gsm8k_query(q))

        answer_strs.append(raw_answer)
        answer_numbers.append(_parse_gsm8k_answer(raw_answer))

    df = pd.DataFrame(
        {
            "question": questions,
            "answer_str": answer_strs,
            "answer_number": answer_numbers,
        },
    )

    if cfg.qa_dataset_len > 0:
        df = df.head(cfg.qa_dataset_len)

    return df


if __name__ == "__main__":
    cfg = Config()
    print("Loading GSM8K dataset...")
    df = gsm8k(cfg)
    print(df.head())
    print(f"\nTotal samples: {len(df)}")
    null_count = (df["answer_number"] == "NULL").sum()
    print(f"NULL answers: {null_count}")
    print(f"\nSample question:\n{df['question'].iloc[0]}")
    print(f"\nSample answer string:\n{df['answer_str'].iloc[0]}")
    print(f"\nExtracted answer: {df['answer_number'].iloc[0]}")
