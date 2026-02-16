# Data Module

This module handles the loading and processing of datasets used for both reference computations (like MAUVE) and evaluation (Generative QA).

## Reference Data

### FineWeb (FW)
Processes a 10B token subset of the [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset. This is primarily used as the reference distribution for computing MAUVE scores, ensuring that generated text is compared against high-quality web data. In practice, we only use the validation set for evaluation.

## Generative QA Datasets

These datasets are used to evaluate the model's ability to generate accurate and truthful answers in a zero-shot or few-shot setting.
We use both datasets in generative mode.

### TruthfulQA
Processes the [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) dataset.

### CommonSense QA (CSQA)
Processes the [CommonSense QA](https://huggingface.co/datasets/tau/commonsense_qa) dataset.

## Configuration

Datasets can be selected and configured via `src/config.py`:
- `qa_dataset`: Choose between `"truthful_qa"` or `"commonsense_qa"`.
- `qa_dataset_len`: Number of samples to use for evaluation.
- `qa_n_shots`: Number of few-shot examples to include in the prompt.
