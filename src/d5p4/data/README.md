# Data Module

This module handles the loading and processing of datasets used for both reference computations (like MAUVE) and evaluation (Generative QA).

## Reference Data

### FineWeb (FW)
Processes a 10B token subset of the [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset. This is primarily used as the reference distribution for computing MAUVE scores, ensuring that generated text is compared against high-quality web data. In practice, we only use the validation set for evaluation.

### OpenWebText (OWT)
Processes the [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) dataset. Similar to FineWeb, it is used for reference distribution analysis and tokenization benchmarking.

## Generative QA Datasets

These datasets are used to evaluate the model's ability to generate accurate and truthful answers in a zero-shot or few-shot setting. All datasets are used in generative mode (extracting the model's response rather than just multiple-choice scoring).

### TruthfulQA
Processes the [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) dataset.

### CommonSense QA (CSQA)
Processes the [CommonSense QA](https://huggingface.co/datasets/tau/commonsense_qa) dataset.

### AI2 ARC
Processes the [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc) dataset (defaulting to `ARC-Challenge`).

### GSM8K
Processes the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset of grade school math word problems.

## Configuration

Datasets can be selected and configured via `src/config.py`:
- `qa_dataset`: Choose between `"truthful_qa"`, `"commonsense_qa"`, `"ai2_arc"`, or `"gsm8k"`.
- `qa_dataset_len`: Number of samples to use for evaluation (-1 for all).
- `qa_n_shots`: Number of few-shot examples to include in the prompt.
