"""
Compute MAUVE score between a corpus of generated text (JSON from MDLM experiments)
and a reference corpus (stored as .bin file with tokenized text).
"""

import argparse
import json

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from config import CACHE_DIR, SEQUENCE_LENGTH
from mauve.compute_mauve import compute_mauve
from utils import process_model_args


def load_reference_texts(
    bin_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_samples: int = 5000,
    seq_len: int = SEQUENCE_LENGTH,
) -> list[str]:
    """
    Load reference texts from a .bin file.
    The .bin is a numpy memmap of uint16 tokens encoded with a transformers tokenizer.

    In the training data of MDLM, every sequence starts with BOS and ends with EOS.
    In GPT-2, BOS == EOS. This function filters out the EOS tokens and chunks
    the remaining tokens into sequences of size (seq_len - 2) to match
    the content length used during training.

    Args:
        bin_path: Path to the .bin file containing tokenized data
        tokenizer: Transformers tokenizer to use for decoding
        max_samples: Maximum number of sequences to load
        seq_len: Total sequence length used in training

    Returns:
        List of decoded text strings
    """
    arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
    eos_token_id = tokenizer.eos_token_id
    content_len = seq_len - 2

    buffer_request = max_samples * seq_len * 2
    raw_tokens = np.array(arr[:buffer_request])
    filtered_tokens = raw_tokens[raw_tokens != eos_token_id]

    num_sequences = min(max_samples, len(filtered_tokens) // content_len)
    if num_sequences == 0:
        print(f"Warning: Not enough tokens in {bin_path} to form even one sequence of length {content_len}")
        return []

    packed_tokens = filtered_tokens[: num_sequences * content_len].reshape(num_sequences, content_len)

    # Decode in batch for performance
    # Convert uint16 to int for tokenizer compatibility
    texts = tokenizer.batch_decode(packed_tokens.astype(int).tolist(), skip_special_tokens=False)

    print(f"Loaded {len(texts)} reference texts from {bin_path} (seq_len={seq_len}, content_len={content_len})")
    return texts


def load_samples_from_json(json_path: str) -> list[str]:
    """
    Load generated samples from a JSON file (MDLM experiment format).
    Expected structure: {"text_samples": [[str, ...], ...], "config": {...}, ...}

    When group_size > 1, the text_samples contain duplicated groups where each group
    of `group_size` samples are variants from the same base sequence. This function
    deduplicates by keeping only the first sample from each group.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    text_samples = data.get("text_samples", [])
    config = data.get("config", {})
    group_size = config.get("group_size", 1)

    # Flatten nested lists - text_samples is typically a list of lists
    flattened = []
    for item in text_samples:
        if isinstance(item, list):
            for text in item:
                if isinstance(text, str) and text.strip():
                    flattened.append(text.strip())
        elif isinstance(item, str) and item.strip():
            flattened.append(item.strip())

    original_count = len(flattened)

    # Deduplicate if group_size > 1 AND model is MDLM
    # In MDLM, samples are organized as [g0_s0, g0_s1, ..., g0_s(gs-1), g1_s0, g1_s1, ...] per batch
    # where each group of group_size samples are variants from the same initial sequence.
    # In LLaDA, samples are generated independently (expansion happens mid-generation, not at output).
    model_type = config.get("model", "mdlm")
    if group_size > 1 and model_type == "mdlm" and flattened:
        deduplicated = []
        for i, text in enumerate(flattened):
            if i % group_size == 0:
                deduplicated.append(text)
        flattened = deduplicated
        print(
            f"Loaded {original_count} samples from {json_path}, deduplicated to {len(flattened)} (group_size={group_size}, model={model_type})"
        )
    else:
        print(f"Loaded {len(flattened)} samples from {json_path}")

    return flattened


def main():
    parser = argparse.ArgumentParser(description="Compute MAUVE score between reference and generated text.")
    parser.add_argument("bin_path", type=str, help="Path to the reference .bin file")
    parser.add_argument("json_path", type=str, help="Path to the JSON file with generated samples")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-large",
        help="Model to use for featurization (default: gpt2-large)",
    )
    parser.add_argument(
        "--reference_tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer used to encode the reference .bin file (default: gpt2)",
    )
    parser.add_argument(
        "--max_ref_samples",
        type=int,
        default=5000,
        help="Maximum number of reference samples to load (default: 5000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for featurization (default: 8)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=SEQUENCE_LENGTH,
        help="Sequence length used in training - reference sequences will be truncated to seq_len-2 (default: 1024)",
    )
    args = parser.parse_args()

    # Initialize tokenizer for decoding reference .bin file
    ref_tokenizer_args = process_model_args(args.reference_tokenizer, cache_dir=CACHE_DIR)
    ref_tokenizer = AutoTokenizer.from_pretrained(**ref_tokenizer_args)

    # Load reference texts from .bin
    ref_texts = load_reference_texts(
        args.bin_path, tokenizer=ref_tokenizer, max_samples=args.max_ref_samples, seq_len=args.seq_len
    )

    # Load generated samples from JSON
    gen_texts = load_samples_from_json(args.json_path)

    if not ref_texts or not gen_texts:
        print("Error: No valid texts loaded from one or both sources.")
        return

    # Initialize model and tokenizer for MAUVE computation (following eval_core.py pattern)
    model_args = process_model_args(args.model, cache_dir=CACHE_DIR)
    model = AutoModel.from_pretrained(**model_args)
    tokenizer = AutoTokenizer.from_pretrained(**model_args)

    # Compute MAUVE
    device_id = 0 if torch.cuda.is_available() else -1
    print(f"Computing MAUVE with {len(ref_texts)} reference and {len(gen_texts)} generated texts...")

    result = compute_mauve(
        p_text=ref_texts,
        q_text=gen_texts,
        models=(model, tokenizer),
        device_id=device_id,
        batch_size=args.batch_size,
        verbose=False,
    )

    print("\n" + "=" * 50)
    print(f"MAUVE Score: {result.mauve:.4f}")
    print(f"MAUVE* Score (smoothed): {result.mauve_star:.4f}")
    print(f"Frontier Integral: {result.frontier_integral:.4f}")
    print(f"Frontier Integral* (smoothed): {result.frontier_integral_star:.4f}")
    print(f"Number of Buckets: {result.num_buckets}")
    print("=" * 50)


if __name__ == "__main__":
    main()
