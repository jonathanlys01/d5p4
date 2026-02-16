# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os

import numpy as np
import tiktoken
from datasets import load_dataset

from utils import tqdm


if __name__ == "__main__":
    import argparse

    # number of workers in .map() call
    # good number to use is ~order number of cpu cores // 2
    cpu_count = os.cpu_count()
    num_proc = max(15, cpu_count // 2 if cpu_count is not None else 1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_proc",
        type=int,
        default=num_proc,
        help="number of processes to use for loading the dataset",
    )
    parser.add_argument(
        "--path",
        type=str,
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
    )
    parser.add_argument(
        "--encoding",
        default="gpt2",
        type=str,
        help="encoding to use for the dataset",
    )
    args = parser.parse_args()

    enc = tiktoken.get_encoding(args.encoding)

    if os.path.isdir(args.path):
        # if path is a directory, use the files in the directory
        args.path = os.path.join(args.path, "*.parquet")

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": args.path,
        },
        cache_dir=args.cache_dir,
        num_proc=args.num_proc,
    )

    # fw by default only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)  # type: ignore
    split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val
    print(split_dataset)

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #   train: Dataset({
    #      features: ['text', 'id', 'dump', 'url', 'file_path', 'language',
    #       'language_score', 'token_count', 'score', 'int_score']
    #      num_rows: 9_667_264
    #     })
    #   val: Dataset({
    #      features: ['text', 'id', 'dump', 'url', 'file_path', 'language',
    #       'language_score', 'token_count', 'score', 'int_score']
    #      num_rows: 4_837
    #     })
    # })

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example["text"])  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {"ids": ids, "len": len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=args.num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    print(split_dataset.cleanup_cache_files())

    # train.bin is ~19GB, val.bin ~9.4MB
    # train has ~9.8B tokens (9 899 278 336)
    # val has ~4.9M tokens (4 953 088)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
