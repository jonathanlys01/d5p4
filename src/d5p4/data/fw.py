import os

import numpy as np
import tiktoken
from datasets import load_dataset

from d5p4.utils import tqdm


if __name__ == "__main__":
    import argparse

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
        args.path = os.path.join(args.path, "*.parquet")

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": args.path,
        },
        cache_dir=args.cache_dir,
        num_proc=args.num_proc,
    )

    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)  # type: ignore
    split_dataset["val"] = split_dataset.pop("test")
    print(split_dataset)

    def process(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        out = {"ids": ids, "len": len(ids)}
        return out

    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=args.num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))  # type: ignore
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    print(split_dataset.cleanup_cache_files())
