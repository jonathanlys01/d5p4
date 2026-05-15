"""
Run a single LLaDA generation under torch.profiler and export a Chrome trace.

Examples:
    python -m d5p4.mods.profile.profile_llada_trace --prompt "What is the capital of France?" method=random model=llada

    torchrun --nproc_per_node=8 -m d5p4.mods.profile.profile_llada_trace \
        --trace-dir results/llada_traces \
        --profile-runs 1 \
        --warmup-runs 1 \
        method=random model=llada
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from d5p4.config import RESULTS_DIR, Config
from d5p4.data import get_qa_dataset
from d5p4.mods.profile.diffusion_llada_profile import CUDA_TIMING_SCOPE_ORDER, LLADAProfilerSampler
from d5p4.utils import compile_model, print, seed_all


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to profile. Defaults to qa_dataset row.")
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=0,
        help="Dataset row to use when --prompt is omitted.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path(RESULTS_DIR) / "llada_traces",
        help="Directory that will contain one subdirectory per profiling run.",
    )
    parser.add_argument(
        "--trace-name",
        type=str,
        default=None,
        help="Optional run name. Shared across ranks when distributed.",
    )
    parser.add_argument(
        "--save-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export the Chrome trace JSON. Disabled by default to avoid large output files.",
    )
    parser.add_argument("--warmup-runs", type=int, default=5, help="Unprofiled runs before tracing.")
    parser.add_argument("--profile-runs", type=int, default=1, help="Profiled runs to include in the trace.")
    parser.add_argument(
        "--steps-per-block",
        type=int,
        default=4,
        help=(
            "Override llada_steps to this many diffusion steps per block for quicker comparison traces. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--profile-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include the first compiled invocation in the trace by skipping warmup.",
    )
    parser.add_argument(
        "--record-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable shape recording in torch.profiler.",
    )
    parser.add_argument(
        "--profile-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable memory profiling in torch.profiler.",
    )
    parser.add_argument(
        "--with-stack",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture Python stacks in the trace.",
    )
    parser.add_argument(
        "--with-flops",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture FLOP estimates when supported by the profiler.",
    )
    return parser.parse_known_args()


def load_config(config_args: list[str]) -> Config:
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], *config_args]
        config = Config(model="llada")
    finally:
        sys.argv = original_argv

    if config.model != "llada":
        raise ValueError(f"profile_llada_trace.py requires model=llada, got {config.model}")

    return config


def reduce_sampling_steps(config: Config, steps_per_block: int) -> Config:
    if steps_per_block <= 0:
        return config

    num_blocks = config.gen_length // config.block_length
    target_steps = min(config.llada_steps, steps_per_block * num_blocks)
    if target_steps == config.llada_steps:
        return config

    config_dict = asdict(config)
    config_dict["disable_sys_args"] = True
    config_dict["llada_steps"] = target_steps
    return Config(**config_dict)


def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_prompt(config: Config, prompt: str | None, prompt_index: int) -> str:
    if prompt is not None:
        return prompt

    dataset = get_qa_dataset(config)
    if not 0 <= prompt_index < len(dataset):
        raise IndexError(f"prompt_index {prompt_index} is out of range for dataset of size {len(dataset)}")

    row = dataset.iloc[prompt_index]
    return str(row["question"])


def build_trace_output_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def save_metadata(output_dir: Path, metadata: dict, rank: int):
    metadata_path = output_dir / f"rank{rank}.meta.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)


def print_wall_clock_summary(summary: list[dict[str, float | int | str]]):
    print("Average isolated GPU time per annotated scope (rank 0):", force=True)
    seen = {str(row["name"]) for row in summary}

    for name in CUDA_TIMING_SCOPE_ORDER:
        if name not in seen:
            continue
        row = next(row for row in summary if row["name"] == name)
        print(
            f"  {row['name']}: {row['avg_ms']:.3f} +- {row['stderr_ms']:.3f} ms over {row['count']} calls",
            force=True,
        )


def main():
    args, config_args = parse_args()
    config = reduce_sampling_steps(load_config(config_args), args.steps_per_block)
    sampler = LLADAProfilerSampler(config)
    offset = sampler.distributed_utils.rank if sampler.distributed_utils else 0

    seed_all(config.seed + offset)

    sampler.set_profiling_scopes(True)
    sampler.set_cuda_timing(torch.cuda.is_available())
    sampler.model = compile_model(sampler.model, config, dynamic=True)

    prompt = resolve_prompt(config, args.prompt, args.prompt_index)
    output_dir = build_trace_output_dir(args.trace_dir)
    trace_stem = args.trace_name or (
        f"llada_{config.method}_steps{config.llada_steps}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    trace_path = output_dir / f"{trace_stem}_rank{offset}.trace.json"

    warmup_runs = 0 if args.profile_compile else args.warmup_runs

    for warmup_idx in range(warmup_runs):
        with record_function(f"llada.warmup_{warmup_idx}"):
            _ = sampler.sample(prompt)
            sync_device()

    sampler.reset_cuda_timing()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        with_flops=args.with_flops,
    ) as prof:
        for profile_idx in range(args.profile_runs):
            with record_function(f"llada.profile_run_{profile_idx}"):
                _ = sampler.sample(prompt)
                sync_device()

    if offset == 0:
        summary = sampler.summarize_cuda_timing()
        print_wall_clock_summary(summary)
        prof.export_chrome_trace(str(trace_path))
        save_metadata(
            output_dir,
            {
                "rank": offset,
                "prompt": prompt,
                "trace_path": str(trace_path) if args.save_trace else None,
                "scope_wall_clock_summary": summary,
                "config": asdict(config),
                "profiler_args": {
                    "save_trace": args.save_trace,
                    "warmup_runs": args.warmup_runs,
                    "profile_runs": args.profile_runs,
                    "steps_per_block": args.steps_per_block,
                    "profile_compile": args.profile_compile,
                    "record_shapes": args.record_shapes,
                    "profile_memory": args.profile_memory,
                    "with_stack": args.with_stack,
                    "with_flops": args.with_flops,
                },
            },
            offset,
        )
        if args.save_trace:
            prof.export_chrome_trace(str(trace_path))
            print(f"Saved trace to {trace_path}")

    if sampler.distributed_utils:
        sampler.distributed_utils.cleanup()


if __name__ == "__main__":
    main()
