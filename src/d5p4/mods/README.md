# Modified Samplers

This package contains experimental sampler variants and profiling entry points. These modules are useful for development runs, timing studies, and SMC-style MDLM experiments; the stable release-facing runners remain at the package root.

## Contents

- `esmc/`: MDLM sampler variant with SMC-style expansion and entropy-based final selection.
  - `diffusion_smc_mdlm.py`: `SMC_MDLMSampler`.
  - `single_run_smc_mdlm.py`: single-run entry point using the modified MDLM sampler.
- `profile/`: profiler-oriented LLaDA sampler fork.
  - `diffusion_llada_profile.py`: LLaDA sampler with named profiler/CUDA timing scopes.
  - `profile_llada_trace.py`: trace runner for one prompt or a configured QA dataset row.

## Stable Entry Points

Use the root package entry points for normal experiments:

- `d5p4.single_run_mdlm`: MDLM generation.
- `d5p4.llada_math`: LLaDA GSM8K math evaluation.
- `d5p4.single_run_llada`: non-math LLaDA generative QA.

## Modified Entrypoints

Run modified entry points as modules so imports resolve through the installed package:

```bash
uv run python -m d5p4.mods.esmc.single_run_smc_mdlm method=greedy_map n_groups=4 group_size=2
```

For LLaDA profiling:

```bash
uv run python -m d5p4.mods.profile.profile_llada_trace \
  --prompt "What is the capital of France?" \
  --profile-runs 1 \
  --warmup-runs 1 \
  method=random model=llada
```

Distributed profiling works with `torchrun`:

```bash
torchrun --nproc_per_node=8 -m d5p4.mods.profile.profile_llada_trace \
  --trace-dir results/llada_traces \
  --profile-runs 1 \
  --warmup-runs 1 \
  method=random model=llada
```

Profiler outputs default to `results/llada_traces`. Use `--save-trace` when you want Chrome trace JSONs; otherwise treat the module as a timing/metadata runner.
