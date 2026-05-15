# D5P4: Partition Determinantal Point Process for Diversity in Parallel Discrete Diffusion Decoding

D5P4 implements partitioned Determinantal Point Process (DPP) subset selection for parallel discrete diffusion decoding. The code focuses on improving the quality-diversity tradeoff for diffusion language models such as [MDLM](https://github.com/kuleshov-group/mdlm) and [LLaDA](https://github.com/ML-GSAI/LLaDA), with evaluation utilities for free-form generation and generative QA.

## What Is Included

- **Diffusion model runners** for MDLM and LLaDA.
- **Subset selection methods** for random selection, baseline pass-through, greedy and diverse beam search, exact DPP sampling, exhaustive search, and fast greedy MAP DPP inference.
- **Partitioned/transversal selection**, where candidates are grouped and the selector chooses one item per group to preserve parallel sample coverage.
- **Evaluation utilities** for perplexity, MAUVE, BLEU, cosine similarity, and generative QA metrics.
- **Experiment scripts** for score/embedding correlation analysis and Optuna sweeps over decoding and diversity parameters.
- **Modified samplers** for SMC-style MDLM runs and LLaDA profiling.

## Repository Layout

- `src/d5p4/config.py`: Central OmegaConf-backed configuration dataclass.
- `src/d5p4/single_run_mdlm.py`: MDLM generation entry point.
- `src/d5p4/llada_math.py`: LLaDA GSM8K math evaluation entry point.
- `src/d5p4/single_run_llada.py`: LLaDA generative QA runner for non-math QA datasets.
- `src/d5p4/diffusion_mdlm.py` and `src/d5p4/diffusion_llada.py`: Model-specific samplers.
- `src/d5p4/subsample/`: Subset selectors and selector benchmarks. See [`src/d5p4/subsample/README.md`](src/d5p4/subsample/README.md).
- `src/d5p4/data/`: FineWeb/OpenWebText reference data and QA dataset loaders. See [`src/d5p4/data/README.md`](src/d5p4/data/README.md).
- `src/d5p4/exps/correlation/`: Likelihood, score, and embedding alignment experiments. See [`src/d5p4/exps/correlation/README.md`](src/d5p4/exps/correlation/README.md).
- `src/d5p4/exps/sweeps/`: Optuna sweeps for DPP kernels, CAT temperature, and diverse beam search. See [`src/d5p4/exps/sweeps/README.md`](src/d5p4/exps/sweeps/README.md).
- `src/d5p4/mods/`: Experimental SMC MDLM and LLaDA profiling variants. See [`src/d5p4/mods/README.md`](src/d5p4/mods/README.md).

## Setup

This project uses Python 3.11+ and is configured for `uv`.

```bash
uv sync
```

For editable installation with another environment manager:

```bash
pip install -e .
```

Some workloads require large model weights or gated Hugging Face models. Configure Hugging Face access before running LLaDA or Llama-based evaluation jobs.

## Running Generation

Run MDLM generation:

```bash
uv run python -m d5p4.single_run_mdlm
```

Run LLaDA GSM8K math evaluation:

```bash
uv run python -m d5p4.llada_math qa_dataset=gsm8k
```

Both entry points read defaults from `Config` in `src/d5p4/config.py` and accept OmegaConf-style command-line overrides:

```bash
uv run python -m d5p4.llada_math qa_dataset=gsm8k method=greedy_map n_groups=4 group_size=2 _w_interaction=10.0
```

Results are written to `results/` by default.

## Subsampling Methods

Set the selector with `method=<name>`. Available methods are:

- `baseline`: pass-through baseline, intended for `group_size=1`.
- `random`: uniform random subset selection.
- `greedy_beam`: top-score beam-style selection.
- `diverse_beam`: Maximal Marginal Relevance-style diverse beam search controlled by `_diversity_alpha`.
- `dpp`: exact k-DPP sampling via `dppy`.
- `greedy_map`: fast Triton-backed greedy MAP DPP inference for CUDA workloads.
- `_greedy_map`: plain PyTorch reference implementation of greedy MAP.
- `exhaustive`: exact exhaustive search for small candidate pools.

Partitioned selection is controlled by `transversal=True`. With transversal selection, the candidate pool is split into `n_groups` groups of `group_size`, and the selector chooses one item from each group.

## Data

Reference distribution utilities cover FineWeb and OpenWebText. QA evaluation supports:

- `truthful_qa`
- `commonsense_qa`
- `ai2_arc`
- `gsm8k`

Useful dataset configuration fields include `qa_dataset`, `qa_dataset_len`, and `qa_n_shots`. FineWeb data can be downloaded with:

```bash
bash src/d5p4/data/download.sh <fineweb-output-dir>
```

## Experiments

Run correlation and alignment experiments as modules:

```bash
uv run python -m d5p4.exps.correlation.embeddings_mdlm mdlm_model_path=<model-id-or-dir>
uv run python -m d5p4.exps.correlation.likelihood_llada qa_dataset=truthful_qa
```

Run Optuna sweeps from `src/d5p4/exps/sweeps/`:

```bash
uv run python -m d5p4.exps.sweeps.map_times n_groups=4 group_size=2
```

The sweep scripts cover:

- `map_plus.py`: additive DPP kernel interaction weight.
- `map_times.py`: multiplicative quality-diversity DPP kernel interaction weight.
- `cat.py`: token sampling temperature.
- `divbs.py`: diverse beam search penalty.

## Benchmarks

Compare subset selectors by log-determinant quality, validity, oracle MAE, and timing:

```bash
uv run python -m d5p4.subsample.selector_benchmark
```

Additional selector benchmarks live in `src/d5p4/subsample/reference_rank_benchmark.py`, `scaling_benchmark.py`, and `kernel_method_benchmark.py`.

## Configuration

Configuration lives in `src/d5p4/config.py`. The most commonly changed fields are:

- `model`: `mdlm`, `llada`, or `ar`.
- `method`: subset selection method.
- `n_groups` and `group_size`: candidate pool shape.
- `transversal`: enable one-selection-per-group constraints.
- `_kernel_type`, `_kernel_method`, `_w_interaction`, `_score_method`: DPP kernel and scoring behavior.
- `llada_steps`, `gen_length`, `block_length`, `remasking`, `cfg_scale`: LLaDA decoding controls.
- `ppl_model_id`, `cos_model_id`, `eval_selection_metric`: evaluation controls.

You can also load a config file with `--config`, `-c`, `config`, or `cfg` and then override individual values on the CLI.

## Distributed Execution

The main samplers use `idr_torch` for multi-GPU execution when launcher-provided distributed metadata is present. Distributed utilities handle gathering and redistributing sequences across ranks, while evaluation and result writing are coordinated on the master rank.

Use `standalone_job=True` when a process should ignore launcher-provided distributed metadata.
