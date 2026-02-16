# D5P4: Partition Determinantal Point Process for Diversity in Parallel Discrete Diffusion Decoding

D5P4 implements Partitioned Determinantal Point Processes (DPP) to improve the quality and diversity of text generation in diffusion-based language models like MDLM and LLaDA.

## Key Features

- **Diffusion Model Support**: Integration with [MDLM](https://github.com/kuleshov-group/mdlm) and [LLaDA](https://github.com/ML-GSAI/LLaDA).
- **Subsampling Algorithms**:
  - Fast Greedy MAP Inference for DPPs.
  - Determinantal Point Processes (DPP) via `dppy`.
  - Naive and Diverse Beam Search.
  - Random Subsampling.
- **Partitioned Sampling**: Support for transversal/partitioned selection to maintain sample efficiency.
- **Comprehensive Evaluation**: Metrics for Perplexity, MAUVE, BLEU, Cosine Similarity (Jina BERT), and Wasserstein Distance.

## Project Structure

- `src/`: Core implementation.
  - `data/`: Dataset processing for FineWeb, TruthfulQA, and CommonSense QA.
  - `subsample/`: Implementation of all subsampling algorithms.
  - `exps/`: Scripts for running various experiments and sweeps.
  - `config.py`: Centralized configuration system using OmegaConf.
  - `eval_core.py`: Main evaluation module.
  - `diffusion_llada.py` / `diffusion_mdlm.py`: Samplers for LLaDA and MDLM.

## Setup

1. **Install Dependencies**:
   This project uses `uv`. You can install the environment and dependencies with:
   ```bash
   uv sync
   ```
   Alternatively, you can use `pip`:
   ```bash
   pip install -e . 
   ```

2. **Download External Code**:
   - **LLADA**:
     ```bash
     hf download GSAI-ML/LLaDA-8B-Base --local-dir . --exclude *.safetensors
     ```
   - **MAUVE**: The code is expected in `src/mauve/`.

## Usage

### Single Generation
Run a generation sample using LLaDA:
```bash
python src/single_run_llada.py
```
Or for MDLM:
```bash
python src/single_run_mdlm.py
```

### Experiments and Sweeps
The project includes several experiment scripts in `src/exps/`:
- **Classifier-Free Guidance (CFG)**: `python src/exps/cfg_exp.py`
- **Interaction Weight Sweeps**: `python src/exps/interaction_exp.py`
- **Baselines**: `src/exps/baseline_llada.py`, `src/exps/baseline_mdlm.py`

See [src/exps/README.md](src/exps/README.md) for more details.

### Benchmarks
To compare subsampling methods by log-determinant quality and timing:
```bash
python src/subsample/test.py
```

### Configuration
All parameters can be overridden via command line:
```bash
python src/single_run_llada.py method=greedy_map _w_interaction=10.0
```
See [src/config.py](src/config.py) for all available options.

## Distributed Execution

This project uses `idr_torch` for multi-GPU distributed inference. 
- The `DistributedUtils` class in `src/utils.py` manages sequence gathering and redistribution across ranks.
- Evaluation metrics in `src/eval_core.py` are designed to aggregate results correctly in a distributed setting.
- When running in a multi-GPU environment (e.g., via Slurm), the scripts will automatically initialize the distributed backend.
