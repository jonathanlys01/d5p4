## Advanced Examples

### Comparing Methods

Run all methods with the same seed for comparison:

```bash
# Random baseline
python main.py sampling=random sampling.num_sample_batches=32 seed=42

# BoN with N=8
python main.py sampling=bon sampling.num_particles=8 sampling.num_sample_batches=32 seed=42

# SMC with default settings
python main.py sampling=smc smc.num_particles=8 sampling.num_sample_batches=32 seed=42

# Greedy with default settings
python main.py sampling=greedy greedy.num_candidates=8 greedy.beam_size=1 sampling.num_sample_batches=32 seed=42
```

### Using Different Evaluation Models

```bash
# Use GPT-2-large for evaluation
python main.py sampling=bon \
  eval.gen_ppl_eval_model_name_or_path=gpt2-large \
  seed=42

# Use Llama-3-8B for evaluation
python main.py sampling=smc \
  eval.gen_ppl_eval_model_name_or_path=meta-llama/Meta-Llama-3-8B \
  seed=42
```

### Custom Diffusion Steps

```bash
# Fast generation (fewer steps)
python main.py sampling=greedy \
  sampling.steps=64 \
  greedy.num_candidates=8 \
  seed=42

# High-quality generation (more steps)
python main.py sampling=smc \
  sampling.steps=256 \
  smc.num_particles=16 \
  seed=42
```

### Using HuggingFace Models

```bash
# Use pre-trained MDLM from HuggingFace
python main.py sampling=bon \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  backbone=hf_dit \
  model.length=1024 \
  sampling.num_particles=8 \
  seed=42
```

### Using Local Checkpoints

```bash
# Use local checkpoint
python main.py sampling=greedy \
  eval.checkpoint_path=<checkpoint.ckpt> \
  backbone=dit \
  model.length=1024 \
  greedy.num_candidates=8 \
  seed=42
```

## Output and Results

### Evaluation Metrics

All methods report the following metrics:

- **U_denoise**: Generation uncertainty (lower is better)
  - Integral of entropy trajectory over time
  - Measures uncertainty during generation

- **Sentence Entropy**: Token diversity (higher is better)
  - Based on token frequency distribution
  - Measures text diversity

- **Perplexity**: Text quality (lower is better)
  - Evaluated using external LM (GPT-2-large or Llama)
  - Measures fluency and coherence
