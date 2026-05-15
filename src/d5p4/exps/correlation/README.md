# Correlation and Alignment Experiments

This directory contains scripts to evaluate the alignment between discrete diffusion models (MDLM, LLaDA) and standard autoregressive or semantic models. These experiments focus on two main axes: **Likelihood (Quality)** and **Embeddings (Semantics)**.

## Overview

The core objective is to verify if the internal signals used for sampling and subsampling (like log-likelihood or confidence scores) correlate well with external metrics of text quality and meaning.

### 1. Likelihood and Score Correlation
These experiments check if the internal scores produced by MDLM/LLaDA can serve as proxies for text quality.

- **`likelihood_mdlm.py`**: Correlates MDLM's estimated log-likelihood, entropy, and self-certainty scores against **GPT-2 Perplexity**. It uses Monte Carlo sampling to estimate the log-likelihood of web text sequences.
- **`likelihood_llada.py`**: Correlates LLaDA's scores against the log-likelihood of an **Autoregressive model (e.g., Llama-3)**. This is typically run on Generative QA datasets (TruthfulQA, CSQA) to see if LLaDA "knows" when it is giving a high-quality answer.

### 2. Embedding and Semantic Alignment
These experiments evaluate the "semantic integrity" of the models' hidden representations, especially under different masking conditions.

- **`embeddings_mdlm.py`** & **`embeddings_llada.py`**:
    - Computes **CKA (Centered Kernel Alignment)** between the diffusion model's hidden states and a reference model (e.g., GPT-2 or a dedicated embedding model).
    - Measures **ACS (Average Cosine Similarity)** within batches to track representational collapse.
    - Sweeps over **mask ratios** (0.0 to 1.0) to visualize how semantics degrade during the denoising process.
    - Implements various pooling strategies: `mean`, `pool_non_masked`, `pool_masked`, and `flatten`.

### 3. Shared Utilities (`common.py`)
Common functions used across experiments:
- **CKA**: Centered Kernel Alignment for comparing feature maps across different architectures.
- **Spectral Analysis**: Functions to compute the spectral norm of residual matrices to detect representational "drift".
- **Plotting**: Standardized visualization for correlation sweeps and CKA/ACS curves.

## Running Experiments

Most scripts rely on the global configuration in `src/d5p4/config.py`. You can override parameters via CLI:

```bash
# Example: Run MDLM embedding alignment sweep
python -m d5p4.exps.correlation.embeddings_mdlm mdlm_model_path=<model-id-or-dir>

# Example: Run LLaDA likelihood correlation on TruthfulQA
python -m d5p4.exps.correlation.likelihood_llada qa_dataset=truthful_qa
```

## Metrics Summary

| Metric | Description |
| :--- | :--- |
| **Spearman Rho** | Non-parametric correlation between internal scores and external benchmarks. |
| **CKA** | Measures similarity of representational structures, invariant to linear transformations. |
| **ACS** | Average cosine similarity; indicates how "clustered" or collapsed representations are. |
| **H Spectral Norm** | Used to analyze the residual semantic information in masked tokens. |
