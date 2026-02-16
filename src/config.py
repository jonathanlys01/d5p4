"""Main configuration file"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from omegaconf import OmegaConf


if TYPE_CHECKING:
    import torch

AVAIL = ["dpp", "exhaustive", "greedy_map", "greedy_beam", "diverse_beam", "random", "baseline"]


SEQUENCE_LENGTH = 1_024
HIDDEN_SIZE_MDLM = 768
HIDDEN_SIZE_LLADA = 4_096
HIDDEN_SIZE_AR = 4_096
RESULTS_DIR = "results"
CACHE_DIR = "./.cache"

CONFIG_FLAGS = ("--config", "-c", "config", "cfg")


def env_path_or(env_name: str, suffix: str, fallback: str) -> str:
    val = os.getenv(env_name)
    return str(Path(val) / suffix) if val else fallback


OmegaConf.register_new_resolver("env_path_or", env_path_or, replace=True)


@dataclass(frozen=True)
class Config:
    """
    Central configuration for the D5P4 project.
    Supports loading from command-line arguments and YAML files via OmegaConf.
    """

    disable_sys_args: bool = False
    """If True, skip loading configuration from system arguments."""

    sequence_length: int = SEQUENCE_LENGTH
    """Maximum sequence length for the model."""
    embedding_dim: int = 0  # to be set in __post_init__
    """Dimension of the model embeddings (auto-set based on model type)."""
    model: str = "mdlm"  # "mdlm", "llada", or "ar"
    """The type of model to use: 'mdlm' (Discrete Diffusion), 'llada' (LLaDA), or 'ar' (Autoregressive)."""

    seed: int = 0
    """Random seed for reproducibility."""
    n_runs: int = 16
    """Number of sequences to generate in a single run."""
    compile_model: bool = True
    """If True, use torch.compile to optimize the model."""

    # MDLM
    mdlm_model_path: str = "kuleshov-group/mdlm-owt"
    """Path or HuggingFace ID for the MDLM model."""
    mdlm_tokenizer: str = "gpt2"
    """Tokenizer to use for the MDLM model."""

    # LLaDA
    llada_model_path: str = "GSAI-ML/LLaDA-8B-Base"
    """Path or HuggingFace ID for the LLaDA model."""
    llada_tokenizer: str = "GSAI-ML/LLaDA-8B-Base"
    """Tokenizer to use for the LLaDA model."""
    cfg_scale: float = 0.0
    """Classifier-Free Guidance scale (0.0 means no guidance)."""
    llada_steps: int = 128
    """Number of sampling steps for LLaDA."""
    gen_length: int = 128
    """Number of tokens to generate."""
    block_length: int = 32
    """Number of tokens processed per block in LLaDA."""
    remasking: str = "low_confidence"  # "low_confidence" or "random"
    """Method used for remasking tokens during LLaDA sampling."""
    logits_eos_inf: bool = False
    """If True, set EOS token logits to -infinity."""
    confidence_eos_eot_inf: bool = True
    """If True, treat EOS/EOT tokens as having -infinity confidence (forcing remasking)."""
    guidance_start: int = 0  # step at which to start applying CFG (0-indexed)
    """Step index at which to start applying Classifier-Free Guidance."""
    guidance_end: int = -1  # step at which to stop applying CFG (-1 means steps)
    """Step index at which to stop applying Classifier-Free Guidance."""

    # Autoregressive
    ar_model_path: str = "meta-llama/Meta-Llama-3-8B"
    """Path or HuggingFace ID for the Autoregressive model."""
    ar_tokenizer: str = "meta-llama/Meta-Llama-3-8B"
    """Tokenizer to use for the Autoregressive model."""
    ar_embedding_method: str = "last"  # "last" or "mean" for AR embedding selection
    """Method to extract embeddings from AR models: 'last' token or 'mean' pooling."""

    # sampling
    mdlm_steps: int = SEQUENCE_LENGTH  # number of MDLM sampling steps
    """Number of sampling steps for MDLM."""
    cat_temperature: float = 1.0
    """Temperature for categorical sampling."""

    # Source data
    data_path: str = "path_to.bin"
    """Path to the input data file."""
    initial_mask_ratio: float = 1.0  # ratio of tokens to mask at start of sampling (1.0 = all tokens masked)
    """Fraction of tokens to mask initially (1.0 is standard diffusion)."""
    single_init: bool = True  # sample a single sequence and repeat it across the batch
    """If True, use the same initial noise for all sequences in the batch."""

    # Subset selection ###################################################################################
    method: str = "random"  # subset selection method
    """Subsampling method to use (e.g., 'dpp', 'greedy_map', 'random', 'baseline')."""
    transversal: bool = True  # use transversal sampling
    """If True, perform transversal (partitioned) selection across groups."""

    group_size: int = 2
    """Number of candidates to select per group."""
    n_groups: int = 2
    """Number of parallel groups to maintain."""

    # Subsample parameters (specific to each method)

    _kernel_type: str = "rbf"  # type of kernel to use in DPP
    """Kernel function for DPP similarity: 'cosine', 'rbf', or 'linear'."""
    _kernel_method: str = "additive"  # "additive": w*S + diag(q), "multiplicative": diag(q) @ S @ diag(q)
    """Method to combine quality and diversity in the DPP kernel."""
    _kernel_power: int = 1  # power for eigenvalue modulation
    """Power factor for eigenvalue modulation in the kernel."""
    _w_interaction: float = 0.0  # weight for diversity term in DPP, -1 for no quality term
    """Weight for the diversity (interaction) term. High values favor diversity."""
    _w_split: float = 0.0  # weight for split groups in DPP
    """Weight for split-group interaction in partitioned DPP."""
    _rbf_gamma: float = 1  # RBF kernel gamma parameter (when using RBF kernel)
    """Gamma parameter for the RBF kernel."""
    _temperature: float = 0.0  # temperature for any sampling
    """Temperature parameter for subsampling selectors."""
    _diversity_alpha: float = 0.0  # diversity coefficient for diverse beam search
    """Diversity factor for Diverse Beam Search."""
    _score_method: str = "entropy"  # "entropy" or "self-certainty" (CE with uniform distribution)
    """Method to compute the quality score: 'entropy' (negative) or 'self-certainty'."""
    ######################################################################################################

    # windowing
    subsample_start: int = -1
    """Step at which to start applying subsampling selectors."""
    subsample_end: int = 1024
    """Step at which to stop applying subsampling selectors."""
    subsample_k: int = 0  # if > 0 and < batch_size, subsample k best sequences from pool based on perplexity
    """If > 0, subsample the top-K sequences based on perplexity (baseline only)."""

    # eval
    eval_batch_size: int = 8  # batch size for evaluation (separate from inference batch_size)
    """Batch size used during evaluation of metrics."""
    ppl_model_id: str = "gpt2"
    """Model ID used for computing perplexity."""
    cos_model_id: str = "jinaai/jina-embeddings-v2-base-en"
    """Model ID used for computing cosine similarity embeddings."""

    qa_dataset: str = "truthful_qa"  # "truthful_qa" or "commonsense_qa"
    """The QA dataset to use for evaluation."""
    qa_dataset_len: int = -1  # number of samples to use from qa_dataset (-1 for all)
    """Number of samples to pull from the QA dataset."""
    qa_n_shots: int = 0  # number of few-shot examples for QA
    """Number of shots for in-context learning evaluation."""
    truthful_qa_path: str = "truthfulqa/truthful_qa"
    """Local path or HF ID for TruthfulQA."""
    commonsense_qa_path: str = "tau/commonsense_qa"
    """Local path or HF ID for CommonSense QA."""

    # cache
    cache_dir: str = CACHE_DIR
    """Directory for caching model weights and processed data."""

    # optuna
    n_trials: int = 100  # number of Optuna trials for hyperparameter sweeps
    """Number of trials for Optuna-based hyperparameter optimization."""

    batch_size: int = 0  # to be set in __post_init__
    """Total batch size (n_groups * group_size)."""
    interactive: bool = True
    """If True, enable interactive logging and progress bars."""

    def __post_init__(self):  # noqa: C901, PLR0912
        # Always set model-specific embedding_dim and batch_size first
        if self.model == "mdlm":
            object.__setattr__(self, "embedding_dim", HIDDEN_SIZE_MDLM)
        elif self.model == "llada":
            object.__setattr__(self, "embedding_dim", HIDDEN_SIZE_LLADA)
        elif self.model == "ar":
            object.__setattr__(self, "embedding_dim", HIDDEN_SIZE_AR)
        else:
            raise ValueError(f"Model {self.model} not recognized. Available models: 'mdlm', 'llada', 'ar'")

        object.__setattr__(self, "batch_size", self.n_groups * self.group_size)

        if self.disable_sys_args:
            return

        self_args = OmegaConf.structured(self)
        sys_args = OmegaConf.from_cli()

        # Priority:
        # 1. Command-line args
        # 2. Command-line provided config file (if any)
        # 3. Default args

        if any(flag in sys_args for flag in CONFIG_FLAGS):
            flag = next(flag for flag in CONFIG_FLAGS if flag in sys_args)
            cfg_file = sys_args.pop(flag)  # remove the flag from sys_args (not in struct)
            cfg_args = OmegaConf.load(cfg_file)
            add_args = OmegaConf.merge(cfg_args, sys_args)
        else:
            add_args = sys_args

        args = OmegaConf.merge(self_args, add_args)
        self.__dict__.update(args)

        assert 0 < self.initial_mask_ratio <= 1.0, "initial_mask_ratio must be in (0, 1]"

        if self.subsample_k > 0:
            assert self.method == "baseline", "subsample_k only makes sense for baseline method"

        # Re-set embedding_dim and batch_size in case model/n_groups/group_size changed via CLI
        if self.model == "mdlm":
            object.__setattr__(self, "embedding_dim", HIDDEN_SIZE_MDLM)
        elif self.model == "llada":
            object.__setattr__(self, "embedding_dim", HIDDEN_SIZE_LLADA)
        elif self.model == "ar":
            object.__setattr__(self, "embedding_dim", HIDDEN_SIZE_AR)
        else:
            raise ValueError(f"Model {self.model} not recognized. Available models: 'mdlm', 'llada', 'ar'")

        object.__setattr__(self, "batch_size", self.n_groups * self.group_size)

        if self.n_runs == 1:
            object.__setattr__(self, "interactive", True)

        assert self.method in AVAIL, f"Method {self.method} not recognized. Available methods: {list(AVAIL)}"

        if self.model == "llada":
            assert self.remasking in ["low_confidence", "random"], f"Remasking method {self.remasking} not recognized."
            assert self.gen_length % self.block_length == 0, "gen_length must be divisible by block_length"
            num_blocks = self.gen_length // self.block_length
            assert self.llada_steps % num_blocks == 0, "llada_steps must be divisible by num_blocks"

            # Set guidance_end to steps if not explicitly set
            if self.guidance_end == -1:
                object.__setattr__(self, "guidance_end", self.llada_steps)

            # Validate guidance range
            assert 0 <= self.guidance_start < self.guidance_end, (
                f"guidance_start ({self.guidance_start}) must be >= 0 and < guidance_end ({self.guidance_end})"
            )
            assert self.guidance_end <= self.llada_steps, (
                f"guidance_end ({self.guidance_end}) must be <= llada_steps ({self.llada_steps})"
            )

    def __str__(self) -> str:
        return OmegaConf.to_yaml(OmegaConf.structured(self))


@dataclass
class Cache:
    x: Optional["torch.Tensor"] = None
    log_p_x0: Optional["torch.Tensor"] = None
    embeddings: Optional["torch.Tensor"] = None


# Utils
def _expand_path(value: str) -> str:
    """Expand environment variables and user home in a path string."""
    return os.path.expandvars(os.path.expanduser(value))


def _is_likely_path(value: str) -> bool:
    """Check if a string is likely to be a path using multiple heuristics."""
    if not isinstance(value, str) or not value:
        return False

    # Check if it's an existing directory (definitive proof it's a path)
    if os.path.isdir(_expand_path(value)):
        return True

    if "/" in value or "\\" in value:
        return True
    path_extensions = (".bin", ".pt", ".pth", ".yaml", ".yml", ".json", ".txt", ".csv", ".log")
    if any(value.endswith(ext) for ext in path_extensions):
        return True
    path_keywords = ("path", "dir", "file", "cache", "results")
    return any(kw in value.lower() for kw in path_keywords)


if __name__ == "__main__":
    config = Config()
    config_dict = OmegaConf.to_container(OmegaConf.structured(config))
    statuses = ["\033[91m(NOK)\033[0m", "\033[92m(OK)\033[0m"]

    print("Verifying paths...")
    print("=" * 50)

    assert hasattr(config_dict, "items")
    for key, value in config_dict.items():
        if isinstance(value, str) and _is_likely_path(value):
            exists = os.path.exists(_expand_path(value))
            print(f"{key}: {value} {statuses[exists]}")
        else:
            print(f"{key}: {value}")

    print("\n" + "=" * 50)
