import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import idr_torch
from omegaconf import OmegaConf


if TYPE_CHECKING:
    import torch

AVAIL = ["dpp", "exhaustive", "greedy_map", "greedy_beam", "diverse_beam", "random", "baseline", "_greedy_map"]


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
    disable_sys_args: bool = False

    sequence_length: int = SEQUENCE_LENGTH
    embedding_dim: int = 0  # to be set in __post_init__
    model: str = "mdlm"  # "mdlm", "llada", or "ar"

    seed: int = 0
    n_runs: int = 16
    compile_model: bool = True

    # MDLM
    mdlm_model_path: str = "kuleshov-group/mdlm-owt"
    mdlm_tokenizer: str = "gpt2"

    # LLaDA
    llada_model_path: str = "GSAI-ML/LLaDA-8B-Base"
    llada_tokenizer: str = "GSAI-ML/LLaDA-8B-Base"
    cfg_scale: float = 0.0
    llada_steps: int = 128
    gen_length: int = 128
    block_length: int = 32
    remasking: str = "low_confidence"  # "low_confidence", "selection_temperature", or "random"
    selection_temperature: float = 1.0
    logits_eos_inf: bool = False
    confidence_eos_eot_inf: bool = True
    guidance_start: int = 0  # step at which to start applying CFG (0-indexed)
    guidance_end: int = -1  # step at which to stop applying CFG (-1 means steps)

    # Autoregressive
    ar_model_path: str = "meta-llama/Meta-Llama-3-8B"
    ar_tokenizer: str = "meta-llama/Meta-Llama-3-8B"
    ar_embedding_method: str = "last"  # "last" or "mean" for AR embedding selection

    # sampling
    mdlm_steps: int = SEQUENCE_LENGTH  # number of MDLM sampling steps
    cat_temperature: float = 1.0

    # Source data
    data_path: str = "path_to.bin"
    initial_mask_ratio: float = 1.0  # ratio of tokens to mask at start of sampling (1.0 = all tokens masked)
    single_init: bool = True  # sample a single sequence and repeat it across the batch

    # Subset selection ###################################################################################
    method: str = "random"  # subset selection method
    transversal: bool = True  # use transversal sampling

    group_size: int = 2
    n_groups: int = 2

    # Subsample parameters (specific to each method)

    _kernel_type: str = "rbf"  # type of kernel to use in DPP
    _kernel_method: str = "additive"  # "additive": w*S + diag(q), "multiplicative": diag(q) @ S @ diag(q)
    _kernel_power: int = 1  # power for eigenvalue modulation
    _w_interaction: float = 0.0  # weight for diversity term in DPP, -1 for no quality term
    _w_split: float = 0.0  # weight for split groups in DPP
    _rbf_gamma: float = 1  # RBF kernel gamma parameter (when using RBF kernel)
    _temperature: float = 0.0  # temperature for any sampling
    _diversity_alpha: float = 0.0  # diversity coefficient for diverse beam search
    _score_method: str = "entropy"  # "entropy" or "self-certainty" (CE with uniform distribution)
    ######################################################################################################

    # windowing
    subsample_start: int = -1
    subsample_end: int = 1024
    subsample_k: int = 0  # if > 0 and < batch_size, subsample k best sequences from pool based on perplexity

    # eval
    eval_batch_size: int = 8  # batch size for evaluation (separate from inference batch_size)
    skip_eval: bool = False
    ppl_model_id: str = "gpt2"
    cos_model_id: str = "jinaai/jina-embeddings-v2-base-en"
    eval_selection_metric: str = "ppl"  # "ppl", "f1", or "int" for final sequence selection
    eval_transversal_group_representatives: bool = False  # pick one final representative per transversal group

    qa_dataset: str = "truthful_qa"  # "truthful_qa", "commonsense_qa", "ai2_arc", or "gsm8k"
    qa_dataset_len: int = -1  # number of samples to use from qa_dataset (-1 for all)
    qa_n_shots: int = 0  # number of few-shot examples for QA
    truthful_qa_path: str = "truthfulqa/truthful_qa"
    commonsense_qa_path: str = "tau/commonsense_qa"
    ai2_arc_path: str = "allenai/ai2_arc"
    ai2_arc_subset: str = "ARC-Challenge"
    gsm8k_path: str = "openai/gsm8k"

    # cache
    cache_dir: str = CACHE_DIR
    results_dir: str = RESULTS_DIR

    # optuna
    n_trials: int = 100  # number of Optuna trials for hyperparameter sweeps
    comment: str = ""

    batch_size: int = 0  # to be set in __post_init__
    interactive: bool = True
    minimal_log: bool = False
    quiet: bool = False
    standalone_job: bool = False  # ignore launcher-provided distributed metadata for this process

    def __post_init__(self):
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
        assert self.eval_selection_metric in {"ppl", "f1", "int"}, (
            f"eval_selection_metric must be 'ppl', 'f1', or 'int', got {self.eval_selection_metric!r}"
        )

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
            assert self.remasking in ["low_confidence", "selection_temperature", "random"], (
                f"Remasking method {self.remasking} not recognized."
            )
            if self.eval_transversal_group_representatives:
                assert self.transversal, "eval_transversal_group_representatives requires transversal=True"
                assert self.group_size > 1, "eval_transversal_group_representatives requires group_size > 1"
            assert self.selection_temperature >= 0.0, "selection_temperature must be non-negative"
            assert self.gen_length % self.block_length == 0, "gen_length must be divisible by block_length"
            num_blocks = self.gen_length // self.block_length
            assert self.llada_steps % num_blocks == 0, "llada_steps must be divisible by num_blocks"
            if self.remasking == "selection_temperature" and self.cat_temperature != 0.0 and idr_torch.rank == 0:
                print(
                    "Warning: remasking=selection_temperature with cat_temperature != 0.0. "
                    "This mixes stochastic token sampling with stochastic remasking.",
                )

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
