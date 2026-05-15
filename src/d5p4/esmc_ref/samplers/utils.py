"""
Utility functions for sampling.
"""

import itertools
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import diffusion


def load_model(config, tokenizer):
    """Load model from checkpoint."""
    if config.backbone in {"hf_dit", "mdlm_ref"}:
        model = diffusion.Diffusion(config, tokenizer=tokenizer).to('cuda')
    else:
        if not os.path.exists(config.eval.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {config.eval.checkpoint_path}. "
                f"If you're trying to use a local pretrained model directory, make sure to set backbone=mdlm_ref"
            )
        model = diffusion.Diffusion.load_from_checkpoint(
            config.eval.checkpoint_path,
            tokenizer=tokenizer,
            config=config
        )
    
    model.gen_ppl_metric.reset()
    
    if config.eval.disable_ema:
        model.ema = None
    
    if model.ema:
        model.ema.store(itertools.chain(
            model.backbone.parameters(),
            model.noise.parameters()))
        model.ema.copy_to(itertools.chain(
            model.backbone.parameters(),
            model.noise.parameters()))
    
    model.backbone.eval()
    model.noise.eval()
    
    return model


def restore_model(model):
    """Restore model to training state."""
    if model.ema:
        model.ema.restore(itertools.chain(
            model.backbone.parameters(),
            model.noise.parameters()))
    model.backbone.train()
    model.noise.train()


def generate_particle_seeds(base_seed: int, num_particles: int, run_id: int = 0):
    """Generate seeds for particles."""
    run_seed = base_seed + run_id * 10000
    np.random.seed(run_seed)
    particle_seeds = [run_seed + j for j in range(num_particles)]
    return particle_seeds


def compute_sentence_entropy(token_ids: torch.Tensor) -> float:
    """Compute sentence entropy based on token frequency distribution."""
    if token_ids.dim() > 1:
        token_ids = token_ids.squeeze()
    
    tokens = token_ids.cpu().numpy()
    unique_tokens, counts = np.unique(tokens, return_counts=True)
    L = len(tokens)
    probabilities = counts / L
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    
    return float(entropy)


def compute_generative_perplexity(
    model, 
    text_sample: str, 
    eval_model_name: str = "gpt2-large"
) -> float:
    """Compute generative perplexity for a single text sample."""
    cache_dir = model.config.data.cache_dir
    model_args = {
        "pretrained_model_name_or_path": eval_model_name,
        "cache_dir": cache_dir,
    }
    tokenizer_args = {
        "pretrained_model_name_or_path": eval_model_name,
        "cache_dir": cache_dir,
    }
    if os.path.isdir(eval_model_name):
        model_args["local_files_only"] = True
        tokenizer_args["local_files_only"] = True

    eval_model = AutoModelForCausalLM.from_pretrained(
        **model_args).eval().to(model.device)
    eval_tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)
    
    if eval_tokenizer.pad_token is None:
        eval_tokenizer.pad_token = eval_tokenizer.eos_token
        eval_tokenizer.pad_token_id = eval_tokenizer.eos_token_id
    
    tokenized = eval_tokenizer(
        text_sample,
        return_tensors='pt',
        return_attention_mask=True,
        truncation=True,
        padding=True,
        max_length=model.config.model.length
    )
    
    input_ids = tokenized['input_ids'].to(model.device)
    attention_mask = tokenized['attention_mask'].to(model.device)
    
    with torch.no_grad():
        logits = eval_model(input_ids, attention_mask=attention_mask)[0]
        logits = logits.transpose(-1, -2)
        
        nlls = F.cross_entropy(
            logits[..., :-1],
            input_ids[..., 1:],
            reduction='none'
        )
        
        first_eos = (input_ids == eval_tokenizer.eos_token_id).cumsum(-1) == 1
        token_mask = (input_ids != eval_tokenizer.eos_token_id)
        valid_mask = first_eos[..., 1:] + token_mask[..., 1:]
        
        total_nll = (nlls * valid_mask).sum()
        total_tokens = valid_mask.sum()
        
        if total_tokens > 0:
            avg_nll = total_nll / total_tokens
            perplexity = torch.exp(avg_nll).item()
        else:
            perplexity = float('inf')
    
    del eval_model
    torch.cuda.empty_cache()
    
    return perplexity
