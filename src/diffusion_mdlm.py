"""
Minimalist diffusion sampler, adapted from MDLM codebase.
"""

from typing import Optional

import torch
from torch import nn

from config import Cache, Config
from mdlm_ref.modeling_mdlm import MDLM, MDLMConfig
from subsample import get_subsample_selector
from utils import get_initial_data, get_tokenizer, process_model_args, sample_categorical, tqdm


NEG_INFINITY = -1_000_000.0
EPS = 1e-5
torch.set_float32_matmul_precision("high")


class MDLMSampler(nn.Module):
    """
    Discrete Diffusion Model sampler for MDLM.
    Implements the reverse diffusion process based on the MDLM codebase.
    """

    def __init__(self, config: Config):
        super().__init__()

        model_args = process_model_args(config.mdlm_model_path, cache_dir=config.cache_dir)
        self.model = MDLM.from_pretrained(**model_args)
        self.selector = get_subsample_selector(config)
        self.config = config
        self.tokenizer = get_tokenizer(config, "mdlm")

        model_config: MDLMConfig = self.model.config
        self.mask_index = model_config.vocab_size - 1
        self.model_length = model_config.model_length

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.distributed_utils = self.selector.distributed_utils if self.selector.distributed_utils else None

    def update_config(self, config: Config):
        """Update model and selector config (for reusing model across sweep trials)."""
        self.config = config
        self.selector.config = config

    def _subs_parameterization(self, logits, xt):
        with torch.no_grad():
            logits[:, :, self.mask_index] = NEG_INFINITY
            logits = logits / self.config.cat_temperature
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            unmasked_indices = xt != self.mask_index
            logits[unmasked_indices] = NEG_INFINITY
            logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _forward_model(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):  # type: ignore
            out = self.model.forward(x, return_dict=True, output_hidden_states=True)
            logits = out.logits
            embeddings = out.hidden_states
        return self._subs_parameterization(logits=logits, xt=x), embeddings

    def _sample_prior(self, *batch_dims) -> torch.Tensor:
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

    def _ddpm_update(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        step: int,
    ) -> torch.Tensor | None:
        if t.ndim > 1:
            t = t.squeeze(-1)

        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]

        assert move_chance_t.ndim == 3, move_chance_t.shape

        log_p_x0, out = self._forward_model(x)
        embeddings = out[-1] if out is not None else None
        cache = Cache(log_p_x0=log_p_x0, embeddings=embeddings, x=x)

        subsample_step = self.config.subsample_start <= step <= self.config.subsample_end
        last_step = step == -1

        slice_idx = (
            self.selector.subsample(cache)
            if subsample_step or last_step
            else torch.arange(x.size(0), device=self.device)
        )

        if slice_idx is None:
            ret = None

        else:
            copy_flag = (x != self.mask_index).to(x.dtype)

            assert cache.log_p_x0 is not None
            p_x0 = cache.log_p_x0.exp()
            p_x0 = p_x0[slice_idx]  # k x L x V

            assert move_chance_t.ndim == p_x0.ndim

            # equiv to move_chance_s * one_hot_mask + (move_chance_t - move_chance_s) * p_x0
            q_xs = p_x0 * (move_chance_t - move_chance_s)[slice_idx]  # k x L x V
            q_xs[:, :, self.mask_index] = move_chance_s[slice_idx, :, 0]

            _x = sample_categorical(q_xs, expand=self.config.group_size if (subsample_step or last_step) else None)

            # Slice and possibly repeat intermediate tensors
            copy_flag = copy_flag[slice_idx]
            original_x = x[slice_idx]

            if (subsample_step or last_step) and self.config.group_size > 1:
                copy_flag = copy_flag.repeat_interleave(self.config.group_size, dim=0)
                original_x = original_x.repeat_interleave(self.config.group_size, dim=0)

            ret = _x * (1 - copy_flag) + original_x * copy_flag

        if self.distributed_utils and (subsample_step or last_step):
            ret = self.distributed_utils.dispatch_sequences(ret, last=last_step)

        return ret

    @torch.no_grad()
    def sample(
        self,
        init_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Samples a sequence from the MDLM model.

        Args:
            init_x: Initial tensor for sampling (if None, initialized with masks).

        Returns:
            A tensor of sampled token IDs.
        """
        num_steps = self.config.mdlm_steps

        if init_x is None:
            if self.config.initial_mask_ratio == 1.0:
                init_x = self._sample_prior(self.config.batch_size, self.model_length)
            else:
                init_x = get_initial_data(self.tokenizer, self.mask_index, self.config)

        x = init_x.to(self.device)

        timesteps = torch.linspace(1, EPS, num_steps + 1, device=self.device)
        dt = (1 - EPS) / num_steps

        disable = False
        if self.distributed_utils:
            disable = self.distributed_utils.rank != 0
        for i in tqdm(range(num_steps), desc="Generating", disable=disable):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            x = self._ddpm_update(x=x, t=t, dt=dt, step=i)

        assert x is not None

        # last step cleanup: sample from p(x0 | xt) to fill remaining masks
        t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
        x = self._ddpm_update(x=x, t=t, dt=timesteps[-1].item(), step=-1)

        return x


if __name__ == "__main__":
    # load and return distribution of first step (all mask)

    config = Config(n_groups=1, group_size=1)  # batch size = 1
    model = MDLMSampler(config)
    model.eval()
    with torch.no_grad():
        init_x = model._sample_prior(config.batch_size, model.model_length).to(model.device)
        t = torch.ones(config.batch_size, device=model.device)
        dt = 1 - 1e-5
        logits, _ = model._forward_model(init_x)
        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        p_x0 = logits.exp()
        q_xs = p_x0 * (move_chance_t - move_chance_s)  # B x L x V
        q_xs[:, :, model.mask_index] = move_chance_s[:, :, 0]
        q_xs /= config.cat_temperature
        print("Logits at first step:", q_xs[0, 0, :])
        probs = torch.softmax(q_xs, dim=-1)
        print("Probs at first step:", probs[0, 0, :])
