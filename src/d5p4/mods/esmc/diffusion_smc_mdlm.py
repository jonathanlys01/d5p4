"""
ESMC diffusion sampler (MDLM-based).
"""

from typing import Optional

import torch
from torch import nn

from d5p4.config import Cache, Config
from d5p4.mdlm_ref.modeling_mdlm import MDLM, MDLMConfig
from d5p4.subsample import get_subsample_selector
from d5p4.utils import configure_runtime, get_initial_data, get_tokenizer, process_model_args, sample_categorical, tqdm


NEG_INFINITY = -1_000_000.0
EPS = 1e-5
torch.set_float32_matmul_precision("high")


class SMC_MDLMSampler(nn.Module):
    """Discrete Diffusion Model base class. (MDLM version)"""

    def __init__(self, config: Config):
        super().__init__()
        configure_runtime(config)

        model_args = process_model_args(config.mdlm_model_path, cache_dir=config.cache_dir)
        self.model = MDLM.from_pretrained(**model_args)
        self.selector = get_subsample_selector(config)
        self.config = config
        self.tokenizer = get_tokenizer(config, "mdlm")

        model_config: MDLMConfig = self.model.config
        self.vocab_size = model_config.vocab_size
        self.mask_index = model_config.vocab_size - 1
        self.model_length = model_config.model_length

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.distributed_utils = self.selector.distributed_utils if self.selector.distributed_utils else None

    def update_config(self, config: Config):
        """Update model and selector config (for reusing model across sweep trials)."""
        configure_runtime(config)
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
        use_selector: bool = True,
    ) -> torch.Tensor | None:
        if t.ndim > 1:
            t = t.squeeze(-1)

        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]

        assert move_chance_t.ndim == 3, move_chance_t.shape

        log_p_x0, out = self._forward_model(x)
        embeddings = out[-1] if out is not None else None
        cache = Cache(log_p_x0=log_p_x0, embeddings=embeddings, x=x)

        subsample_step = use_selector and self.config.subsample_start <= step <= self.config.subsample_end
        last_step = use_selector and step == -1

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

        if self.distributed_utils and use_selector and (subsample_step or last_step):
            ret = self.distributed_utils.dispatch_sequences(ret, last=last_step)

        return ret

    def _compute_mask_conditional_entropy(self, x):
        """
        from https://github.com/LINs-lab/DenoisingEntropy/blob/main/diffusion.py
        Compute the Mask-Conditional Entropy H_MC(t) for current state.

        Args:
        x: Current latent sequence with shape (batch_size, seq_len)

        Returns:
        entropy: Mask-conditional entropy for each sample in the batch
        """
        log_p_x0, _ = self._forward_model(x)

        probs = log_p_x0.float().exp()
        masked_positions = x == self.mask_index

        log_probs = torch.log2(probs.clamp_min(1e-10))
        shannon_entropy = -torch.sum(probs * log_probs, dim=-1)

        masked_counts = masked_positions.sum(dim=-1)
        entropy_sums = (shannon_entropy * masked_positions).sum(dim=-1)
        avg_entropy = entropy_sums / masked_counts.clamp_min(1)

        return torch.where(masked_counts > 0, avg_entropy, torch.zeros_like(avg_entropy))

    def compute_entropy_reward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paper-inspired terminal reward: lower mask-conditional entropy means higher reward.
        """
        with torch.no_grad():
            entropies = self._compute_mask_conditional_entropy(x)
            max_entropy = torch.log2(torch.tensor(self.vocab_size, dtype=torch.float32, device=x.device))
            rewards = max_entropy - entropies
            rewards = torch.clamp(rewards / max_entropy, 0.0, 1.0)
        return rewards

    def _select_best_final_particle(self, x: torch.Tensor) -> torch.Tensor:
        """
        Score the current particle pool and keep only the highest-reward sequence.

        The reward is computed before the explicit cleanup step, when masked positions
        still carry entropy information. After cleanup every particle has zero masked
        tokens, so the entropy-based reward would become uninformative.
        """
        if self.distributed_utils:
            x = self.distributed_utils.dispatch_sequences(x, last=True).to(dtype=x.dtype)

        rewards = self.compute_entropy_reward(x)
        best_idx = torch.argmax(rewards)
        return x[best_idx : best_idx + 1]

    def sample(
        self,
        init_x: Optional[torch.Tensor] = None,
        select_best_final: bool = False,
    ):
        with torch.no_grad():
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

            if select_best_final:
                x = self._select_best_final_particle(x)

            # last step cleanup: sample from p(x0 | xt) to fill remaining masks
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            x = self._ddpm_update(
                x=x,
                t=t,
                dt=timesteps[-1].item(),
                step=-1,
                use_selector=not select_best_final,
            )

            return x


if __name__ == "__main__":
    # load and return distribution of first step (all mask)

    config = Config(n_groups=1, group_size=1)  # batch size = 1
    model = SMC_MDLMSampler(config)
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
