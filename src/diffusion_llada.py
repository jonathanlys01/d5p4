r"""
Minimalist LLaDA diffusion sampler, adapted from the LLaDA codebase
"""

import torch
import torch.nn.functional as F
from torch import nn

from config import Cache, Config
from data import get_qa_dataset
from llada_ref.modeling_llada import LLaDAConfig, LLaDAModelLM
from subsample import get_subsample_selector
from utils import get_tokenizer, process_model_args, sample_categorical, tqdm


MASK_TOKEN_ID = 126336


class LLADASampler(nn.Module):
    """
    Discrete Diffusion Model sampler for LLaDA.
    Implements the reverse diffusion process with support for Classifier-Free Guidance.
    """

    def __init__(self, config: Config):
        super().__init__()

        model_args = process_model_args(config.llada_model_path, cache_dir=config.cache_dir, dtype="auto")
        self.model = LLaDAModelLM.from_pretrained(**model_args)
        self.selector = get_subsample_selector(config)
        self.config: Config = config
        self.tokenizer = get_tokenizer(config, "llada")

        model_config: LLaDAConfig = self.model.config
        self.mask_index = model_config.mask_token_id
        sequence_length = config.sequence_length
        assert sequence_length <= model_config.max_sequence_length, "Requested sequence length exceeds model's maximum."
        self.sequence_length = sequence_length

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.distributed_utils = self.selector.distributed_utils if self.selector.distributed_utils else None

    def update_config(self, config: Config):
        """Update model and selector config (for reusing model across sweep trials)."""
        self.config = config
        self.selector.config = config

    def _forward_model(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):  # type: ignore
            out = self.model.forward(x, return_dict=True, output_hidden_states=True)
            logits = out.logits
            embeddings = out.hidden_states
        return logits, embeddings

    def _get_block_transfer_tokens(self, mask_index, steps):
        """
        In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
        Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
        the expected number of tokens transitioned at each step should be consistent.

        This function is designed to precompute the number of tokens that need to be transitioned at each step.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)

        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, : remainder[i]] += 1

        return num_transfer_tokens

    def _preprocess_prompt(self, prompt: str) -> torch.Tensor:
        """Apply chat template if needed, and tokenize the prompt."""
        if "instruct" in self.config.llada_model_path.lower():
            message = [{"role": "user", "content": prompt}]
            prompt_str = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        else:
            prompt_str = prompt

        encoded_outputs = self.tokenizer(
            [prompt_str],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        prompt_tokens = encoded_outputs["input_ids"].to(self.device)
        return prompt_tokens

    def _get_slice(self, t: int, cache: Cache) -> tuple[bool, torch.Tensor | None]:
        subsample_step = self.config.subsample_start <= t <= self.config.subsample_end
        last_step = t == -1

        assert cache.x is not None

        slice_idx = (
            self.selector.subsample(cache)
            if subsample_step or last_step
            else torch.arange(cache.x.size(0), device=self.device)
        )

        return subsample_step, slice_idx

    def _block_sample(self, logits: torch.Tensor, subsample_step: bool) -> torch.Tensor:
        temperature = self.config.cat_temperature
        expand = self.config.group_size if subsample_step else 1

        if temperature == 0.0:
            x0_ = torch.argmax(logits, dim=-1)
            x0 = torch.repeat_interleave(x0_, repeats=expand, dim=0)
        else:
            logits = logits.to(torch.float64) / temperature
            probs = F.softmax(logits, dim=-1)
            x0 = sample_categorical(probs, expand=expand)
        return x0

    def _get_confidence(
        self,
        logits: torch.Tensor,
        x0: torch.Tensor,
        num_block: int,
        prompt_len: int,
        is_log_probs: bool = False,
    ) -> torch.Tensor:
        if self.config.confidence_eos_eot_inf:
            logits[:, :, 126348] = -torch.inf

        if self.config.remasking == "low_confidence":
            p = torch.exp(logits) if is_log_probs else F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
        elif self.config.remasking == "random":
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise ValueError(f"Invalid remasking method: {self.config.remasking}")

        x0_p[:, prompt_len + (num_block + 1) * self.config.block_length :] = -torch.inf
        return x0_p

    @torch.no_grad()
    def sample(self, prompt: str) -> torch.Tensor:
        """
        Samples a sequence from the LLaDA model given a prompt.

        Args:
            prompt: The input text to guide generation.

        Returns:
            A tensor of sampled token IDs.
        """
        num_blocks = self.config.gen_length // self.config.block_length
        steps = self.config.llada_steps // num_blocks
        batch_size = self.config.batch_size
        assert self.config.cfg_scale >= 0, f"cfg_scale must be non-negative, got {self.config.cfg_scale}"

        prompt_tokens = self._preprocess_prompt(prompt)
        prompt_len = prompt_tokens.shape[1]
        prompt_tokens = prompt_tokens.repeat(batch_size, 1)

        # Setup generation buffer
        x = torch.full(
            (batch_size, prompt_len + self.config.gen_length),
            self.mask_index,
            dtype=torch.long,
        ).to(self.device)
        x[:, :prompt_len] = prompt_tokens.clone()

        prompt_index = x != self.mask_index

        disable = False
        if self.distributed_utils:
            disable = self.distributed_utils.rank != 0

        # When there's only one block, show progress for steps instead
        single_block = num_blocks == 1
        block_iter = range(num_blocks) if single_block else tqdm(range(num_blocks), desc="Blocks", disable=disable)

        for num_block in block_iter:
            start = prompt_len + num_block * self.config.block_length
            end = prompt_len + (num_block + 1) * self.config.block_length
            block_mask_index = x[:, start:end] == self.mask_index

            num_transfer_tokens = self._get_block_transfer_tokens(block_mask_index, steps)

            step_iter = tqdm(range(steps), desc="Steps", disable=disable) if single_block else range(steps)
            for step in step_iter:
                mask_index = x == self.mask_index

                # Apply CFG only if step is within the guidance range
                apply_cfg = (
                    self.config.cfg_scale != 1.0 and self.config.guidance_start <= step < self.config.guidance_end
                )

                if apply_cfg:
                    un_x = x.clone()
                    un_x[prompt_index] = self.mask_index
                    x_ = torch.cat([x, un_x], dim=0)

                    logits_all, out_all = self._forward_model(x_)
                    embeddings_all = out_all[-1]

                    cond_logits, uncond_logits = torch.chunk(logits_all, 2, dim=0)
                    embeddings, _ = torch.chunk(embeddings_all, 2, dim=0)  # cond logits

                    logits = uncond_logits + self.config.cfg_scale * (cond_logits - uncond_logits)
                else:
                    logits, out = self._forward_model(x)
                    embeddings = out[-1]

                if self.config.logits_eos_inf:
                    logits[:, :, 126081] = -torch.inf

                log_p_x0 = F.log_softmax(logits, dim=-1)

                cache = Cache(
                    log_p_x0=log_p_x0[:, start:end],
                    embeddings=embeddings[:, start:end],
                    x=x[:, start:end],
                )
                subsample_step, slice_idx = self._get_slice(step, cache)

                assert slice_idx is not None

                # Capture logits for sampling BEFORE expansion
                logits_to_sample = torch.index_select(log_p_x0, 0, slice_idx)

                if subsample_step:
                    # Expand indices
                    expanded_idx = slice_idx.repeat_interleave(self.config.group_size)

                    # Expand state
                    x = x[expanded_idx]
                    log_p_x0 = log_p_x0[expanded_idx]
                    mask_index = mask_index[expanded_idx]
                    num_transfer_tokens = num_transfer_tokens[expanded_idx]
                    prompt_index = prompt_index[expanded_idx]

                # Pass log_probs to _block_sample (softmax is invariant to shift, so log_probs work same as logits)
                x0 = self._block_sample(logits_to_sample, subsample_step)

                # Pass log_probs to _get_confidence
                x0_p = self._get_confidence(log_p_x0, x0, num_block, prompt_len, is_log_probs=True)

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -torch.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(x.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, step])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        if self.distributed_utils:
            x = self.distributed_utils.all_gather_sequences(x)

        return x


def main_block():
    cfg = Config(
        disable_sys_args=True,
        qa_dataset_len=50,
    )
    sampler = LLADASampler(cfg)
    dataset = get_qa_dataset(cfg)

    samples = []
    prompts = []

    limit = cfg.qa_dataset_len if cfg.qa_dataset_len > 0 else len(dataset)
    for i, row in enumerate(dataset.itertuples()):
        if i >= limit:
            break

        prompt: str = row.question  # type: ignore

        samples.extend(sampler.sample(prompt=prompt))
        prompts.extend([prompt] * cfg.batch_size)

    if sampler.distributed_utils:
        sampler.distributed_utils.cleanup()

    with open(f"llada_block_{cfg.cfg_scale}.log", "w") as f:
        for i, sample in enumerate(samples):
            decoded_text = sampler.tokenizer.decode(sample.tolist(), skip_special_tokens=False)
            f.write(f"{decoded_text}\n\n")
            f.write("=" * 80 + "\n\n")

    print("Done")


if __name__ == "__main__":
    main_block()
