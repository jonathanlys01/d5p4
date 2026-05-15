r"""
Profiler-oriented fork of the LLaDA diffusion sampler.

This keeps the production sampler untouched and exposes profiler scopes around:
- model execution
- CFG/guidance
- subsample selection
- token sampling
- token transfer / state updates
"""

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import record_function

from d5p4.config import Cache, Config
from d5p4.llada_ref.modeling_llada import LLaDAConfig, LLaDAModelLM
from d5p4.subsample import get_subsample_selector
from d5p4.utils import configure_runtime, get_tokenizer, process_model_args, sample_categorical, tqdm


CUDA_TIMING_SCOPE_ORDER = [
    "llada.block_transfer_plan",
    "llada.guidance",
    "llada.forward_pass",
    "llada.log_softmax",
    "llada.selection.slice",
    "llada.selection.expand",
    "llada.sampling",
    "llada.selection.confidence",
    "llada.selection.mask",
    "llada.selection.transfer",
    "llada.state_update",
]


class CUDAScopeTimer:
    def __init__(self):
        self.enabled = torch.cuda.is_available()
        self._events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            name: [] for name in CUDA_TIMING_SCOPE_ORDER
        }

    def set_enabled(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()

    def reset(self):
        self._events = {name: [] for name in CUDA_TIMING_SCOPE_ORDER}

    def scope(self, name: str):
        if not self.enabled or name not in self._events:
            return nullcontext()
        return _CUDAScopeTimerContext(self._events[name])

    def summarize(self) -> list[dict[str, float | int | str]]:
        if not self.enabled:
            return []

        torch.cuda.synchronize()
        summary = []
        for name in CUDA_TIMING_SCOPE_ORDER:
            durations_ms = [start.elapsed_time(end) for start, end in self._events[name]]
            count = len(durations_ms)
            if count == 0:
                continue

            avg_ms = sum(durations_ms) / count
            if count > 1:
                variance = sum((value - avg_ms) ** 2 for value in durations_ms) / (count - 1)
                stderr_ms = math.sqrt(variance) / math.sqrt(count)
            else:
                stderr_ms = 0.0

            summary.append(
                {
                    "name": name,
                    "count": count,
                    "avg_ms": avg_ms,
                    "stderr_ms": stderr_ms,
                    "avg_us": avg_ms * 1000.0,
                    "stderr_us": stderr_ms * 1000.0,
                },
            )

        return summary


class _CUDAScopeTimerContext:
    def __init__(self, sink: list[tuple[torch.cuda.Event, torch.cuda.Event]]):
        self.sink = sink
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.end.record()
        self.sink.append((self.start, self.end))
        return False


class LLADAProfilerSampler(nn.Module):
    """Fork of the LLaDA sampler with explicit torch profiler scopes."""

    def __init__(self, config: Config):
        super().__init__()
        configure_runtime(config)

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
        self.enable_profiling_scopes = False
        self.cuda_timer = CUDAScopeTimer()

    def update_config(self, config: Config):
        configure_runtime(config)
        self.config = config
        self.selector.config = config

    def set_profiling_scopes(self, enabled: bool = True):
        self.enable_profiling_scopes = enabled

    def set_cuda_timing(self, enabled: bool = True):
        self.cuda_timer.set_enabled(enabled)

    def reset_cuda_timing(self):
        self.cuda_timer.reset()

    def summarize_cuda_timing(self) -> list[dict[str, float | int | str]]:
        return self.cuda_timer.summarize()

    def _scope(self, name: str):
        if not self.enable_profiling_scopes:
            return nullcontext()
        return record_function(name)

    def _forward_model(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with (
            self._scope("llada.model_forward"),
            torch.amp.autocast(  # type: ignore
                device_type=self.device,
                dtype=torch.bfloat16,
            ),
        ):
            out = self.model.forward(x, return_dict=True, output_hidden_states=True)
            logits = out.logits
            embeddings = out.hidden_states
        return logits, embeddings

    def _get_block_transfer_tokens(self, mask_index, steps):
        mask_num = mask_index.sum(dim=1, keepdim=True)

        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, : remainder[i]] += 1

        return num_transfer_tokens

    def _preprocess_prompt(self, prompt: str) -> torch.Tensor:
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
            logits[:, :, 126081] = -torch.inf

        if self.config.remasking in {"low_confidence", "selection_temperature"}:
            p = torch.exp(logits) if is_log_probs else F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
        elif self.config.remasking == "random":
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise ValueError(f"Invalid remasking method: {self.config.remasking}")

        x0_p[:, prompt_len + (num_block + 1) * self.config.block_length :] = -torch.inf
        return x0_p

    def sample(self, prompt: str):
        with torch.no_grad():
            num_blocks = self.config.gen_length // self.config.block_length
            steps = self.config.llada_steps // num_blocks
            batch_size = self.config.batch_size
            assert self.config.cfg_scale >= 0, f"cfg_scale must be non-negative, got {self.config.cfg_scale}"

            with self._scope("llada.prompt_preprocess"):
                prompt_tokens = self._preprocess_prompt(prompt)

            prompt_len = prompt_tokens.shape[1]
            prompt_tokens = prompt_tokens.repeat(batch_size, 1)

            with self._scope("llada.init_generation"):
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

            single_block = num_blocks == 1
            block_iter = range(num_blocks) if single_block else tqdm(range(num_blocks), desc="Blocks", disable=disable)

            for num_block in block_iter:
                start = prompt_len + num_block * self.config.block_length
                end = prompt_len + (num_block + 1) * self.config.block_length
                block_mask_index = x[:, start:end] == self.mask_index

                with self._scope("llada.block_transfer_plan"), self.cuda_timer.scope("llada.block_transfer_plan"):
                    num_transfer_tokens = self._get_block_transfer_tokens(block_mask_index, steps)

                step_iter = tqdm(range(steps), desc="Steps", disable=disable) if single_block else range(steps)
                for step in step_iter:
                    with self._scope("llada.diffusion_step"):
                        mask_index = x == self.mask_index

                        apply_cfg = (
                            self.config.cfg_scale != 1.0
                            and self.config.guidance_start <= step < self.config.guidance_end
                        )

                        if apply_cfg:
                            with self._scope("llada.guidance"), self.cuda_timer.scope("llada.guidance"):
                                un_x = x.clone()
                                un_x[prompt_index] = self.mask_index
                                x_ = torch.cat([x, un_x], dim=0)

                                logits_all, out_all = self._forward_model(x_)
                                embeddings_all = out_all[-1]

                                cond_logits, uncond_logits = torch.chunk(logits_all, 2, dim=0)
                                embeddings, _ = torch.chunk(embeddings_all, 2, dim=0)

                                logits = uncond_logits + self.config.cfg_scale * (cond_logits - uncond_logits)
                        else:
                            with self._scope("llada.forward_pass"), self.cuda_timer.scope("llada.forward_pass"):
                                logits, out = self._forward_model(x)
                                embeddings = out[-1]

                        if self.config.logits_eos_inf:
                            logits[:, :, 126081] = -torch.inf

                        with self._scope("llada.log_softmax"), self.cuda_timer.scope("llada.log_softmax"):
                            log_p_x0 = F.log_softmax(logits, dim=-1)

                        cache = Cache(
                            log_p_x0=log_p_x0[:, start:end],
                            embeddings=embeddings[:, start:end],
                            x=x[:, start:end],
                        )
                        with self._scope("llada.selection.slice"), self.cuda_timer.scope("llada.selection.slice"):
                            subsample_step, slice_idx = self._get_slice(step, cache)

                        assert slice_idx is not None

                        with self._scope("llada.selection.index"):
                            logits_to_sample = torch.index_select(log_p_x0, 0, slice_idx)

                        if subsample_step:
                            with self._scope("llada.selection.expand"), self.cuda_timer.scope("llada.selection.expand"):
                                expanded_idx = slice_idx.repeat_interleave(self.config.group_size)
                                x = torch.index_select(x, 0, expanded_idx)
                                log_p_x0 = torch.index_select(log_p_x0, 0, expanded_idx)
                                mask_index = torch.index_select(mask_index, 0, expanded_idx)
                                num_transfer_tokens = torch.index_select(num_transfer_tokens, 0, expanded_idx)
                                prompt_index = torch.index_select(prompt_index, 0, expanded_idx)

                                assert x.size(0) == self.config.batch_size, (
                                    f"Expanded batch size mismatch: {x.size(0)} != {self.config.batch_size}"
                                )

                        with self._scope("llada.sampling"), self.cuda_timer.scope("llada.sampling"):
                            x0 = self._block_sample(logits_to_sample, subsample_step)

                        with (
                            self._scope("llada.selection.confidence"),
                            self.cuda_timer.scope("llada.selection.confidence"),
                        ):
                            x0_p = self._get_confidence(log_p_x0, x0, num_block, prompt_len, is_log_probs=True)

                        with self._scope("llada.selection.mask"), self.cuda_timer.scope("llada.selection.mask"):
                            x0 = torch.where(mask_index, x0, x)
                            confidence = torch.where(mask_index, x0_p, -torch.inf)

                        with self._scope("llada.selection.transfer"), self.cuda_timer.scope("llada.selection.transfer"):
                            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                            for j in range(x.shape[0]):
                                k = int(num_transfer_tokens[j, step].item())
                                if k <= 0:
                                    continue

                                if self.config.remasking == "selection_temperature":
                                    valid_mask = torch.isfinite(confidence[j])
                                    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

                                    if valid_indices.numel() <= k:
                                        select_index = valid_indices
                                    else:
                                        candidate_count = min(2 * k, valid_indices.numel())
                                        top_vals, top_pos = torch.topk(confidence[j], k=candidate_count)

                                        sel_temp = self.config.selection_temperature
                                        if sel_temp <= 0:
                                            select_index = top_pos[:k]
                                        else:
                                            probs = F.softmax(top_vals / sel_temp, dim=-1)
                                            sampled_rel = torch.multinomial(probs, num_samples=k, replacement=False)
                                            select_index = top_pos[sampled_rel]
                                else:
                                    _, select_index = torch.topk(confidence[j], k=k)

                                transfer_index[j, select_index] = True

                        with self._scope("llada.state_update"), self.cuda_timer.scope("llada.state_update"):
                            x[transfer_index] = x0[transfer_index]

            if self.distributed_utils:
                with self._scope("llada.distributed_gather"):
                    x = self.distributed_utils.all_gather_sequences(x)

            return x
