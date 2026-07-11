# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import os
import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _token_id(tokenizer, token: str):
    qwen_vl_token_ids = {
        "<|vision_start|>": 151652,
        "<|vision_end|>": 151653,
        "<|image_pad|>": 151655,
    }
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return qwen_vl_token_ids.get(token)
    token_id = convert(token)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", object()):
        return qwen_vl_token_ids.get(token)
    return int(token_id)


def _load_image_processor(model_path: str):
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return getattr(processor, "image_processor", None)
    except Exception:
        return None


def _images_from_item(item):
    if isinstance(item, dict):
        images = item.get("image")
        if images is not None:
            if isinstance(images, (list, tuple, np.ndarray)):
                return list(images)
            return [images]
    return []


def _limit_item_images(item, max_images: int):
    if not isinstance(item, dict) or max_images <= 0:
        return item, 0
    images = item.get("image")
    if images is None:
        return item, 0
    image_list = _images_from_item(item)
    if len(image_list) <= max_images:
        return item, 0
    limited = dict(item)
    limited["image"] = image_list[:max_images]
    return limited, len(image_list) - max_images


def _image_token_counts(image_processor, item) -> tuple[List[int], int]:
    images = _images_from_item(item)
    if image_processor is None or not images:
        return [], len(images)
    image_inputs = image_processor(images, return_tensors="pt")
    image_grid_thw = image_inputs.get("image_grid_thw")
    if image_grid_thw is None:
        return [], len(images)
    merge_size = int(getattr(image_processor, "merge_size", 1))
    merge_length = merge_size ** 2
    token_counts = [max(1, int(grid.prod().item() // merge_length)) for grid in image_grid_thw]
    return token_counts, len(images)


def _expected_image_token_count(image_processor, item) -> tuple[int, int]:
    token_counts, image_count = _image_token_counts(image_processor, item)
    return sum(token_counts), image_count


def _collapse_consecutive_token_ids(token_ids: List[int], token_id: int | None) -> List[int]:
    if token_id is None:
        return token_ids
    collapsed = []
    previous_is_token = False
    for current in token_ids:
        if current == token_id:
            if previous_is_token:
                continue
            previous_is_token = True
        else:
            previous_is_token = False
        collapsed.append(current)
    return collapsed


def _collapse_vision_spans_to_single_image_token(
    token_ids: List[int],
    vision_start_id: int | None,
    image_token_id: int | None,
    vision_end_id: int | None,
) -> List[int]:
    if vision_start_id is None or image_token_id is None or vision_end_id is None:
        return _collapse_consecutive_token_ids(token_ids, image_token_id)

    output = []
    cursor = 0
    while cursor < len(token_ids):
        if token_ids[cursor] == image_token_id:
            while cursor < len(token_ids) and token_ids[cursor] == image_token_id:
                cursor += 1
            output.extend([vision_start_id, image_token_id, vision_end_id])
            continue
        if token_ids[cursor] != vision_start_id:
            output.append(token_ids[cursor])
            cursor += 1
            continue
        try:
            end = token_ids.index(vision_end_id, cursor + 1)
        except ValueError:
            output.append(token_ids[cursor])
            cursor += 1
            continue
        if image_token_id in token_ids[cursor:end + 1]:
            output.extend([vision_start_id, image_token_id, vision_end_id])
        else:
            output.extend(token_ids[cursor:end + 1])
        cursor = end + 1
    return output


def _drop_vision_spans(
    token_ids: List[int],
    vision_start_id: int | None,
    image_token_id: int | None,
    vision_end_id: int | None,
) -> List[int]:
    if vision_start_id is None or image_token_id is None or vision_end_id is None:
        return [token_id for token_id in token_ids if token_id != image_token_id]

    output = []
    cursor = 0
    while cursor < len(token_ids):
        if token_ids[cursor] != vision_start_id:
            output.append(token_ids[cursor])
            cursor += 1
            continue
        try:
            end = token_ids.index(vision_end_id, cursor + 1)
        except ValueError:
            output.append(token_ids[cursor])
            cursor += 1
            continue
        if image_token_id in token_ids[cursor:end + 1]:
            cursor = end + 1
        else:
            output.extend(token_ids[cursor:end + 1])
            cursor = end + 1
    return output


def _rewrite_vision_spans_to_token_counts(
    token_ids: List[int],
    token_counts: List[int],
    vision_start_id: int | None,
    image_token_id: int | None,
    vision_end_id: int | None,
) -> tuple[List[int], int, int]:
    if not token_counts or image_token_id is None:
        return token_ids, 0, 0

    output = []
    cursor = 0
    image_index = 0
    spans_seen = 0
    changed = 0

    def append_expected_span(count: int):
        output.extend([vision_start_id] if vision_start_id is not None else [])
        output.extend([image_token_id] * int(count))
        output.extend([vision_end_id] if vision_end_id is not None else [])

    while cursor < len(token_ids):
        current = token_ids[cursor]
        if (
            vision_start_id is not None
            and vision_end_id is not None
            and current == vision_start_id
        ):
            try:
                end = token_ids.index(vision_end_id, cursor + 1)
            except ValueError:
                output.append(current)
                cursor += 1
                continue
            span = token_ids[cursor:end + 1]
            if image_token_id in span:
                spans_seen += 1
                if image_index < len(token_counts):
                    expected = int(token_counts[image_index])
                    append_expected_span(expected)
                    if span.count(image_token_id) != expected:
                        changed += 1
                    image_index += 1
                else:
                    changed += 1
                cursor = end + 1
                continue
            output.extend(span)
            cursor = end + 1
            continue

        if current == image_token_id and image_index < len(token_counts):
            start = cursor
            while cursor < len(token_ids) and token_ids[cursor] == image_token_id:
                cursor += 1
            spans_seen += 1
            expected = int(token_counts[image_index])
            append_expected_span(expected)
            if cursor - start != expected:
                changed += 1
            image_index += 1
            continue

        output.append(current)
        cursor += 1

    while image_index < len(token_counts):
        append_expected_span(int(token_counts[image_index]))
        image_index += 1
        changed += 1

    return output, spans_seen, changed


def _align_multimodal_data(multi_modal_data, batch_size: int):
    multi_modal_data = np.asarray(multi_modal_data, dtype=object)
    if len(multi_modal_data) == batch_size:
        return multi_modal_data
    if len(multi_modal_data) < batch_size and len(multi_modal_data) > 0:
        if batch_size % len(multi_modal_data) == 0:
            return np.repeat(multi_modal_data, batch_size // len(multi_modal_data), axis=0)
        pad_values = np.repeat(multi_modal_data[:1], batch_size - len(multi_modal_data), axis=0)
        return np.concatenate([multi_modal_data, pad_values], axis=0)
    if len(multi_modal_data) > batch_size:
        return multi_modal_data[:batch_size]
    return multi_modal_data


def _sanitize_multimodal_alignment(
    idx_list: List[List[int]],
    multi_modal_data,
    *,
    image_processor,
    image_token_id: int | None,
    vision_start_id: int | None,
    vision_end_id: int | None,
):
    if image_token_id is None:
        return idx_list, multi_modal_data

    sanitized = list(multi_modal_data)
    dropped = 0
    aligned_rows = 0
    capped_images = 0
    max_images = int(os.getenv("VLLM_MAX_IMAGES_PER_PROMPT", "4"))
    for row, item in enumerate(sanitized):
        item, capped = _limit_item_images(item, max_images)
        capped_images += capped
        sanitized[row] = item
        images = _images_from_item(item)
        actual_tokens = idx_list[row].count(image_token_id)
        if actual_tokens <= 0:
            if images:
                sanitized[row] = {}
                aligned_rows += 1
            continue
        if not images:
            if idx_list[row].count(image_token_id):
                idx_list[row] = _drop_vision_spans(idx_list[row], vision_start_id, image_token_id, vision_end_id)
                dropped += 1
            continue
        if len(images) > actual_tokens:
            limited = dict(item)
            limited["image"] = images[:actual_tokens]
            sanitized[row] = limited
            capped_images += len(images) - actual_tokens
            aligned_rows += 1
            continue
        if len(images) == actual_tokens:
            if capped:
                aligned_rows += 1
            continue
        sanitized[row] = {}
        idx_list[row] = _drop_vision_spans(idx_list[row], vision_start_id, image_token_id, vision_end_id)
        dropped += 1

    if aligned_rows or capped_images:
        print(
            f"VLLM_MM_ALIGN_IMAGES: rows={aligned_rows}/{len(sanitized)} "
            f"capped_images={capped_images}",
            flush=True,
        )
    if dropped:
        print(
            f"[warn] dropped mismatched multimodal rows before vLLM generation: {dropped}/{len(sanitized)}",
            flush=True,
        )
    return idx_list, np.array(sanitized, dtype=object)


def _vllm_output_to_tensors(output, response_length: int, pad_token_id: int, device: torch.device):
    if isinstance(output, (tuple, list)) and len(output) == 2 and hasattr(output[0], "to"):
        return output[0].to(device), output[1].to(device)

    responses = []
    for request_output in output:
        completions = getattr(request_output, "outputs", None) or []
        for completion in completions:
            responses.append(list(getattr(completion, "token_ids", []) or []))

    if not responses:
        responses = [[]]

    width = min(response_length, max(1, max(len(tokens) for tokens in responses)))
    response = torch.full((len(responses), width), pad_token_id, dtype=torch.long, device=device)
    for row, tokens in enumerate(responses):
        tokens = tokens[:width]
        if tokens:
            response[row, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
    log_probs = torch.zeros_like(response, dtype=torch.float32, device=device)
    return response, log_probs


def _strip_generated_vision_tokens(
    response: torch.Tensor,
    log_probs: torch.Tensor,
    forbidden_token_ids,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    forbidden = {int(token_id) for token_id in forbidden_token_ids if token_id is not None}
    if not forbidden or response.numel() == 0:
        return response, log_probs, 0

    cleaned_response = torch.full_like(response, pad_token_id)
    cleaned_log_probs = torch.zeros_like(log_probs)
    removed = 0

    for row_idx in range(response.shape[0]):
        row = response[row_idx]
        valid_mask = row != pad_token_id
        if not valid_mask.any():
            continue
        keep_mask = valid_mask.clone()
        for token_id in forbidden:
            keep_mask &= row != token_id
        removed += int(valid_mask.sum().item() - keep_mask.sum().item())
        kept_tokens = row[keep_mask]
        if kept_tokens.numel():
            width = min(kept_tokens.numel(), response.shape[1])
            cleaned_response[row_idx, :width] = kept_tokens[:width]
            if log_probs.shape == response.shape:
                cleaned_log_probs[row_idx, :width] = log_probs[row_idx][keep_mask][:width]

    if removed == 0:
        return response, log_probs, 0
    return cleaned_response, cleaned_log_probs, removed


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        # NOTE: Using max_model_len parameter in LLM() instead of model_hf_config assertion

        llm_kwargs = {}
        if vllm_version not in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
            llm_kwargs['enable_sleep_mode'] = True
        max_images_per_prompt = int(os.getenv("VLLM_MAX_IMAGES_PER_PROMPT", "4"))
        if max_images_per_prompt > 0:
            llm_kwargs["limit_mm_per_prompt"] = {"image": max_images_per_prompt}
        max_num_seqs = int(os.getenv("VLLM_MAX_NUM_SEQS", "1"))
        if max_num_seqs > 0:
            llm_kwargs["max_num_seqs"] = max_num_seqs
        disable_mm_cache = os.getenv("VLLM_DISABLE_MM_PREPROCESSOR_CACHE", "true").strip().lower()
        llm_kwargs["disable_mm_preprocessor_cache"] = disable_mm_cache in {"1", "true", "yes", "on"}
        max_model_len = config.max_model_len if config.max_model_len else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        self.inference_engine = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            **llm_kwargs,
        )

        # NOTE: offload_model_weights() not available in standard vLLM

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = self.inference_engine.get_tokenizer().pad_token_id
        tokenizer = self.inference_engine.get_tokenizer()
        self.tokenizer = tokenizer
        self.image_token_id = _token_id(tokenizer, "<|image_pad|>")
        self.vision_start_id = _token_id(tokenizer, "<|vision_start|>")
        self.vision_end_id = _token_id(tokenizer, "<|vision_end|>")
        self.image_processor = _load_image_processor(model_path)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        raw_prompt_ids = prompts.non_tensor_batch.get('raw_prompt_ids') if prompts.non_tensor_batch else None
        if raw_prompt_ids is not None and len(raw_prompt_ids) == batch_size:
            idx_list = []
            for raw_ids in raw_prompt_ids:
                if hasattr(raw_ids, "tolist"):
                    raw_ids = raw_ids.tolist()
                idx_list.append([int(token_id) for token_id in raw_ids])
        else:
            idx_list = []
            # parse idx from torch.Tensor to List[List[str]]
            for i in range(batch_size):
                idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        multi_modal_data = prompts.non_tensor_batch.get('multi_modal_data') if prompts.non_tensor_batch else None
        if multi_modal_data is not None:
            multi_modal_data = _align_multimodal_data(multi_modal_data, batch_size)
            before_lengths = [len(token_ids) for token_ids in idx_list]
            idx_list = [
                _collapse_vision_spans_to_single_image_token(
                    token_ids,
                    self.vision_start_id,
                    self.image_token_id,
                    self.vision_end_id,
                )
                for token_ids in idx_list
            ]
            collapsed_rows = sum(before != len(after) for before, after in zip(before_lengths, idx_list))
            if collapsed_rows:
                print(
                    f"VLLM_MM_PROMPT_COLLAPSE: rows={collapsed_rows}/{batch_size} "
                    f"tokens={sum(before_lengths)}->{sum(len(token_ids) for token_ids in idx_list)}",
                    flush=True,
                )
            idx_list, multi_modal_data = _sanitize_multimodal_alignment(
                idx_list,
                multi_modal_data,
                image_processor=self.image_processor,
                image_token_id=self.image_token_id,
                vision_start_id=self.vision_start_id,
                vision_end_id=self.vision_end_id,
            )

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            if multi_modal_data is not None:
                vllm_inputs = [
                    {
                        'prompt': self.tokenizer.decode(prompt_token_ids, skip_special_tokens=False),
                        'multi_modal_data': multi_modal_data[i],
                    }
                    for i, prompt_token_ids in enumerate(idx_list)
                ]
                output = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False)
            else:
                output = self.inference_engine.generate(
                    prompts=None,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    prompt_token_ids=idx_list,
                    use_tqdm=False)

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
        response, log_probs = _vllm_output_to_tensors(output, self.config.response_length, self.pad_token_id, idx.device)
        response, log_probs, removed_vision_tokens = _strip_generated_vision_tokens(
            response,
            log_probs,
            (self.vision_start_id, self.image_token_id, self.vision_end_id),
            self.pad_token_id,
        )
        if removed_vision_tokens:
            print(
                f"VLLM_STRIP_GENERATED_VISION_TOKENS: tokens={removed_vision_tokens}",
                flush=True,
            )

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)
            # Ensure tensors stay on correct device after padding
            response = response.to(idx.device)
            log_probs = log_probs.to(idx.device)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            if multi_modal_data is not None:
                repeated = []
                for item in multi_modal_data:
                    for _ in range(self.config.n):
                        repeated.append(item)
                prompts.non_tensor_batch['multi_modal_data'] = np.array(repeated, dtype=object)
            if raw_prompt_ids is not None and len(raw_prompt_ids) == batch_size:
                repeated_raw_prompt_ids = []
                for raw_ids in raw_prompt_ids:
                    for _ in range(self.config.n):
                        repeated_raw_prompt_ids.append(raw_ids)
                prompts.non_tensor_batch['raw_prompt_ids'] = np.array(repeated_raw_prompt_ids, dtype=object)
            batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
