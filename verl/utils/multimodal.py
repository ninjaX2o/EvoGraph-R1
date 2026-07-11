"""Small multimodal helpers shared by RL dataset, rollout, and actor code."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from io import BytesIO
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch


GENERATION_NON_TENSOR_KEYS = (
    "raw_prompt_ids",
    "multi_modal_data",
    "image_urls",
    "image_paths",
    "data_source",
    "dataset",
)


def process_image(image: Any, max_pixels: int = 672 * 672 * 2, min_pixels: int = 512 * 512):
    """Normalize a dataset image to an RGB PIL image."""
    from PIL import Image

    if isinstance(image, Mapping) and "bytes" in image:
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, (str, Path)):
        image = resolve_image_path(image)
        image = Image.open(image)

    if not isinstance(image, Image.Image):
        raise TypeError(f"unsupported image value for multimodal RL data: {type(image)!r}")

    if image.width * image.height > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        size = (max(1, int(image.width * resize_factor)), max(1, int(image.height * resize_factor)))
        image = image.resize(size, resample=Image.Resampling.NEAREST)

    if image.width * image.height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        size = (max(1, int(image.width * resize_factor)), max(1, int(image.height * resize_factor)))
        image = image.resize(size, resample=Image.Resampling.NEAREST)

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def resolve_image_path(image: str | Path) -> Path:
    path = Path(image)
    if path.is_file():
        return path

    text = str(image)
    windows_path = PureWindowsPath(text)
    parts = list(windows_path.parts)
    if "datasets_mm" in parts:
        rel_parts = parts[parts.index("datasets_mm") :]
        candidate = Path.cwd().joinpath(*rel_parts)
        if candidate.is_file():
            return candidate

    candidate = Path.cwd() / text
    if candidate.is_file():
        return candidate
    return path


def normalize_images(images: Any) -> list[Any]:
    if images is None:
        return []
    if isinstance(images, np.ndarray):
        images = images.tolist()
    elif not isinstance(images, (list, tuple)):
        images = [images]
    return [process_image(image) for image in images]


def encode_prompt_images(prompt: str, processor: Any, images: list[Any]) -> tuple[str, torch.Tensor | None]:
    """Expand <image> placeholders using the processor's image grid metadata."""
    if processor is None or not images:
        return prompt, None

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return prompt, None

    image_inputs = image_processor(images, return_tensors="pt")
    image_grid_thw = image_inputs.get("image_grid_thw")
    if image_grid_thw is None:
        return prompt, None

    image_token = getattr(processor, "image_token", "<|image_pad|>")
    merge_size = getattr(image_processor, "merge_size", 1)
    merge_length = merge_size**2

    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt

    for grid in image_grid_thw:
        token_count = max(1, int(grid.prod().item() // merge_length))
        replacement = "<|vision_start|>" + (image_token * token_count) + "<|vision_end|>"
        prompt = prompt.replace("<image>", replacement, 1)

    return prompt, image_grid_thw


def _processor_token_id(processor: Any, token: str) -> int | None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return None
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    token_id = convert(token)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", object()):
        return None
    return int(token_id)


def _qwen_vl_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    image_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Compute Qwen-VL mRoPE ids for image+text prompts."""
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)

    position_ids = torch.ones(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    image_index = 0

    for batch_index, row_input_ids in enumerate(input_ids):
        valid_input_ids = row_input_ids[attention_mask[batch_index].to(row_input_ids.device) == 1]
        vision_start_indices = torch.argwhere(valid_input_ids == vision_start_token_id).squeeze(1)
        if vision_start_indices.numel() == 0:
            text_position_ids = torch.arange(valid_input_ids.shape[0], dtype=input_ids.dtype, device=input_ids.device)
            position_ids[:, batch_index, attention_mask[batch_index] == 1] = text_position_ids.expand(3, -1)
            continue

        vision_tokens = valid_input_ids[vision_start_indices + 1]
        image_count = int((vision_tokens == image_token_id).sum().item())
        input_tokens = valid_input_ids.tolist()
        llm_pos_ids_list = []
        start = 0

        for _ in range(image_count):
            end = input_tokens.index(image_token_id, start)
            t, h, w = image_grid_thw[image_index]
            image_index += 1
            llm_grid_t = int(t.item())
            llm_grid_h = int(h.item()) // spatial_merge_size
            llm_grid_w = int(w.item()) // spatial_merge_size
            text_len = end - start

            start_index = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(
                torch.arange(text_len, dtype=input_ids.dtype, device=input_ids.device).view(1, -1).expand(3, -1)
                + start_index
            )

            t_index = torch.arange(llm_grid_t, dtype=input_ids.dtype, device=input_ids.device).view(-1, 1).expand(
                -1, llm_grid_h * llm_grid_w
            ).flatten()
            h_index = torch.arange(llm_grid_h, dtype=input_ids.dtype, device=input_ids.device).view(1, -1, 1).expand(
                llm_grid_t, -1, llm_grid_w
            ).flatten()
            w_index = torch.arange(llm_grid_w, dtype=input_ids.dtype, device=input_ids.device).view(1, 1, -1).expand(
                llm_grid_t, llm_grid_h, -1
            ).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + start_index)
            start = end + llm_grid_t * llm_grid_h * llm_grid_w

        if start < len(input_tokens):
            start_index = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - start
            llm_pos_ids_list.append(
                torch.arange(text_len, dtype=input_ids.dtype, device=input_ids.device).view(1, -1).expand(3, -1)
                + start_index
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[:, batch_index, attention_mask[batch_index] == 1] = llm_positions.to(position_ids.device)

    return position_ids


def compute_position_ids(
    *,
    processor: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
) -> torch.Tensor:
    """Return 1D text position ids or 3D Qwen-VL mRoPE ids when image metadata is available."""
    from verl.utils.model import compute_position_id_with_mask

    if image_grid_thw is None or processor is None:
        return compute_position_id_with_mask(attention_mask)

    image_token_id = _processor_token_id(processor, getattr(processor, "image_token", "<|image_pad|>"))
    vision_start_token_id = _processor_token_id(processor, "<|vision_start|>")
    image_processor = getattr(processor, "image_processor", None)
    spatial_merge_size = int(getattr(image_processor, "merge_size", 1))

    if image_token_id is None or vision_start_token_id is None:
        return compute_position_id_with_mask(attention_mask)

    position_ids = _qwen_vl_position_ids(
        input_ids,
        attention_mask,
        image_grid_thw,
        image_token_id=image_token_id,
        vision_start_token_id=vision_start_token_id,
        spatial_merge_size=spatial_merge_size,
    )
    return position_ids[:, 0, :] if position_ids.shape[1] == 1 else position_ids


def build_multi_modal_inputs(processor: Any, multi_modal_data: Iterable[Any]) -> np.ndarray:
    if processor is None:
        raise RuntimeError("multimodal batch contains images but no processor is available")

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("processor does not expose image_processor")

    processed = []
    for item in multi_modal_data:
        images = item.get("image") if isinstance(item, Mapping) else None
        if images is None:
            processed.append({})
            continue
        image_inputs = image_processor(list(images), return_tensors="pt")
        processed.append({key: value for key, value in image_inputs.items()})
    return np.array(processed, dtype=object)


def collate_multi_modal_inputs(micro_batch: Mapping[str, Any], device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
    if "multi_modal_inputs" not in micro_batch:
        return {}

    inputs = list(micro_batch["multi_modal_inputs"])
    if not inputs:
        return {}

    keys = []
    for item in inputs:
        for key in item.keys():
            if key not in keys:
                keys.append(key)
    if not keys:
        return {}

    collated = {}
    for key in keys:
        values = [item[key] for item in inputs if key in item]
        if not values:
            continue
        if isinstance(values[0], torch.Tensor):
            tensor = torch.cat(values, dim=0)
            collated[key] = tensor.to(device) if device is not None else tensor
    return collated


def generation_non_tensor_keys(data: Any) -> list[str]:
    return [key for key in GENERATION_NON_TENSOR_KEYS if key in getattr(data, "non_tensor_batch", {})]


def select_non_tensors(non_tensor_batch: Mapping[str, np.ndarray], mask: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    return {key: value[mask] for key, value in non_tensor_batch.items()}


def pad_non_tensors(non_tensor_batch: Mapping[str, np.ndarray], padding_size: int) -> dict[str, np.ndarray]:
    if padding_size <= 0:
        return dict(non_tensor_batch)

    padded = {}
    for key, value in non_tensor_batch.items():
        if len(value) == 0:
            padded[key] = value
            continue
        pad_values = np.repeat(value[:1], padding_size, axis=0)
        padded[key] = np.concatenate([value, pad_values], axis=0)
    return padded
