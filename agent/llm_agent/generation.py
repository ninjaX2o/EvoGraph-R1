"""
Tool generation manager for LLM agents
"""

import torch
import re
import json
import os
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from copy import deepcopy

import random

from .tensor_helper import TensorHelper, TensorConfig
from agent.tool.tool_env import ToolEnv, step, step_batch
from verl import DataProto
from verl.utils.tracking import Tracking
from verl.utils.multimodal import pad_non_tensors, select_non_tensors
from verl.utils.multimodal import process_image

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

@dataclass
class ToolGenerationConfig:
    """Configuration for tool-based generation"""
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_tool_response_length: int  # Renamed from max_obs_length
    num_gpus: int
    # use_parallel_tool_calls: bool = False
    use_batch_tool_calls: bool = False  # New option for batch execution
    tool_call_start: str = "<tool_call>"
    tool_call_end: str = "</tool_call>"
    tool_response_start: str = "<knowledge>"
    tool_response_end: str = "</knowledge>"
    # Knowledge pipeline tools use different tags to avoid confusion
    pipeline_response_start: str = "<pipeline>"
    pipeline_response_end: str = "</pipeline>"
    tool_custom_response_template: str = ""
    force_first_image_search: bool = True


@dataclass
class TrajectoryState:
    """Per-trajectory source of truth for dynamic multimodal rollout state."""

    prompt_token_ids: List[int]
    multi_modal_data: Dict[str, Any] | None = None
    tool_return_images: List[Any] = field(default_factory=list)
    
class ToolGenerationManager:
    """Manager for handling LLM tool-based generation and interaction"""
    
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: ToolGenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_tool_response_length=config.max_tool_response_length,  # Renamed
            max_start_length=config.max_start_length,
        ))
        self.image_processor = self._load_image_processor()

    def _load_image_processor(self):
        model_path = getattr(self.tokenizer, "name_or_path", None)
        if not model_path:
            return None
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            return getattr(processor, "image_processor", None)
        except Exception as exc:
            print(f"[warn] failed to load image processor for final mm alignment: {exc}", flush=True)
            return None

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _process_tool_call(self, responses_str) -> Tuple[List[str], List[bool]]:
        """
        Process a list of response strings to extract the first tool call
        while preserving the rest of the string content.
        
        Args:
            responses_str (List[str]): List of response strings potentially containing tool calls
            
        Returns:
            List[str]: Processed responses with only first tool call preserved
        """
        def process_single_response(resp):
            tool_pattern = r'<tool_call>(.*?)</tool_call>'
            match = re.search(tool_pattern, resp, re.DOTALL)

            if not match:
                # A dangling tool-call start often makes HF rollout spend a full
                # max_new_tokens pass on every retry. End the trajectory instead
                # of letting a malformed call dominate validation/training time.
                return resp + self.tokenizer.eos_token, False

            resp = resp.split(self.config.tool_call_end)[0] + self.config.tool_call_end

            return resp + self.tokenizer.eos_token, True
        
        # Process each response string (single pass)
        processed = [process_single_response(resp) for resp in responses_str]
        return [p[0] for p in processed], [p[1] for p in processed]

    def _forced_image_search_response(self) -> str:
        return (
            '<think>Identify the image entity before doing text lookup.</think>\n'
            '<tool_call>{"tool":"kb_search","args":{"query":"<img>"}}</tool_call>'
            + self.tokenizer.eos_token
        )

    def _build_forced_image_search_batch(self, active_mask: torch.Tensor) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        responses_str = [self._forced_image_search_response() for _ in range(int(active_mask.sum().item()))]
        responses_ids = self._batch_tokenize(responses_str)
        return responses_ids, responses_str, torch.ones(len(responses_str), dtype=torch.bool)

    def _should_force_first_image_search(self, step: int, rollings: DataProto) -> bool:
        return (
            self.config.force_first_image_search
            and step == 0
            and self._has_multimodal_data(rollings.non_tensor_batch)
        )

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to extract tool calls."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # Extract the first tool call from each response
        responses_str, active_masks = self._process_tool_call(responses_str)
        
        # Tokenize processed responses
        cleaned_token_ids = self._batch_tokenize(responses_str)
        
        return cleaned_token_ids, responses_str, torch.tensor(active_masks, dtype=torch.bool)
    
    def _process_tool_responses(self, tool_responses: List[str]) -> torch.Tensor:
        """Process tool responses to token ids"""
        
        tool_responses_ids = self.tokenizer(
            tool_responses, 
            padding='longest',
            return_tensors='pt'
        )['input_ids']
        
        if tool_responses_ids.shape[1] > self.config.max_tool_response_length:
            print("[WARNING] TOOL RESPONSE TOO LONG, CONSIDER CHANGING YOUR CONFIG")
            tool_responses_ids = tool_responses_ids[:, :self.config.max_tool_response_length]
            
        return tool_responses_ids

    def _to_token_list(self, token_ids: Any) -> List[int]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        elif hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return [int(token_id) for token_id in token_ids]

    def _row_valid_token_list(self, row: torch.Tensor) -> List[int]:
        valid = row[row != self.tokenizer.pad_token_id]
        if valid.numel() == 0:
            return []
        return self._to_token_list(valid)

    def _images_from_item(self, item: Any) -> List[Any]:
        if isinstance(item, dict):
            images = item.get("image")
            if images is not None:
                if isinstance(images, (list, tuple, np.ndarray)):
                    return list(images)
                return [images]
        return []

    def _init_trajectory_states(self, gen_batch: DataProto) -> List[TrajectoryState]:
        batch_size = gen_batch.batch["input_ids"].shape[0]
        non_tensors = gen_batch.non_tensor_batch or {}
        raw_prompt_ids = non_tensors.get("raw_prompt_ids")
        multi_modal_data = non_tensors.get("multi_modal_data")

        states = []
        for row_idx in range(batch_size):
            if raw_prompt_ids is not None and row_idx < len(raw_prompt_ids):
                prompt_token_ids = self._to_token_list(raw_prompt_ids[row_idx])
            else:
                prompt_token_ids = self._row_valid_token_list(gen_batch.batch["input_ids"][row_idx])

            item = None
            if multi_modal_data is not None and row_idx < len(multi_modal_data):
                item = multi_modal_data[row_idx]
            item = deepcopy(item) if isinstance(item, dict) else None
            states.append(TrajectoryState(prompt_token_ids=prompt_token_ids, multi_modal_data=item))
        return states

    def _states_to_non_tensors(self, states: List[TrajectoryState]) -> Dict[str, Any]:
        non_tensors = {
            "raw_prompt_ids": np.array(
                [list(state.prompt_token_ids) for state in states],
                dtype=object,
            )
        }
        if any(state.multi_modal_data is not None for state in states):
            non_tensors["multi_modal_data"] = np.array(
                [
                    deepcopy(state.multi_modal_data) if state.multi_modal_data is not None else {}
                    for state in states
                ],
                dtype=object,
            )
        return non_tensors

    def _sync_rollings_from_states(self, rollings: DataProto, states: List[TrajectoryState]) -> DataProto:
        non_tensors = dict(rollings.non_tensor_batch or {})
        non_tensors.update(self._states_to_non_tensors(states))
        return DataProto.from_dict(
            {key: value for key, value in rollings.batch.items()},
            non_tensors=non_tensors,
        )

    def _prepare_tool_responses_multimodal(
        self,
        tool_responses: List[str],
    ) -> Tuple[List[str], List[List[Any]]]:
        prepared_responses = []
        response_images = []
        for tool_response in tool_responses:
            image_paths = self._extract_tool_response_image_paths(tool_response)
            images = self._load_tool_response_images(image_paths)
            tool_response = self._strip_tool_response_image_fields(tool_response)
            if images:
                tool_response = self._append_image_placeholders(tool_response, len(images))
            prepared_responses.append(tool_response)
            response_images.append(images)
        return prepared_responses, response_images

    def _strip_tool_response_image_fields(self, tool_response: str) -> str:
        """Hide local image locators from the model-visible tool response."""
        start_tag = self.config.tool_response_start
        end_tag = self.config.tool_response_end

        def sanitize_payload(payload: str) -> str:
            try:
                parsed = json.loads(payload)
            except Exception:
                return re.sub(
                    r',?\s*"(?:image_path|image_url)"\s*:\s*"[^"]*"',
                    "",
                    payload,
                )
            return json.dumps(self._remove_image_locator_fields(parsed), ensure_ascii=False)

        pattern = re.escape(start_tag) + r"\s*(.*?)\s*" + re.escape(end_tag)

        def replace_match(match: re.Match) -> str:
            return f"{start_tag}{sanitize_payload(match.group(1))}{end_tag}"

        updated, count = re.subn(pattern, replace_match, tool_response, flags=re.DOTALL)
        if count:
            return updated
        return sanitize_payload(tool_response)

    def _remove_image_locator_fields(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self._remove_image_locator_fields(item)
                for key, item in value.items()
                if key not in {"image_path", "image_url"}
            }
        if isinstance(value, list):
            return [self._remove_image_locator_fields(item) for item in value]
        return value

    def _extract_tool_response_image_paths(self, tool_response: str) -> List[str]:
        limit = int(os.getenv("TOOL_RESPONSE_IMAGE_LIMIT", "3"))
        if limit <= 0:
            return []

        payloads = re.findall(
            re.escape(self.config.tool_response_start) + r"\s*(.*?)\s*" + re.escape(self.config.tool_response_end),
            tool_response,
            flags=re.DOTALL,
        ) or [tool_response]

        paths = []
        seen = set()
        for payload in payloads:
            parsed_items = []
            try:
                parsed_items.append(json.loads(payload))
            except Exception:
                parsed_items = []

            for parsed in parsed_items:
                for result in self._iter_json_results(parsed):
                    if not isinstance(result, dict):
                        continue
                    image_path = str(result.get("image_path") or "").strip()
                    if image_path and image_path not in seen:
                        seen.add(image_path)
                        paths.append(image_path)
                        if len(paths) >= limit:
                            return paths

            if not parsed_items:
                for match in re.finditer(r'"image_path"\s*:\s*"([^"]+)"', payload):
                    image_path = match.group(1).strip()
                    if image_path and image_path not in seen:
                        seen.add(image_path)
                        paths.append(image_path)
                        if len(paths) >= limit:
                            return paths
        return paths

    def _iter_json_results(self, parsed: Any):
        if isinstance(parsed, dict):
            results = parsed.get("results")
            if isinstance(results, list):
                yield from results
            else:
                yield parsed
        elif isinstance(parsed, list):
            for item in parsed:
                yield from self._iter_json_results(item)

    def _load_tool_response_images(self, image_paths: List[str]) -> List[Any]:
        images = []
        for image_path in image_paths:
            try:
                images.append(process_image(image_path))
            except Exception as exc:
                print(
                    f"[warn] failed to load tool-returned image: {image_path} ({exc})",
                    flush=True,
                )
        return images

    def _append_image_placeholders(self, tool_response: str, image_count: int) -> str:
        placeholders = "\nRetrieved images:\n" + "\n".join(
            f"image {index + 1}: {IMAGE_PLACEHOLDER}"
            for index in range(image_count)
        )
        end_tag = self.config.tool_response_end
        if end_tag in tool_response:
            return tool_response.replace(end_tag, placeholders + "\n" + end_tag, 1)
        return tool_response + placeholders
    
    def _execute_tool_calls(self, response_strs: List[str], 
                          envs: List[ToolEnv], 
                          active_mask: torch.Tensor) -> List[str]:
        """Execute tool calls sequentially and return tool responses."""
        # Convert torch tensor to list of booleans if needed
        active_list = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else active_mask
        
        # Initialize result list with empty strings
        tool_responses = [""] * len(response_strs)
        # Process each environment sequentially
        for i, (resp, env, active) in enumerate(zip(response_strs, envs, active_list)):
            if not active:
                continue
                
            # Step the environment using the agent's response
            result = step(env, resp)
            tool_response = result[0]  # Extract observation from (observation, reward, done, info)
            
            # Determine tool type and use appropriate template
            tool_name = self._extract_tool_name(resp)
            if tool_name in ["insert", "update", "delete"]:
                # Use pipeline template for knowledge management tools
                template = self.config.tool_custom_response_template.replace(
                    self.config.tool_response_start, 
                    self.config.pipeline_response_start
                ).replace(
                    self.config.tool_response_end, 
                    self.config.pipeline_response_end
                )
                tool_responses[i] = template.format(tool_response=tool_response)
            else:
                # Use knowledge template for search tools
                tool_responses[i] = self.config.tool_custom_response_template.format(tool_response=tool_response)            
        return tool_responses
    
    def _extract_tool_name(self, response_str: str) -> str:
        """Extract tool name from response string"""
        import re
        import json
        
        # Try to extract tool call
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        tool_call_match = re.search(tool_call_pattern, response_str, re.DOTALL)
        
        if tool_call_match:
            try:
                tool_call_json = tool_call_match.group(1).strip()
                tool_call_data = json.loads(tool_call_json)
                
                # Handle both dictionary and list formats
                if isinstance(tool_call_data, dict):
                    return tool_call_data.get("tool", "unknown")
                elif isinstance(tool_call_data, list) and len(tool_call_data) > 0:
                    # If it's a list, take the first element and extract tool name
                    first_call = tool_call_data[0]
                    if isinstance(first_call, dict):
                        return first_call.get("tool", "unknown")
                    else:
                        return "unknown"
                else:
                    return "unknown"
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                pass
        
        # Try legacy query format
        query_pattern = r'<query>(.*?)</query>'
        query_match = re.search(query_pattern, response_str, re.DOTALL)
        if query_match:
            return "kb_search"  # Legacy format defaults to kb_search
            
        return "unknown"
    
    def _execute_tool_calls_batch(self, response_strs: List[str], 
                                 envs: List[ToolEnv], 
                                 active_mask: torch.Tensor) -> List[str]:
        """Execute tool calls in batch for tools that support batch operations."""
        # Convert torch tensor to list of booleans
        active_list = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else active_mask
        
        # Filter active environments and responses
        active_envs = []
        active_responses = []
        active_indices = []
        
        for i, (env, resp, active) in enumerate(zip(envs, response_strs, active_list)):
            if active:
                active_envs.append(env)
                active_responses.append(resp)
                active_indices.append(i)
        
        # Initialize result list with empty strings
        tool_responses = [""] * len(response_strs)
        
        if not active_envs:
            return tool_responses
            
        # Use the independent step_batch function for active environments with step numbers
        # Generate step numbers for batch processing (every 10 steps)
        step_numbers = [i + 1 for i in range(len(active_envs))]  # Simple step numbering
        batch_results = step_batch(active_envs, active_responses, step_numbers)
        
        # Map results back to original indices
        for idx, result, resp in zip(active_indices, batch_results, active_responses):
            if result is None:
                tool_responses[idx] = ""
            else:
                tool_response = result[0]  # Extract observation from (observation, reward, done, info)
                
                # Determine tool type and use appropriate template
                tool_name = self._extract_tool_name(resp)
                if tool_name in ["insert", "update", "delete"]:
                    # Use pipeline template for knowledge management tools
                    template = self.config.tool_custom_response_template.replace(
                        self.config.tool_response_start, 
                        self.config.pipeline_response_start
                    ).replace(
                        self.config.tool_response_end, 
                        self.config.pipeline_response_end
                    )
                    tool_responses[idx] = template.format(tool_response=tool_response)
                else:
                    # Use knowledge template for search tools
                    tool_responses[idx] = self.config.tool_custom_response_template.format(tool_response=tool_response)
        return tool_responses
    
    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            tool_responses_ids: torch.Tensor,
                            tool_response_images: List[List[Any]] | None = None,
                            trajectory_states: List[TrajectoryState] | None = None) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            tool_responses_ids
        ])
        
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, int(effective_len.item()))

        if self._has_multimodal_data(rollings.non_tensor_batch):
            new_input_ids = self._truncate_preserving_vision_tokens(new_input_ids, max_len)
            new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        else:
            new_input_ids = new_input_ids[:, -max_len:]
            new_attention_mask = new_attention_mask[:, -max_len:]

        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)
        if rollings.batch['position_ids'].dim() == 3:
            new_position_ids = new_position_ids.unsqueeze(1).expand(-1, 3, -1)

        non_tensors = dict(rollings.non_tensor_batch or {})
        if trajectory_states is not None:
            self._update_trajectory_states(
                trajectory_states,
                cur_responses,
                tool_responses_ids,
                tool_response_images=tool_response_images,
                max_len=max_len,
            )
            non_tensors.update(self._states_to_non_tensors(trajectory_states))
        else:
            if "raw_prompt_ids" in non_tensors:
                non_tensors["raw_prompt_ids"] = self._update_raw_prompt_ids(
                    non_tensors["raw_prompt_ids"],
                    cur_responses,
                    tool_responses_ids,
                    max_len=max_len,
                )
            if tool_response_images:
                non_tensors = self._append_tool_images_to_non_tensors(
                    non_tensors,
                    tool_response_images,
                )

        return DataProto.from_dict(
            {
                'input_ids': new_input_ids,
                'position_ids': new_position_ids,
                'attention_mask': new_attention_mask
            },
            non_tensors=non_tensors,
        )

    def _update_trajectory_states(
        self,
        states: List[TrajectoryState],
        cur_responses: torch.Tensor,
        tool_responses_ids: torch.Tensor,
        *,
        tool_response_images: List[List[Any]] | None,
        max_len: int,
    ) -> None:
        appended = 0
        for row_idx, state in enumerate(states):
            response_ids = cur_responses[row_idx]
            response_ids = response_ids[response_ids != self.tokenizer.pad_token_id]
            tool_ids = tool_responses_ids[row_idx]
            tool_ids = tool_ids[tool_ids != self.tokenizer.pad_token_id]

            combined = state.prompt_token_ids + self._to_token_list(response_ids) + self._to_token_list(tool_ids)
            state.prompt_token_ids = self._truncate_token_list_preserving_vision(combined, max_len)

            row_images = tool_response_images[row_idx] if tool_response_images and row_idx < len(tool_response_images) else []
            if row_images:
                item = deepcopy(state.multi_modal_data) if isinstance(state.multi_modal_data, dict) else {}
                images = self._images_from_item(item)
                images.extend(row_images)
                item["image"] = images
                state.multi_modal_data = item
                state.tool_return_images.extend(row_images)
                appended += len(row_images)

        if appended:
            print(f"ROLLOUT_TRAJECTORY_TOOL_IMAGES_APPENDED: images={appended}", flush=True)

    def _append_tool_images_to_non_tensors(
        self,
        non_tensors: Dict[str, Any],
        tool_response_images: List[List[Any]],
    ) -> Dict[str, Any]:
        multi_modal_data = non_tensors.get("multi_modal_data")
        if multi_modal_data is None:
            return non_tensors

        updated = []
        appended = 0
        for row_idx, item in enumerate(multi_modal_data):
            row_images = tool_response_images[row_idx] if row_idx < len(tool_response_images) else []
            if isinstance(item, dict):
                next_item = deepcopy(item)
            else:
                next_item = {}
            if row_images:
                images = list(next_item.get("image") or [])
                images.extend(row_images)
                next_item["image"] = images
                appended += len(row_images)
            updated.append(next_item)

        if appended:
            print(f"ROLLOUT_TOOL_IMAGES_APPENDED: images={appended}", flush=True)
        non_tensors["multi_modal_data"] = np.array(updated, dtype=object)
        return non_tensors

    def _has_multimodal_data(self, non_tensor_batch: Dict[str, Any]) -> bool:
        data = (non_tensor_batch or {}).get("multi_modal_data")
        if data is None:
            return False
        for item in data:
            if isinstance(item, dict) and item.get("image"):
                return True
        return False

    def _align_final_multimodal_data(
        self,
        input_ids: torch.Tensor,
        non_tensors: Dict[str, Any],
    ) -> Dict[str, Any]:
        multi_modal_data = (non_tensors or {}).get("multi_modal_data")
        if multi_modal_data is None:
            return non_tensors

        aligned = []
        trimmed = 0
        dropped = 0
        for row_idx, item in enumerate(multi_modal_data):
            if not isinstance(item, dict):
                aligned.append(item)
                continue

            row = input_ids[row_idx]
            valid = row[row != self.tokenizer.pad_token_id]
            token_ids = valid.detach().cpu().tolist()
            span_count = len(self._find_vision_spans(token_ids))
            images = item.get("image") or []
            if not isinstance(images, (list, tuple, np.ndarray)):
                images = [images]

            if span_count <= 0:
                next_item = dict(item)
                if images:
                    next_item.pop("image", None)
                    dropped += len(images)
                aligned.append(next_item)
                continue

            if len(images) > span_count:
                next_item = dict(item)
                next_item["image"] = list(images[:span_count])
                trimmed += len(images) - span_count
                aligned.append(next_item)
                continue

            aligned.append(item)

        if trimmed or dropped:
            print(
                f"FINAL_MM_ALIGN_IMAGES: trimmed={trimmed} dropped={dropped}",
                flush=True,
            )
        next_non_tensors = dict(non_tensors)
        next_non_tensors["multi_modal_data"] = np.array(aligned, dtype=object)
        return next_non_tensors

    def _image_token_counts(self, images: List[Any]) -> List[int]:
        if self.image_processor is None or not images:
            return [1 for _ in images]
        image_inputs = self.image_processor(images, return_tensors="pt")
        image_grid_thw = image_inputs.get("image_grid_thw")
        if image_grid_thw is None:
            return [1 for _ in images]
        merge_size = int(getattr(self.image_processor, "merge_size", 1))
        merge_length = merge_size ** 2
        return [max(1, int(grid.prod().item() // merge_length)) for grid in image_grid_thw]

    def _rewrite_vision_spans_to_counts(self, token_ids: List[int], token_counts: List[int]) -> Tuple[List[int], int, int]:
        vision_start_id, image_token_id, vision_end_id = self._vision_token_ids()
        if image_token_id is None:
            return token_ids, 0, 0

        output = []
        cursor = 0
        image_index = 0
        spans_seen = 0
        changed = 0

        def append_span(count: int) -> None:
            if vision_start_id is not None:
                output.append(vision_start_id)
            output.extend([image_token_id] * int(count))
            if vision_end_id is not None:
                output.append(vision_end_id)

        while cursor < len(token_ids):
            current = token_ids[cursor]
            if vision_start_id is not None and vision_end_id is not None and current == vision_start_id:
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
                        count = int(token_counts[image_index])
                        append_span(count)
                        if span.count(image_token_id) != count:
                            changed += 1
                        image_index += 1
                    else:
                        changed += 1
                    cursor = end + 1
                    continue

                output.extend(span)
                cursor = end + 1
                continue

            if current == image_token_id:
                start = cursor
                while cursor < len(token_ids) and token_ids[cursor] == image_token_id:
                    cursor += 1
                spans_seen += 1
                if image_index < len(token_counts):
                    count = int(token_counts[image_index])
                    append_span(count)
                    if cursor - start != count:
                        changed += 1
                    image_index += 1
                else:
                    changed += 1
                continue

            output.append(current)
            cursor += 1

        return output, spans_seen, changed

    def _pad_token_lists(self, token_lists: List[List[int]], width: int, *, pad_to_left: bool) -> torch.Tensor:
        output = torch.full(
            (len(token_lists), width),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        for row_idx, token_ids in enumerate(token_lists):
            token_ids = token_ids[-width:] if len(token_ids) > width else token_ids
            if not token_ids:
                continue
            value = torch.tensor(token_ids, dtype=torch.long)
            if pad_to_left:
                output[row_idx, -len(token_ids):] = value
            else:
                output[row_idx, :len(token_ids)] = value
        return output

    def _expand_final_vision_tokens(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        non_tensors: Dict[str, Any] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any] | None]:
        if not non_tensors or "multi_modal_data" not in non_tensors:
            return prompts, responses, non_tensors

        multi_modal_data = list(non_tensors["multi_modal_data"])
        prompt_lists = []
        response_lists = []
        aligned_mm = []
        changed_rows = 0
        trimmed_images = 0
        dropped_spans = 0

        for row_idx in range(prompts.shape[0]):
            prompt_ids = self._row_valid_token_list(prompts[row_idx])
            response_ids = self._row_valid_token_list(responses[row_idx])
            item = multi_modal_data[row_idx] if row_idx < len(multi_modal_data) else {}
            item = deepcopy(item) if isinstance(item, dict) else {}
            images = self._images_from_item(item)

            prompt_span_count = len(self._find_vision_spans(prompt_ids))
            response_span_count = len(self._find_vision_spans(response_ids))
            total_span_count = prompt_span_count + response_span_count
            if len(images) > total_span_count:
                item["image"] = images[:total_span_count]
                images = images[:total_span_count]
                trimmed_images += len(self._images_from_item(multi_modal_data[row_idx])) - len(images)

            token_counts = self._image_token_counts(images)
            prompt_counts = token_counts[:prompt_span_count]
            response_counts = token_counts[prompt_span_count:prompt_span_count + response_span_count]

            new_prompt_ids, prompt_spans_seen, prompt_changed = self._rewrite_vision_spans_to_counts(
                prompt_ids,
                prompt_counts,
            )
            new_response_ids, response_spans_seen, response_changed = self._rewrite_vision_spans_to_counts(
                response_ids,
                response_counts,
            )

            represented_spans = min(prompt_spans_seen, len(prompt_counts)) + min(response_spans_seen, len(response_counts))
            if represented_spans < total_span_count:
                dropped_spans += total_span_count - represented_spans
            if prompt_changed or response_changed:
                changed_rows += 1

            prompt_lists.append(self._truncate_token_list_preserving_vision(new_prompt_ids, prompts.shape[1]))
            response_lists.append(self._truncate_token_list_preserving_vision(new_response_ids, self.config.max_response_length))
            aligned_mm.append(item)

        prompt_width = prompts.shape[1]
        response_width = max(1, min(self.config.max_response_length, max(len(ids) for ids in response_lists)))
        new_prompts = self._pad_token_lists(prompt_lists, prompt_width, pad_to_left=True).to(prompts.device)
        new_responses = self._pad_token_lists(response_lists, response_width, pad_to_left=False).to(responses.device)

        if changed_rows or trimmed_images or dropped_spans:
            print(
                f"FINAL_MM_EXPAND_IMAGES: rows={changed_rows}/{prompts.shape[0]} "
                f"trimmed_images={trimmed_images} dropped_spans={dropped_spans}",
                flush=True,
            )

        next_non_tensors = dict(non_tensors)
        next_non_tensors["multi_modal_data"] = np.array(aligned_mm, dtype=object)
        return new_prompts, new_responses, next_non_tensors

    def _vision_token_ids(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if not callable(convert):
            return None, None, None
        ids = []
        for token in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"):
            token_id = convert(token)
            if token_id is None or token_id == getattr(self.tokenizer, "unk_token_id", object()):
                ids.append(None)
            else:
                ids.append(int(token_id))
        return tuple(ids)

    def _find_vision_spans(self, token_ids: List[int]) -> List[Tuple[int, int]]:
        vision_start_id, image_token_id, vision_end_id = self._vision_token_ids()
        if vision_start_id is None or image_token_id is None or vision_end_id is None:
            return []

        spans = []
        cursor = 0
        while cursor < len(token_ids):
            try:
                start = token_ids.index(vision_start_id, cursor)
            except ValueError:
                break
            try:
                end = token_ids.index(vision_end_id, start + 1) + 1
            except ValueError:
                break
            if image_token_id in token_ids[start:end]:
                spans.append((start, end))
            cursor = end
        return spans

    def _truncate_token_list_preserving_vision(self, token_ids: List[int], max_len: int) -> List[int]:
        """Left-truncate token lists without cutting Qwen-VL vision spans."""
        max_len = int(max_len)
        if len(token_ids) <= max_len:
            return token_ids

        spans = self._find_vision_spans(token_ids)
        span_token_count = sum(end - start for start, end in spans)
        if not spans or span_token_count >= max_len:
            return token_ids[-max_len:]

        keep = [False] * len(token_ids)
        for start, end in spans:
            for pos in range(start, end):
                keep[pos] = True

        tail_budget = max_len - span_token_count
        for pos in range(len(token_ids) - 1, -1, -1):
            if keep[pos]:
                continue
            keep[pos] = True
            tail_budget -= 1
            if tail_budget == 0:
                break

        kept = [token for token, should_keep in zip(token_ids, keep) if should_keep]
        return kept[-max_len:]

    def _update_raw_prompt_ids(
        self,
        raw_prompt_ids,
        cur_responses: torch.Tensor,
        tool_responses_ids: torch.Tensor,
        *,
        max_len: int,
    ):
        updated = []
        for row_idx, raw_ids in enumerate(raw_prompt_ids):
            if isinstance(raw_ids, torch.Tensor):
                ids = raw_ids.detach().cpu().tolist()
            elif hasattr(raw_ids, "tolist"):
                ids = raw_ids.tolist()
            else:
                ids = list(raw_ids)

            response_ids = cur_responses[row_idx]
            response_ids = response_ids[response_ids != self.tokenizer.pad_token_id].detach().cpu().tolist()
            tool_ids = tool_responses_ids[row_idx]
            tool_ids = tool_ids[tool_ids != self.tokenizer.pad_token_id].detach().cpu().tolist()

            combined = [int(token) for token in ids + response_ids + tool_ids]
            updated.append(self._truncate_token_list_preserving_vision(combined, max_len))
        return updated

    def _truncate_preserving_vision_tokens(self, input_ids: torch.Tensor, max_len: int) -> torch.Tensor:
        """Left-truncate prompts without cutting Qwen-VL vision spans."""
        max_len = int(max_len)
        if max_len >= input_ids.shape[1]:
            return input_ids

        output = torch.full(
            (input_ids.shape[0], max_len),
            self.tokenizer.pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        for row_idx, row in enumerate(input_ids):
            valid = row[row != self.tokenizer.pad_token_id]
            if valid.numel() == 0:
                continue
            if valid.numel() <= max_len:
                output[row_idx, -valid.numel():] = valid
                continue

            token_ids = valid.detach().cpu().tolist()
            spans = self._find_vision_spans(token_ids)
            span_token_count = sum(end - start for start, end in spans)
            if not spans or span_token_count >= max_len:
                output[row_idx] = valid[-max_len:]
                continue

            keep = torch.zeros(valid.shape[0], dtype=torch.bool, device=valid.device)
            for start, end in spans:
                keep[start:end] = True

            tail_budget = max_len - span_token_count
            for pos in range(valid.shape[0] - 1, -1, -1):
                if keep[pos]:
                    continue
                keep[pos] = True
                tail_budget -= 1
                if tail_budget == 0:
                    break

            kept = valid[keep]
            if kept.numel() > max_len:
                kept = kept[-max_len:]
            output[row_idx, -kept.numel():] = kept

        return output

    def _env_enabled(self, name: str, default: str = "false") -> bool:
        return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

    def _drop_vision_spans_for_generation(self, active_batch: DataProto, step: int) -> DataProto:
        """Remove image spans before vLLM tool-turn generation.

        vLLM 0.7.x can misalign Qwen2.5-VL image features and placeholder
        tokens after the tool loop expands/repeats multimodal prompts. The tool
        response already carries textual evidence, so later turns can generate
        from text-only context while the original trajectory remains unchanged.
        """
        if step == 0 or not self._env_enabled("DROP_MM_IN_VLLM_TOOL_TURNS"):
            return active_batch
        if not self._has_multimodal_data(active_batch.non_tensor_batch):
            return active_batch

        input_ids = active_batch.batch["input_ids"]
        output = torch.full_like(input_ids, self.tokenizer.pad_token_id)
        dropped = 0

        for row_idx, row in enumerate(input_ids):
            valid = row[row != self.tokenizer.pad_token_id]
            if valid.numel() == 0:
                continue

            token_ids = valid.detach().cpu().tolist()
            spans = self._find_vision_spans(token_ids)
            if spans:
                keep = torch.ones(valid.shape[0], dtype=torch.bool, device=valid.device)
                for start, end in spans:
                    keep[start:end] = False
                valid = valid[keep]
                dropped += len(spans)

            if valid.numel() > input_ids.shape[1]:
                valid = valid[-input_ids.shape[1]:]
            if valid.numel() > 0:
                output[row_idx, -valid.numel():] = valid

        if dropped == 0:
            return active_batch

        attention_mask = self.tensor_fn.create_attention_mask(output)
        position_ids = self.tensor_fn.create_position_ids(attention_mask)
        if active_batch.batch["position_ids"].dim() == 3:
            position_ids = position_ids.unsqueeze(1).expand(-1, 3, -1)

        non_tensors = dict(active_batch.non_tensor_batch or {})
        non_tensors.pop("multi_modal_data", None)
        non_tensors.pop("multi_modal_inputs", None)
        non_tensors.pop("raw_prompt_ids", None)

        print(
            f"ROLLOUT_DROP_MM_FOR_VLLM: step={step} rows={input_ids.shape[0]} spans={dropped}",
            flush=True,
        )
        return DataProto.from_dict(
            {
                "input_ids": output,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            },
            non_tensors=non_tensors,
        )

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          tool_responses_ids: torch.Tensor) -> Dict:
        """Update right side state."""
        responses = self.tensor_fn.concatenate_with_padding([
            right_side['responses'],
            cur_responses,
            tool_responses_ids
        ], pad_to_left=False)
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len]}


    def _flush_deferred_env_tools(self, envs: List[Any] = None) -> None:
        seen = set()
        for env in envs or []:
            if env is not None and id(env) not in seen:
                seen.add(id(env))
                flush_deferred_tools = getattr(env, "flush_deferred_tools", None)
                if callable(flush_deferred_tools):
                    flush_deferred_tools()
        from agent.tool.tools.graphr1_base_tool import GraphR1BaseTool
        GraphR1BaseTool.flush_deferred_io()


    def _select_active_rollings(self, rollings: DataProto, active_mask: torch.Tensor) -> DataProto:
        return DataProto.from_dict(
            {k: v[active_mask] for k, v in rollings.batch.items()},
            non_tensors=select_non_tensors(rollings.non_tensor_batch, active_mask),
        )


    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
            
        padded_active_batch = DataProto.from_dict(
            padded_batch,
            non_tensors=pad_non_tensors(active_batch.non_tensor_batch, padding_size),
        )
        
        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
        if getattr(padded_output, 'non_tensor_batch', None):
            padded_output.non_tensor_batch = {
                k: v[:-padding_size] for k, v in padded_output.non_tensor_batch.items()
            }
            
        padded_output.batch = trimmed_batch
        return padded_output
    
    def run_llm_loop(self, gen_batch, envs: List[Any] = None,
                    initial_input_ids: torch.Tensor = None) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        initial_position_ids = gen_batch.batch.get('position_ids')
        if initial_position_ids is not None and initial_position_ids.dim() == 3:
            initial_position_ids = initial_position_ids[..., -self.config.max_start_length:]
        elif initial_position_ids is not None:
            initial_position_ids = initial_position_ids[:, -self.config.max_start_length:]

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        if initial_position_ids is not None:
            original_left_side['position_ids'] = initial_position_ids
        original_right_side = {'responses': initial_input_ids[:, []]}
        
        batch_size = gen_batch.batch['input_ids'].shape[0]
        
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns = torch.zeros(batch_size, dtype=torch.int32)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        trajectory_states = self._init_trajectory_states(gen_batch)
        rollings = self._sync_rollings_from_states(rollings, trajectory_states)

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            print(f"ROLLOUT_TURN_START: step={step} active={active_mask.sum().item()}", flush=True)
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings_active = self._select_active_rollings(rollings, active_mask)
            if self._should_force_first_image_search(step, rollings):
                responses_ids, responses_str, new_active_masks = self._build_forced_image_search_batch(active_mask)
                meta_info = dict(getattr(rollings_active, "meta_info", {}) or {})
                print(f"ROLLOUT_TURN_FORCED_IMAGE_SEARCH: step={step} active={active_mask.sum().item()}", flush=True)
            else:
                rollings_active = self._drop_vision_spans_for_generation(rollings_active, step)
                gen_output = self._generate_with_gpu_padding(rollings_active)
                print(f"ROLLOUT_TURN_GENERATED: step={step} active={active_mask.sum().item()}", flush=True)

                meta_info = gen_output.meta_info
                responses_ids, responses_str, new_active_masks = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            active_mask[active_mask.clone()] = new_active_masks

            turns[active_mask] += 1

            if self.config.use_batch_tool_calls:
                # Use batch execution for tool calls
                tool_responses = self._execute_tool_calls_batch(responses_str, envs, active_mask)
            else:
                # Use sequential execution for tool calls
                tool_responses = self._execute_tool_calls(responses_str, envs, active_mask)

            active_num_list.append(active_mask.sum().item())
            tool_responses, tool_response_images = self._prepare_tool_responses_multimodal(tool_responses)
            tool_responses_ids = self._process_tool_responses(tool_responses)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                tool_responses_ids,
                tool_response_images=tool_response_images,
                trajectory_states=trajectory_states,
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                tool_responses_ids
            )
        self._flush_deferred_env_tools(envs)
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        original_right_side['turns'] = turns
        
        # Save trajectory and return final output
        output_non_tensors = {}
        if not self.is_validation:
            output_non_tensors.update(self._states_to_non_tensors(trajectory_states))
            output_non_tensors.pop("raw_prompt_ids", None)
        return self._compose_final_output(
            original_left_side,
            original_right_side,
            meta_info,
            non_tensors=output_non_tensors,
        )


    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            non_tensors: Dict = None) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        prompts = left_side['input_ids']
        responses = right_side['responses']
        prompts, responses, non_tensors = self._expand_final_vision_tokens(
            prompts,
            responses,
            non_tensors,
        )
        final_output['prompts'] = prompts
        final_output['responses'] = responses

        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            prompts,
            responses
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(prompts),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        position_ids = self.tensor_fn.create_position_ids(final_output['attention_mask'])
        left_position_ids = left_side.get('position_ids')
        if left_position_ids is not None and left_position_ids.dim() == 3:
            prompt_attention_mask = self.tensor_fn.create_attention_mask(prompts)
            left_position_ids = self.tensor_fn.create_position_ids(prompt_attention_mask)
            left_position_ids = left_position_ids.unsqueeze(1).expand(-1, 3, -1)
            prompt_len = prompts.shape[1]
            response_position_ids = position_ids[:, prompt_len:].unsqueeze(1).expand(-1, 3, -1)
            position_ids = torch.cat([left_position_ids, response_position_ids], dim=-1)
        final_output['position_ids'] = position_ids

        if non_tensors and "multi_modal_data" in non_tensors:
            non_tensors = self._align_final_multimodal_data(
                final_output["input_ids"],
                non_tensors,
            )
        
        final_output = DataProto.from_dict(final_output, non_tensors=non_tensors)
        final_output.meta_info.update(meta_info)

        return final_output
