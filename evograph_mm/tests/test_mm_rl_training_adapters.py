from types import SimpleNamespace
import importlib.util

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

HAS_TENSORDICT = importlib.util.find_spec("tensordict") is not None
pytestmark = pytest.mark.skipif(
    not HAS_TENSORDICT,
    reason="tensordict is required for DataProto-backed training adapter tests",
)

if HAS_TENSORDICT:
    from agent.llm_agent.generation import ToolGenerationManager
    from verl import DataProto
    from verl.utils.dataset.rl_dataset import ToolRLDataset
    from verl.utils.multimodal import (
        build_multi_modal_inputs,
        collate_multi_modal_inputs,
        generation_non_tensor_keys,
    )


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    eos_token = "<eos>"

    def __init__(self):
        self.last_prompt = ""

    def apply_chat_template(self, chat, tools=None, add_generation_prompt=True, tokenize=False):
        if hasattr(chat, "tolist"):
            chat = chat.tolist()
        text = "\n".join(item["content"] for item in chat)
        if add_generation_prompt:
            text += "\nassistant:"
        return text

    def __call__(self, prompt, return_tensors="pt", add_special_tokens=False):
        self.last_prompt = prompt
        ids = torch.arange(1, len(prompt) + 1, dtype=torch.long).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


class _DummyImageProcessor:
    merge_size = 1

    def __call__(self, images, return_tensors="pt"):
        count = len(images)
        return {
            "pixel_values": torch.ones(count, 3),
            "image_grid_thw": torch.tensor([[1, 1, 2]] * count, dtype=torch.long),
        }


class _DummyProcessor:
    image_token = "<|image_pad|>"

    def __init__(self):
        self.image_processor = _DummyImageProcessor()


class _DummyToolEnv:
    tool_desc = []

    def tools_format_func(self):
        return "tools"


def test_tool_rl_dataset_keeps_multimodal_data_and_expands_image_tokens():
    tokenizer = _DummyTokenizer()
    processor = _DummyProcessor()
    image = Image.new("L", (8, 8), color=128)

    dataset = ToolRLDataset.__new__(ToolRLDataset)
    dataset.dataframe = pd.DataFrame(
        [
            {
                "prompt": [{"role": "user", "content": "Question with <image>"}],
                "images": [image],
                "extra_info": {"index": 7},
            }
        ]
    )
    dataset.prompt_key = "prompt"
    dataset.image_key = "images"
    dataset.tokenizer = tokenizer
    dataset.processor = processor
    dataset.max_prompt_length = 256
    dataset.truncation = "error"
    dataset.return_raw_chat = False
    dataset.tool_env = _DummyToolEnv()
    dataset.tools = []
    dataset.use_custom_tool_format_func = False

    item = dataset[0]

    assert "multi_modal_data" in item
    assert "images" not in item
    assert item["multi_modal_data"]["image"][0].mode == "RGB"
    assert "<image>" not in tokenizer.last_prompt
    assert tokenizer.last_prompt.count(processor.image_token) == 2
    assert item["index"] == 7


def test_multimodal_inputs_are_built_and_collated_from_non_tensor_data():
    processor = _DummyProcessor()
    images = np.array(
        [
            {"image": [Image.new("RGB", (4, 4))]},
            {"image": [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]},
        ],
        dtype=object,
    )

    inputs = build_multi_modal_inputs(processor, images)
    collated = collate_multi_modal_inputs({"multi_modal_inputs": inputs}, device=torch.device("cpu"))

    assert len(inputs) == 2
    assert collated["pixel_values"].shape == (3, 3)
    assert collated["image_grid_thw"].shape == (3, 3)


def test_generation_non_tensor_keys_select_only_prompt_context_fields():
    batch = SimpleNamespace(
        non_tensor_batch={
            "multi_modal_data": [{"image": ["a"]}, {"image": ["b"]}],
            "image_urls": ["url-a", "url-b"],
            "uid": ["keep-on-training-batch", "not-needed-for-generation"],
        },
    )

    assert generation_non_tensor_keys(batch) == ["multi_modal_data", "image_urls"]


def test_tool_generation_active_and_padding_keep_multimodal_non_tensors():
    class _FakeRolloutWG:
        def __init__(self):
            self.seen = None

        def generate_sequences(self, batch):
            self.seen = batch
            size = batch.batch["input_ids"].shape[0]
            return DataProto.from_dict(
                tensors={"responses": torch.ones(size, 1, dtype=torch.long)}
            )

    manager = ToolGenerationManager.__new__(ToolGenerationManager)
    manager.config = SimpleNamespace(num_gpus=4)
    manager.actor_rollout_wg = _FakeRolloutWG()

    rollings = DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(9, dtype=torch.long).reshape(3, 3),
            "attention_mask": torch.ones(3, 3, dtype=torch.long),
            "position_ids": torch.arange(9, dtype=torch.long).reshape(3, 3),
        },
        non_tensors={
            "multi_modal_data": [
                {"image": ["image-a"]},
                {"image": ["image-b"]},
                {"image": ["image-c"]},
            ],
        },
    )

    selected = manager._select_active_rollings(
        rollings, torch.tensor([True, False, True])
    )
    output = manager._generate_with_gpu_padding(selected)

    assert list(selected.non_tensor_batch["multi_modal_data"]) == [
        {"image": ["image-a"]},
        {"image": ["image-c"]},
    ]
    seen_mm = list(manager.actor_rollout_wg.seen.non_tensor_batch["multi_modal_data"])
    assert seen_mm == [
        {"image": ["image-a"]},
        {"image": ["image-c"]},
        {"image": ["image-a"]},
        {"image": ["image-a"]},
    ]
    assert output.batch["responses"].shape[0] == 2
