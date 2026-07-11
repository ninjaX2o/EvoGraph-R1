import numpy as np
import torch
from PIL import Image
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_tensor_helper():
    helper_path = ROOT / "agent" / "llm_agent" / "tensor_helper.py"
    spec = importlib.util.spec_from_file_location("_tensor_helper_under_test", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TensorConfig, module.TensorHelper


def _load_multimodal_helpers():
    helper_path = ROOT / "verl" / "utils" / "multimodal.py"
    spec = importlib.util.spec_from_file_location("_multimodal_under_test", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        module.build_multi_modal_inputs,
        module.collate_multi_modal_inputs,
        module.pad_non_tensors,
        module.select_non_tensors,
    )


TensorConfig, TensorHelper = _load_tensor_helper()
(
    build_multi_modal_inputs,
    collate_multi_modal_inputs,
    pad_non_tensors,
    select_non_tensors,
) = _load_multimodal_helpers()


class _DummyImageProcessor:
    def __call__(self, images, return_tensors="pt"):
        count = len(images)
        return {
            "pixel_values": torch.ones(count, 3),
            "image_grid_thw": torch.tensor([[1, 1, 2]] * count, dtype=torch.long),
        }


class _DummyProcessor:
    image_processor = _DummyImageProcessor()


def test_non_tensor_selection_and_padding_preserve_multimodal_data():
    non_tensors = {
        "multi_modal_data": np.array(
            [{"image": ["a"]}, {"image": ["b"]}, {"image": ["c"]}],
            dtype=object,
        ),
    }

    selected = select_non_tensors(non_tensors, torch.tensor([True, False, True]))
    padded = pad_non_tensors(selected, padding_size=2)

    assert list(selected["multi_modal_data"]) == [
        {"image": ["a"]},
        {"image": ["c"]},
    ]
    assert list(padded["multi_modal_data"]) == [
        {"image": ["a"]},
        {"image": ["c"]},
        {"image": ["a"]},
        {"image": ["a"]},
    ]


def test_build_and_collate_multimodal_inputs_from_non_tensor_batch():
    images = np.array(
        [
            {"image": [Image.new("RGB", (4, 4))]},
            {"image": [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]},
        ],
        dtype=object,
    )

    inputs = build_multi_modal_inputs(_DummyProcessor(), images)
    collated = collate_multi_modal_inputs(
        {"multi_modal_inputs": inputs},
        device=torch.device("cpu"),
    )

    assert len(inputs) == 2
    assert collated["pixel_values"].shape == (3, 3)
    assert collated["image_grid_thw"].shape == (3, 3)


def test_collate_multimodal_inputs_does_not_depend_on_first_item():
    images = np.array(
        [
            {},
            {"image": [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]},
        ],
        dtype=object,
    )

    inputs = build_multi_modal_inputs(_DummyProcessor(), images)
    collated = collate_multi_modal_inputs({"multi_modal_inputs": inputs})

    assert collated["pixel_values"].shape == (2, 3)
    assert collated["image_grid_thw"].shape == (2, 3)


def test_tensor_helper_cuts_three_dimensional_position_ids_on_sequence_axis():
    helper = TensorHelper(
        TensorConfig(
            pad_token_id=0,
            max_prompt_length=8,
            max_tool_response_length=8,
            max_start_length=8,
        )
    )
    position_ids = torch.arange(2 * 3 * 5).reshape(2, 3, 5)
    tensor_dict = {
        "input_ids": torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]]),
        "position_ids": position_ids,
    }

    result = helper.cut_to_effective_len(tensor_dict, keys=["input_ids", "position_ids"])

    assert result["input_ids"].shape == (2, 4)
    assert result["position_ids"].shape == (2, 3, 4)
    torch.testing.assert_close(result["position_ids"], position_ids[..., -4:])
