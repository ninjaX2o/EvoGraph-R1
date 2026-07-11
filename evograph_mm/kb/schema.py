"""Canonical metadata-only multimodal QA record schema."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from .vqa_prompt import build_vqa_user_prompt
from typing import Any


MM_SCHEMA_VERSION = "evograph-mm-metadata-record-v2"
MM_REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "context",
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
        "data_id",
        "image_id",
        "image_path",
        "image_urls",
    }
)
MM_REQUIRED_EXTRA_INFO_FIELDS = frozenset(
    {
        "dataset",
        "split",
        "index",
        "question",
        "question_original",
        "answer",
        "golden_answers",
        "image_id",
        "image_path",
        "image_missing",
        "subset_root",
        "source_path",
        "source_csv",
        "row_index",
        "source_row",
        "original_metadata",
        "mm_schema_version",
    }
)


class MultimodalSchemaError(ValueError):
    """Raised when a metadata-only multimodal record violates the schema."""


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MultimodalSchemaError(f"{field} must be an object")
    return value


def _require_non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MultimodalSchemaError(f"{field} must be a non-empty string")
    return value


def _normalize_extra_info_index(value: object) -> str:
    if isinstance(value, bool):
        raise MultimodalSchemaError(
            "extra_info.index must be a non-empty string or integer"
        )
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise MultimodalSchemaError(
        "extra_info.index must be a non-empty string or integer"
    )


def _require_sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MultimodalSchemaError(f"{field} must be a sequence")
    return value


def _normalize_answer_sequence(answers: Sequence[object]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for answer in answers:
        if answer is None:
            continue
        text = str(answer).strip()
        if text and text not in seen:
            normalized.append(text)
            seen.add(text)
    return normalized


def _validate_golden_answers(value: object) -> None:
    golden_answers = _require_sequence(value, "extra_info.golden_answers")
    if not golden_answers:
        raise MultimodalSchemaError(
            "extra_info.golden_answers must contain at least one answer"
        )
    for index, answer in enumerate(golden_answers):
        _require_non_empty_string(answer, f"extra_info.golden_answers[{index}]")


def validate_mm_record(record: Mapping[str, object]) -> dict[str, object]:
    """Validate and return a JSON-compatible metadata-only MM QA record."""

    if not isinstance(record, Mapping):
        raise MultimodalSchemaError("record must be an object")

    if "images" in record:
        raise MultimodalSchemaError(
            "metadata-only metadata-only records must not include an images field"
        )

    missing = MM_REQUIRED_TOP_LEVEL_FIELDS.difference(record)
    if missing:
        raise MultimodalSchemaError(
            f"missing required top-level field(s): {', '.join(sorted(missing))}"
        )

    _require_non_empty_string(record["data_source"], "data_source")
    _require_non_empty_string(record["ability"], "ability")
    _require_non_empty_string(record["data_id"], "data_id")
    _require_non_empty_string(record["image_id"], "image_id")
    _require_non_empty_string(record["image_path"], "image_path")

    prompt = _require_sequence(record["prompt"], "prompt")
    if not prompt:
        raise MultimodalSchemaError("prompt must contain at least one message")
    for index, message in enumerate(prompt):
        message_map = _require_mapping(message, f"prompt[{index}]")
        _require_non_empty_string(message_map.get("role"), f"prompt[{index}].role")
        content = _require_non_empty_string(
            message_map.get("content"), f"prompt[{index}].content"
        )
        if "<image>" in content:
            raise MultimodalSchemaError("prompt content must not contain <image>")

    reward_model = _require_mapping(record["reward_model"], "reward_model")
    _require_non_empty_string(reward_model.get("style"), "reward_model.style")
    if "ground_truth" not in reward_model:
        raise MultimodalSchemaError("reward_model.ground_truth is required")

    extra_info = _require_mapping(record["extra_info"], "extra_info")
    extra_missing = MM_REQUIRED_EXTRA_INFO_FIELDS.difference(extra_info)
    if extra_missing:
        raise MultimodalSchemaError(
            f"missing required extra_info field(s): {', '.join(sorted(extra_missing))}"
        )
    normalized_index = _normalize_extra_info_index(extra_info["index"])
    _require_non_empty_string(extra_info["dataset"], "extra_info.dataset")
    _require_non_empty_string(extra_info["split"], "extra_info.split")
    _require_non_empty_string(extra_info["question"], "extra_info.question")
    _require_non_empty_string(
        extra_info["question_original"], "extra_info.question_original"
    )
    _require_non_empty_string(extra_info["image_id"], "extra_info.image_id")
    _require_non_empty_string(extra_info["image_path"], "extra_info.image_path")
    _require_non_empty_string(extra_info["subset_root"], "extra_info.subset_root")
    _require_non_empty_string(extra_info["source_csv"], "extra_info.source_csv")
    _require_mapping(extra_info["source_row"], "extra_info.source_row")
    _require_mapping(extra_info["original_metadata"], "extra_info.original_metadata")
    if not isinstance(extra_info["image_missing"], bool):
        raise MultimodalSchemaError("extra_info.image_missing must be a boolean")
    _validate_golden_answers(extra_info["golden_answers"])
    if extra_info["mm_schema_version"] != MM_SCHEMA_VERSION:
        raise MultimodalSchemaError(
            f"extra_info.mm_schema_version must be {MM_SCHEMA_VERSION}"
        )

    normalized_record = dict(record)
    normalized_extra_info = dict(extra_info)
    normalized_extra_info["index"] = normalized_index
    normalized_record["extra_info"] = normalized_extra_info

    try:
        payload = json.loads(json.dumps(normalized_record, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise MultimodalSchemaError(f"record must be JSON serializable: {exc}") from exc
    return payload


def parse_e_vqa_answers(raw_answer: object) -> list[str]:
    """Return de-duplicated answer strings from E-VQA answer syntax."""

    if raw_answer is None:
        return []
    values: list[str] = []
    seen: set[str] = set()
    for pipe_part in str(raw_answer).split("|"):
        for answer_part in pipe_part.split("&&"):
            answer = answer_part.strip()
            if answer and answer not in seen:
                values.append(answer)
                seen.add(answer)
    return values


def build_multimodal_qa_record(
    *,
    data_source: str,
    data_id: str | None = None,
    dataset: str | None = None,
    split: str,
    index: int | str,
    question: str,
    question_original: str | None = None,
    answers: Sequence[object],
    raw_answer: object = None,
    image_id: str,
    image_path: str,
    image_missing: bool,
    subset_root: str = "metadata-only",
    source_path: str | None = None,
    source_csv: str = "metadata-only",
    row_index: int | str | None = None,
    source_row: Mapping[str, object] | None = None,
    context: Sequence[object] = (),
    original_metadata: Mapping[str, object] | None = None,
    image_urls: object | None = None,
    ability: str = "multimodal_qa",
) -> dict[str, object]:
    """Build and validate a metadata-only metadata-only multimodal QA record."""

    golden_answers = _normalize_answer_sequence(answers)
    if not golden_answers:
        golden_answers = parse_e_vqa_answers(raw_answer)
    dataset_name = data_source if dataset is None else dataset
    normalized_index = _normalize_extra_info_index(index)
    record_id = data_id or f"{dataset_name}:{split}:{normalized_index}:{image_id}"
    original_question = question if question_original is None else question_original
    answer_payload = list(golden_answers) if raw_answer is None else raw_answer
    prompt_text = build_vqa_user_prompt(
        question=question,
        image_id=image_id,
        image_path=image_path,
    )
    record: dict[str, object] = {
        "context": [str(value) for value in context if str(value).strip()],
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": golden_answers},
        "extra_info": {
            "dataset": dataset_name,
            "split": split,
            "index": normalized_index,
            "question": question,
            "question_original": original_question,
            "answer": answer_payload,
            "golden_answers": golden_answers,
            "image_id": image_id,
            "image_path": image_path,
            "image_missing": image_missing,
            "subset_root": subset_root,
            "source_path": source_path,
            "source_csv": source_csv,
            "row_index": None if row_index is None else str(row_index),
            "source_row": dict(source_row or {}),
            "original_metadata": dict(original_metadata or {}),
            "mm_schema_version": MM_SCHEMA_VERSION,
        },
        "data_id": record_id,
        "image_id": image_id,
        "image_path": image_path,
        "image_urls": image_urls,
    }
    return validate_mm_record(record)
