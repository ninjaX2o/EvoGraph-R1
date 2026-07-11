"""Store record builders for multimodal KB artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import math
import re


@dataclass(frozen=True)
class RecordBundle:
    text_documents: list[dict[str, Any]]
    visual_records: list[dict[str, Any]]
    links: list[dict[str, Any]]


def text_embedding_id(text_doc_id: str) -> str:
    return f"text_embedding::{text_doc_id}"


def visual_embedding_id(image_id: str) -> str:
    return f"visual_embedding::{image_id}"


def fused_embedding_id(data_id: str) -> str:
    return f"fused_embedding::{data_id}"


def build_records_from_rows(rows: Iterable[dict[str, Any]]) -> RecordBundle:
    text_documents: list[dict[str, Any]] = []
    visual_records: list[dict[str, Any]] = []
    seen_image_ids: set[str] = set()
    links: list[dict[str, Any]] = []

    for raw_row in rows:
        row = _json_safe(raw_row)
        data_id = _string_value(row.get("data_id"))
        extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        reward_model = row.get("reward_model") if isinstance(row.get("reward_model"), dict) else {}
        source_row = (
            extra_info.get("source_row")
            if isinstance(extra_info.get("source_row"), dict)
            else {}
        )
        split = _string_value(_field(row, extra_info, "split"))
        question = _string_value(_field(row, extra_info, "question"))
        question_original = _string_value(_field(row, extra_info, "question_original"))
        answer = _string_value(_field(row, extra_info, "answer"))
        context = _field(row, extra_info, "context")
        generated_tar_qa = _is_tar_anchored_source_row(source_row)
        if generated_tar_qa:
            question = ""
            question_original = ""
            answer = ""
            context = ""
        image_id = _string_value(row.get("image_id"))
        image_path = _string_value(row.get("image_path"))
        text_doc_id = f"text::{data_id}"
        visual_record_id = f"visual::{image_id}"
        golden_answers = (
            []
            if generated_tar_qa
            else _as_list(
                _first_present(
                    extra_info.get("golden_answers"),
                    reward_model.get("ground_truth"),
                    row.get("golden_answers"),
                )
            )
        )
        source_metadata = _source_metadata(row, extra_info)
        image_missing = _bool_value(
            _first_present(
                extra_info.get("image_missing"),
                row.get("image_missing"),
                None,
            ),
            default=not bool(image_path),
        )

        text_doc = {
            "text_doc_id": text_doc_id,
            "data_id": data_id,
            "split": split,
            "question": question,
            "question_original": question_original,
            "context": context,
            "answer": answer,
            "golden_answers": golden_answers,
            "image_id": image_id,
            "image_path": image_path,
            "source_metadata": source_metadata,
            "contents": _build_contents(
                wikipedia_title=_string_value(source_row.get("wikipedia_title")),
                wikipedia_url=_string_value(source_row.get("wikipedia_url")),
                evidence_section_title=_string_value(
                    source_row.get("evidence_section_title")
                ),
                evidence=_string_value(source_row.get("evidence")),
            ),
        }
        text_documents.append(text_doc)

        if image_id not in seen_image_ids:
            seen_image_ids.add(image_id)
            visual_records.append(
                {
                    "visual_record_id": visual_record_id,
                    "image_id": image_id,
                    "image_path": image_path,
                    "split": split,
                    "data_id": data_id,
                    "source_metadata": source_metadata,
                    "image_missing": image_missing,
                }
            )

        fused_id = fused_embedding_id(data_id)
        links.extend(
            [
                _link(data_id, visual_record_id, "has_image", data_id, split),
                _link(data_id, text_doc_id, "has_text_document", data_id, split),
                _link(
                    visual_record_id,
                    visual_embedding_id(image_id),
                    "has_visual_embedding",
                    data_id,
                    split,
                ),
                _link(
                    text_doc_id,
                    text_embedding_id(text_doc_id),
                    "has_text_embedding",
                    data_id,
                    split,
                ),
                _link(
                    data_id,
                    fused_id,
                    "has_fused_embedding",
                    data_id,
                    split,
                ),
                _link(
                    text_doc_id,
                    fused_id,
                    "has_fused_embedding",
                    data_id,
                    split,
                ),
                _link(
                    visual_record_id,
                    fused_id,
                    "has_fused_embedding",
                    data_id,
                    split,
                ),
            ]
        )

    return RecordBundle(
        text_documents=text_documents,
        visual_records=visual_records,
        links=links,
    )


def write_store(layout: Any, bundle: RecordBundle) -> dict[str, int]:
    layout.ensure_directories()

    _write_jsonl(
        layout.graphr1_text_root / "text_documents.jsonl",
        bundle.text_documents,
    )
    _write_jsonl(
        layout.graphr1_text_root / "corpus.jsonl",
        (_corpus_record(doc) for doc in bundle.text_documents),
    )
    _write_json(
        layout.graphr1_text_root / "metadata.json",
        {
            "text_document_count": len(bundle.text_documents),
            "corpus_count": len(bundle.text_documents),
        },
    )

    _write_jsonl(layout.visual_store_root / "image_records.jsonl", bundle.visual_records)
    _write_json(
        layout.visual_store_root / "metadata.json",
        {"visual_record_count": len(bundle.visual_records)},
    )

    _write_jsonl(layout.link_store_root / "links.jsonl", bundle.links)
    _write_json(
        layout.link_store_root / "metadata.json",
        {"link_count": len(bundle.links)},
    )

    return {
        "text_document_count": len(bundle.text_documents),
        "visual_record_count": len(bundle.visual_records),
        "link_count": len(bundle.links),
    }


def _build_contents(
    *,
    wikipedia_title: str,
    wikipedia_url: str,
    evidence_section_title: str,
    evidence: str,
) -> str:
    evidence = _trim_evidence(evidence)
    fields = [wikipedia_title, evidence_section_title, evidence]
    return "\n\n".join(value for value in fields if value)


def _trim_evidence(value: str) -> str:
    return re.sub(r"\s+", " ", _string_value(value)).strip()


def _corpus_record(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": doc["text_doc_id"],
        "contents": doc["contents"],
    }


def _source_metadata(row: dict[str, Any], extra_info: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("source_metadata")
    source_metadata = dict(metadata) if isinstance(metadata, dict) else {}
    for key in ("source_csv", "row_index", "source_path", "source_row", "original_metadata"):
        value = _first_present(extra_info.get(key), row.get(key))
        if value is not None:
            source_metadata[key] = value
    for key in ("wikipedia_title", "wikipedia_url", "wikipedia_section"):
        if key in row and key not in source_metadata:
            source_metadata[key] = row[key]
    return source_metadata


def _is_tar_anchored_source_row(source_row: dict[str, Any]) -> bool:
    return _string_value(source_row.get("question_type")).lower() == "tar_anchored"


def _field(row: dict[str, Any], extra_info: dict[str, Any], key: str) -> Any:
    return _first_present(extra_info.get(key), row.get(key))


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _context_contents(context: Any) -> str:
    context = _json_safe(context)
    if isinstance(context, list):
        return " ".join(_string_value(item) for item in context)
    return _string_value(context)


def _bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _link(
    source_id: str,
    target_id: str,
    relation: str,
    data_id: str,
    split: str,
) -> dict[str, str]:
    return {
        "source_id": source_id,
        "target_id": target_id,
        "relation": relation,
        "data_id": data_id,
        "split": split,
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_list(value: Any) -> list[Any]:
    value = _json_safe(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _string_value(value: Any) -> str:
    value = _json_safe(value)
    if value is None:
        return ""
    return str(value)


def _json_safe(value: Any) -> Any:
    if _is_missing_scalar(value):
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(_json_safe(key)): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if value != value:
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _is_missing_scalar(value: Any) -> bool:
    value_type = type(value)
    module_name = getattr(value_type, "__module__", "")
    if module_name.startswith("pandas._libs.missing"):
        return True
    try:
        import pandas as pd
    except ImportError:
        return False
    return value is pd.NA
