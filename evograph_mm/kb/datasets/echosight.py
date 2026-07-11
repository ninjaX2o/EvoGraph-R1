"""Guarded EchoSight/E-VQA/InfoSeek adapter scaffolding."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from ..schema import (
    MM_SCHEMA_VERSION,
    MultimodalSchemaError,
    build_multimodal_qa_record,
    parse_e_vqa_answers,
)
from ..audit_echosight import audit_echosight_assets
from ..full_image_layout import dataset_full_image_roots
from ..prepare_echosight import DATASET_E_VQA, DATASET_INFOSEEK, SUPPORTED_DATASETS
from ..validate_echosight import validate_echosight_full_assets


READINESS_VERSION = "echosight-readiness-v1"
SCHEMA_FREEZE_GUARD = (
    "Metadata-only preprocessing does not finalize the real E-VQA/InfoSeek "
    "runtime schema without local asset review."
)
SUBSET_PARQUET_COLUMNS = [
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
]
E_VQA_SUBSET_SPLIT_FILES = {
    "train": "qa_train.csv",
    "test": "qa_test.csv",
}
E_VQA_FULL_SPLIT_FILES = {
    "train": "qa_train.csv",
    "val": "qa_dev.csv",
    "test": "qa_test.csv",
}
INFOSEEK_FULL_SPLIT_FILES = {
    "train": "qa_train.csv",
    "test": "qa_test.csv",
}
E_VQA_REQUIRED_CSV_FIELDS = ("question", "answer", "dataset_image_ids")
FULL_REQUIRED_CSV_FIELDS = ("question", "answer")
E_VQA_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


def _normalize_datasets(datasets: Iterable[str] | str | None) -> tuple[str, ...]:
    if datasets is None:
        return SUPPORTED_DATASETS
    if isinstance(datasets, str):
        selected = (datasets,)
    else:
        selected = tuple(datasets)
    unknown = tuple(dataset for dataset in selected if dataset not in SUPPORTED_DATASETS)
    if unknown:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"unsupported EchoSight dataset(s): {unknown}; expected {supported}")
    return selected


def _dataset_report(
    report: Mapping[str, object] | None,
    dataset: str,
) -> Mapping[str, object] | None:
    if report is None:
        return None
    for item in report.get("dataset_reports", []):
        if isinstance(item, Mapping) and item.get("dataset") == dataset:
            return item
    return None


def _blocker(
    *,
    dataset: str,
    stage: str,
    message: str,
    details: object | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "dataset": dataset,
        "stage": stage,
        "message": message,
    }
    if details:
        value["details"] = details
    return value


def _metadata_only_policy() -> dict[str, bool]:
    return {
        "metadata_only": True,
        "downloads_enabled": False,
        "remote_urls_accessed": False,
        "image_bytes_embedded": False,
        "text_dataset_roots_written": False,
        "text_expr_roots_written": False,
    }


def evaluate_echosight_readiness(
    root: str | Path = ".",
    *,
    datasets: Iterable[str] | str | None = None,
    audit_report: Mapping[str, object] | None = None,
    validation_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return guarded readiness blockers without freezing real schemas."""

    selected = _normalize_datasets(datasets)
    root_path = Path(root)
    if audit_report is None:
        audit_report = audit_echosight_assets(root_path, datasets=selected)
    if validation_report is None:
        validation_report = validate_echosight_full_assets(root_path, datasets=selected)

    blockers: list[dict[str, object]] = []
    evidence: dict[str, object] = {
        "b2_audit_version": audit_report.get("audit_version"),
        "b3_validation_version": validation_report.get("validation_version"),
        "datasets": {},
    }

    for dataset in selected:
        audit_dataset = _dataset_report(audit_report, dataset)
        validation_dataset = _dataset_report(validation_report, dataset)
        audit_ready = bool(
            audit_dataset
            and audit_dataset.get("task_c_readiness", {}).get(
                "representative_real_audit_complete"
            )
        )
        validation_status = None
        if validation_dataset:
            validation_status = validation_dataset.get("summary", {}).get("status")
        validation_ready = validation_status == "complete"

        evidence["datasets"][dataset] = {
            "b2_representative_real_audit_complete": audit_ready,
            "b3_full_validation_status": validation_status,
        }

        if not audit_ready:
            details = None
            if audit_dataset:
                details = audit_dataset.get("task_c_readiness", {}).get("blockers")
            blockers.append(
                _blocker(
                    dataset=dataset,
                    stage="audit",
                    message=(
                        "Representative local audit evidence is incomplete; "
                        "real E-VQA/InfoSeek runtime schema review is not ready."
                    ),
                    details=details,
                )
            )
        if not validation_ready:
            details = validation_dataset.get("blockers") if validation_dataset else None
            blockers.append(
                _blocker(
                    dataset=dataset,
                    stage="validation",
                    message=(
                        "Full local validation is not complete; real "
                        "E-VQA/InfoSeek runtime schema review is not ready."
                    ),
                    details=details,
                )
            )
        blockers.append(
            _blocker(
                dataset=dataset,
                stage="metadata-only",
                message=SCHEMA_FREEZE_GUARD,
            )
        )

    return {
        "readiness_version": READINESS_VERSION,
        "dataset_family": "echosight",
        "datasets": list(selected),
        "checked_root": str(root_path),
        "schema_freeze_ready": False,
        "real_schema_status": "blocked",
        "blockers": blockers,
        "evidence": evidence,
        "policy": {
            "schema_freeze": False,
            "metadata_only_scaffolding": True,
            "real_payloads_processed": False,
            "downloads_enabled": False,
            "archives_extracted": False,
            "payload_files_written": False,
        },
    }


def _echosight_subset_root(root: str | Path, dataset: str, subset: str) -> Path:
    return Path(root) / "datasets_mm" / dataset / "subsets" / subset


def _echosight_processed_root(
    root: str | Path,
    dataset: str,
    subset: str | None,
    *,
    subset_mode: str,
    output_root: str | Path | None = None,
) -> Path:
    processed_base = Path(root) / "datasets_mm" / dataset / "processed"
    if output_root is None:
        if subset_mode == "none":
            return processed_base
        if subset is None:
            raise ValueError("subset is required when subset_mode is paper")
        return processed_base / subset

    resolved_base = processed_base.resolve()
    resolved_output = Path(output_root).resolve()
    try:
        resolved_output.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(
            "output_root must resolve inside "
            f"{resolved_base}; got {resolved_output}"
        ) from exc
    return resolved_output


def _load_manifest_image_index(subset_root: Path) -> dict[str, dict[str, object]]:
    manifest_path = subset_root / "manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index: dict[str, dict[str, object]] = {}
    for item in manifest.get("copied_images", []):
        if isinstance(item, Mapping):
            image_id = str(item.get("image_id", "")).strip()
            if image_id and image_id not in index:
                index[image_id] = dict(item)
    return index


def _manifest_subset_image_path(
    root: Path,
    subset_root: Path,
    subset_path: object,
) -> Path | None:
    if subset_path is None:
        return None
    text = str(subset_path).strip()
    if not text:
        return None

    manifest_path = Path(text)
    if manifest_path.is_absolute():
        candidates = (manifest_path,)
    else:
        candidates = (
            root.joinpath(*manifest_path.parts),
            subset_root.joinpath(*manifest_path.parts),
        )
    resolved_subset_root = subset_root.resolve()
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        try:
            resolved_candidate.relative_to(resolved_subset_root)
        except ValueError:
            continue
        if resolved_candidate.is_file():
            return resolved_candidate
    return None


def _safe_fallback_image_candidate(
    resolved_image_dir: Path,
    candidate: Path,
) -> Path | None:
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_image_dir)
    except ValueError:
        return None
    return resolved_candidate


def _resolve_e_vqa_image_path(
    root: Path,
    subset_root: Path,
    image_id: str,
    manifest_entry: Mapping[str, object] | None,
) -> Path:
    if manifest_entry is not None:
        manifest_path = _manifest_subset_image_path(
            root,
            subset_root,
            manifest_entry.get("subset_path"),
        )
        if manifest_path is not None:
            return manifest_path

    image_dir = subset_root / "images"
    resolved_image_dir = image_dir.resolve()
    missing_fallback: Path | None = None
    for suffix in E_VQA_IMAGE_EXTENSIONS:
        candidate = image_dir / f"{image_id}{suffix}"
        safe_candidate = _safe_fallback_image_candidate(
            resolved_image_dir,
            candidate,
        )
        if safe_candidate is None:
            continue
        if missing_fallback is None:
            missing_fallback = safe_candidate
        if safe_candidate.is_file():
            return safe_candidate
    if image_dir.is_dir():
        for candidate in sorted(image_dir.iterdir(), key=lambda path: path.name):
            safe_candidate = _safe_fallback_image_candidate(
                resolved_image_dir,
                candidate,
            )
            if (
                safe_candidate is not None
                and safe_candidate.is_file()
                and safe_candidate.stem == image_id
            ):
                return safe_candidate
    if missing_fallback is not None:
        return missing_fallback

    safe_missing = _safe_fallback_image_candidate(
        resolved_image_dir,
        image_dir / "__missing_image__.jpg",
    )
    if safe_missing is not None:
        return safe_missing
    raise ValueError("fallback image path could not be constrained inside images")


def _first_image_id(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    for separator in ("|", "&&", ",", ";"):
        if separator in text:
            text = text.split(separator)[0].strip()
    return text


def _full_row_image_id(row: Mapping[str, object]) -> tuple[str, str]:
    dataset_image_id = _first_image_id(row.get("dataset_image_ids"))
    if dataset_image_id:
        return dataset_image_id, "dataset_image_ids"
    fallback_image_id = _first_image_id(row.get("image_id"))
    if fallback_image_id:
        return fallback_image_id, "image_id"
    return "", "dataset_image_ids"


def _csv_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _is_missing_csv_value(value: object) -> bool:
    return _csv_text(value) == ""


def _row_context(row: Mapping[str, object]) -> list[str]:
    values: list[str] = []
    for key in ("wikipedia_title", "wikipedia_url", "evidence_section_title", "evidence"):
        value = _csv_text(row.get(key, ""))
        if value:
            values.append(value)
    return values


def _load_full_id2name_index(raw_root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    mapping_root = raw_root / "id2name"
    if not mapping_root.is_dir():
        return index
    for mapping_path in sorted(mapping_root.glob("*_id2name.json")):
        try:
            payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        for key, value in payload.items():
            image_id = _csv_text(key)
            relative_path = _csv_text(value)
            if image_id and relative_path:
                index.setdefault(image_id, relative_path)
    return index


def _load_full_image_stem_index(image_roots: Iterable[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for image_root in image_roots:
        if not image_root.is_dir():
            continue
        for candidate in image_root.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in E_VQA_IMAGE_EXTENSIONS:
                continue
            index.setdefault(candidate.stem, candidate)
            index.setdefault(candidate.name, candidate)
    return index


def _full_image_direct_candidates(
    *,
    image_roots: Iterable[Path],
    split: str,
    image_id: str,
    id2name: Mapping[str, str],
) -> Iterable[Path]:
    relative_values: list[str] = []
    mapped_path = id2name.get(image_id)
    if mapped_path:
        relative_values.append(mapped_path)
    relative_values.extend([image_id, f"{split}/{image_id}"])
    for extension in E_VQA_IMAGE_EXTENSIONS:
        relative_values.extend([f"{image_id}{extension}", f"{split}/{image_id}{extension}"])

    for image_root in image_roots:
        for relative_value in relative_values:
            relative_path = Path(relative_value)
            if relative_path.is_absolute():
                continue
            yield image_root / relative_path


def _resolve_full_image_path(
    *,
    image_roots: tuple[Path, ...],
    split: str,
    image_id: str,
    id2name: Mapping[str, str],
    image_stem_index: Mapping[str, Path],
) -> Path:
    for candidate in _full_image_direct_candidates(
        image_roots=image_roots,
        split=split,
        image_id=image_id,
        id2name=id2name,
    ):
        if candidate.is_file():
            return candidate
    indexed_candidate = image_stem_index.get(image_id)
    if indexed_candidate is not None:
        return indexed_candidate

    fallback_root = image_roots[0] if image_roots else Path("missing_full_images")
    mapped_path = id2name.get(image_id)
    if mapped_path:
        mapped_relative = Path(mapped_path)
        if not mapped_relative.is_absolute():
            return fallback_root / mapped_relative
    return fallback_root / f"{split}" / f"{image_id}.jpg"


def _empty_split_stats() -> dict[str, int]:
    return {
        "rows_read": 0,
        "rows_written": 0,
        "missing_csv": 0,
        "missing_required_fields": 0,
        "missing_images": 0,
        "missing_manifest_entries": 0,
        "empty_answers": 0,
        "validation_failures": 0,
    }


def _normalize_parquet_extra_info(extra_info: object) -> object:
    if not isinstance(extra_info, dict):
        return extra_info

    normalized = dict(extra_info)
    changed = False
    if normalized.get("source_row") == {}:
        normalized["source_row"] = None
        changed = True
    if normalized.get("original_metadata") == {}:
        normalized["original_metadata"] = None
        changed = True
    return normalized if changed else extra_info


def _ordered_subset_parquet_row(record: Mapping[str, object]) -> dict[str, object]:
    row = {column: record[column] for column in SUBSET_PARQUET_COLUMNS}
    row["extra_info"] = _normalize_parquet_extra_info(row.get("extra_info"))
    return row


def _process_e_vqa_split(
    *,
    root: Path,
    subset: str,
    subset_root: Path,
    processed_root: Path,
    split: str,
    csv_name: str,
    manifest_images: Mapping[str, Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int], list[dict[str, object]]]:
    del processed_root
    csv_path = subset_root / csv_name
    stats = _empty_split_stats()
    blockers: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []

    if not csv_path.is_file():
        stats["missing_csv"] = 1
        blockers.append(
            _blocker(
                dataset=DATASET_E_VQA,
                stage="metadata-only",
                message=f"missing CSV for E-VQA {split} split",
                details={"split": split, "source_csv": str(csv_path)},
            )
        )
        return rows, stats, blockers

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            stats["rows_read"] += 1
            missing_fields = [
                field
                for field in E_VQA_REQUIRED_CSV_FIELDS
                if field not in row or _is_missing_csv_value(row.get(field))
            ]
            if missing_fields:
                stats["missing_required_fields"] += 1
                if "answer" in missing_fields:
                    stats["empty_answers"] += 1
                blockers.append(
                    _blocker(
                        dataset=DATASET_E_VQA,
                        stage="metadata-only",
                        message="missing required E-VQA CSV field(s)",
                        details={
                            "split": split,
                            "row_index": idx,
                            "fields": missing_fields,
                            "source_csv": str(csv_path),
                        },
                    )
                )
                continue

            question = _csv_text(row.get("question", ""))
            raw_answer = row.get("answer")
            answers = parse_e_vqa_answers(raw_answer)
            if not answers:
                stats["empty_answers"] += 1
                blockers.append(
                    _blocker(
                        dataset=DATASET_E_VQA,
                        stage="metadata-only",
                        message="empty answers in E-VQA CSV row",
                        details={
                            "split": split,
                            "row_index": idx,
                            "source_csv": str(csv_path),
                        },
                    )
                )
                continue

            image_id = _first_image_id(row.get("dataset_image_ids"))
            if not image_id:
                stats["missing_required_fields"] += 1
                blockers.append(
                    _blocker(
                        dataset=DATASET_E_VQA,
                        stage="metadata-only",
                        message="missing required E-VQA image id",
                        details={
                            "split": split,
                            "row_index": idx,
                            "field": "dataset_image_ids",
                            "source_csv": str(csv_path),
                        },
                    )
                )
                continue

            manifest_entry = manifest_images.get(image_id)
            image_path = _resolve_e_vqa_image_path(
                root,
                subset_root,
                image_id,
                manifest_entry,
            )
            image_missing = not image_path.is_file()
            if image_missing:
                stats["missing_images"] += 1
                blockers.append(
                    _blocker(
                        dataset=DATASET_E_VQA,
                        stage="metadata-only",
                        message="missing image file for E-VQA CSV row",
                        details={
                            "split": split,
                            "row_index": idx,
                            "image_id": image_id,
                            "image_path": str(image_path),
                        },
                    )
                )

            if manifest_entry is None:
                stats["missing_manifest_entries"] += 1
                source_path = None
                row_index: object = idx
                original_metadata: dict[str, object] = {}
            else:
                source_path = (
                    None
                    if manifest_entry.get("source_path") is None
                    else str(manifest_entry.get("source_path"))
                )
                row_index = manifest_entry.get("row_index", idx)
                original_metadata = {
                    "sha256": manifest_entry.get("sha256"),
                    "manifest": dict(manifest_entry),
                }

            try:
                record = build_multimodal_qa_record(
                    data_source=f"{DATASET_E_VQA}/{subset}",
                    data_id=f"E-VQA:{subset}:{split}:{idx}:{image_id}",
                    dataset=DATASET_E_VQA,
                    split=split,
                    index=idx,
                    question=question,
                    question_original=_csv_text(
                        row.get("question_original")
                    )
                    or question,
                    answers=answers,
                    raw_answer=raw_answer,
                    image_id=image_id,
                    image_path=str(image_path),
                    image_missing=image_missing,
                    subset_root=str(subset_root),
                    source_path=source_path,
                    source_csv=csv_name,
                    row_index=row_index,
                    source_row=dict(row),
                    context=_row_context(row),
                    original_metadata=original_metadata,
                )
            except MultimodalSchemaError as exc:
                stats["validation_failures"] += 1
                blockers.append(
                    _blocker(
                        dataset=DATASET_E_VQA,
                        stage="metadata-only",
                        message="validation failure for E-VQA metadata-only row",
                        details={
                            "split": split,
                            "row_index": idx,
                            "image_id": image_id,
                            "error": str(exc),
                        },
                    )
                )
                continue

            rows.append(_ordered_subset_parquet_row(record))
            stats["rows_written"] += 1

    return rows, stats, blockers


def _load_full_split_rows(
    *,
    dataset: str,
    raw_root: Path,
    split_files: Mapping[str, str],
) -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, dict[str, int]],
    list[dict[str, object]],
]:
    split_rows: dict[str, list[dict[str, object]]] = {}
    statistics: dict[str, dict[str, int]] = {}
    blockers: list[dict[str, object]] = []

    for split, csv_name in split_files.items():
        csv_path = raw_root / csv_name
        stats = _empty_split_stats()
        if not csv_path.is_file():
            stats["missing_csv"] = 1
            blockers.append(
                _blocker(
                    dataset=dataset,
                    stage="metadata-only",
                    message=f"missing CSV for {dataset} {split} split",
                    details={"split": split, "source_csv": str(csv_path)},
                )
            )
            split_rows[split] = []
            statistics[split] = stats
            continue

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            loaded_rows = [dict(row) for row in reader]
        stats["rows_read"] = len(loaded_rows)
        split_rows[split] = loaded_rows
        statistics[split] = stats

    return split_rows, statistics, blockers


def _process_full_split(
    *,
    root_path: Path,
    dataset: str,
    raw_root: Path,
    split: str,
    csv_name: str,
    split_input_rows: list[dict[str, object]],
    stats: dict[str, int],
    image_roots: tuple[Path, ...],
    id2name: Mapping[str, str],
    image_stem_index: Mapping[str, Path],
) -> tuple[list[dict[str, object]], dict[str, int], list[dict[str, object]]]:
    blockers: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    csv_path = raw_root / csv_name

    if stats["missing_csv"]:
        return rows, stats, blockers

    for idx, row in enumerate(split_input_rows):
        missing_fields = [
            field
            for field in FULL_REQUIRED_CSV_FIELDS
            if field not in row or _is_missing_csv_value(row.get(field))
        ]
        if missing_fields:
            stats["missing_required_fields"] += 1
            if "answer" in missing_fields:
                stats["empty_answers"] += 1
            blockers.append(
                _blocker(
                    dataset=dataset,
                    stage="metadata-only",
                    message=f"missing required {dataset} CSV field(s)",
                    details={
                        "split": split,
                        "row_index": idx,
                        "fields": missing_fields,
                        "source_csv": str(csv_path),
                    },
                )
            )
            continue

        question = _csv_text(row.get("question", ""))
        raw_answer = row.get("answer")
        answers = parse_e_vqa_answers(raw_answer)
        if not answers:
            stats["empty_answers"] += 1
            blockers.append(
                _blocker(
                    dataset=dataset,
                    stage="metadata-only",
                    message=f"empty answers in {dataset} CSV row",
                    details={
                        "split": split,
                        "row_index": idx,
                        "source_csv": str(csv_path),
                    },
                )
            )
            continue

        image_id, image_field = _full_row_image_id(row)
        if not image_id:
            stats["missing_required_fields"] += 1
            blockers.append(
                _blocker(
                    dataset=dataset,
                    stage="metadata-only",
                    message=f"missing required {dataset} image id",
                    details={
                        "split": split,
                        "row_index": idx,
                        "field": image_field,
                        "source_csv": str(csv_path),
                    },
                )
            )
            continue

        image_path = _resolve_full_image_path(
            image_roots=image_roots,
            split=split,
            image_id=image_id,
            id2name=id2name,
            image_stem_index=image_stem_index,
        )
        image_missing = not image_path.is_file()
        if image_missing:
            stats["missing_images"] += 1
            blockers.append(
                _blocker(
                    dataset=dataset,
                    stage="metadata-only",
                    message=f"missing image file for {dataset} CSV row",
                    details={
                        "split": split,
                        "row_index": idx,
                        "image_id": image_id,
                        "image_path": str(image_path),
                    },
                )
            )

        try:
            record = build_multimodal_qa_record(
                data_source=f"{dataset}/full",
                data_id=_csv_text(row.get("data_id"))
                or f"{dataset}:full:{split}:{idx}:{image_id}",
                dataset=dataset,
                split=split,
                index=idx,
                question=question,
                question_original=_csv_text(row.get("question_original")) or question,
                answers=answers,
                raw_answer=raw_answer,
                image_id=image_id,
                image_path=str(image_path),
                image_missing=image_missing,
                subset_root=str(raw_root),
                source_path=str(csv_path),
                source_csv=csv_name,
                row_index=idx,
                source_row=dict(row),
                context=_row_context(row),
                original_metadata={
                    "source_mode": "full_raw",
                    "image_root_candidates": [str(path) for path in image_roots],
                },
            )
        except MultimodalSchemaError as exc:
            stats["validation_failures"] += 1
            blockers.append(
                _blocker(
                    dataset=dataset,
                    stage="metadata-only",
                    message=f"validation failure for {dataset} metadata-only row",
                    details={
                        "split": split,
                        "row_index": idx,
                        "image_id": image_id,
                        "error": str(exc),
                    },
                )
            )
            continue

        rows.append(_ordered_subset_parquet_row(record))
        stats["rows_written"] += 1

    return rows, stats, blockers


def _write_e_vqa_parquet(rows: list[dict[str, object]], output_path: Path) -> None:
    from datasets import Dataset

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output_path))


def _load_subset_manifest(subset_root: Path) -> dict[str, object]:
    manifest_path = subset_root / "manifest.json"
    if not manifest_path.is_file():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _infoseek_split_provenance_key(
    split: object,
    image_id: str,
) -> tuple[str, str] | None:
    split_text = _csv_text(split)
    image_id_text = _csv_text(image_id)
    if not split_text or not image_id_text:
        return None
    return (split_text, image_id_text)


def _infoseek_provenance_signature(candidate: Mapping[str, object]) -> tuple[str, str] | None:
    source_path = candidate.get("source_path")
    subset_path = candidate.get("subset_path")
    if source_path is not None:
        return ("source", str(source_path))
    if subset_path is not None:
        return ("subset", str(subset_path))
    return None


def _select_infoseek_canonical_provenance(
    candidates: list[dict[str, object]],
) -> dict[str, object] | None:
    if not candidates:
        return None
    valid_candidates = [
        candidate for candidate in candidates if candidate.get("subset_path") is not None
    ]
    if not valid_candidates:
        return None
    canonical = valid_candidates[0]
    canonical_signature = _infoseek_provenance_signature(canonical)
    if canonical_signature is None:
        return None

    for candidate in candidates:
        if candidate == canonical:
            continue
        if _infoseek_provenance_signature(candidate) != canonical_signature:
            return None
    return canonical


def _load_infoseek_subset_provenance(
    root: Path,
    subset_root: Path,
) -> tuple[
    dict[object, dict[str, object]],
    list[dict[str, object]],
]:
    manifest = _load_subset_manifest(subset_root)
    blockers: list[dict[str, object]] = []
    split_groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    fallback_groups: dict[str, list[dict[str, object]]] = {}

    for item in manifest.get("copied_images", []):
        if not isinstance(item, Mapping):
            continue
        image_id = str(item.get("image_id", "")).strip()
        if not image_id:
            continue
        manifest_path = _manifest_subset_image_path(
            root,
            subset_root,
            item.get("subset_path"),
        )
        candidate = {
            "image_id": image_id,
            "source_path": (
                None
                if item.get("source_path") is None
                else str(item.get("source_path")).strip() or None
            ),
            "subset_path": (
                None if manifest_path is None else str(manifest_path)
            ),
            "manifest": dict(item),
        }

        split_key = _infoseek_split_provenance_key(item.get("split"), image_id)
        if split_key is None:
            fallback_groups.setdefault(image_id, []).append(candidate)
            continue

        split_groups.setdefault(split_key, []).append(candidate)

    index: dict[object, dict[str, object]] = {}
    for split_key, candidates in split_groups.items():
        if len(candidates) == 1:
            index[split_key] = candidates[0]
            continue
        canonical = _select_infoseek_canonical_provenance(candidates)
        if canonical is not None:
            index[split_key] = canonical
            continue

        split, image_id = split_key
        blockers.append(
            _blocker(
                dataset=DATASET_INFOSEEK,
                stage="metadata-only",
                message="ambiguous InfoSeek subset provenance for image id",
                details={
                    "split": split,
                    "image_id": image_id,
                    "subset_root": str(subset_root),
                },
            )
        )

    for image_id, candidates in fallback_groups.items():
        if len(candidates) != 1:
            blockers.append(
                _blocker(
                    dataset=DATASET_INFOSEEK,
                    stage="metadata-only",
                    message="ambiguous InfoSeek legacy subset provenance for image id",
                    details={
                        "image_id": image_id,
                        "subset_root": str(subset_root),
                    },
                )
            )
            continue
        canonical = _select_infoseek_canonical_provenance(candidates)
        index[image_id] = canonical if canonical is not None else candidates[0]
    return index, blockers


def _process_infoseek_subset_split(
    *,
    dataset: str,
    subset: str,
    subset_root: Path,
    split: str,
    csv_name: str,
    split_input_rows: list[dict[str, object]],
    stats: dict[str, int],
    manifest_provenance: Mapping[object, Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int], list[dict[str, object]]]:
    blockers: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    csv_path = subset_root / csv_name

    if stats["missing_csv"]:
        return rows, stats, blockers

    for idx, row in enumerate(split_input_rows):
        missing_fields = [
            field
            for field in FULL_REQUIRED_CSV_FIELDS
            if field not in row or _is_missing_csv_value(row.get(field))
        ]
        if missing_fields:
            stats["missing_required_fields"] += 1
            if "answer" in missing_fields:
                stats["empty_answers"] += 1
            blockers.append(
                _blocker(
                    dataset=dataset,
                    stage="metadata-only",
                    message=f"missing required {dataset} CSV field(s)",
                    details={
                        "split": split,
                        "row_index": idx,
                        "fields": missing_fields,
                        "source_csv": str(csv_path),
                    },
                )
            )
            continue

        question = _csv_text(row.get("question", ""))
        raw_answer = row.get("answer")
        answers = parse_e_vqa_answers(raw_answer)
        if not answers:
            stats["empty_answers"] += 1
            blockers.append(
                _blocker(
                    dataset=dataset,
                    stage="metadata-only",
                    message=f"empty answers in {dataset} CSV row",
                    details={
                        "split": split,
                        "row_index": idx,
                        "source_csv": str(csv_path),
                    },
                )
            )
            continue

        image_id, image_field = _full_row_image_id(row)
        if not image_id:
            stats["missing_required_fields"] += 1
            blockers.append(
                _blocker(
                    dataset=dataset,
                    stage="metadata-only",
                    message=f"missing required {dataset} image id",
                    details={
                        "split": split,
                        "row_index": idx,
                        "field": image_field,
                        "source_csv": str(csv_path),
                    },
                )
            )
            continue

        split_key = _infoseek_split_provenance_key(split, image_id)
        provenance = (
            None
            if split_key is None
            else manifest_provenance.get(split_key)
        )
        if provenance is None:
            provenance = manifest_provenance.get(image_id)
        if provenance is None:
            stats["missing_manifest_entries"] += 1
            blockers.append(
                _blocker(
                    dataset=dataset,
                    stage="metadata-only",
                    message="missing InfoSeek subset provenance for image id",
                    details={
                        "split": split,
                        "row_index": idx,
                        "image_id": image_id,
                        "subset_root": str(subset_root),
                    },
                )
            )
            continue

        subset_image_path = provenance.get("subset_path")
        if subset_image_path is None:
            stats["missing_manifest_entries"] += 1
            blockers.append(
                _blocker(
                    dataset=dataset,
                    stage="metadata-only",
                    message="invalid InfoSeek subset provenance path",
                    details={
                        "split": split,
                        "row_index": idx,
                        "image_id": image_id,
                        "subset_root": str(subset_root),
                    },
                )
            )
            continue

        image_path = Path(str(subset_image_path))
        image_missing = not image_path.is_file()
        if image_missing:
            stats["missing_images"] += 1
            blockers.append(
                _blocker(
                    dataset=dataset,
                    stage="metadata-only",
                    message=f"missing image file for {dataset} CSV row",
                    details={
                        "split": split,
                        "row_index": idx,
                        "image_id": image_id,
                        "image_path": str(image_path),
                    },
                )
            )
            source_path = provenance.get("source_path")
        else:
            source_path = provenance.get("source_path")

        try:
            record = build_multimodal_qa_record(
                data_source=f"{dataset}/{subset}",
                data_id=_csv_text(row.get("data_id"))
                or f"{dataset}:{subset}:{split}:{idx}:{image_id}",
                dataset=dataset,
                split=split,
                index=idx,
                question=question,
                question_original=_csv_text(row.get("question_original")) or question,
                answers=answers,
                raw_answer=raw_answer,
                image_id=image_id,
                image_path=str(image_path),
                image_missing=image_missing,
                subset_root=str(subset_root),
                source_path=source_path,
                source_csv=csv_name,
                row_index=idx,
                source_row=dict(row),
                context=_row_context(row),
                original_metadata={"manifest": dict(provenance.get("manifest", {}))},
            )
        except MultimodalSchemaError as exc:
            stats["validation_failures"] += 1
            blockers.append(
                _blocker(
                    dataset=dataset,
                    stage="metadata-only",
                    message=f"validation failure for {dataset} metadata-only row",
                    details={
                        "split": split,
                        "row_index": idx,
                        "image_id": image_id,
                        "error": str(exc),
                    },
                )
            )
            continue

        rows.append(_ordered_subset_parquet_row(record))
        stats["rows_written"] += 1

    return rows, stats, blockers


def _process_full_metadata_only(
    *,
    root_path: Path,
    dataset: str,
    output_root: str | Path | None,
) -> dict[str, object]:
    raw_root = root_path / "datasets_mm" / dataset / "raw"
    processed_root = _echosight_processed_root(
        root_path,
        dataset,
        None,
        subset_mode="none",
        output_root=output_root,
    )
    split_files = (
        E_VQA_FULL_SPLIT_FILES
        if dataset == DATASET_E_VQA
        else INFOSEEK_FULL_SPLIT_FILES
    )
    split_input_rows, statistics, blockers = _load_full_split_rows(
        dataset=dataset,
        raw_root=raw_root,
        split_files=split_files,
    )
    image_roots = dataset_full_image_roots(root_path, dataset)
    id2name = _load_full_id2name_index(raw_root)
    image_stem_index = _load_full_image_stem_index(image_roots)

    split_rows: dict[str, list[dict[str, object]]] = {}
    split_counts: dict[str, int] = {}
    output_files: dict[str, str] = {}

    for split, csv_name in split_files.items():
        rows, stats, split_blockers = _process_full_split(
            root_path=root_path,
            dataset=dataset,
            raw_root=raw_root,
            split=split,
            csv_name=csv_name,
            split_input_rows=split_input_rows.get(split, []),
            stats=statistics[split],
            image_roots=image_roots,
            id2name=id2name,
            image_stem_index=image_stem_index,
        )
        split_rows[split] = rows
        split_counts[split] = len(rows)
        statistics[split] = stats
        if rows:
            output_path = processed_root / f"{split}.parquet"
            _write_e_vqa_parquet(rows, output_path)
            output_files[split] = str(output_path)
        blockers.extend(split_blockers)

    if not any(split_rows.values()):
        blockers.append(
            _blocker(
                dataset=dataset,
                stage="metadata-only",
                message=f"no complete {dataset} full rows were written",
                details={"raw_root": str(raw_root)},
            )
        )

    return {
        "dataset": dataset,
        "dataset_family": "echosight",
        "mode": "metadata_only",
        "schema_version": MM_SCHEMA_VERSION,
        "format": "parquet",
        "metadata_only": True,
        "subset_mode": "none",
        "subset": None,
        "subset_root": str(raw_root),
        "processed_root": str(processed_root),
        "split_counts": split_counts,
        "output_files": output_files,
        "statistics": statistics,
        "blockers": blockers,
        "policy": _metadata_only_policy(),
    }


def _process_subset_metadata_only(
    *,
    root_path: Path,
    dataset: str,
    subset: str,
    output_root: str | Path | None,
) -> dict[str, object]:
    if dataset == DATASET_INFOSEEK:
        return _process_infoseek_subset_metadata_only(
            root_path=root_path,
            dataset=dataset,
            subset=subset,
            output_root=output_root,
        )

    subset_root = _echosight_subset_root(root_path, dataset, subset)
    processed_root = _echosight_processed_root(
        root_path,
        dataset,
        subset,
        subset_mode="paper",
        output_root=output_root,
    )
    manifest_images = _load_manifest_image_index(subset_root)

    split_rows: dict[str, list[dict[str, object]]] = {}
    split_counts: dict[str, int] = {}
    statistics: dict[str, dict[str, int]] = {}
    output_files: dict[str, str] = {}
    blockers: list[dict[str, object]] = []

    for split, csv_name in E_VQA_SUBSET_SPLIT_FILES.items():
        rows, stats, split_blockers = _process_e_vqa_split(
            root=root_path,
            subset=subset,
            subset_root=subset_root,
            processed_root=processed_root,
            split=split,
            csv_name=csv_name,
            manifest_images=manifest_images,
        )
        split_rows[split] = rows
        split_counts[split] = len(rows)
        statistics[split] = stats
        if rows:
            output_path = processed_root / f"{split}.parquet"
            _write_e_vqa_parquet(rows, output_path)
            output_files[split] = str(output_path)
        blockers.extend(split_blockers)

    if not any(split_rows.values()):
        blockers.append(
            _blocker(
                dataset=dataset,
                stage="metadata-only",
                message="no complete E-VQA rows were written",
                details={"subset": subset, "subset_root": str(subset_root)},
            )
        )

    return {
        "dataset": dataset,
        "dataset_family": "echosight",
        "mode": "metadata_only",
        "schema_version": MM_SCHEMA_VERSION,
        "format": "parquet",
        "metadata_only": True,
        "subset_mode": "paper",
        "subset": subset,
        "subset_root": str(subset_root),
        "processed_root": str(processed_root),
        "split_counts": split_counts,
        "output_files": output_files,
        "statistics": statistics,
        "blockers": blockers,
        "policy": _metadata_only_policy(),
    }


def _process_infoseek_subset_metadata_only(
    *,
    root_path: Path,
    dataset: str,
    subset: str,
    output_root: str | Path | None,
) -> dict[str, object]:
    subset_root = _echosight_subset_root(root_path, dataset, subset)
    processed_root = _echosight_processed_root(
        root_path,
        dataset,
        subset,
        subset_mode="paper",
        output_root=output_root,
    )
    split_files = dict(E_VQA_SUBSET_SPLIT_FILES)
    manifest_provenance, provenance_blockers = _load_infoseek_subset_provenance(
        root_path,
        subset_root,
    )
    split_input_rows, statistics, blockers = _load_full_split_rows(
        dataset=dataset,
        raw_root=subset_root,
        split_files=split_files,
    )
    blockers.extend(provenance_blockers)

    split_rows: dict[str, list[dict[str, object]]] = {}
    split_counts: dict[str, int] = {}
    output_files: dict[str, str] = {}

    for split, csv_name in split_files.items():
        rows, stats, split_blockers = _process_infoseek_subset_split(
            dataset=dataset,
            subset=subset,
            subset_root=subset_root,
            split=split,
            csv_name=csv_name,
            split_input_rows=split_input_rows.get(split, []),
            stats=statistics[split],
            manifest_provenance=manifest_provenance,
        )
        split_rows[split] = rows
        split_counts[split] = len(rows)
        statistics[split] = stats
        if rows:
            output_path = processed_root / f"{split}.parquet"
            _write_e_vqa_parquet(rows, output_path)
            output_files[split] = str(output_path)
        blockers.extend(split_blockers)

    if not any(split_rows.values()):
        blockers.append(
            _blocker(
                dataset=dataset,
                stage="metadata-only",
                message=f"no complete {dataset} rows were written",
                details={"subset": subset, "subset_root": str(subset_root)},
            )
        )

    return {
        "dataset": dataset,
        "dataset_family": "echosight",
        "mode": "metadata_only",
        "schema_version": MM_SCHEMA_VERSION,
        "format": "parquet",
        "metadata_only": True,
        "subset_mode": "paper",
        "subset": subset,
        "subset_root": str(subset_root),
        "processed_root": str(processed_root),
        "split_counts": split_counts,
        "output_files": output_files,
        "statistics": statistics,
        "blockers": blockers,
        "policy": _metadata_only_policy(),
    }
def process_echosight_dataset(
    root: str | Path = ".",
    *,
    dataset: str = DATASET_E_VQA,
    subset: str | None = None,
    subset_mode: str = "paper",
    output_root: str | Path | None = None,
    output_format: str = "parquet",
    metadata_only: bool = False,
    audit_report: Mapping[str, object] | None = None,
    validation_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Process or report readiness for real EchoSight datasets."""

    if dataset not in {DATASET_E_VQA, DATASET_INFOSEEK}:
        raise ValueError(f"unsupported EchoSight dataset: {dataset}")
    if subset_mode not in {"none", "paper"}:
        raise ValueError(f"unsupported subset_mode: {subset_mode}")
    if (
        not metadata_only
        or (subset_mode == "paper" and subset is None)
    ):
        readiness = evaluate_echosight_readiness(
            root,
            datasets=[dataset],
            audit_report=audit_report,
            validation_report=validation_report,
        )
        return {
            "dataset": dataset,
            "dataset_family": "echosight",
            "mode": "readiness_only",
            "output_files": {},
            "readiness": readiness,
            "policy": readiness["policy"],
        }

    if output_format != "parquet":
        raise ValueError("EchoSight metadata-only preprocessing supports only parquet")

    if subset_mode == "none":
        return _process_full_metadata_only(
            root_path=Path(root),
            dataset=dataset,
            output_root=output_root,
        )

    return _process_subset_metadata_only(
        root_path=Path(root),
        dataset=dataset,
        subset=subset,
        output_root=output_root,
    )
