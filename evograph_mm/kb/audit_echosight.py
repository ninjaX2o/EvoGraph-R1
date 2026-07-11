"""Lightweight EchoSight/E-VQA/InfoSeek local asset audit.

The lightweight audit gathers representative evidence from files the user has already
placed under ``datasets_mm``. It never downloads assets, extracts archives, or
claims schema-freeze readiness.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

from .prepare_echosight import (
    DATASET_E_VQA,
    DATASET_INFOSEEK,
    NO_DOWNLOAD_POLICY,
    SUPPORTED_DATASETS,
    AssetDefinition,
    asset_catalog,
)

AUDIT_VERSION = "echosight-b2-lightweight-audit-v1"
FIELD_AUDIT_REPORT_PATH = "evograph_mm/kb/echosight-data-audit.md"
AUDIT_POLICY_NOTE = (
    "The lightweight audit only inspects local evidence from user-provided "
    "assets. It does not download, extract full archives, validate full "
    "payloads, or finalize runtime schemas."
)

ANSWER_FIELD_HINTS = (
    "answer",
    "answers",
    "answer_text",
    "gold_answer",
    "label",
    "entity",
    "entity_name",
)
SPLIT_FIELD_HINTS = ("split", "subset", "partition", "phase")
IMAGE_ID_FIELD_HINTS = (
    "image_id",
    "imageid",
    "img_id",
    "imgid",
    "image_uid",
    "image",
)
IMAGE_PATH_FIELD_HINTS = (
    "image_path",
    "image_name",
    "image_file",
    "file_name",
    "filename",
    "path",
    "url",
)
QUESTION_FIELD_HINTS = ("question", "query", "prompt")
TEXT_SAMPLE_SUFFIXES = {".json", ".jsonl", ".txt", ".csv", ".tsv"}
INDEX_SUFFIXES = {".faiss", ".index", ".idx"}


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
        raise ValueError(f"unsupported dataset(s): {unknown}; expected one of {supported}")
    return selected


def _repo_relative_path(root: Path, local_path: str) -> Path:
    return root.joinpath(*PurePosixPath(local_path).parts)


def _asset_exists(path: Path, expected_kind: str) -> bool:
    if expected_kind == "directory":
        return path.is_dir()
    if expected_kind == "file":
        return path.is_file()
    raise ValueError(f"unsupported expected_kind: {expected_kind}")


def _safe_preview(value: object, *, limit: int = 200) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, sort_keys=True, ensure_ascii=True)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _field_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _candidate_fields(field_names: Sequence[str], hints: Sequence[str]) -> list[str]:
    hint_keys = {_field_key(hint) for hint in hints}
    candidates: list[str] = []
    for field_name in field_names:
        key = _field_key(field_name)
        if key in hint_keys or any(hint in key for hint in hint_keys):
            candidates.append(field_name)
    return candidates


def _asset_status(asset: AssetDefinition, path: Path) -> dict[str, object]:
    exists = _asset_exists(path, asset.expected_kind)
    return {
        "dataset": asset.dataset,
        "asset_id": asset.asset_id,
        "asset_type": asset.asset_type,
        "local_path": asset.local_path,
        "expected_kind": asset.expected_kind,
        "required": asset.required,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
    }


def _missing_asset(asset: AssetDefinition) -> dict[str, object]:
    return {
        "dataset": asset.dataset,
        "asset_id": asset.asset_id,
        "asset_type": asset.asset_type,
        "local_path": asset.local_path,
        "expected_kind": asset.expected_kind,
        "required": asset.required,
        "source_filename": asset.source_filename,
        "manual_remediation_hint": asset.source_note,
        "reason": f"missing_{asset.expected_kind}",
    }


def _audit_csv_asset(asset: AssetDefinition, path: Path, sample_rows: int) -> dict[str, object]:
    report = _asset_status(asset, path)
    report.update(
        {
            "filename": Path(asset.local_path).name,
            "field_names": [],
            "sample_rows": [],
            "sampled_row_count": 0,
            "answer_field_candidates": [],
            "split_field_candidates": [],
            "image_id_field_candidates": [],
            "image_path_field_candidates": [],
            "question_field_candidates": [],
            "error": None,
        }
    )
    if not report["exists"]:
        return report

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            field_names = list(reader.fieldnames or [])
            rows: list[dict[str, str]] = []
            for row in reader:
                rows.append(
                    {
                        str(key): _safe_preview(value or "")
                        for key, value in row.items()
                        if key is not None
                    }
                )
                if len(rows) >= sample_rows:
                    break
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    report.update(
        {
            "field_names": field_names,
            "sample_rows": rows,
            "sampled_row_count": len(rows),
            "answer_field_candidates": _candidate_fields(field_names, ANSWER_FIELD_HINTS),
            "split_field_candidates": _candidate_fields(field_names, SPLIT_FIELD_HINTS),
            "image_id_field_candidates": _candidate_fields(field_names, IMAGE_ID_FIELD_HINTS),
            "image_path_field_candidates": _candidate_fields(field_names, IMAGE_PATH_FIELD_HINTS),
            "question_field_candidates": _candidate_fields(field_names, QUESTION_FIELD_HINTS),
        }
    )
    return report


def _json_type(value: object) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    return type(value).__name__


def _sample_json_pairs(value: object, sample_items: int) -> list[dict[str, object]]:
    if not isinstance(value, dict):
        return []
    pairs: list[dict[str, object]] = []
    for key in sorted(value, key=str)[:sample_items]:
        item = value[key]
        pairs.append(
            {
                "key": str(key),
                "value_type": _json_type(item) if isinstance(item, (dict, list)) else type(item).__name__,
                "value_preview": _safe_preview(item),
            }
        )
    return pairs


def _sample_json_items(value: object, sample_items: int) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, object]] = []
    for index, item in enumerate(value[:sample_items]):
        items.append(
            {
                "index": index,
                "value_type": _json_type(item) if isinstance(item, (dict, list)) else type(item).__name__,
                "value_preview": _safe_preview(item),
            }
        )
    return items


def _audit_json_mapping_asset(
    asset: AssetDefinition,
    path: Path,
    sample_items: int,
    max_json_bytes: int,
) -> dict[str, object]:
    report = _asset_status(asset, path)
    report.update(
        {
            "json_top_level_type": None,
            "entry_count": None,
            "sample_pairs": [],
            "sample_items": [],
            "mapping_strategy": "not_observed",
            "error": None,
        }
    )
    if not report["exists"]:
        return report

    size_bytes = int(report["size_bytes"] or 0)
    if size_bytes > max_json_bytes:
        report["error"] = (
            f"file_too_large_for_lightweight_json_parse: {size_bytes} bytes "
            f"> {max_json_bytes} bytes"
        )
        return report

    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    top_level_type = _json_type(value)
    entry_count = len(value) if isinstance(value, (dict, list)) else None
    sample_pairs = _sample_json_pairs(value, sample_items)
    sample_items_report = _sample_json_items(value, sample_items)

    mapping_strategy = "unknown_json_shape"
    if isinstance(value, dict):
        mapping_strategy = "json_object_key_to_value"
    elif (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) for item in value[:sample_items])
    ):
        mapping_strategy = "json_array_records"

    report.update(
        {
            "json_top_level_type": top_level_type,
            "entry_count": entry_count,
            "sample_pairs": sample_pairs,
            "sample_items": sample_items_report,
            "mapping_strategy": mapping_strategy,
        }
    )
    return report


def _zip_member_report(info: zipfile.ZipInfo) -> dict[str, object]:
    return {
        "filename": info.filename,
        "file_size": info.file_size,
        "compress_size": info.compress_size,
        "is_dir": info.is_dir(),
        "suffix": Path(info.filename).suffix.lower(),
    }


def _sample_zip_text(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    max_bytes: int,
) -> dict[str, object] | None:
    suffix = Path(info.filename).suffix.lower()
    if info.is_dir() or suffix not in TEXT_SAMPLE_SUFFIXES or info.file_size > max_bytes:
        return None

    with archive.open(info, "r") as handle:
        raw = handle.read(max_bytes)
    text = raw.decode("utf-8", errors="replace")
    sample: dict[str, object] = {
        "filename": info.filename,
        "suffix": suffix,
        "bytes_sampled": len(raw),
        "preview": _safe_preview(text),
    }
    if suffix == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            sample["json_top_level_type"] = "invalid_json_sample"
        else:
            sample["json_top_level_type"] = _json_type(parsed)
            if isinstance(parsed, dict):
                sample["json_keys"] = sorted(str(key) for key in parsed)[:10]
    return sample


def _audit_zip_asset(
    asset: AssetDefinition,
    path: Path,
    *,
    sample_members: int,
    sample_bytes: int,
) -> dict[str, object]:
    report = _asset_status(asset, path)
    report.update(
        {
            "archive_type": "zip",
            "member_count": 0,
            "total_uncompressed_size": 0,
            "total_compressed_size": 0,
            "sample_members": [],
            "small_text_samples": [],
            "index_candidate_members": [],
            "metadata_candidate_members": [],
            "error": None,
        }
    )
    if not report["exists"]:
        return report

    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = sorted(archive.infolist(), key=lambda info: info.filename)
            report["member_count"] = len(infos)
            report["total_uncompressed_size"] = sum(info.file_size for info in infos)
            report["total_compressed_size"] = sum(info.compress_size for info in infos)
            report["sample_members"] = [
                _zip_member_report(info)
                for info in infos[:sample_members]
            ]
            report["index_candidate_members"] = [
                info.filename
                for info in infos
                if Path(info.filename).suffix.lower() in INDEX_SUFFIXES
            ][:sample_members]
            report["metadata_candidate_members"] = [
                info.filename
                for info in infos
                if "metadata" in Path(info.filename).name.lower()
                or Path(info.filename).suffix.lower() in {".json", ".jsonl", ".csv", ".tsv"}
            ][:sample_members]

            text_samples: list[dict[str, object]] = []
            for info in infos:
                sample = _sample_zip_text(archive, info, sample_bytes)
                if sample is not None:
                    text_samples.append(sample)
                if len(text_samples) >= sample_members:
                    break
            report["small_text_samples"] = text_samples
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _audit_directory_asset(
    asset: AssetDefinition,
    path: Path,
    sample_entries: int,
) -> dict[str, object]:
    report = _asset_status(asset, path)
    report.update(
        {
            "sample_entries": [],
            "sampled_entry_count": 0,
            "error": None,
        }
    )
    if not report["exists"]:
        return report

    try:
        entries = sorted(path.iterdir(), key=lambda entry: entry.name)[:sample_entries]
        sample = [
            {
                "name": entry.name,
                "is_dir": entry.is_dir(),
                "suffix": entry.suffix.lower(),
                "size_bytes": entry.stat().st_size if entry.is_file() else None,
            }
            for entry in entries
        ]
    except OSError as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    report["sample_entries"] = sample
    report["sampled_entry_count"] = len(sample)
    return report


def _audit_faiss_directory(raw_root: Path, sample_entries: int) -> dict[str, object]:
    faiss_root = raw_root / "faiss"
    report: dict[str, object] = {
        "local_path": faiss_root.as_posix(),
        "exists": faiss_root.is_dir(),
        "index_files": [],
        "metadata_files": [],
    }
    if not faiss_root.is_dir():
        return report

    index_files: list[dict[str, object]] = []
    metadata_files: list[dict[str, object]] = []
    for path in sorted(faiss_root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(faiss_root).as_posix()
        suffix = path.suffix.lower()
        entry = {
            "relative_path": relative,
            "suffix": suffix,
            "size_bytes": path.stat().st_size,
        }
        if suffix in INDEX_SUFFIXES:
            index_files.append(entry)
        elif "metadata" in path.name.lower() or suffix in {".json", ".jsonl", ".csv", ".tsv"}:
            metadata_files.append(entry)
        if len(index_files) >= sample_entries and len(metadata_files) >= sample_entries:
            break

    report["index_files"] = index_files[:sample_entries]
    report["metadata_files"] = metadata_files[:sample_entries]
    return report


def _merge_candidates(reports: Sequence[dict[str, object]], key: str) -> list[str]:
    merged: list[str] = []
    for report in reports:
        for field in report.get(key, []):
            if isinstance(field, str) and field not in merged:
                merged.append(field)
    return merged


def _has_kb_sample_evidence(reports: Sequence[dict[str, object]]) -> bool:
    return any(
        report.get("exists")
        and not report.get("error")
        and int(report.get("member_count") or 0) > 0
        and bool(report.get("small_text_samples"))
        for report in reports
    )


def _has_faiss_metadata_evidence(reports: Sequence[dict[str, object]]) -> bool:
    return any(
        report.get("exists")
        and not report.get("error")
        and int(report.get("member_count") or 0) > 0
        and bool(report.get("index_candidate_members"))
        and bool(report.get("metadata_candidate_members"))
        for report in reports
    )


def _has_id_mapping_evidence(report: dict[str, object]) -> bool:
    if not report.get("exists") or report.get("error"):
        return False
    if not int(report.get("entry_count") or 0) > 0:
        return False
    strategy = report.get("mapping_strategy")
    if strategy == "json_object_key_to_value":
        return bool(report.get("sample_pairs"))
    if strategy == "json_array_records":
        return bool(report.get("sample_items"))
    return False


def _unreadable_asset_blockers(reports: Sequence[dict[str, object]]) -> list[str]:
    blockers: list[str] = []
    for report in reports:
        if report.get("exists") and report.get("error"):
            blockers.append(
                f"{report.get('asset_id', 'unknown_asset')} unreadable at "
                f"{report.get('local_path', 'unknown_path')}: {report['error']}"
            )
    return blockers


def _dataset_blockers(dataset_report: dict[str, object]) -> list[str]:
    blockers: list[str] = []
    for missing in dataset_report["missing_assets"]:
        blockers.append(
            f"{missing['asset_id']} missing at {missing['local_path']}"
        )

    if not dataset_report["answer_field_candidates"]:
        blockers.append("no answer field candidate observed in present VQA CSV headers")
    if not dataset_report["image_id_field_candidates"]:
        blockers.append("no image id field candidate observed in present VQA CSV headers")
    if not dataset_report["image_id_path_mapping_strategy_evidence"]["id_mapping_assets"]:
        blockers.append("no id mapping JSON shape evidence observed")
    blockers.extend(_unreadable_asset_blockers(dataset_report["kb_sample_evidence"]))
    if not _has_kb_sample_evidence(dataset_report["kb_sample_evidence"]):
        blockers.append("no readable KB archive sample evidence observed")
    blockers.extend(_unreadable_asset_blockers(dataset_report["faiss_metadata_evidence"]))
    if not _has_faiss_metadata_evidence(dataset_report["faiss_metadata_evidence"]):
        blockers.append("no readable FAISS archive or index metadata evidence observed")
    return blockers


def _dataset_audit(
    root_path: Path,
    dataset: str,
    *,
    sample_rows: int,
    sample_items: int,
    zip_sample_members: int,
    zip_sample_bytes: int,
    image_sample_entries: int,
    max_json_bytes: int,
) -> dict[str, object]:
    assets = asset_catalog([dataset], required_only=True)
    raw_root = _repo_relative_path(root_path, f"datasets_mm/{dataset}/raw")

    inspected_files: list[dict[str, object]] = []
    missing_assets: list[dict[str, object]] = []
    vqa_csvs: list[dict[str, object]] = []
    id_mappings: list[dict[str, object]] = []
    kb_archives: list[dict[str, object]] = []
    faiss_archives: list[dict[str, object]] = []
    image_sources: list[dict[str, object]] = []

    for asset in assets:
        path = _repo_relative_path(root_path, asset.local_path)
        status = _asset_status(asset, path)
        inspected_files.append(status)
        if asset.required and not status["exists"]:
            missing_assets.append(_missing_asset(asset))

        if asset.asset_type == "vqa_csv":
            vqa_csvs.append(_audit_csv_asset(asset, path, sample_rows))
        elif asset.asset_type == "id_mapping":
            id_mappings.append(
                _audit_json_mapping_asset(asset, path, sample_items, max_json_bytes)
            )
        elif asset.asset_type == "kb_archive":
            kb_archives.append(
                _audit_zip_asset(
                    asset,
                    path,
                    sample_members=zip_sample_members,
                    sample_bytes=zip_sample_bytes,
                )
            )
        elif asset.asset_type == "faiss_archive":
            faiss_archives.append(
                _audit_zip_asset(
                    asset,
                    path,
                    sample_members=zip_sample_members,
                    sample_bytes=zip_sample_bytes,
                )
            )
        elif asset.asset_type == "image_source":
            image_sources.append(_audit_directory_asset(asset, path, image_sample_entries))

    observed_field_names = {
        report["filename"]: report["field_names"]
        for report in vqa_csvs
        if report.get("exists")
    }
    answer_candidates = _merge_candidates(vqa_csvs, "answer_field_candidates")
    split_candidates = _merge_candidates(vqa_csvs, "split_field_candidates")
    image_id_candidates = _merge_candidates(vqa_csvs, "image_id_field_candidates")
    image_path_candidates = _merge_candidates(vqa_csvs, "image_path_field_candidates")

    dataset_report: dict[str, object] = {
        "dataset": dataset,
        "raw_root": f"datasets_mm/{dataset}/raw",
        "field_audit_report_path": FIELD_AUDIT_REPORT_PATH,
        "inspected_files": inspected_files,
        "missing_assets": missing_assets,
        "observed_field_names": observed_field_names,
        "vqa_csvs": vqa_csvs,
        "id_mappings": id_mappings,
        "kb_sample_evidence": kb_archives,
        "faiss_metadata_evidence": faiss_archives,
        "faiss_directory_evidence": _audit_faiss_directory(raw_root, image_sample_entries),
        "image_sources": image_sources,
        "answer_field_candidates": answer_candidates,
        "split_field_candidates": split_candidates,
        "image_id_field_candidates": image_id_candidates,
        "image_path_field_candidates": image_path_candidates,
        "image_id_path_mapping_strategy_evidence": {
            "csv_image_id_fields": image_id_candidates,
            "csv_image_path_fields": image_path_candidates,
            "id_mapping_assets": [
                {
                    "asset_id": report["asset_id"],
                    "local_path": report["local_path"],
                    "json_top_level_type": report["json_top_level_type"],
                    "entry_count": report["entry_count"],
                    "mapping_strategy": report["mapping_strategy"],
                    "sample_pairs": report["sample_pairs"],
                    "sample_items": report["sample_items"],
                }
                for report in id_mappings
                if _has_id_mapping_evidence(report)
            ],
        },
    }

    representative_blockers = _dataset_blockers(dataset_report)
    representative_complete = not representative_blockers
    schema_freeze_blocker = (
        "metadata-only runtime schema review is not ready from B2 alone; this report is "
        "lightweight evidence and must be reviewed against real representative assets."
    )
    dataset_report["task_c_readiness"] = {
        "representative_real_audit_complete": representative_complete,
        "schema_freeze_ready": False,
        "blockers": representative_blockers + [schema_freeze_blocker],
    }
    return dataset_report


def audit_echosight_assets(
    root: str | Path = ".",
    *,
    datasets: Iterable[str] | str | None = None,
    sample_rows: int = 3,
    sample_items: int = 3,
    zip_sample_members: int = 10,
    zip_sample_bytes: int = 4096,
    image_sample_entries: int = 10,
    max_json_bytes: int = 8 * 1024 * 1024,
) -> dict[str, object]:
    """Return a JSON-serializable B2 lightweight local asset audit report."""

    selected = _normalize_datasets(datasets)
    root_path = Path(root)
    dataset_reports = [
        _dataset_audit(
            root_path,
            dataset,
            sample_rows=sample_rows,
            sample_items=sample_items,
            zip_sample_members=zip_sample_members,
            zip_sample_bytes=zip_sample_bytes,
            image_sample_entries=image_sample_entries,
            max_json_bytes=max_json_bytes,
        )
        for dataset in selected
    ]

    blockers: list[str] = []
    for dataset_report in dataset_reports:
        blockers.extend(
            f"{dataset_report['dataset']}: {blocker}"
            for blocker in dataset_report["task_c_readiness"]["blockers"]
        )

    representative_complete = all(
        dataset_report["task_c_readiness"]["representative_real_audit_complete"]
        for dataset_report in dataset_reports
    )
    return {
        "audit_version": AUDIT_VERSION,
        "field_audit_report_path": FIELD_AUDIT_REPORT_PATH,
        "checked_root": str(root_path),
        "datasets": list(selected),
        "policy": {
            "downloads_enabled": False,
            "extract_archives": False,
            "lightweight_only": True,
            "schema_freeze": False,
            "note": f"{NO_DOWNLOAD_POLICY} {AUDIT_POLICY_NOTE}",
        },
        "dataset_reports": dataset_reports,
        "task_c_readiness": {
            "representative_real_audit_complete": representative_complete,
            "schema_freeze_ready": False,
            "blockers": blockers,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit local EchoSight/E-VQA/InfoSeek assets without downloads.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository or staging root containing datasets_mm/.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=(DATASET_E_VQA, DATASET_INFOSEEK),
        help="Dataset to audit. May be provided more than once.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=3,
        help="Maximum CSV rows to sample from each present VQA CSV.",
    )
    parser.add_argument(
        "--sample-items",
        type=int,
        default=3,
        help="Maximum JSON mapping items to include as shape evidence.",
    )
    parser.add_argument(
        "--zip-sample-members",
        type=int,
        default=10,
        help="Maximum ZIP members to include as archive metadata evidence.",
    )
    parser.add_argument(
        "--zip-sample-bytes",
        type=int,
        default=4096,
        help="Maximum bytes to read from small text/JSON ZIP members.",
    )
    parser.add_argument(
        "--image-sample-entries",
        type=int,
        default=10,
        help="Maximum image source directory entries to list.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = audit_echosight_assets(
        args.root,
        datasets=args.dataset or None,
        sample_rows=args.sample_rows,
        sample_items=args.sample_items,
        zip_sample_members=args.zip_sample_members,
        zip_sample_bytes=args.zip_sample_bytes,
        image_sample_entries=args.image_sample_entries,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
