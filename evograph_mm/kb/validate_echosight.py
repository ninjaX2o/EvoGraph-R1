"""Full local EchoSight/E-VQA/InfoSeek asset validation.

The validator only inspects files that are already present under
``datasets_mm``. It never downloads assets, extracts archives, or writes
payload files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import deque
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

from PIL import Image, UnidentifiedImageError

from .full_image_layout import candidate_full_image_roots
from .prepare_echosight import (
    DATASET_E_VQA,
    DATASET_INFOSEEK,
    NO_DOWNLOAD_POLICY,
    SUPPORTED_DATASETS,
    AssetDefinition,
    asset_catalog,
)

VALIDATION_VERSION = "echosight-b3-full-local-validation-v1"
VALIDATION_POLICY_NOTE = (
    "Validation only inspects user-supplied local EchoSight assets. It does not "
    "download files, extract full archives, or create payload artifacts."
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
INDEX_SUFFIXES = {".faiss", ".index", ".idx"}
METADATA_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv"}
PICKLE_METADATA_SUFFIXES = {".pkl", ".pickle"}
DOWNLOAD_BLOCKING_STATUSES = {"download_failed", "insufficient_disk"}
SUBSET_MANIFEST_VERSION = "echosight-paper-subset-v1"
TAR_SUBSET_MANIFEST_VERSION = "echosight-tar-subset-v1"
LOCAL_INFOSEEK_SUBSET_MANIFEST_VERSION = "infoseek-local-subset-v1"
SUPPORTED_SUBSET_MANIFEST_VERSIONS = {
    LOCAL_INFOSEEK_SUBSET_MANIFEST_VERSION,
    SUBSET_MANIFEST_VERSION,
    TAR_SUBSET_MANIFEST_VERSION,
}


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


def _root_relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        pass
    try:
        return path.resolve(strict=False).relative_to(
            root.resolve(strict=False),
        ).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_for_validation_root(root_path: Path, candidate_path: Path) -> dict[str, object]:
    root_resolved = root_path.resolve(strict=False)
    candidate_resolved = candidate_path.resolve(strict=False)
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError:
        return {
            "inside": False,
            "root_resolved": root_resolved,
            "candidate_resolved": candidate_resolved,
        }
    return {
        "inside": True,
        "root_resolved": root_resolved,
        "candidate_resolved": candidate_resolved,
    }


def _actual_kind(path: Path) -> str:
    if path.is_file():
        return "file"
    if path.is_dir():
        return "directory"
    if path.exists():
        return "other"
    return "missing"


def _asset_exists(path: Path, expected_kind: str) -> bool:
    if expected_kind == "directory":
        return path.is_dir()
    if expected_kind == "file":
        return path.is_file()
    raise ValueError(f"unsupported expected_kind: {expected_kind}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _summary_split_count(summary: Mapping[str, object], split: str) -> int | None:
    selected_by_split = summary.get("selected_by_split")
    if not isinstance(selected_by_split, Mapping):
        return None
    return _safe_int(selected_by_split.get(split))


def _safe_sha256(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if len(normalized) == 64 and all(ch in "0123456789abcdef" for ch in normalized):
        return normalized
    return None


def _download_sidecar_path(path: Path, expected_kind: str) -> Path:
    if expected_kind == "directory":
        return path / ".download.json"
    return path.with_suffix(path.suffix + ".download.json")


def _download_sidecar_local_path(asset: AssetDefinition) -> str:
    if asset.expected_kind == "directory":
        return f"{asset.local_path.rstrip('/')}/.download.json"
    return f"{asset.local_path}.download.json"


def _read_download_sidecar(
    asset: AssetDefinition,
    path: Path,
    *,
    root_path: Path,
    max_json_bytes: int,
) -> dict[str, object]:
    sidecar = _download_sidecar_path(path, asset.expected_kind)
    report: dict[str, object] = {
        "local_path": _download_sidecar_local_path(asset),
        "exists": False,
        "inside_validation_root": None,
        "resolved_path": None,
        "readable": False,
        "valid_json": False,
        "status": None,
        "blocking": False,
        "dataset": None,
        "asset_id": None,
        "bytes_downloaded": None,
        "source_url": None,
        "remediation": None,
        "error": None,
        "required_bytes": None,
        "free_bytes": None,
        "subset_mode": None,
        "requested_root": None,
        "probed_path": None,
        "payload": None,
    }
    if not sidecar.exists():
        return report

    containment = _resolve_for_validation_root(root_path, sidecar)
    report["inside_validation_root"] = containment["inside"]
    report["resolved_path"] = str(containment["candidate_resolved"])
    if not containment["inside"]:
        report.update(
            {
                "exists": True,
                "status": "download_sidecar_outside_validation_root",
                "blocking": True,
                "error": (
                    "download sidecar resolves outside validation root: "
                    f"{containment['candidate_resolved']}"
                ),
            }
        )
        return report

    report["exists"] = sidecar.is_file()
    if not sidecar.is_file():
        report.update(
            {
                "exists": False,
                "status": "download_sidecar_not_a_file",
                "blocking": True,
                "error": "download sidecar path exists but is not a file",
            }
        )
        return report

    try:
        if sidecar.stat().st_size > max_json_bytes:
            report.update(
                {
                    "status": "download_sidecar_unreadable",
                    "blocking": True,
                    "error": (
                        f"download sidecar too large to parse: "
                        f"{sidecar.stat().st_size} bytes > {max_json_bytes} bytes"
                    ),
                }
            )
            return report
        document = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        report.update(
            {
                "status": "download_sidecar_unreadable",
                "blocking": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return report

    if not isinstance(document, Mapping):
        report.update(
            {
                "readable": True,
                "status": "download_sidecar_invalid",
                "blocking": True,
                "error": "download sidecar top-level JSON is not an object",
                "payload": document,
            }
        )
        return report

    status = document.get("status")
    normalized_status = status if isinstance(status, str) and status else None
    report.update(
        {
            "readable": True,
            "valid_json": True,
            "status": normalized_status,
            "blocking": normalized_status in DOWNLOAD_BLOCKING_STATUSES,
            "dataset": document.get("dataset"),
            "asset_id": document.get("asset_id"),
            "bytes_downloaded": document.get("bytes_downloaded"),
            "source_url": document.get("source_url"),
            "remediation": document.get("remediation"),
            "error": document.get("error"),
            "required_bytes": document.get("required_bytes"),
            "free_bytes": document.get("free_bytes"),
            "subset_mode": document.get("subset_mode"),
            "requested_root": document.get("requested_root"),
            "probed_path": document.get("probed_path"),
            "payload": dict(document),
        }
    )
    if normalized_status is None:
        report.update(
            {
                "status": "download_sidecar_invalid",
                "blocking": True,
                "error": "download sidecar does not contain a non-empty status",
            }
        )
    return report


def _apply_download_sidecar(
    report: dict[str, object],
    sidecar: Mapping[str, object],
) -> None:
    report["download_sidecar"] = dict(sidecar)
    if not sidecar.get("blocking"):
        return

    status = str(sidecar.get("status") or "download_sidecar_unreadable")
    report["status"] = status
    reason = (
        f"{report['asset_id']} download sidecar reports {status} at "
        f"{sidecar['local_path']}"
    )
    if sidecar.get("error"):
        reason = f"{reason}: {sidecar['error']}"
    report["blockers"].append(reason)


def _as_metadata_entries(document: object) -> list[dict[str, object]]:
    if not isinstance(document, Mapping):
        return []

    raw_assets = document.get("assets")
    if isinstance(raw_assets, list):
        return [dict(item) for item in raw_assets if isinstance(item, Mapping)]

    if isinstance(raw_assets, Mapping):
        entries: list[dict[str, object]] = []
        for asset_id, value in raw_assets.items():
            if isinstance(value, Mapping):
                entry = dict(value)
            else:
                entry = {"value": value}
            entry.setdefault("asset_id", str(asset_id))
            entries.append(entry)
        return entries

    entries = []
    for key, value in document.items():
        if key in {"manifest_version", "datasets", "policy", "local_roots"}:
            continue
        if isinstance(value, Mapping):
            entry = dict(value)
            entry.setdefault("asset_id", str(key))
            entries.append(entry)
    return entries


def _metadata_report(
    *,
    root_path: Path,
    raw_root: Path,
    dataset: str,
    filename: str,
    assets: Sequence[AssetDefinition],
    recognized_assets: Sequence[AssetDefinition] | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    local_path = f"datasets_mm/{dataset}/raw/{filename}"
    path = raw_root / filename
    report: dict[str, object] = {
        "local_path": local_path,
        "exists": False,
        "inside_validation_root": None,
        "resolved_path": None,
        "blocking": False,
        "status": "not_yet_supplied",
        "valid_json": False,
        "asset_count": 0,
        "catalog_missing_asset_ids": [],
        "extra_asset_ids": [],
        "duplicate_asset_ids": [],
        "path_mismatches": [],
        "kind_mismatches": [],
        "error": None,
    }
    containment = _resolve_for_validation_root(root_path, path)
    report["inside_validation_root"] = containment["inside"]
    report["resolved_path"] = str(containment["candidate_resolved"])
    if not containment["inside"]:
        report.update(
            {
                "status": "metadata_outside_validation_root",
                "blocking": True,
                "error": (
                    "metadata file resolves outside validation root: "
                    f"{containment['candidate_resolved']}"
                ),
            }
        )
        return report, {}

    if not path.exists():
        return report, {}
    report["exists"] = path.is_file()
    if not path.is_file():
        report["exists"] = False
        report["status"] = "incomplete"
        report["blocking"] = True
        report["error"] = "metadata_path_is_not_a_file"
        return report, {}

    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        report["status"] = "incomplete"
        report["blocking"] = True
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report, {}

    entries_by_id: dict[str, dict[str, object]] = {}
    duplicate_asset_ids: list[str] = []
    for entry in _as_metadata_entries(document):
        asset_id = entry.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            continue
        if asset_id in entries_by_id:
            duplicate_asset_ids.append(asset_id)
            continue
        entries_by_id[asset_id] = entry

    catalog_by_id = {asset.asset_id: asset for asset in assets}
    recognized_catalog_by_id = {
        asset.asset_id: asset
        for asset in (recognized_assets if recognized_assets is not None else assets)
    }
    missing_ids = sorted(asset_id for asset_id in catalog_by_id if asset_id not in entries_by_id)
    extra_ids = sorted(
        asset_id for asset_id in entries_by_id if asset_id not in recognized_catalog_by_id
    )
    path_mismatches: list[dict[str, object]] = []
    kind_mismatches: list[dict[str, object]] = []

    for asset_id, asset in catalog_by_id.items():
        entry = entries_by_id.get(asset_id)
        if entry is None:
            continue
        entry_path = entry.get("local_path")
        if isinstance(entry_path, str) and entry_path != asset.local_path:
            path_mismatches.append(
                {
                    "asset_id": asset_id,
                    "expected": asset.local_path,
                    "observed": entry_path,
                }
            )
        entry_kind = entry.get("expected_kind")
        if isinstance(entry_kind, str) and entry_kind != asset.expected_kind:
            kind_mismatches.append(
                {
                    "asset_id": asset_id,
                    "expected": asset.expected_kind,
                    "observed": entry_kind,
                }
            )

    blockers = missing_ids or extra_ids or duplicate_asset_ids or path_mismatches or kind_mismatches
    report.update(
        {
            "status": "incomplete" if blockers else "complete",
            "blocking": bool(blockers),
            "valid_json": True,
            "asset_count": len(entries_by_id),
            "catalog_missing_asset_ids": missing_ids,
            "extra_asset_ids": extra_ids,
            "duplicate_asset_ids": sorted(duplicate_asset_ids),
            "path_mismatches": path_mismatches,
            "kind_mismatches": kind_mismatches,
        }
    )
    return report, entries_by_id


def _expected_metadata(
    asset: AssetDefinition,
    manifest_entries: Mapping[str, Mapping[str, object]],
    checksum_entries: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    manifest_entry = manifest_entries.get(asset.asset_id, {})
    checksum_entry = checksum_entries.get(asset.asset_id, {})

    expected_size = _safe_int(checksum_entry.get("size_bytes"))
    if expected_size is None:
        expected_size = _safe_int(manifest_entry.get("size_bytes"))
    if expected_size is None:
        expected_size = asset.size_bytes

    expected_sha = _safe_sha256(checksum_entry.get("sha256"))
    if expected_sha is None:
        expected_sha = _safe_sha256(manifest_entry.get("sha256"))
    if expected_sha is None:
        expected_sha = _safe_sha256(asset.sha256)

    checksum_source = None
    if expected_sha is not None:
        checksum_source = (
            "checksums.json"
            if _safe_sha256(checksum_entry.get("sha256")) is not None
            else "manifest.json"
        )

    return {
        "manifest_entry_present": bool(manifest_entry),
        "checksum_entry_present": bool(checksum_entry),
        "expected_size_bytes": expected_size,
        "expected_sha256": expected_sha,
        "checksum_source": checksum_source,
    }


def _zip_member_report(info: zipfile.ZipInfo) -> dict[str, object]:
    return {
        "filename": info.filename,
        "file_size": info.file_size,
        "compress_size": info.compress_size,
        "is_dir": info.is_dir(),
        "suffix": Path(info.filename).suffix.lower(),
    }


def _validate_zip(path: Path, *, sample_members: int) -> dict[str, object]:
    report: dict[str, object] = {
        "readable": False,
        "member_count": 0,
        "total_uncompressed_size": 0,
        "total_compressed_size": 0,
        "sample_members": [],
        "index_candidate_members": [],
        "metadata_candidate_members": [],
        "error": None,
    }
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = sorted(archive.infolist(), key=lambda item: item.filename)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    index_candidates = [
        info.filename
        for info in infos
        if not info.is_dir() and Path(info.filename).suffix.lower() in INDEX_SUFFIXES
    ]
    metadata_candidates = []
    for info in infos:
        if info.is_dir():
            continue
        name = Path(info.filename).name.lower()
        suffix = Path(info.filename).suffix.lower()
        has_metadata_name = "metadata" in name or "ids" in name
        if (
            has_metadata_name
            or suffix in METADATA_SUFFIXES
            or (suffix in PICKLE_METADATA_SUFFIXES and has_metadata_name)
        ):
            metadata_candidates.append(info.filename)
    report.update(
        {
            "readable": True,
            "member_count": len(infos),
            "total_uncompressed_size": sum(info.file_size for info in infos),
            "total_compressed_size": sum(info.compress_size for info in infos),
            "sample_members": [_zip_member_report(info) for info in infos[:sample_members]],
            "index_candidate_members": index_candidates[:sample_members],
            "metadata_candidate_members": metadata_candidates[:sample_members],
        }
    )
    return report


def _validate_csv(path: Path, *, sample_rows: int) -> dict[str, object]:
    report: dict[str, object] = {
        "readable": False,
        "field_names": [],
        "sampled_row_count": 0,
        "sample_rows": [],
        "error": None,
    }
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            field_names = list(reader.fieldnames or [])
            rows: list[dict[str, str]] = []
            for row in reader:
                rows.append({str(key): value or "" for key, value in row.items() if key})
                if len(rows) >= sample_rows:
                    break
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    report.update(
        {
            "readable": True,
            "field_names": field_names,
            "sampled_row_count": len(rows),
            "sample_rows": rows,
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


def _validate_json_mapping(path: Path, *, max_json_bytes: int) -> dict[str, object]:
    size_bytes = path.stat().st_size
    report: dict[str, object] = {
        "readable": False,
        "status": "not_checked",
        "json_top_level_type": None,
        "entry_count": None,
        "size_bytes": size_bytes,
        "max_json_bytes": max_json_bytes,
        "error": None,
    }
    if size_bytes > max_json_bytes:
        report.update(
            {
                "status": "parse_skipped_size_limit",
                "error": (
                    "file_too_large_for_json_parse: "
                    f"{size_bytes} bytes > {max_json_bytes} bytes"
                ),
            }
        )
        return report

    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        report["status"] = "invalid"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    except (OSError, UnicodeDecodeError) as exc:
        report["status"] = "unreadable"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    report.update(
        {
            "readable": True,
            "status": "parsed",
            "json_top_level_type": _json_type(value),
            "entry_count": len(value) if isinstance(value, (dict, list)) else None,
        }
    )
    return report


def _empty_directory_report() -> dict[str, object]:
    return {
        "sample_entries": [],
        "sampled_entry_count": 0,
        "sampled_image_count": 0,
        "readable_sampled_image_count": 0,
        "invalid_sample_paths": [],
        "error": None,
    }


def _image_source_dir_names(dataset: str) -> set[str]:
    return {
        PurePosixPath(asset.local_path).name
        for asset in asset_catalog([dataset], include_optional=True)
        if asset.asset_type == "image_source"
    }


def _accept_parent_image_root_candidate(
    candidate: Path,
    *,
    asset: AssetDefinition,
    default_path: Path,
) -> bool:
    if candidate != default_path.parent:
        return True

    image_source_dir_names = _image_source_dir_names(asset.dataset)
    try:
        children = sorted(candidate.iterdir(), key=lambda item: item.name)
    except OSError:
        return True

    if any(
        child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
        for child in children
    ):
        return True
    if any(
        child.is_dir() and child.name not in image_source_dir_names
        for child in children
    ):
        return True
    return False


def _scan_image_directory(
    root_path: Path,
    path: Path | None,
    *,
    sample_entries: int,
) -> tuple[dict[str, object], dict[str, object]]:
    directory_report = _empty_directory_report()
    image_stats: dict[str, object] = {
        "root": None,
        "file_count": 0,
        "sample_paths": [],
    }
    if path is None or not path.is_dir():
        return directory_report, image_stats

    sample_budget = max(sample_entries, 0)
    traversal_budget = max(sample_budget, 1) * 1000
    pending: deque[Path] = deque([path])
    samples: list[dict[str, object]] = []
    sampled_image_count = 0
    readable_sampled_image_count = 0
    invalid_sample_paths: list[str] = []
    sample_paths: list[str] = []
    file_count = 0
    while pending:
        current = pending.popleft()
        try:
            children = sorted(current.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            directory_report["error"] = f"{type(exc).__name__}: {exc}"
            break
        for child in children:
            relative = child.relative_to(path).as_posix()
            is_dir = child.is_dir()
            is_file = child.is_file()
            is_image_candidate = is_file and child.suffix.lower() in IMAGE_SUFFIXES
            if sampled_image_count < sample_budget and len(samples) < traversal_budget:
                samples.append(
                    {
                        "relative_path": relative,
                        "is_dir": is_dir,
                        "suffix": child.suffix.lower(),
                        "size_bytes": child.stat().st_size if is_file else None,
                        "is_image_candidate": is_image_candidate,
                    }
                )
                if is_image_candidate:
                    sampled_image_count += 1
                    if _is_decodable_image_file(child):
                        readable_sampled_image_count += 1
                    elif len(invalid_sample_paths) < sample_budget:
                        invalid_sample_paths.append(relative)
            if child.is_dir():
                pending.append(child)
                continue
            if not is_image_candidate:
                continue
            file_count += 1
            if len(sample_paths) < sample_budget:
                sample_paths.append(relative)

    directory_report.update(
        {
            "sample_entries": samples,
            "sampled_entry_count": len(samples),
            "sampled_image_count": sampled_image_count,
            "readable_sampled_image_count": readable_sampled_image_count,
            "invalid_sample_paths": invalid_sample_paths,
        }
    )
    image_stats["root"] = _root_relative_path(root_path, path)
    image_stats["file_count"] = file_count
    image_stats["sample_paths"] = sample_paths
    return directory_report, image_stats


def _resolve_image_source_path(
    root_path: Path,
    asset: AssetDefinition,
    default_path: Path,
) -> tuple[Path, bool]:
    try:
        candidates = candidate_full_image_roots(root_path, asset.dataset, asset.asset_id)
    except ValueError:
        return default_path, default_path.exists()

    for candidate in candidates:
        if candidate.is_dir() and _accept_parent_image_root_candidate(
            candidate,
            asset=asset,
            default_path=default_path,
        ):
            return candidate, True
    for candidate in candidates:
        if candidate.exists() and not candidate.is_dir():
            return candidate, True
    return default_path, False


def _base_asset_report(
    asset: AssetDefinition,
    path: Path,
    *,
    raw_root_exists: bool,
    expected: Mapping[str, object],
) -> dict[str, object]:
    path_exists = path.exists()
    exists = _asset_exists(path, asset.expected_kind)
    actual_kind = _actual_kind(path)
    size_bytes = path.stat().st_size if exists and path.is_file() else None
    expected_size = expected["expected_size_bytes"]
    size_matches = None
    if expected_size is not None and size_bytes is not None:
        size_matches = size_bytes == expected_size

    status = "complete"
    blockers: list[str] = []
    if not exists:
        if path_exists:
            status = "incomplete"
            blockers.append(
                f"{asset.asset_id} expected {asset.expected_kind} at {asset.local_path} "
                f"but found {actual_kind}"
            )
        else:
            status = "missing" if raw_root_exists else "not_yet_supplied"
            blockers.append(
                f"{asset.asset_id} missing {asset.expected_kind} at {asset.local_path}"
            )

    if size_matches is False:
        blockers.append(
            f"{asset.asset_id} size mismatch at {asset.local_path}: "
            f"expected {expected_size}, observed {size_bytes}"
        )
        status = "incomplete"

    return {
        "dataset": asset.dataset,
        "asset_id": asset.asset_id,
        "asset_type": asset.asset_type,
        "local_path": asset.local_path,
        "expected_kind": asset.expected_kind,
        "required": asset.required,
        "path_exists": path_exists,
        "exists": exists,
        "actual_kind": actual_kind,
        "size_bytes": size_bytes,
        "expected_size_bytes": expected_size,
        "size_matches": size_matches,
        "manifest_entry_present": expected["manifest_entry_present"],
        "checksum_entry_present": expected["checksum_entry_present"],
        "checksum": {
            "status": "not_available",
            "algorithm": "sha256",
            "source": expected["checksum_source"],
            "expected_sha256": expected["expected_sha256"],
            "observed_sha256": None,
        },
        "archive": None,
        "csv": None,
        "json_mapping": None,
        "directory": None,
        "image_stats": None,
        "download_sidecar": None,
        "status": status,
        "blockers": blockers,
    }


def _validate_file_checksum(
    report: dict[str, object],
    path: Path,
    *,
    compute_sha256: bool,
) -> None:
    checksum = report["checksum"]
    expected_sha = checksum["expected_sha256"]
    if expected_sha is None and not compute_sha256:
        checksum["status"] = "not_requested"
        return

    observed_sha = _sha256(path)
    checksum["observed_sha256"] = observed_sha
    if expected_sha is None:
        checksum["status"] = "computed"
        return
    if observed_sha == expected_sha:
        checksum["status"] = "matched"
        return

    checksum["status"] = "mismatch"
    report["status"] = "checksum_mismatch"
    report["blockers"].append(
        f"{report['asset_id']} checksum mismatch at {report['local_path']}"
    )


def _validate_asset(
    asset: AssetDefinition,
    path: Path,
    *,
    root_path: Path,
    raw_root_exists: bool,
    expected: Mapping[str, object],
    compute_sha256: bool,
    csv_sample_rows: int,
    zip_sample_members: int,
    image_sample_entries: int,
    max_json_bytes: int,
    require_images: bool,
) -> dict[str, object]:
    if asset.asset_type == "image_source":
        path, candidate_root_found = _resolve_image_source_path(root_path, asset, path)
    else:
        candidate_root_found = False
    report = _base_asset_report(
        asset,
        path,
        raw_root_exists=raw_root_exists,
        expected=expected,
    )
    download_sidecar = _read_download_sidecar(
        asset,
        path,
        root_path=root_path,
        max_json_bytes=max_json_bytes,
    )
    if asset.asset_type == "image_source":
        resolved_root = path if path.is_dir() else None
        report["directory"], report["image_stats"] = _scan_image_directory(
            root_path,
            resolved_root,
            sample_entries=image_sample_entries,
        )
        if not report["exists"] and require_images and not candidate_root_found:
            report["status"] = "missing_required_image_root"
            report["blockers"] = ["missing_required_image_root"]
    if not report["exists"]:
        _apply_download_sidecar(report, download_sidecar)
        return report

    if path.is_file():
        if int(report["size_bytes"] or 0) == 0:
            report["status"] = "incomplete"
            report["blockers"].append(
                f"{asset.asset_id} is an empty file at {asset.local_path}"
            )
        _validate_file_checksum(report, path, compute_sha256=compute_sha256)
        if report["status"] == "checksum_mismatch":
            return report

    if asset.asset_type in {"kb_archive", "faiss_archive"}:
        archive = _validate_zip(path, sample_members=zip_sample_members)
        report["archive"] = archive
        if not archive["readable"]:
            report["status"] = "archive_unreadable"
            report["blockers"].append(
                f"{asset.asset_id} archive unreadable at {asset.local_path}: "
                f"{archive['error']}"
            )
        elif archive["member_count"] == 0:
            report["status"] = "incomplete"
            report["blockers"].append(
                f"{asset.asset_id} archive has no members at {asset.local_path}"
            )
        elif asset.asset_type == "faiss_archive" and (
            not archive["index_candidate_members"]
            or not archive["metadata_candidate_members"]
        ):
            report["status"] = "incomplete"
            report["blockers"].append(
                f"{asset.asset_id} lacks FAISS index or metadata members at {asset.local_path}"
            )

    elif asset.asset_type == "vqa_csv":
        csv_report = _validate_csv(path, sample_rows=csv_sample_rows)
        report["csv"] = csv_report
        if not csv_report["readable"] or not csv_report["field_names"]:
            report["status"] = "incomplete"
            report["blockers"].append(
                f"{asset.asset_id} VQA CSV header unreadable or empty at {asset.local_path}"
            )

    elif asset.asset_type == "id_mapping":
        json_report = _validate_json_mapping(path, max_json_bytes=max_json_bytes)
        report["json_mapping"] = json_report
        if (
            json_report["status"] != "parse_skipped_size_limit"
            and (
                not json_report["readable"]
                or json_report["json_top_level_type"] not in {"object", "array"}
                or not json_report["entry_count"]
            )
        ):
            report["status"] = "incomplete"
            report["blockers"].append(
                f"{asset.asset_id} id mapping JSON unreadable or empty at {asset.local_path}"
            )

    elif asset.asset_type == "image_source":
        if report["directory"]["error"] or report["image_stats"]["file_count"] == 0:
            report["status"] = "incomplete"
            report["blockers"].append(
                f"{asset.asset_id} image source has no sampled image files at {asset.local_path}"
            )
        elif report["directory"]["invalid_sample_paths"]:
            report["status"] = "incomplete"
            report["blockers"].append(
                f"{asset.asset_id} image source has unreadable sampled image files at {asset.local_path}"
            )

    _apply_download_sidecar(report, download_sidecar)
    return report


def _metadata_blockers(
    dataset: str,
    report: Mapping[str, object],
    *,
    label: str,
) -> list[dict[str, object]]:
    if (not report["exists"] and not report.get("blocking")) or report["status"] == "complete":
        return []
    status = str(report.get("status") or "incomplete")
    reason = (
        f"{label}_outside_validation_root"
        if status == "metadata_outside_validation_root"
        else f"{label}_inconsistent_or_unreadable"
    )
    return [
        {
            "dataset": dataset,
            "asset_id": None,
            "local_path": report["local_path"],
            "status": status,
            "reason": reason,
            "error": report.get("error"),
        }
    ]


def _is_decodable_image_file(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError):
        return False
    return True


def _asset_blocker_objects(report: Mapping[str, object]) -> list[dict[str, object]]:
    if not report["required"] or report["status"] == "complete":
        return []
    download_sidecar = report.get("download_sidecar")
    sidecar_blocker = (
        dict(download_sidecar)
        if isinstance(download_sidecar, Mapping) and download_sidecar.get("blocking")
        else None
    )
    blockers = []
    for reason in report["blockers"]:
        blocker = {
            "dataset": report["dataset"],
            "asset_id": report["asset_id"],
            "local_path": report["local_path"],
            "status": report["status"],
            "reason": reason,
        }
        if sidecar_blocker is not None:
            blocker["download_sidecar"] = sidecar_blocker
        blockers.append(blocker)
    return blockers


def _status_counts(asset_reports: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for report in asset_reports:
        status = str(report["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def _has_complete_subset_report(subset_reports: Mapping[str, object]) -> bool:
    summary = subset_reports.get("summary")
    if not isinstance(summary, Mapping):
        return False
    complete_count = _safe_int(summary.get("complete_subset_count"))
    return bool(complete_count and complete_count > 0)


def _target_subset(
    *,
    subset_sample_train: int | None,
    subset_sample_test: int | None,
    subset_seed: int | None,
) -> dict[str, int | None] | None:
    if (
        subset_sample_train is None
        and subset_sample_test is None
        and subset_seed is None
    ):
        return None
    return {
        "sample_train": subset_sample_train,
        "sample_test": subset_sample_test,
        "seed": subset_seed,
    }


def _subset_report_matches_target(
    report: Mapping[str, object],
    target_subset: Mapping[str, int | None] | None,
) -> bool:
    if target_subset is None:
        return True
    for key, expected in target_subset.items():
        if expected is None:
            continue
        if _safe_int(report.get(key)) != expected:
            return False
    return True


def _image_source_informational(
    asset_report: Mapping[str, object],
    *,
    require_complete_subset: bool,
    subset_has_complete_report: bool,
    require_images: bool,
) -> bool:
    return (
        require_complete_subset
        and subset_has_complete_report
        and asset_report.get("asset_type") == "image_source"
    )


def _subset_manifest_image_path(
    root_path: Path,
    subset_root: Path,
    subset_path: object,
) -> Path | None:
    if subset_path is None:
        return None
    text = str(subset_path).strip()
    if not text:
        return None

    raw_path = Path(text)
    if raw_path.is_absolute():
        candidates = (raw_path,)
    else:
        relative_path = PurePosixPath(text.replace("\\", "/"))
        candidates = (
            root_path.joinpath(*relative_path.parts),
            subset_root.joinpath(*relative_path.parts),
        )

    resolved_subset_root = subset_root.resolve()
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        try:
            resolved_candidate.relative_to(resolved_subset_root)
        except ValueError:
            continue
        return candidate
    return None


def _subset_copied_image_stats(
    root_path: Path,
    subset_root: Path,
    copied_images: object,
    *,
    sample_limit: int = 10,
) -> dict[str, object] | None:
    if not isinstance(copied_images, list):
        return None

    stats: dict[str, object] = {
        "copied_image_count": 0,
        "present_image_count": 0,
        "missing_image_count": 0,
        "outside_image_count": 0,
        "missing_samples": [],
        "outside_samples": [],
    }
    missing_samples: list[dict[str, object]] = []
    outside_samples: list[dict[str, object]] = []

    for index, item in enumerate(copied_images):
        if not isinstance(item, Mapping):
            continue
        stats["copied_image_count"] = int(stats["copied_image_count"]) + 1
        image_id = item.get("image_id")
        manifest_path = item.get("subset_path")
        image_path = _subset_manifest_image_path(root_path, subset_root, manifest_path)
        sample = {
            "index": index,
            "image_id": image_id,
            "subset_path": manifest_path,
        }
        if image_path is None:
            stats["outside_image_count"] = int(stats["outside_image_count"]) + 1
            if len(outside_samples) < sample_limit:
                outside_samples.append(sample)
            continue
        if image_path.is_file():
            stats["present_image_count"] = int(stats["present_image_count"]) + 1
            continue
        stats["missing_image_count"] = int(stats["missing_image_count"]) + 1
        if len(missing_samples) < sample_limit:
            sample["resolved_path"] = str(image_path)
            missing_samples.append(sample)

    stats["missing_samples"] = missing_samples
    stats["outside_samples"] = outside_samples
    return stats


def _dataset_status(
    *,
    raw_root_exists: bool,
    required_reports: Sequence[Mapping[str, object]],
    blockers: Sequence[Mapping[str, object]],
) -> str:
    if not blockers:
        return "complete"
    if not raw_root_exists and all(
        report["status"] == "not_yet_supplied" for report in required_reports
    ) and all(
        blocker.get("status") == "not_yet_supplied" for blocker in blockers
    ):
        return "not_yet_supplied"
    return "incomplete"


def _subset_manifest_report(
    root_path: Path,
    dataset: str,
    manifest_path: Path,
    *,
    max_json_bytes: int,
) -> dict[str, object]:
    report: dict[str, object] = {
        "manifest_path": _root_relative_path(root_path, manifest_path),
        "subset_root": _root_relative_path(root_path, manifest_path.parent),
        "inside_validation_root": None,
        "resolved_path": None,
        "readable": False,
        "valid_json": False,
        "manifest_version": None,
        "dataset": None,
        "status": "invalid_manifest",
        "sample_train": None,
        "sample_test": None,
        "seed": None,
        "summary": {},
        "blockers": [],
    }

    containment = _resolve_for_validation_root(root_path, manifest_path)
    report["inside_validation_root"] = containment["inside"]
    report["resolved_path"] = str(containment["candidate_resolved"])
    if not containment["inside"]:
        report["status"] = "subset_manifest_outside_validation_root"
        report["blockers"].append(
            {
                "reason": "manifest_outside_validation_root",
                "resolved_path": str(containment["candidate_resolved"]),
                "validation_root": str(containment["root_resolved"]),
            }
        )
        return report

    try:
        size_bytes = manifest_path.stat().st_size
        if size_bytes > max_json_bytes:
            report["blockers"].append(
                {
                    "reason": "manifest_too_large_for_json_parse",
                    "size_bytes": size_bytes,
                    "max_json_bytes": max_json_bytes,
                }
            )
            return report
        document = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        report["blockers"].append(
            {
                "reason": "manifest_unreadable",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return report

    report["readable"] = True
    if not isinstance(document, Mapping):
        report["blockers"].append(
            {
                "reason": "manifest_top_level_not_object",
                "observed_type": _json_type(document),
            }
        )
        return report

    summary = document.get("summary")
    summary = dict(summary) if isinstance(summary, Mapping) else {}
    source_blockers = document.get("blockers")
    source_blockers = list(source_blockers) if isinstance(source_blockers, list) else []
    manifest_version = document.get("manifest_version")
    observed_dataset = document.get("dataset")
    sample_train = _safe_int(document.get("sample_train"))
    sample_test = _safe_int(document.get("sample_test"))
    seed = _safe_int(document.get("seed"))
    requested_train = _safe_int(summary.get("requested_train"))
    requested_test = _safe_int(summary.get("requested_test"))
    complete_train = _safe_int(summary.get("complete_train"))
    complete_test = _safe_int(summary.get("complete_test"))
    selected_train = _safe_int(summary.get("selected_train"))
    selected_test = _safe_int(summary.get("selected_test"))
    if complete_train is None:
        complete_train = _summary_split_count(summary, "train")
    if complete_test is None:
        complete_test = _summary_split_count(summary, "test")
    if complete_train is None:
        complete_train = selected_train
    if complete_test is None:
        complete_test = selected_test
    if sample_train is None:
        sample_train = requested_train
    if sample_test is None:
        sample_test = requested_test
    extracted_images = _safe_int(summary.get("extracted_images"))
    image_stats = _subset_copied_image_stats(
        root_path,
        manifest_path.parent,
        document.get("copied_images"),
    )
    if extracted_images is None and image_stats is not None:
        extracted_images = _safe_int(image_stats.get("present_image_count"))
    summary_status = summary.get("status")
    normalized_summary_status = (
        summary_status if isinstance(summary_status, str) and summary_status else None
    )

    blockers: list[object] = []
    if manifest_version not in SUPPORTED_SUBSET_MANIFEST_VERSIONS:
        blockers.append(
            {
                "reason": "unexpected_manifest_version",
                "expected": sorted(SUPPORTED_SUBSET_MANIFEST_VERSIONS),
                "observed": manifest_version,
            }
        )
    if observed_dataset != dataset:
        blockers.append(
            {
                "reason": "dataset_mismatch",
                "expected": dataset,
                "observed": observed_dataset,
            }
        )
    if normalized_summary_status != "complete":
        blockers.append(
            {
                "reason": "summary_status_not_complete",
                "observed": normalized_summary_status,
            }
        )
    if source_blockers:
        blockers.append({"reason": "source_manifest_blockers", "blockers": source_blockers})

    requested_train = requested_train if requested_train is not None else sample_train
    requested_test = requested_test if requested_test is not None else sample_test
    if (
        requested_train is not None
        and complete_train is not None
        and complete_train != requested_train
    ):
        blockers.append(
            {
                "reason": "train_count_mismatch",
                "requested": requested_train,
                "complete": complete_train,
            }
        )
    if (
        requested_test is not None
        and complete_test is not None
        and complete_test != requested_test
    ):
        blockers.append(
            {
                "reason": "test_count_mismatch",
                "requested": requested_test,
                "complete": complete_test,
            }
        )
    if normalized_summary_status == "complete":
        if requested_train is not None and complete_train is None:
            blockers.append({"reason": "train_count_missing", "requested": requested_train})
        if requested_test is not None and complete_test is None:
            blockers.append({"reason": "test_count_missing", "requested": requested_test})
        if (
            extracted_images is not None
            and complete_train is not None
            and complete_test is not None
            and extracted_images < complete_train + complete_test
        ):
            blockers.append(
                {
                    "reason": "extracted_image_count_mismatch",
                    "expected_at_least": complete_train + complete_test,
                    "observed": extracted_images,
                }
            )
        if image_stats is not None:
            missing_image_count = _safe_int(image_stats.get("missing_image_count")) or 0
            outside_image_count = _safe_int(image_stats.get("outside_image_count")) or 0
            if missing_image_count:
                blockers.append(
                    {
                        "reason": "copied_subset_images_missing",
                        "missing_image_count": missing_image_count,
                        "samples": image_stats.get("missing_samples", []),
                    }
                )
            if outside_image_count:
                blockers.append(
                    {
                        "reason": "copied_subset_images_outside_subset_root",
                        "outside_image_count": outside_image_count,
                        "samples": image_stats.get("outside_samples", []),
                    }
                )

    report.update(
        {
            "valid_json": True,
            "manifest_version": manifest_version,
            "dataset": observed_dataset,
            "sample_train": sample_train,
            "sample_test": sample_test,
            "seed": seed,
            "summary": summary,
            "image_stats": image_stats,
            "blockers": blockers,
            "status": "complete" if not blockers else "subset_incomplete",
        }
    )
    return report


def _subset_reports(
    root_path: Path,
    dataset: str,
    *,
    max_json_bytes: int,
    subset_sample_train: int | None = None,
    subset_sample_test: int | None = None,
    subset_seed: int | None = None,
) -> dict[str, object]:
    target_subset = _target_subset(
        subset_sample_train=subset_sample_train,
        subset_sample_test=subset_sample_test,
        subset_seed=subset_seed,
    )
    subsets_root = root_path / "datasets_mm" / dataset / "subsets"
    subsets_root_exists = subsets_root.is_dir()
    reports: list[dict[str, object]] = []
    if subsets_root_exists:
        containment = _resolve_for_validation_root(root_path, subsets_root)
        if containment["inside"]:
            reports = [
                _subset_manifest_report(
                    root_path,
                    dataset,
                    manifest_path,
                    max_json_bytes=max_json_bytes,
                )
                for manifest_path in sorted(subsets_root.glob("*/manifest.json"))
            ]
        else:
            reports = [
                {
                    "manifest_path": f"datasets_mm/{dataset}/subsets/*/manifest.json",
                    "subset_root": f"datasets_mm/{dataset}/subsets",
                    "inside_validation_root": False,
                    "resolved_path": str(containment["candidate_resolved"]),
                    "readable": False,
                    "valid_json": False,
                    "manifest_version": None,
                    "dataset": None,
                    "status": "subsets_root_outside_validation_root",
                    "sample_train": None,
                    "sample_test": None,
                    "seed": None,
                    "summary": {},
                    "blockers": [
                        {
                            "reason": "subsets_root_outside_validation_root",
                            "resolved_path": str(containment["candidate_resolved"]),
                            "validation_root": str(containment["root_resolved"]),
                        }
                    ],
                }
            ]
    matching_reports = [
        report
        for report in reports
        if _subset_report_matches_target(report, target_subset)
    ]
    complete_reports = [
        report for report in matching_reports if report["status"] == "complete"
    ]
    status_counts: dict[str, int] = {}
    for report in reports:
        status = str(report["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    summary: dict[str, object] = {
        "total_subset_count": len(reports),
        "complete_subset_count": len(complete_reports),
        "status_counts": status_counts,
    }
    if target_subset is not None:
        summary["target_subset"] = target_subset
        summary["matching_subset_count"] = len(matching_reports)
    return {
        "subsets_root": f"datasets_mm/{dataset}/subsets",
        "subsets_root_exists": subsets_root_exists,
        "reports": reports,
        "complete_reports": complete_reports,
        "summary": summary,
    }


def _subset_blockers(
    dataset: str,
    subset_reports: Mapping[str, object],
    *,
    require_complete_subset: bool,
    target_subset: Mapping[str, int | None] | None = None,
) -> list[dict[str, object]]:
    if not require_complete_subset:
        return []

    reports = [
        report
        for report in subset_reports.get("reports", [])
        if isinstance(report, Mapping)
    ]
    if target_subset is not None:
        matching_reports = [
            report
            for report in reports
            if _subset_report_matches_target(report, target_subset)
        ]
        if any(report.get("status") == "complete" for report in matching_reports):
            return []
        return [
            {
                "dataset": dataset,
                "asset_id": None,
                "local_path": f"datasets_mm/{dataset}/subsets",
                "status": "missing",
                "reason": "no_matching_complete_subset",
                "target_subset": dict(target_subset),
                "matching_subset_count": len(matching_reports),
                "remediation": (
                    "Build a complete paper subset manifest matching the targeted "
                    "sample size and seed."
                ),
            }
        ]

    if not reports:
        return [
            {
                "dataset": dataset,
                "asset_id": None,
                "local_path": f"datasets_mm/{dataset}/subsets",
                "status": "missing",
                "reason": "no_subset_manifest",
                "remediation": (
                    "Build a paper subset manifest before requiring complete subset "
                    "validation."
                ),
            }
        ]

    blockers: list[dict[str, object]] = []
    for report in reports:
        if report.get("status") == "complete":
            continue
        blockers.append(
            {
                "dataset": dataset,
                "asset_id": None,
                "local_path": report.get("manifest_path"),
                "status": report.get("status"),
                "reason": "subset_manifest_incomplete",
                "details": report.get("blockers", []),
                "remediation": (
                    "Re-run the paper subset builder after resolving missing rows or "
                    "images, then validate again."
                ),
            }
        )
    return blockers


def _dataset_validation(
    root_path: Path,
    dataset: str,
    *,
    include_optional: bool,
    compute_sha256: bool,
    csv_sample_rows: int,
    zip_sample_members: int,
    image_sample_entries: int,
    max_json_bytes: int,
    require_complete_subset: bool,
    require_images: bool,
    subset_sample_train: int | None = None,
    subset_sample_test: int | None = None,
    subset_seed: int | None = None,
) -> dict[str, object]:
    assets = asset_catalog([dataset], include_optional=include_optional)
    recognized_assets = asset_catalog([dataset], include_optional=True)
    raw_root = _repo_relative_path(root_path, f"datasets_mm/{dataset}/raw")
    raw_root_exists = raw_root.is_dir()
    manifest, manifest_entries = _metadata_report(
        root_path=root_path,
        raw_root=raw_root,
        dataset=dataset,
        filename="manifest.json",
        assets=assets,
        recognized_assets=recognized_assets,
    )
    checksums, checksum_entries = _metadata_report(
        root_path=root_path,
        raw_root=raw_root,
        dataset=dataset,
        filename="checksums.json",
        assets=assets,
        recognized_assets=recognized_assets,
    )

    asset_reports = [
        _validate_asset(
            asset,
            _repo_relative_path(root_path, asset.local_path),
            root_path=root_path,
            raw_root_exists=raw_root_exists,
            expected=_expected_metadata(asset, manifest_entries, checksum_entries),
            compute_sha256=compute_sha256,
            csv_sample_rows=csv_sample_rows,
            zip_sample_members=zip_sample_members,
            image_sample_entries=image_sample_entries,
            max_json_bytes=max_json_bytes,
            require_images=require_images,
        )
        for asset in assets
    ]
    required_reports = [report for report in asset_reports if report["required"]]
    subsets = _subset_reports(
        root_path,
        dataset,
        max_json_bytes=max_json_bytes,
        subset_sample_train=subset_sample_train,
        subset_sample_test=subset_sample_test,
        subset_seed=subset_seed,
    )
    subset_has_complete_report = _has_complete_subset_report(subsets)
    target_subset = _target_subset(
        subset_sample_train=subset_sample_train,
        subset_sample_test=subset_sample_test,
        subset_seed=subset_seed,
    )

    blockers: list[dict[str, object]] = []
    blockers.extend(_metadata_blockers(dataset, manifest, label="manifest"))
    blockers.extend(_metadata_blockers(dataset, checksums, label="checksums"))
    for asset_report in asset_reports:
        if _image_source_informational(
            asset_report,
            require_complete_subset=require_complete_subset,
            subset_has_complete_report=subset_has_complete_report,
            require_images=require_images,
        ):
            continue
        blockers.extend(_asset_blocker_objects(asset_report))
    blockers.extend(
        _subset_blockers(
            dataset,
            subsets,
            require_complete_subset=require_complete_subset,
            target_subset=target_subset,
        )
    )

    status = _dataset_status(
        raw_root_exists=raw_root_exists,
        required_reports=required_reports,
        blockers=blockers,
    )
    return {
        "dataset": dataset,
        "raw_root": f"datasets_mm/{dataset}/raw",
        "raw_root_exists": raw_root_exists,
        "manifest": manifest,
        "checksums": checksums,
        "assets": asset_reports,
        "subset_reports": subsets,
        "blockers": blockers,
        "summary": {
            "status": status,
            "total_assets": len(asset_reports),
            "required_assets": len(required_reports),
            "complete_required_assets": sum(
                1 for report in required_reports if report["status"] == "complete"
            ),
            "status_counts": _status_counts(asset_reports),
        },
    }


def _overall_status(dataset_reports: Sequence[Mapping[str, object]]) -> str:
    statuses = [report["summary"]["status"] for report in dataset_reports]
    if all(status == "complete" for status in statuses):
        return "complete"
    if all(status == "not_yet_supplied" for status in statuses):
        return "not_yet_supplied"
    return "incomplete"


def validate_echosight_full_assets(
    root: str | Path = ".",
    *,
    datasets: Iterable[str] | str | None = None,
    include_optional: bool = False,
    compute_sha256: bool = False,
    csv_sample_rows: int = 3,
    zip_sample_members: int = 10,
    image_sample_entries: int = 10,
    max_json_bytes: int = 64 * 1024 * 1024,
    require_complete_subset: bool = False,
    require_images: bool = False,
    subset_sample_train: int | None = None,
    subset_sample_test: int | None = None,
    subset_seed: int | None = None,
) -> dict[str, object]:
    """Return a JSON-serializable B3 full local asset validation report."""

    selected = _normalize_datasets(datasets)
    root_path = Path(root)
    target_subset = _target_subset(
        subset_sample_train=subset_sample_train,
        subset_sample_test=subset_sample_test,
        subset_seed=subset_seed,
    )
    dataset_reports = [
        _dataset_validation(
            root_path,
            dataset,
            include_optional=include_optional,
            compute_sha256=compute_sha256,
            csv_sample_rows=csv_sample_rows,
            zip_sample_members=zip_sample_members,
            image_sample_entries=image_sample_entries,
            max_json_bytes=max_json_bytes,
            require_complete_subset=require_complete_subset,
            require_images=require_images,
            subset_sample_train=subset_sample_train,
            subset_sample_test=subset_sample_test,
            subset_seed=subset_seed,
        )
        for dataset in selected
    ]
    blockers = [
        blocker
        for dataset_report in dataset_reports
        for blocker in dataset_report["blockers"]
    ]
    policy: dict[str, object] = {
        "downloads_enabled": False,
        "extract_archives": False,
        "writes_payloads": False,
        "require_complete_subset": require_complete_subset,
        "require_images": require_images,
        "schema_freeze": False,
        "note": f"{NO_DOWNLOAD_POLICY} {VALIDATION_POLICY_NOTE}",
    }
    if target_subset is not None:
        policy["target_subset"] = target_subset
    return {
        "validation_version": VALIDATION_VERSION,
        "checked_root": str(root_path),
        "datasets": list(selected),
        "policy": policy,
        "large_payload_safety": {
            "downloads_attempted": False,
            "archives_extracted": False,
            "payload_files_written": False,
        },
        "dataset_reports": dataset_reports,
        "summary": {
            "status": _overall_status(dataset_reports),
            "blockers": blockers,
            "dataset_count": len(dataset_reports),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate local EchoSight/E-VQA/InfoSeek assets without downloads.",
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
        help="Dataset to validate. May be provided more than once.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional reranker assets in the report without blocking on them.",
    )
    parser.add_argument(
        "--compute-sha256",
        action="store_true",
        help="Compute sha256 for present files even when no checksum is expected.",
    )
    parser.add_argument(
        "--csv-sample-rows",
        type=int,
        default=3,
        help="Maximum CSV rows to sample while checking VQA CSV readability.",
    )
    parser.add_argument(
        "--zip-sample-members",
        type=int,
        default=10,
        help="Maximum ZIP members to include as archive metadata evidence.",
    )
    parser.add_argument(
        "--image-sample-entries",
        type=int,
        default=10,
        help="Maximum image source entries to sample recursively.",
    )
    parser.add_argument(
        "--max-json-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Maximum id mapping JSON bytes to parse for validation evidence.",
    )
    parser.add_argument(
        "--require-complete-subset",
        action="store_true",
        help="Treat incomplete or missing paper subset manifests as dataset blockers.",
    )
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Treat missing canonical or compatibility full-image roots as blockers.",
    )
    parser.add_argument(
        "--sample-train",
        type=int,
        default=None,
        help="Target subset train row count for --require-complete-subset.",
    )
    parser.add_argument(
        "--sample-test",
        type=int,
        default=None,
        help="Target subset test row count for --require-complete-subset.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=None,
        help="Target subset seed for --require-complete-subset.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = validate_echosight_full_assets(
        args.root,
        datasets=args.dataset or None,
        include_optional=args.include_optional,
        compute_sha256=args.compute_sha256,
        csv_sample_rows=args.csv_sample_rows,
        zip_sample_members=args.zip_sample_members,
        image_sample_entries=args.image_sample_entries,
        max_json_bytes=args.max_json_bytes,
        require_complete_subset=args.require_complete_subset,
        require_images=args.require_images,
        subset_sample_train=args.sample_train,
        subset_sample_test=args.sample_test,
        subset_seed=args.subset_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
