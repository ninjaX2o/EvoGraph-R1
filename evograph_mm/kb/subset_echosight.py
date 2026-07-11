"""Build deterministic EchoSight paper-sized QA/image subsets."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Mapping, Sequence

from .prepare_echosight import DATASETS_MM_ROOT, SUPPORTED_DATASETS, raw_dataset_root


SUBSET_MANIFEST_VERSION = "echosight-paper-subset-v1"
IMAGE_ID_FIELD_CANDIDATES = (
    "image_id",
    "dataset_image_ids",
    "image",
    "img_id",
    "img",
    "id",
)
ANSWER_FIELD_CANDIDATES = ("answer", "answers", "answer_eval")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MAPPING_VALUE_KEYS = ("file_name", "filename", "path", "image_path", "image", "img")
UNRESOLVED_ROW_SAMPLE_LIMIT = 50


def build_paper_subset(
    root: str | Path,
    dataset: str,
    *,
    sample_train: int = 5120,
    sample_test: int = 128,
    seed: int = 0,
    image_roots: Iterable[str | Path] | None = None,
) -> dict[str, object]:
    """Build a deterministic local QA subset and copy only resolved images."""

    if dataset not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"unsupported EchoSight dataset: {dataset}; expected {supported}")
    sample_train = _require_nonnegative_int("sample_train", sample_train)
    sample_test = _require_nonnegative_int("sample_test", sample_test)
    seed = _require_int("seed", seed)

    root_path = Path(root)
    raw_root = root_path / raw_dataset_root(dataset)
    subsets_root = root_path / DATASETS_MM_ROOT / dataset / "subsets"
    subset_rel = (
        Path(DATASETS_MM_ROOT)
        / dataset
        / "subsets"
        / f"paper_{sample_train}_{sample_test}_seed{seed}"
    )
    subset_root = root_path / subset_rel
    subsets_root_resolved = subsets_root.resolve()
    subset_resolved = _require_under(
        subsets_root_resolved,
        subset_root,
        "subset output must stay under the dataset subsets root",
    )

    train_rows, train_fieldnames = _read_csv(raw_root / "qa_train.csv")
    test_rows, test_fieldnames = _read_csv(raw_root / "qa_test.csv")
    mappings = _load_id_mappings(raw_root)
    search_roots = _image_search_roots(root_path, raw_root, image_roots)

    images_dir = subset_resolved / "images"
    _prepare_subset_root(subset_resolved, images_dir)

    rng = random.Random(seed)
    used_image_names: set[str] = set()
    split_reports: dict[str, dict[str, object]] = {}
    copied_images: list[dict[str, object]] = []

    train_complete, split_reports["train"] = _build_split(
        dataset=dataset,
        split="train",
        rows=train_rows,
        target_count=sample_train,
        rng=rng,
        mappings=mappings,
        search_roots=search_roots,
        images_dir=images_dir,
        subset_rel=subset_rel,
        used_image_names=used_image_names,
    )
    copied_images.extend(split_reports["train"]["copied_images"])

    test_complete, split_reports["test"] = _build_split(
        dataset=dataset,
        split="test",
        rows=test_rows,
        target_count=sample_test,
        rng=rng,
        mappings=mappings,
        search_roots=search_roots,
        images_dir=images_dir,
        subset_rel=subset_rel,
        used_image_names=used_image_names,
    )
    copied_images.extend(split_reports["test"]["copied_images"])

    _write_csv(subset_resolved / "qa_train.csv", train_fieldnames, train_complete)
    _write_csv(subset_resolved / "qa_test.csv", test_fieldnames, test_complete)

    blockers = _blockers_from_split_reports(dataset, split_reports)
    status = "complete" if not blockers else "subset_incomplete"
    summary = {
        "status": status,
        "complete_train": len(train_complete),
        "complete_test": len(test_complete),
        "requested_train": sample_train,
        "requested_test": sample_test,
    }
    manifest = {
        "manifest_version": SUBSET_MANIFEST_VERSION,
        "dataset": dataset,
        "subset_root": subset_rel.as_posix(),
        "seed": seed,
        "sample_train": sample_train,
        "sample_test": sample_test,
        "source_files": {
            "train": str(raw_root / "qa_train.csv"),
            "test": str(raw_root / "qa_test.csv"),
        },
        "image_roots": [str(path) for path in search_roots],
        "field_candidates": {
            "image_id": list(IMAGE_ID_FIELD_CANDIDATES),
            "answer": list(ANSWER_FIELD_CANDIDATES),
        },
        "summary": summary,
        "splits": split_reports,
        "copied_images": copied_images,
        "blockers": blockers,
    }
    (subset_resolved / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "dataset": dataset,
        "subset_root": subset_rel.as_posix(),
        "manifest_path": (subset_rel / "manifest.json").as_posix(),
        "summary": summary,
        "blockers": blockers,
        "splits": split_reports,
    }


def _require_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _require_nonnegative_int(name: str, value: object) -> int:
    value = _require_int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _require_under(root_resolved: Path, target: Path, message: str) -> Path:
    target_resolved = target.resolve()
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{message}: {target}") from exc
    return target_resolved


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def _write_csv(
    path: Path,
    source_fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(source_fieldnames, rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _csv_fieldnames(
    source_fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> list[str]:
    fieldnames: list[str] = []
    seen: set[object] = set()
    for field in source_fieldnames:
        if field is None or field in seen:
            continue
        seen.add(field)
        fieldnames.append(field)
    for row in rows:
        for field in row:
            if field is None or field in seen:
                continue
            seen.add(field)
            fieldnames.append(field)
    return fieldnames


def _prepare_subset_root(subset_root: Path, images_dir: Path) -> None:
    subset_root.mkdir(parents=True, exist_ok=True)
    for filename in ("qa_train.csv", "qa_test.csv", "manifest.json"):
        target = _require_under(
            subset_root,
            subset_root / filename,
            "subset file must stay under subset root",
        )
        if target.exists():
            target.unlink()
    images_resolved = _require_under(
        subset_root,
        images_dir,
        "subset images directory must stay under subset root",
    )
    if images_resolved.exists():
        shutil.rmtree(images_resolved)
    images_resolved.mkdir(parents=True, exist_ok=True)


def _image_search_roots(
    root_path: Path,
    raw_root: Path,
    image_roots: Iterable[str | Path] | None,
) -> list[Path]:
    candidates: list[tuple[Path, bool]] = []
    for image_root in image_roots or ():
        candidate = Path(image_root)
        if not candidate.is_absolute():
            candidate = root_path / candidate
        candidates.append((candidate, True))
    candidates.extend(
        [
            (raw_root / "images", False),
            (raw_root / "image", False),
            (raw_root / "imgs", False),
            (raw_root, False),
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate, explicit in candidates:
        if not _usable_image_search_root(candidate, raw_root, explicit=explicit):
            continue
        key = str(candidate.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _usable_image_search_root(
    candidate: Path,
    raw_root: Path,
    *,
    explicit: bool,
) -> bool:
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        return False
    if not resolved.is_dir():
        return False
    if explicit:
        return True

    raw_root_resolved = raw_root.resolve()
    if resolved == raw_root_resolved:
        return _raw_root_has_image_layout(resolved)
    if candidate.parent.resolve() == raw_root_resolved:
        return _directory_has_image_like_entries(resolved)
    return False


def _raw_root_has_image_layout(raw_root: Path) -> bool:
    if _directory_has_direct_image_file(raw_root):
        return True
    for dirname in ("train", "val", "valid", "test"):
        child = raw_root / dirname
        if child.is_dir() and _directory_has_image_like_entries(child):
            return True
    return False


def _directory_has_image_like_entries(path: Path) -> bool:
    try:
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                return True
            if child.is_dir():
                return True
    except OSError:
        return False
    return False


def _directory_has_direct_image_file(path: Path) -> bool:
    try:
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                return True
    except OSError:
        return False
    return False


def _build_split(
    *,
    dataset: str,
    split: str,
    rows: Sequence[Mapping[str, str]],
    target_count: int,
    rng: random.Random,
    mappings: Mapping[str, str],
    search_roots: Sequence[Path],
    images_dir: Path,
    subset_rel: Path,
    used_image_names: set[str],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    order = list(range(len(rows)))
    rng.shuffle(order)

    if target_count > 0 and not search_roots:
        return _build_split_without_image_roots(
            dataset=dataset,
            split=split,
            rows=rows,
            order=order,
            target_count=target_count,
        )

    complete_rows: list[dict[str, str]] = []
    unresolved_rows: list[dict[str, object]] = []
    unresolved_total = 0
    copied_images: list[dict[str, object]] = []

    for row_index in order:
        if len(complete_rows) >= target_count:
            break
        row = dict(rows[row_index])
        image_field, image_id = _image_id_from_row(row)
        if not image_id:
            unresolved_total = _record_unresolved_row(
                unresolved_rows,
                unresolved_total,
                _unresolved_row(split, row_index, image_field, image_id, "missing image id"),
            )
            continue
        image_path = _resolve_image(image_id, mappings, search_roots)
        if image_path is None:
            unresolved_total = _record_unresolved_row(
                unresolved_rows,
                unresolved_total,
                _unresolved_row(
                    split,
                    row_index,
                    image_field,
                    image_id,
                    "unresolved image",
                )
            )
            continue

        copied = _copy_image(
            image_path=image_path,
            image_id=image_id,
            split=split,
            row_index=row_index,
            images_dir=images_dir,
            subset_rel=subset_rel,
            used_image_names=used_image_names,
        )
        copied_images.append(copied)
        complete_rows.append(row)

    return complete_rows, {
        "dataset": dataset,
        "split": split,
        "requested": target_count,
        "source_rows": len(rows),
        "complete": len(complete_rows),
        "unresolved": unresolved_total,
        "unresolved_total": unresolved_total,
        "unresolved_sample_limit": UNRESOLVED_ROW_SAMPLE_LIMIT,
        "unresolved_rows": unresolved_rows,
        "copied_images": copied_images,
    }


def _build_split_without_image_roots(
    *,
    dataset: str,
    split: str,
    rows: Sequence[Mapping[str, str]],
    order: Sequence[int],
    target_count: int,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    unresolved_rows: list[dict[str, object]] = []
    sample_order = order[:UNRESOLVED_ROW_SAMPLE_LIMIT]
    for row_index in sample_order:
        row = rows[row_index]
        image_field, image_id = _image_id_from_row(row)
        reason = "missing image id" if not image_id else "no usable image roots"
        unresolved_rows.append(_unresolved_row(split, row_index, image_field, image_id, reason))
    unresolved_total = len(rows)
    return [], {
        "dataset": dataset,
        "split": split,
        "requested": target_count,
        "source_rows": len(rows),
        "complete": 0,
        "unresolved": unresolved_total,
        "unresolved_total": unresolved_total,
        "unresolved_sample_limit": UNRESOLVED_ROW_SAMPLE_LIMIT,
        "unresolved_rows": unresolved_rows,
        "copied_images": [],
    }


def _blockers_from_split_reports(
    dataset: str,
    split_reports: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    for split, report in split_reports.items():
        requested = int(report["requested"])
        complete = int(report["complete"])
        if complete >= requested:
            continue
        blockers.append(
            {
                "dataset": dataset,
                "split": split,
                "stage": "image_resolution",
                "message": (
                    f"unresolved image rows prevented complete {split} subset: "
                    f"{complete}/{requested} complete"
                ),
                "details": {
                    "requested": requested,
                    "complete": complete,
                    "source_rows": report["source_rows"],
                    "unresolved_total": report.get("unresolved_total", report["unresolved"]),
                    "unresolved_sample_limit": report.get("unresolved_sample_limit"),
                    "unresolved_rows": report["unresolved_rows"],
                },
            }
        )
    return blockers


def _image_id_from_row(row: Mapping[str, str]) -> tuple[str | None, str | None]:
    seen_field: str | None = None
    for field in IMAGE_ID_FIELD_CANDIDATES:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            continue
        seen_field = field
        for image_id in _image_ids_from_value(value):
            if _safe_relative_image_name(image_id):
                return field, image_id
    return seen_field, None


def _image_ids_from_value(value: object) -> list[str]:
    raw = str(value).strip()
    if not raw:
        return []

    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            return _flatten_image_id_values(parsed)

    tokens = re.split(r"[,;|\s]+", raw)
    return [_strip_token_quotes(token) for token in tokens if _strip_token_quotes(token)]


def _flatten_image_id_values(value: object) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_image_id_values(item))
        return flattened
    return []


def _strip_token_quotes(value: str) -> str:
    return value.strip().strip("'\"")


def _record_unresolved_row(
    unresolved_rows: list[dict[str, object]],
    unresolved_total: int,
    row: dict[str, object],
) -> int:
    unresolved_total += 1
    if len(unresolved_rows) < UNRESOLVED_ROW_SAMPLE_LIMIT:
        unresolved_rows.append(row)
    return unresolved_total


def _unresolved_row(
    split: str,
    row_index: int,
    image_field: str | None,
    image_id: str | None,
    reason: str,
) -> dict[str, object]:
    return {
        "split": split,
        "row_index": row_index,
        "image_field": image_field,
        "image_id": image_id,
        "reason": reason,
    }


def _resolve_image(
    image_id: str,
    mappings: Mapping[str, str],
    search_roots: Sequence[Path],
) -> Path | None:
    mapped_name = mappings.get(image_id)
    relative_candidates = _relative_image_candidates(image_id)
    if mapped_name:
        relative_candidates = _relative_image_candidates(mapped_name) + relative_candidates

    for search_root in search_roots:
        try:
            search_root_resolved = search_root.resolve(strict=True)
        except FileNotFoundError:
            continue
        if not search_root_resolved.is_dir():
            continue
        seen: set[str] = set()
        for relative_candidate in relative_candidates:
            key = relative_candidate.as_posix()
            if key in seen:
                continue
            seen.add(key)
            candidate = search_root / relative_candidate
            try:
                candidate_resolved = candidate.resolve(strict=True)
                candidate_resolved.relative_to(search_root_resolved)
            except (FileNotFoundError, ValueError):
                continue
            if candidate_resolved.is_file():
                return candidate_resolved
    return None


def _relative_image_candidates(name: str) -> list[Path]:
    normalized = _safe_relative_image_name(name)
    if not normalized:
        return []
    path = Path(normalized)

    candidates: list[Path] = []
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        candidates.append(path)
    else:
        candidates.extend(Path(f"{normalized}{extension}") for extension in IMAGE_EXTENSIONS)
    return candidates


def _safe_relative_image_name(name: str) -> str | None:
    raw = str(name).strip()
    if not raw or raw.startswith(("/", "\\")):
        return None
    normalized = raw.replace("\\", "/")
    if normalized.startswith("/"):
        return None

    windows_path = PureWindowsPath(raw)
    posix_path = PurePosixPath(normalized)
    if windows_path.anchor or windows_path.drive or windows_path.root:
        return None
    if posix_path.anchor or posix_path.root:
        return None
    if ".." in windows_path.parts or ".." in posix_path.parts:
        return None
    return normalized


def _copy_image(
    *,
    image_path: Path,
    image_id: str,
    split: str,
    row_index: int,
    images_dir: Path,
    subset_rel: Path,
    used_image_names: set[str],
) -> dict[str, object]:
    destination_name = _unique_image_name(image_id, image_path.suffix, used_image_names)
    destination = images_dir / destination_name
    shutil.copy2(image_path, destination)
    relative_destination = subset_rel / "images" / destination_name
    return {
        "split": split,
        "row_index": row_index,
        "image_id": image_id,
        "source_path": str(image_path),
        "subset_path": relative_destination.as_posix(),
        "sha256": _sha256(destination),
    }


def _unique_image_name(image_id: str, suffix: str, used_image_names: set[str]) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", image_id.replace("\\", "/")).strip("._")
    if not cleaned:
        cleaned = "image"
    if Path(cleaned).suffix.lower() not in IMAGE_EXTENSIONS:
        cleaned = f"{cleaned}{suffix.lower() if suffix else '.jpg'}"

    candidate = cleaned
    stem = Path(cleaned).stem
    ext = Path(cleaned).suffix
    counter = 1
    while candidate in used_image_names:
        candidate = f"{stem}_{counter}{ext}"
        counter += 1
    used_image_names.add(candidate)
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_id_mappings(raw_root: Path) -> dict[str, str]:
    mappings: dict[str, str] = {}
    for path in _mapping_files(raw_root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        _merge_mapping_payload(mappings, payload)
    return mappings


def _mapping_files(raw_root: Path) -> list[Path]:
    candidates: list[Path] = []
    id2name = raw_root / "id2name"
    if id2name.is_dir():
        candidates.extend(sorted(id2name.glob("*.json")))
    candidates.extend(sorted(raw_root.glob("*id2name*.json")))
    return candidates


def _merge_mapping_payload(target: dict[str, str], payload: object) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            mapped = _mapping_value(value)
            if mapped:
                target[str(key)] = mapped
        return
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            image_id = _first_mapping_item_value(item, IMAGE_ID_FIELD_CANDIDATES)
            mapped = _first_mapping_item_value(item, MAPPING_VALUE_KEYS)
            if image_id and mapped:
                target[image_id] = mapped


def _mapping_value(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _first_mapping_item_value(value, MAPPING_VALUE_KEYS)
    return None


def _first_mapping_item_value(
    item: Mapping[str, object],
    keys: Sequence[str],
) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return None


__all__ = [
    "ANSWER_FIELD_CANDIDATES",
    "IMAGE_ID_FIELD_CANDIDATES",
    "SUBSET_MANIFEST_VERSION",
    "build_paper_subset",
]
