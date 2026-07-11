"""Build local-image-anchored InfoSeek subsets without remote access."""

from __future__ import annotations

import csv
import json
import random
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from PIL import Image, UnidentifiedImageError

from .full_image_layout import dataset_full_image_roots
from .prepare_echosight import DATASET_INFOSEEK


MANIFEST_VERSION = "infoseek-local-subset-v1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
IMAGE_ID_KEYS = ("dataset_image_ids", "image_id", "id", "oven_id")
MAPPING_PATH_KEYS = ("relative_path", "path", "image_path", "impath", "file_name", "filename")
UNRESOLVED_ROW_SAMPLE_LIMIT = 50
SAFE_SUBSET_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def build_infoseek_local_image_subset(
    *,
    root: str | Path,
    subset_name: str = "paper_local_5120_128_seed0",
    sample_train: int = 5120,
    sample_test: int = 128,
    seed: int = 0,
    copy_images: bool = False,
) -> dict[str, object]:
    sample_train = _require_nonnegative_int("sample_train", sample_train)
    sample_test = _require_nonnegative_int("sample_test", sample_test)
    seed = _require_int("seed", seed)

    root_path = Path(root)
    raw_root = root_path / "datasets_mm" / DATASET_INFOSEEK / "raw"
    subsets_root = root_path / "datasets_mm" / DATASET_INFOSEEK / "subsets"
    safe_subset_name = _validate_subset_name(subset_name)
    subset_root = _validated_subset_root(subsets_root, safe_subset_name)
    subset_rel = Path("datasets_mm") / DATASET_INFOSEEK / "subsets" / safe_subset_name

    image_roots = _existing_image_roots(root_path)
    id_mappings = _load_infoseek_id_mappings(raw_root)
    resolution_cache: dict[str, tuple[Path | None, str | None]] = {}
    image_validity_cache: dict[Path, bool] = {}

    train_rows, train_fields = _read_csv(raw_root / "qa_train.csv")
    test_rows, test_fields = _read_csv(raw_root / "qa_test.csv")
    target_image_ids = _collect_target_image_ids((train_rows, test_rows))
    fallback_matches = _scan_target_image_matches(image_roots, target_image_ids)

    rng = random.Random(seed)
    train_selected, train_report = _resolve_split(
        split="train",
        rows=train_rows,
        requested=sample_train,
        rng=rng,
        image_roots=image_roots,
        id_mappings=id_mappings,
        fallback_matches=fallback_matches,
        resolution_cache=resolution_cache,
        image_validity_cache=image_validity_cache,
    )
    test_selected, test_report = _resolve_split(
        split="test",
        rows=test_rows,
        requested=sample_test,
        rng=rng,
        image_roots=image_roots,
        id_mappings=id_mappings,
        fallback_matches=fallback_matches,
        resolution_cache=resolution_cache,
        image_validity_cache=image_validity_cache,
    )

    copied_images: list[dict[str, object]] = []
    _prepare_subset_root(subset_root, subsets_root=subsets_root, copy_images=copy_images)
    if copy_images:
        copied_images = _copy_selected_images(
            subset_root=subset_root,
            subset_rel=subset_rel,
            selected=(train_selected + test_selected),
        )

    _write_csv(subset_root / "qa_train.csv", train_fields, [item["row"] for item in train_selected])
    _write_csv(subset_root / "qa_test.csv", test_fields, [item["row"] for item in test_selected])

    blockers = []
    for split_name, split_report in (("train", train_report), ("test", test_report)):
        shortfall = int(split_report["shortfall"])
        if shortfall > 0:
            blockers.append(
                {
                    "split": split_name,
                    "message": "requested sample count exceeds locally resolvable rows",
                    "shortfall": shortfall,
                    "available": int(split_report["available"]),
                    "requested": int(split_report["requested"]),
                }
            )

    status = "complete" if not blockers else "subset_incomplete"
    summary = {
        "status": status,
        "requested_train": sample_train,
        "requested_test": sample_test,
        "available_train": train_report["available"],
        "available_test": test_report["available"],
        "selected_train": train_report["selected"],
        "selected_test": test_report["selected"],
        "remote_urls_accessed": False,
    }
    image_root_strings = [path.as_posix() for path in image_roots]
    report = {
        "manifest_version": MANIFEST_VERSION,
        "dataset": DATASET_INFOSEEK,
        "subset_root": subset_rel.as_posix(),
        "seed": seed,
        "copy_images": copy_images,
        "image_roots": image_root_strings,
        "summary": summary,
        "splits": {
            "train": train_report,
            "test": test_report,
        },
        "blockers": blockers,
    }
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dataset": DATASET_INFOSEEK,
        "subset_root": subset_rel.as_posix(),
        "seed": seed,
        "copy_images": copy_images,
        "image_roots": image_root_strings,
        "summary": summary,
        "splits": {
            "train": {
                "requested": train_report["requested"],
                "available": train_report["available"],
                "selected": train_report["selected"],
            },
            "test": {
                "requested": test_report["requested"],
                "available": test_report["available"],
                "selected": test_report["selected"],
            },
        },
        "copied_images": copied_images,
        "blockers": blockers,
    }

    (subset_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (subset_root / "local_subset_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "dataset": DATASET_INFOSEEK,
        "subset_root": subset_rel.as_posix(),
        "manifest_path": (subset_rel / "manifest.json").as_posix(),
        "report_path": (subset_rel / "local_subset_report.json").as_posix(),
        "summary": summary,
        "splits": report["splits"],
        "blockers": blockers,
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


def _validate_subset_name(subset_name: object) -> str:
    if type(subset_name) is not str:
        raise ValueError("subset_name must be a string")
    normalized = subset_name.strip()
    if not normalized:
        raise ValueError("subset_name must be a non-empty trimmed string")
    if normalized != subset_name:
        raise ValueError("subset_name must be a non-empty trimmed string")
    candidate = PurePosixPath(normalized.replace("\\", "/"))
    if (
        candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.parts[0] in {"", ".", ".."}
        or not SAFE_SUBSET_NAME_RE.fullmatch(candidate.parts[0])
    ):
        raise ValueError("subset_name must be a path-safe single directory name")
    return candidate.parts[0]


def _validated_subset_root(subsets_root: Path, subset_name: str) -> Path:
    subset_root = subsets_root / subset_name
    resolved_subsets_root = subsets_root.resolve(strict=False)
    resolved_subset_root = subset_root.resolve(strict=False)
    try:
        resolved_subset_root.relative_to(resolved_subsets_root)
    except ValueError as exc:
        raise ValueError("subset target must stay under datasets_mm/InfoSeek/subsets") from exc
    return resolved_subset_root


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _write_csv(path: Path, source_fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    fieldnames = _csv_fieldnames(source_fieldnames, rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _csv_fieldnames(source_fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
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


def _prepare_subset_root(subset_root: Path, *, subsets_root: Path, copy_images: bool) -> None:
    resolved_subsets_root = subsets_root.resolve(strict=False)
    try:
        subset_root.relative_to(resolved_subsets_root)
    except ValueError as exc:
        raise ValueError("subset target must stay under datasets_mm/InfoSeek/subsets") from exc
    if subset_root.exists():
        shutil.rmtree(subset_root)
    subset_root.mkdir(parents=True, exist_ok=True)
    if copy_images:
        (subset_root / "images").mkdir(parents=True, exist_ok=True)


def _existing_image_roots(root: Path) -> tuple[Path, ...]:
    candidates = list(dataset_full_image_roots(root, DATASET_INFOSEEK))
    fallback_root = root / "datasets_mm" / DATASET_INFOSEEK / "raw" / "images"
    candidates.append(fallback_root)
    ordered: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if candidate.exists() and resolved not in seen:
            seen.add(resolved)
            ordered.append(candidate)
    return tuple(ordered)


def _load_infoseek_id_mappings(raw_root: Path) -> dict[str, str]:
    mapping_path = raw_root / "images" / "oven" / "metadata" / "ovenid2impath.csv"
    mappings: dict[str, str] = {}
    if not mapping_path.is_file():
        return mappings
    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        id_key = _first_present(fieldnames, ("oven_id", "image_id", "dataset_image_ids", "id"))
        path_key = _first_present(fieldnames, MAPPING_PATH_KEYS)
        if id_key is None or path_key is None:
            return mappings
        for row in reader:
            image_id = _csv_text(row.get(id_key))
            mapped = _csv_text(row.get(path_key))
            if image_id and mapped and image_id not in mappings:
                mappings[image_id] = mapped
    return mappings


def _first_present(fieldnames: Sequence[str], candidates: Sequence[str]) -> str | None:
    present = set(fieldnames)
    for candidate in candidates:
        if candidate in present:
            return candidate
    return None


def _resolve_split(
    *,
    split: str,
    rows: Sequence[Mapping[str, str]],
    requested: int,
    rng: random.Random,
    image_roots: Sequence[Path],
    id_mappings: Mapping[str, str],
    fallback_matches: Mapping[str, Sequence[Path]],
    resolution_cache: dict[str, tuple[Path | None, str | None]],
    image_validity_cache: dict[Path, bool],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    resolved: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        image_field, image_id = _image_id_from_row(row)
        if not image_id:
            unresolved_rows.append(
                _unresolved_row(split, row_index, image_field, image_id, "missing image id")
            )
            continue
        image_path, failure_reason = _resolve_local_image_path(
            image_id=image_id,
            image_roots=image_roots,
            id_mappings=id_mappings,
            fallback_matches=fallback_matches,
            resolution_cache=resolution_cache,
            image_validity_cache=image_validity_cache,
        )
        if image_path is None:
            unresolved_rows.append(
                _unresolved_row(
                    split,
                    row_index,
                    image_field,
                    image_id,
                    failure_reason or "no local image file",
                )
            )
            continue
        resolved.append(
            {
                "row": dict(row),
                "split": split,
                "row_index": row_index,
                "image_id": image_id,
                "image_path": image_path,
            }
        )

    ordered = list(resolved)
    rng.shuffle(ordered)
    selected = ordered[:requested]
    selected.sort(key=lambda item: int(item["row_index"]))
    report = {
        "requested": requested,
        "available": len(resolved),
        "selected": len(selected),
        "shortfall": max(0, requested - len(selected)),
        "unresolved_rows": unresolved_rows[:UNRESOLVED_ROW_SAMPLE_LIMIT],
    }
    return selected, report


def _image_id_from_row(row: Mapping[str, str]) -> tuple[str | None, str | None]:
    for key in IMAGE_ID_KEYS:
        image_id = _first_image_id(row.get(key))
        if image_id:
            return key, image_id
    return IMAGE_ID_KEYS[0], None


def _first_image_id(value: object) -> str:
    text = _csv_text(value)
    if not text:
        return ""
    for separator in ("|", "&&", ",", ";"):
        if separator in text:
            text = text.split(separator)[0].strip()
    return text


def _resolve_local_image_path(
    *,
    image_id: str,
    image_roots: Sequence[Path],
    id_mappings: Mapping[str, str],
    fallback_matches: Mapping[str, Sequence[Path]],
    resolution_cache: dict[str, tuple[Path | None, str | None]],
    image_validity_cache: dict[Path, bool],
) -> tuple[Path | None, str | None]:
    cached = resolution_cache.get(image_id)
    if cached is not None:
        return cached

    mapped_relative = _safe_relative_path(id_mappings.get(image_id))
    mapped_matches: list[Path] = []
    if mapped_relative is not None:
        for image_root in image_roots:
            candidate = _resolve_relative_under_root(image_root, mapped_relative)
            if candidate is not None and candidate.is_file():
                mapped_matches.append(candidate)
        unique_mapped = _unique_paths(mapped_matches)
        valid_mapped = _valid_image_candidates(unique_mapped, image_validity_cache)
        if len(valid_mapped) == 1:
            resolution_cache[image_id] = (valid_mapped[0], None)
            return resolution_cache[image_id]
        if len(valid_mapped) > 1:
            resolution_cache[image_id] = (None, "ambiguous mapped local image path")
            return resolution_cache[image_id]
        if unique_mapped:
            resolution_cache[image_id] = (None, "invalid local image file")
            return resolution_cache[image_id]

    stem_matches = _unique_paths(fallback_matches.get(image_id, ()))
    valid_stem_matches = _valid_image_candidates(stem_matches, image_validity_cache)
    if len(valid_stem_matches) == 1:
        resolution_cache[image_id] = (valid_stem_matches[0], None)
        return resolution_cache[image_id]
    if len(valid_stem_matches) > 1:
        resolution_cache[image_id] = (None, "ambiguous duplicate local image matches")
        return resolution_cache[image_id]
    if stem_matches:
        resolution_cache[image_id] = (None, "invalid local image file")
        return resolution_cache[image_id]

    resolution_cache[image_id] = (None, "no local image file")
    return resolution_cache[image_id]


def _collect_target_image_ids(
    row_groups: Sequence[Sequence[Mapping[str, str]]],
) -> set[str]:
    target_ids: set[str] = set()
    for rows in row_groups:
        for row in rows:
            _, image_id = _image_id_from_row(row)
            if image_id:
                target_ids.add(image_id)
    return target_ids


def _scan_target_image_matches(
    image_roots: Sequence[Path],
    target_image_ids: set[str],
) -> dict[str, list[Path]]:
    matches: dict[str, list[Path]] = {image_id: [] for image_id in target_image_ids}
    if not target_image_ids:
        return matches
    for image_root in image_roots:
        for candidate in sorted(image_root.rglob("*"), key=lambda path: path.as_posix()):
            if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_id = candidate.stem
            if image_id not in target_image_ids:
                continue
            matches[image_id].append(candidate)
    return {image_id: _unique_paths(paths) for image_id, paths in matches.items()}


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    ordered: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def _valid_image_candidates(
    paths: Sequence[Path],
    image_validity_cache: dict[Path, bool],
) -> list[Path]:
    return [path for path in paths if _is_decodable_image(path, image_validity_cache)]


def _is_decodable_image(path: Path, image_validity_cache: dict[Path, bool]) -> bool:
    cached = image_validity_cache.get(path)
    if cached is not None:
        return cached
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError):
        image_validity_cache[path] = False
    else:
        image_validity_cache[path] = True
    return image_validity_cache[path]


def _safe_relative_path(value: object) -> PurePosixPath | None:
    text = _csv_text(value).replace("\\", "/")
    if not text or "://" in text:
        return None
    candidate = PurePosixPath(text)
    if candidate.is_absolute():
        return None
    if ".." in candidate.parts or any(part in {"", "."} for part in candidate.parts):
        return None
    return candidate


def _resolve_relative_under_root(root: Path, relative_path: PurePosixPath) -> Path | None:
    candidate = root.joinpath(*relative_path.parts)
    resolved_root = root.resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved_candidate


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


def _copy_selected_images(
    *,
    subset_root: Path,
    subset_rel: Path,
    selected: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    images_dir = subset_root / "images"
    copied: list[dict[str, object]] = []
    used_names: set[str] = set()
    for item in selected:
        image_path = item["image_path"]
        assert isinstance(image_path, Path)
        image_id = str(item["image_id"])
        destination_name = _unique_destination_name(image_id, image_path.suffix.lower(), used_names)
        destination = images_dir / destination_name
        shutil.copy2(image_path, destination)
        copied.append(
            {
                "split": item["split"],
                "row_index": item["row_index"],
                "image_id": image_id,
                "source_path": str(image_path),
                "subset_path": (subset_rel / "images" / destination_name).as_posix(),
            }
        )
    return copied


def _unique_destination_name(image_id: str, suffix: str, used_names: set[str]) -> str:
    suffix = suffix or ".jpg"
    candidate = f"{image_id}{suffix}"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    index = 1
    while True:
        candidate = f"{image_id}_{index}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1


def _csv_text(value: object) -> str:
    return "" if value is None else str(value).strip()
