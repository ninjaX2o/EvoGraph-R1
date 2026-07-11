"""Selected-image candidate planning for EchoSight subsets."""

from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping, Sequence

from .prepare_echosight import (
    DATASET_E_VQA,
    DATASET_INFOSEEK,
    DATASETS_MM_ROOT,
    SUPPORTED_DATASETS,
    raw_dataset_root,
)
from .subset_echosight import _image_id_from_row, _read_csv


SELECTED_CANDIDATE_MANIFEST_VERSION = "echosight-selected-image-candidates-v1"
HF_TOKEN_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
)
DEFAULT_HF_TOKEN_ENV_FILE = Path("docs") / "huggingface.env"
DEFAULT_E_VQA_SOURCES = ("gldv2", "inaturalist_2021")
DEFAULT_INFOSEEK_SOURCES = ("oven",)
SKIPPED_ROW_SAMPLE_LIMIT = 50


@dataclass(frozen=True)
class CandidatePlanConfig:
    sample_train: int = 5120
    sample_test: int = 128
    seed: int = 0
    buffer_ratio: float = 0.25
    enabled_sources: tuple[str, ...] | None = None


@dataclass(frozen=True)
class HFToken:
    value: str | None = field(repr=False)
    source: str | None


def classify_image_source(dataset: str, image_id: str) -> str:
    """Return the selected-image source family for an EchoSight image id."""

    normalized = str(image_id).strip()
    if dataset == DATASET_E_VQA:
        if normalized.isdigit():
            return "inaturalist_2021"
        if re.fullmatch(r"[0-9a-fA-F]{16}", normalized):
            return "gldv2"
        return "unknown"
    if dataset == DATASET_INFOSEEK:
        if normalized.startswith("oven_"):
            return "oven"
        return "unknown"
    return "unknown"


def load_hf_token(env_file: str | Path | None = None) -> HFToken:
    """Load a Hugging Face token from environment variables, then an env file."""

    for name in HF_TOKEN_ENV_NAMES:
        value = os.environ.get(name)
        if value and value.strip():
            return HFToken(value=value.strip(), source=f"env:{name}")

    path = DEFAULT_HF_TOKEN_ENV_FILE if env_file is None else Path(env_file)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return HFToken(value=None, source=None)
    except (OSError, UnicodeError) as exc:
        names = ", ".join(HF_TOKEN_ENV_NAMES)
        raise ValueError(
            f"unable to read Hugging Face token env file {path}; "
            f"accepted variable names: {names}"
        ) from exc

    for line in lines:
        key, value = _parse_env_line(line)
        if key in HF_TOKEN_ENV_NAMES and value:
            return HFToken(value=value, source=f"file:{key}")
    return HFToken(value=None, source=None)


def token_report(token: HFToken) -> dict[str, object]:
    """Return redacted token metadata suitable for reports and logs."""

    return {
        "present": bool(token.value),
        "source": token.source,
        "redacted": bool(token.value),
    }


def plan_selected_image_candidates(
    root: str | Path,
    dataset: str,
    config: CandidatePlanConfig,
) -> dict[str, object]:
    """Write and return a deterministic selected-image candidate manifest."""

    if dataset not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"unsupported EchoSight dataset: {dataset}; expected {supported}")

    normalized_config = _normalize_config(config)
    root_path = Path(root)
    raw_root = root_path / raw_dataset_root(dataset)
    train_rows, _ = _read_csv(raw_root / "qa_train.csv")
    test_rows, _ = _read_csv(raw_root / "qa_test.csv")
    enabled_sources = _enabled_sources(dataset, normalized_config.enabled_sources)

    subset_rel = _subset_relative_path(dataset, normalized_config)
    manifest_rel = _safe_relative_path(
        subset_rel / "selected_image_candidates.json",
        "candidate manifest path must be relative and stay under the subset root",
    )
    manifest_path = _manifest_output_path(root_path, dataset, manifest_rel)

    rng = random.Random(normalized_config.seed)
    candidates: list[dict[str, object]] = []
    split_reports: dict[str, dict[str, object]] = {}

    split_reports["train"] = _plan_split(
        dataset=dataset,
        split="train",
        rows=train_rows,
        requested=normalized_config.sample_train,
        buffer_ratio=normalized_config.buffer_ratio,
        enabled_sources=enabled_sources,
        rng=rng,
        candidates=candidates,
    )
    split_reports["test"] = _plan_split(
        dataset=dataset,
        split="test",
        rows=test_rows,
        requested=normalized_config.sample_test,
        buffer_ratio=normalized_config.buffer_ratio,
        enabled_sources=enabled_sources,
        rng=rng,
        candidates=candidates,
    )

    manifest = {
        "manifest_version": SELECTED_CANDIDATE_MANIFEST_VERSION,
        "dataset": dataset,
        "subset_root": subset_rel.as_posix(),
        "manifest_path": manifest_rel.as_posix(),
        "seed": normalized_config.seed,
        "sample_train": normalized_config.sample_train,
        "sample_test": normalized_config.sample_test,
        "buffer_ratio": normalized_config.buffer_ratio,
        "enabled_sources": list(enabled_sources),
        "source_files": {
            "train": str(raw_root / "qa_train.csv"),
            "test": str(raw_root / "qa_test.csv"),
        },
        "splits": split_reports,
        "summary": _candidate_summary(candidates, split_reports),
        "candidates": candidates,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _normalize_config(config: CandidatePlanConfig) -> CandidatePlanConfig:
    return CandidatePlanConfig(
        sample_train=_require_nonnegative_int("sample_train", config.sample_train),
        sample_test=_require_nonnegative_int("sample_test", config.sample_test),
        seed=_require_int("seed", config.seed),
        buffer_ratio=_require_nonnegative_float("buffer_ratio", config.buffer_ratio),
        enabled_sources=config.enabled_sources,
    )


def _require_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _require_nonnegative_int(name: str, value: object) -> int:
    value = _require_int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _require_nonnegative_float(name: str, value: object) -> float:
    if type(value) not in {float, int}:
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _enabled_sources(dataset: str, configured: tuple[str, ...] | None) -> tuple[str, ...]:
    allowed_sources = _allowed_sources(dataset)
    if configured is None:
        configured = allowed_sources
    enabled: list[str] = []
    seen: set[str] = set()
    for source in configured:
        source_name = str(source).strip()
        if not source_name or source_name in seen:
            continue
        if source_name not in allowed_sources:
            supported = ", ".join(allowed_sources)
            raise ValueError(
                f"unsupported selected-image source for {dataset}: {source_name}; "
                f"expected one of {supported}"
            )
        enabled.append(source_name)
        seen.add(source_name)
    return tuple(enabled)


def _allowed_sources(dataset: str) -> tuple[str, ...]:
    if dataset == DATASET_E_VQA:
        return DEFAULT_E_VQA_SOURCES
    if dataset == DATASET_INFOSEEK:
        return DEFAULT_INFOSEEK_SOURCES
    supported = ", ".join(SUPPORTED_DATASETS)
    raise ValueError(f"unsupported EchoSight dataset: {dataset}; expected {supported}")


def _subset_relative_path(dataset: str, config: CandidatePlanConfig) -> Path:
    return _safe_relative_path(
        Path(DATASETS_MM_ROOT)
        / dataset
        / "subsets"
        / f"paper_{config.sample_train}_{config.sample_test}_seed{config.seed}",
        "subset path must be relative and stay under datasets_mm",
    )


def _safe_relative_path(path: str | Path, message: str) -> Path:
    raw = str(path).replace("\\", "/")
    posix_path = PurePosixPath(raw)
    windows_path = PureWindowsPath(str(path))
    if (
        not raw
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError(f"{message}: {path}")
    return Path(*posix_path.parts)


def _manifest_output_path(root_path: Path, dataset: str, manifest_rel: Path) -> Path:
    subsets_root = root_path / DATASETS_MM_ROOT / dataset / "subsets"
    target = root_path / manifest_rel
    subsets_resolved = subsets_root.resolve()
    target_resolved = target.resolve()
    try:
        target_resolved.relative_to(subsets_resolved)
    except ValueError as exc:
        raise ValueError(
            f"candidate manifest output must stay under {subsets_root}: {target}"
        ) from exc
    return target_resolved


def _plan_split(
    *,
    dataset: str,
    split: str,
    rows: Sequence[Mapping[str, str]],
    requested: int,
    buffer_ratio: float,
    enabled_sources: tuple[str, ...],
    rng: random.Random,
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    order = list(range(len(rows)))
    rng.shuffle(order)

    target_candidates = math.ceil(requested * (1 + buffer_ratio))
    skipped_counts = {
        "missing_image_id": 0,
        "unknown_source": 0,
        "disabled_source": 0,
    }
    skipped_rows: list[dict[str, object]] = []
    source_counts: dict[str, int] = {}
    scanned_rows = 0
    split_start = len(candidates)

    for row_index in order:
        if len(candidates) - split_start >= target_candidates:
            break
        scanned_rows += 1
        row = rows[row_index]
        image_field, image_id = _image_id_from_row(row)
        if not image_id:
            skipped_counts["missing_image_id"] += 1
            _append_skipped_row(
                skipped_rows,
                split,
                row_index,
                image_field,
                image_id,
                "missing image id",
            )
            continue
        source = classify_image_source(dataset, image_id)
        if source == "unknown":
            skipped_counts["unknown_source"] += 1
            _append_skipped_row(
                skipped_rows,
                split,
                row_index,
                image_field,
                image_id,
                "unknown source",
            )
            continue
        if source not in enabled_sources:
            skipped_counts["disabled_source"] += 1
            _append_skipped_row(
                skipped_rows,
                split,
                row_index,
                image_field,
                image_id,
                f"source disabled: {source}",
            )
            continue

        candidate = {
            "dataset": dataset,
            "split": split,
            "row_index": row_index,
            "image_field": image_field,
            "image_id": image_id,
            "source": source,
        }
        candidates.append(candidate)
        source_counts[source] = source_counts.get(source, 0) + 1

    split_candidates = candidates[split_start:]
    return {
        "dataset": dataset,
        "split": split,
        "requested": requested,
        "target_candidates": target_candidates,
        "buffer_candidates": max(0, target_candidates - requested),
        "source_rows": len(rows),
        "scanned_rows": scanned_rows,
        "candidate_count": len(split_candidates),
        "unique_image_ids": len({str(item["image_id"]) for item in split_candidates}),
        "source_counts": source_counts,
        "skipped": skipped_counts,
        "skipped_rows": skipped_rows,
        "complete": len(split_candidates) >= target_candidates,
    }


def _append_skipped_row(
    skipped_rows: list[dict[str, object]],
    split: str,
    row_index: int,
    image_field: str | None,
    image_id: str | None,
    reason: str,
) -> None:
    if len(skipped_rows) >= SKIPPED_ROW_SAMPLE_LIMIT:
        return
    skipped_rows.append(
        {
            "split": split,
            "row_index": row_index,
            "image_field": image_field,
            "image_id": image_id,
            "reason": reason,
        }
    )


def _candidate_summary(
    candidates: Sequence[Mapping[str, object]],
    splits: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    source_counts: dict[str, int] = {}
    unique_by_source: dict[str, set[str]] = {}
    unique_ids: set[str] = set()
    for candidate in candidates:
        source = str(candidate["source"])
        image_id = str(candidate["image_id"])
        source_counts[source] = source_counts.get(source, 0) + 1
        unique_by_source.setdefault(source, set()).add(image_id)
        unique_ids.add(image_id)

    return {
        "candidate_count": len(candidates),
        "unique_image_ids": len(unique_ids),
        "source_counts": source_counts,
        "unique_source_counts": {
            source: len(image_ids)
            for source, image_ids in sorted(unique_by_source.items())
        },
        "complete": all(bool(split["complete"]) for split in splits.values()),
    }


def _parse_env_line(line: str) -> tuple[str | None, str | None]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None, None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if key.startswith("export "):
        key = key[len("export ") :].strip()
    value = value.strip()
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value.startswith(("'", '"'))
    ):
        value = value[1:-1]
    return key, value or None


__all__ = [
    "CandidatePlanConfig",
    "HFToken",
    "classify_image_source",
    "load_hf_token",
    "plan_selected_image_candidates",
    "token_report",
]
