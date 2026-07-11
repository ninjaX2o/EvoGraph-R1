"""EchoSight/E-VQA/InfoSeek asset manifest planning.

This module defines the local asset contract only. It does not download,
extract, parse, audit, or freeze any concrete EchoSight dataset schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from .selected_image_download import SelectedDownloadLimits


def execute_selected_downloads(*args, **kwargs):
    from .selected_image_download import execute_selected_downloads as _execute

    return _execute(*args, **kwargs)


def extract_selected_tar_members(*args, **kwargs):
    from .selected_image_download import extract_selected_tar_members as _extract

    return _extract(*args, **kwargs)


def build_infoseek_local_image_subset(*args, **kwargs):
    from .infoseek_local_subset import (
        build_infoseek_local_image_subset as _build_infoseek_local_image_subset,
    )

    return _build_infoseek_local_image_subset(*args, **kwargs)


DATASET_E_VQA = "E-VQA"
DATASET_INFOSEEK = "InfoSeek"
SUPPORTED_DATASETS = (DATASET_E_VQA, DATASET_INFOSEEK)
DATASETS_MM_ROOT = "datasets_mm"
MANIFEST_VERSION = "echosight-b1-asset-manifest-v1"
NO_DOWNLOAD_POLICY = (
    "This command only plans and validates local user-provided assets; downloads "
    "and sample audits are opt-in."
)
DEFAULT_HF_TOKEN_ENV_FILE = ".env"
DEFAULT_SELECTED_IMAGE_BUFFER_RATIO = 0.25
DEFAULT_SELECTED_IMAGE_MAX_TRANSFER_GB = 120.0
DEFAULT_SELECTED_IMAGE_REQUEST_DELAY_SECONDS = 1.0
DEFAULT_SELECTED_IMAGE_RATE_LIMIT_COOLDOWN_SECONDS = 600.0
DEFAULT_OVEN_MAX_SHARDS = 5
SELECTED_METADATA_MAX_BYTES = 2_000_000_000
GLDV2_METADATA_REL = (
    PurePosixPath(DATASETS_MM_ROOT)
    / DATASET_E_VQA
    / "raw"
    / "images"
    / "google_landmarks_v2"
    / "metadata"
    / "train.csv"
)
OVEN_MAPPING_REL = (
    PurePosixPath(DATASETS_MM_ROOT)
    / DATASET_INFOSEEK
    / "raw"
    / "images"
    / "oven"
    / "metadata"
    / "ovenid2impath.csv"
)
EVQA_INATURALIST_IMAGE_ROOT_REL = (
    PurePosixPath(DATASETS_MM_ROOT)
    / DATASET_E_VQA
    / "raw"
    / "images"
)


@dataclass(frozen=True)
class AssetDefinition:
    """Static B1 asset catalog entry."""

    dataset: str
    asset_id: str
    asset_type: str
    source_note: str
    source_filename: str | None
    expected_filename: str
    local_path: str
    required: bool
    expected_kind: str
    size_bytes: int | None = None
    sha256: str | None = None
    unpacked_state: str = "not_applicable"
    sampled_state: str = "not_audited"

    def to_manifest_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_checksum_plan_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "local_path": self.local_path,
            "expected_kind": self.expected_kind,
            "required": self.required,
            "algorithm": "sha256",
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "status": "placeholder_pending_manual_verification",
        }


@dataclass(frozen=True)
class MissingAsset:
    dataset: str
    asset_id: str
    local_path: str
    expected_kind: str
    required: bool
    source_filename: str | None = None
    manual_remediation_hint: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AssetStatus:
    dataset: str
    asset_id: str
    local_path: str
    expected_kind: str
    required: bool
    exists: bool
    size_bytes: int | None
    sha256: str | None
    download_attempted: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    checked_root: str
    checked_assets: tuple[AssetStatus, ...]
    missing_required: tuple[MissingAsset, ...]
    policy: str = NO_DOWNLOAD_POLICY

    @property
    def present_assets(self) -> tuple[AssetStatus, ...]:
        return tuple(status for status in self.checked_assets if status.exists)

    def to_dict(self) -> dict[str, object]:
        return {
            "checked_root": self.checked_root,
            "policy": self.policy,
            "checked_assets": [status.to_dict() for status in self.checked_assets],
            "missing_required": [missing.to_dict() for missing in self.missing_required],
        }


class MissingRequiredAssetsError(ValueError):
    """Raised when required B1 assets are absent from the local layout."""

    def __init__(self, missing_assets: Sequence[MissingAsset]) -> None:
        self.missing_assets = tuple(missing_assets)
        lines = [
            f"Missing required EchoSight assets ({len(self.missing_assets)}):",
        ]
        lines.extend(
            f"- {asset.asset_id}: {asset.local_path} ({asset.expected_kind}); "
            f"source_filename={asset.source_filename or 'manual_directory'}"
            for asset in self.missing_assets
        )
        lines.append("Manual remediation hints:")
        lines.extend(
            f"- {asset.asset_id}: {asset.manual_remediation_hint}"
            for asset in self.missing_assets
        )
        lines.append("Place these assets manually or use the explicit download options.")
        super().__init__("\n".join(lines))


def _require_supported_dataset(dataset: str) -> str:
    if dataset not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"unsupported dataset: {dataset}; expected one of {supported}")
    return dataset


def _normalize_relative_path(relative_path: str | Path | PurePosixPath) -> PurePosixPath:
    normalized = str(relative_path).replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"relative_path must stay under the dataset raw root: {relative_path}")
    return path


def _raw_root(dataset: str) -> PurePosixPath:
    _require_supported_dataset(dataset)
    return PurePosixPath(DATASETS_MM_ROOT) / dataset / "raw"


def raw_dataset_root(dataset: str) -> str:
    """Return the canonical EchoSight raw root for a supported dataset."""

    return str(_raw_root(dataset))


def local_raw_asset_path(dataset: str, relative_path: str | Path | PurePosixPath) -> str:
    """Normalize a raw-root-relative asset path to the B1 local layout."""

    return str(_raw_root(dataset) / _normalize_relative_path(relative_path))


def _asset(
    *,
    dataset: str,
    asset_id: str,
    asset_type: str,
    source_note: str,
    source_filename: str | None,
    relative_path: str,
    required: bool,
    expected_kind: str = "file",
    unpacked_state: str = "not_applicable",
) -> AssetDefinition:
    local_path = PurePosixPath(local_raw_asset_path(dataset, relative_path))
    expected_filename = _normalize_relative_path(relative_path).name
    if expected_kind == "directory":
        expected_filename += "/"

    return AssetDefinition(
        dataset=dataset,
        asset_id=asset_id,
        asset_type=asset_type,
        source_note=source_note,
        source_filename=source_filename,
        expected_filename=expected_filename,
        local_path=str(local_path),
        required=required,
        expected_kind=expected_kind,
        unpacked_state=unpacked_state,
    )


def _catalog() -> tuple[AssetDefinition, ...]:
    e_vqa = DATASET_E_VQA
    infoseek = DATASET_INFOSEEK
    return (
        _asset(
            dataset=e_vqa,
            asset_id="e_vqa_kb_archive",
            asset_type="kb_archive",
            source_note=(
                "Manual EchoSight E-VQA 2M Knowledge Base archive; place the "
                "downloaded asset under raw/kb."
            ),
            source_filename="encyclopedic_kb_wiki.zip",
            relative_path="kb/encyclopedic_kb_wiki.zip",
            required=True,
            unpacked_state="planned",
        ),
        _asset(
            dataset=e_vqa,
            asset_id="e_vqa_faiss_archive",
            asset_type="faiss_archive",
            source_note=(
                "Manual EchoSight E-VQA KB Images FAISS archive; place the "
                "downloaded asset under raw/faiss."
            ),
            source_filename="evqa_2M_faiss_index.zip",
            relative_path="faiss/evqa_2M_faiss_index.zip",
            required=True,
            unpacked_state="planned",
        ),
        _asset(
            dataset=e_vqa,
            asset_id="e_vqa_vqa_train_csv",
            asset_type="vqa_csv",
            source_note=(
                "Manual EchoSight E-VQA train CSV; normalize the local name "
                "to qa_train.csv without auditing its columns in asset planning."
            ),
            source_filename="train_full_image_cleaned.csv",
            relative_path="qa_train.csv",
            required=True,
        ),
        _asset(
            dataset=e_vqa,
            asset_id="e_vqa_vqa_dev_csv",
            asset_type="vqa_csv",
            source_note=(
                "Manual EchoSight E-VQA validation CSV; normalize the local "
                "name to qa_dev.csv without auditing its columns in asset planning."
            ),
            source_filename="val.csv",
            relative_path="qa_dev.csv",
            required=True,
        ),
        _asset(
            dataset=e_vqa,
            asset_id="e_vqa_vqa_test_csv",
            asset_type="vqa_csv",
            source_note=(
                "Manual EchoSight E-VQA test CSV; normalize the local name "
                "to qa_test.csv without auditing its columns in asset planning."
            ),
            source_filename="test.csv",
            relative_path="qa_test.csv",
            required=True,
        ),
        _asset(
            dataset=e_vqa,
            asset_id="e_vqa_image_source_inaturalist_2021",
            asset_type="image_source",
            source_note=(
                "Manual iNaturalist 2021 image source directory referenced by "
                "E-VQA. asset planning records the local directory contract only."
            ),
            source_filename=None,
            relative_path="images/inaturalist_2021",
            required=True,
            expected_kind="directory",
        ),
        _asset(
            dataset=e_vqa,
            asset_id="e_vqa_image_source_google_landmarks_v2",
            asset_type="image_source",
            source_note=(
                "Manual Google Landmarks Dataset V2 image source directory "
                "referenced by E-VQA. asset planning records the local directory "
                "contract only."
            ),
            source_filename=None,
            relative_path="images/google_landmarks_v2",
            required=True,
            expected_kind="directory",
        ),
        _asset(
            dataset=e_vqa,
            asset_id="e_vqa_train_id2name",
            asset_type="id_mapping",
            source_note="Manual EchoSight E-VQA train image id-to-name mapping.",
            source_filename="train_id2name.json",
            relative_path="id2name/train_id2name.json",
            required=True,
        ),
        _asset(
            dataset=e_vqa,
            asset_id="e_vqa_val_id2name",
            asset_type="id_mapping",
            source_note="Manual EchoSight E-VQA validation image id-to-name mapping.",
            source_filename="val_id2name.json",
            relative_path="id2name/val_id2name.json",
            required=True,
        ),
        _asset(
            dataset=e_vqa,
            asset_id="e_vqa_echosight_hard_negatives",
            asset_type="hard_negative_file",
            source_note=(
                "Optional EchoSight hard negative file for later reranker "
                "work; asset planning does not require or audit it."
            ),
            source_filename=None,
            relative_path="reranker/hard_negatives.jsonl",
            required=False,
        ),
        _asset(
            dataset=e_vqa,
            asset_id="e_vqa_reranker_checkpoint",
            asset_type="reranker_checkpoint",
            source_note=(
                "Optional EchoSight reranker checkpoint directory for later "
                "work; asset planning records only the local placement."
            ),
            source_filename=None,
            relative_path="reranker/checkpoint",
            required=False,
            expected_kind="directory",
        ),
        _asset(
            dataset=e_vqa,
            asset_id="e_vqa_reranker_training_artifacts",
            asset_type="reranker_training_artifacts",
            source_note=(
                "Optional EchoSight reranker training artifacts directory; "
                "not required for asset planning."
            ),
            source_filename=None,
            relative_path="reranker/training_artifacts",
            required=False,
            expected_kind="directory",
        ),
        _asset(
            dataset=e_vqa,
            asset_id="e_vqa_reranker_inference_artifacts",
            asset_type="reranker_inference_artifacts",
            source_note=(
                "Optional EchoSight reranker inference artifacts directory; "
                "not required for asset planning."
            ),
            source_filename=None,
            relative_path="reranker/inference_artifacts",
            required=False,
            expected_kind="directory",
        ),
        _asset(
            dataset=infoseek,
            asset_id="infoseek_kb_archive",
            asset_type="kb_archive",
            source_note=(
                "Manual EchoSight InfoSeek 100K Knowledge Base archive; place "
                "the downloaded asset under raw/kb."
            ),
            source_filename="infoseek_100k_wiki.zip",
            relative_path="kb/infoseek_100k_wiki.zip",
            required=True,
            unpacked_state="planned",
        ),
        _asset(
            dataset=infoseek,
            asset_id="infoseek_faiss_archive",
            asset_type="faiss_archive",
            source_note=(
                "Manual EchoSight InfoSeek KB Images FAISS archive; place the "
                "downloaded asset under raw/faiss."
            ),
            source_filename="infoseek_100k_faiss_index.zip",
            relative_path="faiss/infoseek_100k_faiss_index.zip",
            required=True,
            unpacked_state="planned",
        ),
        _asset(
            dataset=infoseek,
            asset_id="infoseek_vqa_train_csv",
            asset_type="vqa_csv",
            source_note=(
                "Manual EchoSight InfoSeek train CSV; normalize the local "
                "name to qa_train.csv without auditing its columns in asset planning."
            ),
            source_filename="infoseek_train_filtered.csv",
            relative_path="qa_train.csv",
            required=True,
        ),
        _asset(
            dataset=infoseek,
            asset_id="infoseek_vqa_test_csv",
            asset_type="vqa_csv",
            source_note=(
                "Manual EchoSight InfoSeek test CSV; normalize the local "
                "name to qa_test.csv without auditing its columns in asset planning."
            ),
            source_filename="infoseek_test_filtered.csv",
            relative_path="qa_test.csv",
            required=True,
        ),
        _asset(
            dataset=infoseek,
            asset_id="infoseek_image_source_oven",
            asset_type="image_source",
            source_note=(
                "Manual Oven image downloads directory referenced by "
                "InfoSeek. asset planning records the local directory contract only."
            ),
            source_filename=None,
            relative_path="images/oven",
            required=True,
            expected_kind="directory",
        ),
        _asset(
            dataset=infoseek,
            asset_id="infoseek_echosight_hard_negatives",
            asset_type="hard_negative_file",
            source_note=(
                "Optional EchoSight hard negative file for later InfoSeek "
                "reranker work; asset planning does not require or audit it."
            ),
            source_filename=None,
            relative_path="reranker/hard_negatives.jsonl",
            required=False,
        ),
        _asset(
            dataset=infoseek,
            asset_id="infoseek_reranker_checkpoint",
            asset_type="reranker_checkpoint",
            source_note=(
                "Optional EchoSight reranker checkpoint directory for later "
                "InfoSeek work; asset planning records only the local placement."
            ),
            source_filename=None,
            relative_path="reranker/checkpoint",
            required=False,
            expected_kind="directory",
        ),
        _asset(
            dataset=infoseek,
            asset_id="infoseek_reranker_training_artifacts",
            asset_type="reranker_training_artifacts",
            source_note=(
                "Optional EchoSight reranker training artifacts directory for "
                "InfoSeek; not required for asset planning."
            ),
            source_filename=None,
            relative_path="reranker/training_artifacts",
            required=False,
            expected_kind="directory",
        ),
        _asset(
            dataset=infoseek,
            asset_id="infoseek_reranker_inference_artifacts",
            asset_type="reranker_inference_artifacts",
            source_note=(
                "Optional EchoSight reranker inference artifacts directory for "
                "InfoSeek; not required for asset planning."
            ),
            source_filename=None,
            relative_path="reranker/inference_artifacts",
            required=False,
            expected_kind="directory",
        ),
    )


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


def asset_catalog(
    datasets: Iterable[str] | str | None = None,
    *,
    include_optional: bool = True,
    required_only: bool = False,
) -> tuple[AssetDefinition, ...]:
    """Return the static B1 asset catalog."""

    selected = set(_normalize_datasets(datasets))
    assets = tuple(asset for asset in _catalog() if asset.dataset in selected)
    if required_only:
        return tuple(asset for asset in assets if asset.required)
    if not include_optional:
        return tuple(asset for asset in assets if asset.required)
    return assets


def manifest_document(
    datasets: Iterable[str] | str | None = None,
    *,
    include_optional: bool = True,
) -> dict[str, object]:
    """Build a JSON-serializable manifest document."""

    selected = _normalize_datasets(datasets)
    return {
        "manifest_version": MANIFEST_VERSION,
        "datasets": list(selected),
        "local_roots": {
            dataset: str(_raw_root(dataset))
            for dataset in selected
        },
        "policy": {
            "downloads_enabled": False,
            "schema_freeze": False,
            "real_sample_audit": False,
            "note": NO_DOWNLOAD_POLICY,
        },
        "assets": [
            asset.to_manifest_dict()
            for asset in asset_catalog(selected, include_optional=include_optional)
        ],
    }


def checksum_plan_document(
    datasets: Iterable[str] | str | None = None,
    *,
    include_optional: bool = True,
) -> dict[str, object]:
    """Build a JSON-serializable checksum planning document."""

    selected = _normalize_datasets(datasets)
    return {
        "manifest_version": MANIFEST_VERSION,
        "datasets": list(selected),
        "policy": {
            "downloads_enabled": False,
            "checksum_values_are_placeholders": True,
            "note": (
                "asset planning records checksum placeholders and can compute hashes "
                "for files that the user has already placed locally."
            ),
        },
        "assets": [
            asset.to_checksum_plan_dict()
            for asset in asset_catalog(selected, include_optional=include_optional)
        ],
    }


def _repo_relative_path(root: Path, local_path: str) -> Path:
    return root.joinpath(*PurePosixPath(local_path).parts)


def write_planning_files(
    root: str | Path = ".",
    *,
    datasets: Iterable[str] | str | None = None,
    include_optional: bool = True,
    overwrite: bool = True,
) -> dict[str, Mapping[str, Path]]:
    """Write manifest.json and checksums.json planning files.

    Only planning JSON files and their parent raw directories are created. No
    payload directories, archives, CSVs, indexes, images, or model files are
    created or downloaded.
    """

    root_path = Path(root)
    written: dict[str, Mapping[str, Path]] = {}
    for dataset in _normalize_datasets(datasets):
        raw_root = _repo_relative_path(root_path, str(_raw_root(dataset)))
        raw_root.mkdir(parents=True, exist_ok=True)
        manifest_path = raw_root / "manifest.json"
        checksums_path = raw_root / "checksums.json"

        if not overwrite:
            existing = [
                str(path)
                for path in (manifest_path, checksums_path)
                if path.exists()
            ]
            if existing:
                raise FileExistsError(
                    "planning file(s) already exist: " + ", ".join(existing)
                )

        manifest_path.write_text(
            json.dumps(
                manifest_document([dataset], include_optional=include_optional),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        checksums_path.write_text(
            json.dumps(
                checksum_plan_document([dataset], include_optional=include_optional),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        written[dataset] = {
            "manifest": manifest_path,
            "checksums": checksums_path,
        }
    return written


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_exists(path: Path, expected_kind: str) -> bool:
    if expected_kind == "directory":
        return path.is_dir()
    if expected_kind == "file":
        return path.is_file()
    raise ValueError(f"unsupported expected_kind: {expected_kind}")


def validate_required_assets(
    root: str | Path = ".",
    *,
    datasets: Iterable[str] | str | None = None,
    compute_checksums: bool = False,
) -> ValidationReport:
    """Validate required local assets and raise a detailed missing report."""

    root_path = Path(root)
    statuses: list[AssetStatus] = []
    missing: list[MissingAsset] = []

    for asset in asset_catalog(datasets, required_only=True):
        target = _repo_relative_path(root_path, asset.local_path)
        exists = _asset_exists(target, asset.expected_kind)
        size_bytes = target.stat().st_size if exists and target.is_file() else None
        sha256 = _sha256(target) if compute_checksums and exists and target.is_file() else None

        statuses.append(
            AssetStatus(
                dataset=asset.dataset,
                asset_id=asset.asset_id,
                local_path=asset.local_path,
                expected_kind=asset.expected_kind,
                required=asset.required,
                exists=exists,
                size_bytes=size_bytes,
                sha256=sha256,
            )
        )
        if not exists:
            missing.append(
                MissingAsset(
                    dataset=asset.dataset,
                    asset_id=asset.asset_id,
                    local_path=asset.local_path,
                    expected_kind=asset.expected_kind,
                    required=asset.required,
                    source_filename=asset.source_filename,
                    manual_remediation_hint=asset.source_note,
                )
            )

    report = ValidationReport(
        checked_root=str(root_path),
        checked_assets=tuple(statuses),
        missing_required=tuple(missing),
    )
    if missing:
        raise MissingRequiredAssetsError(missing)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or validate EchoSight/E-VQA/InfoSeek local assets.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository or staging root containing datasets_mm/.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=SUPPORTED_DATASETS,
        help="Dataset to include. May be provided more than once.",
    )
    parser.add_argument(
        "--write-plans",
        action="store_true",
        help="Write manifest.json and checksums.json planning files.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate required assets that were manually placed locally.",
    )
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="Omit optional assets from generated planning documents.",
    )
    parser.add_argument(
        "--compute-checksums",
        action="store_true",
        help="Compute sha256 values for present local files during validation.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download required EchoSight direct-file assets into the local layout.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the download/subset plan without writing payload files.",
    )
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        "--include-images",
        action="store_true",
        help="Include image-source providers in the download plan.",
    )
    image_group.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image-source providers. This is the default for paper subsets.",
    )
    image_group.add_argument(
        "--download-full-images",
        action="store_true",
        help="Plan download roots for canonical full-image mirrors.",
    )
    parser.add_argument(
        "--subset-mode",
        choices=("none", "paper"),
        default="none",
        help="Build no subset or the paper-sized deterministic subset.",
    )
    parser.add_argument(
        "--sample-train",
        type=int,
        default=5120,
        help="Training rows requested for --subset-mode paper.",
    )
    parser.add_argument(
        "--sample-test",
        type=int,
        default=128,
        help="Test rows requested for --subset-mode paper.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=0,
        help="Random seed for deterministic paper subset sampling.",
    )
    parser.add_argument(
        "--require-complete-subset",
        action="store_true",
        help="Return exit code 2 unless selected datasets have a complete subset.",
    )
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Treat missing canonical or compatibility image roots as validation blockers.",
    )
    parser.add_argument(
        "--force-large-downloads",
        action="store_true",
        help="Bypass full-image free-space guards in the downloader.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="Optional minimum free-space requirement, in decimal GB, for real downloads.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Append a read-only full asset validation report to JSON output.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="HTTP timeout per request for direct-file downloads.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024,
        help="Streaming chunk size in bytes for direct-file downloads.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum direct-file download attempts for failed plan items.",
    )
    parser.add_argument(
        "--download-selected-images",
        action="store_true",
        help=(
            "Plan and download only images selected for --subset-mode paper, "
            "then build the subset."
        ),
    )
    parser.add_argument(
        "--selected-image-buffer-ratio",
        type=float,
        default=DEFAULT_SELECTED_IMAGE_BUFFER_RATIO,
        help=(
            "Extra selected-image candidates to plan beyond sample counts. "
            "0 plans the exact paper subset counts."
        ),
    )
    parser.add_argument(
        "--max-selected-image-transfer-gb",
        type=float,
        default=DEFAULT_SELECTED_IMAGE_MAX_TRANSFER_GB,
        help="Selected-image transfer cap in decimal GB.",
    )
    parser.add_argument(
        "--selected-image-request-delay-seconds",
        type=float,
        default=DEFAULT_SELECTED_IMAGE_REQUEST_DELAY_SECONDS,
        help="Delay between selected-image download requests in seconds.",
    )
    parser.add_argument(
        "--selected-image-rate-limit-cooldown-seconds",
        type=float,
        default=DEFAULT_SELECTED_IMAGE_RATE_LIMIT_COOLDOWN_SECONDS,
        help="Maximum Retry-After cooldown for selected-image 429 responses.",
    )
    parser.add_argument(
        "--hf-token-env-file",
        default=DEFAULT_HF_TOKEN_ENV_FILE,
        help="Env file used only for Hugging Face token lookup.",
    )
    parser.add_argument(
        "--evqa-selected-sources",
        default="all",
        help="E-VQA selected-image sources: all, gldv2, inaturalist, or comma-separated.",
    )
    parser.add_argument(
        "--oven-max-shards",
        type=int,
        default=DEFAULT_OVEN_MAX_SHARDS,
        help="Maximum OVEN source archives allowed for selected-image extraction.",
    )
    parser.add_argument(
        "--keep-selected-source-archives",
        action="store_true",
        help="Retain downloaded selected-image source archives after successful extraction.",
    )
    parser.add_argument(
        "--build-tar-subset",
        action="store_true",
        help="Build an E-VQA subset from local tar shards without downloading images.",
    )
    parser.add_argument(
        "--build-infoseek-local-subset",
        action="store_true",
        help="Build an InfoSeek subset from locally resolvable images only.",
    )
    parser.add_argument(
        "--infoseek-local-subset-name",
        help="Optional subset directory name for --build-infoseek-local-subset.",
    )
    parser.add_argument(
        "--infoseek-copy-images",
        action="store_true",
        help="Copy selected local images into the rebuilt InfoSeek subset.",
    )
    parser.add_argument(
        "--tar-image-root",
        help="Local directory containing GLDv2-style *.tar.gz shards for --build-tar-subset.",
    )
    parser.add_argument(
        "--tar-subset-name",
        help="Optional subset directory name for --build-tar-subset.",
    )
    parser.add_argument(
        "--tar-limit-archives",
        type=int,
        default=None,
        help="Optional maximum number of tar archives to scan for --build-tar-subset.",
    )
    parser.add_argument(
        "--tar-contact-sheet",
        action="store_true",
        help="Request contact-sheet reporting for --build-tar-subset.",
    )
    return parser


def _uses_b3_cli(args: argparse.Namespace) -> bool:
    return (
        args.download
        or args.download_selected_images
        or args.download_full_images
        or args.require_images
        or args.dry_run
        or args.include_images
        or args.skip_images
        or args.subset_mode != "none"
        or args.require_complete_subset
        or args.force_large_downloads
        or args.min_free_gb is not None
        or args.verify
        or args.timeout_seconds != 60.0
        or args.chunk_size != 1024 * 1024
        or args.max_retries != 1
        or args.selected_image_buffer_ratio != DEFAULT_SELECTED_IMAGE_BUFFER_RATIO
        or args.max_selected_image_transfer_gb != DEFAULT_SELECTED_IMAGE_MAX_TRANSFER_GB
        or args.hf_token_env_file != DEFAULT_HF_TOKEN_ENV_FILE
        or args.evqa_selected_sources != "all"
        or args.oven_max_shards != DEFAULT_OVEN_MAX_SHARDS
        or args.keep_selected_source_archives
        or args.build_tar_subset
        or args.build_infoseek_local_subset
        or args.infoseek_local_subset_name is not None
        or args.infoseek_copy_images
        or args.tar_image_root is not None
        or args.tar_subset_name is not None
        or args.tar_limit_archives is not None
        or args.tar_contact_sheet
    )


def _selected_datasets(datasets: Iterable[str] | str | None) -> tuple[str, ...]:
    return _normalize_datasets(datasets)


def _include_images(args: argparse.Namespace) -> bool:
    if args.skip_images:
        return False
    return bool(args.include_images or args.download_full_images)


def _image_mode(args: argparse.Namespace) -> str:
    return "full" if args.download_full_images else "none"


def _effective_require_images(args: argparse.Namespace) -> bool:
    return bool(args.require_images or args.download_full_images)


def _b3_cli_argument_error(args: argparse.Namespace) -> str | None:
    if args.download_full_images and not args.download:
        return "--download-full-images requires --download."
    if args.build_infoseek_local_subset:
        if args.dataset != [DATASET_INFOSEEK]:
            return "--build-infoseek-local-subset requires --dataset InfoSeek."
    elif args.infoseek_local_subset_name is not None or args.infoseek_copy_images:
        return (
            "--infoseek-local-subset-name and --infoseek-copy-images require "
            "--build-infoseek-local-subset."
        )
    return None


def _tar_subset_blocker(
    *,
    reason: str,
    message: str,
    details: object | None = None,
) -> dict[str, object]:
    blocker: dict[str, object] = {
        "dataset": DATASET_E_VQA,
        "reason": reason,
        "message": message,
    }
    if details is not None:
        blocker["details"] = details
    return blocker


def _validate_tar_subset_config(
    args: argparse.Namespace,
    datasets: Sequence[str],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if tuple(datasets) != (DATASET_E_VQA,):
        blockers.append(
            _tar_subset_blocker(
                reason="tar_subset_requires_e_vqa",
                message="--build-tar-subset requires exactly --dataset E-VQA.",
                details={"datasets": list(datasets)},
            )
        )
    if not args.tar_image_root:
        blockers.append(
            _tar_subset_blocker(
                reason="missing_tar_image_root",
                message="--build-tar-subset requires --tar-image-root.",
            )
        )
    return blockers


def _invalid_tar_subset_config_report(
    args: argparse.Namespace,
    datasets: Sequence[str],
    blockers: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "command": "prepare_echosight",
        "root": str(args.root),
        "datasets": list(datasets),
        "options": _b3_options(args),
        "operations": {
            "tar_subset": {
                "summary": {
                    "status": "invalid_config",
                    "blockers": list(blockers),
                },
                "blockers": list(blockers),
            }
        },
        "summary": {
            "status": "invalid_config",
            "exit_code": 2,
        },
    }


def _run_tar_subset(args: argparse.Namespace) -> dict[str, object]:
    from .tar_subset_echosight import build_tar_anchored_e_vqa_subset

    try:
        return build_tar_anchored_e_vqa_subset(
            root=args.root,
            source_tar_root=args.tar_image_root,
            sample_train=args.sample_train,
            sample_test=args.sample_test,
            seed=args.subset_seed,
            subset_name=args.tar_subset_name,
            limit_archives=args.tar_limit_archives,
            contact_sheet=args.tar_contact_sheet,
        )
    except Exception as exc:
        blocker = _tar_subset_blocker(
            reason="tar_subset_build_failed",
            message="Tar-anchored E-VQA subset build failed.",
            details={"error": f"{type(exc).__name__}: {exc}"},
        )
        return {
            "dataset": DATASET_E_VQA,
            "subset_name": args.tar_subset_name,
            "summary": {
                "status": "tar_subset_build_failed",
                "blockers": [blocker],
            },
            "blockers": [blocker],
        }


def _subset_plan(args: argparse.Namespace) -> dict[str, object]:
    return {
        "mode": args.subset_mode,
        "dry_run": bool(args.dry_run),
        "requested": {
            "sample_train": args.sample_train,
            "sample_test": args.sample_test,
            "seed": args.subset_seed,
        },
        "reports": [],
    }


def _subset_build_failed_report(
    dataset: str,
    args: argparse.Namespace,
    exc: Exception,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "subset_root": (
            f"{DATASETS_MM_ROOT}/{dataset}/subsets/"
            f"paper_{args.sample_train}_{args.sample_test}_seed{args.subset_seed}"
        ),
        "manifest_path": (
            f"{DATASETS_MM_ROOT}/{dataset}/subsets/"
            f"paper_{args.sample_train}_{args.sample_test}_seed{args.subset_seed}/"
            "manifest.json"
        ),
        "summary": {
            "status": "subset_build_failed",
            "complete_train": 0,
            "complete_test": 0,
            "requested_train": args.sample_train,
            "requested_test": args.sample_test,
        },
        "blockers": [
            {
                "dataset": dataset,
                "reason": "subset_build_failed",
                "error": str(exc),
                "remediation": (
                    "Download or place the required QA CSV, id mapping, and selected "
                    "image files under the dataset raw root, then rerun paper subset "
                    "build."
                ),
            }
        ],
    }


def _decimal_gb_to_bytes(
    value: float | int | None,
    *,
    option_name: str,
    default_bytes: int | None = None,
    allow_zero: bool = False,
) -> int:
    if value is None:
        if default_bytes is None:
            raise ValueError(f"{option_name} is required")
        return default_bytes
    numeric = float(value)
    if numeric < 0 or (numeric == 0 and not allow_zero):
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{option_name} must be {comparator}")
    return int(numeric * 1_000_000_000)


def _selected_image_limits(args: argparse.Namespace) -> SelectedDownloadLimits:
    from .selected_image_download import SelectedDownloadLimits

    defaults = SelectedDownloadLimits()
    max_transfer_bytes = _decimal_gb_to_bytes(
        args.max_selected_image_transfer_gb,
        option_name="max-selected-image-transfer-gb",
    )
    min_free_bytes = _decimal_gb_to_bytes(
        args.min_free_gb,
        option_name="min-free-gb",
        default_bytes=defaults.min_free_bytes,
    )
    request_delay_seconds = float(args.selected_image_request_delay_seconds)
    if request_delay_seconds < 0:
        raise ValueError("selected-image-request-delay-seconds must be non-negative")
    rate_limit_cooldown_seconds = float(
        args.selected_image_rate_limit_cooldown_seconds
    )
    if rate_limit_cooldown_seconds < 0:
        raise ValueError(
            "selected-image-rate-limit-cooldown-seconds must be non-negative"
        )
    return SelectedDownloadLimits(
        max_transfer_bytes=max_transfer_bytes,
        min_free_bytes=min_free_bytes,
        timeout_seconds=args.timeout_seconds,
        chunk_size=args.chunk_size,
        max_retries=args.max_retries,
        request_delay_seconds=request_delay_seconds,
        rate_limit_cooldown_seconds=rate_limit_cooldown_seconds,
    )


def _evqa_selected_sources(args: argparse.Namespace) -> tuple[str, ...]:
    raw = str(args.evqa_selected_sources or "all")
    parts = [part.strip().lower() for part in raw.replace(";", ",").split(",")]
    normalized: list[str] = []
    for part in parts:
        if not part:
            continue
        if part == "all":
            for source in ("gldv2", "inaturalist_2021"):
                if source not in normalized:
                    normalized.append(source)
            continue
        if part == "inaturalist":
            part = "inaturalist_2021"
        if part not in {"gldv2", "inaturalist_2021"}:
            raise ValueError(
                "evqa-selected-sources must be all, gldv2, inaturalist, "
                "or a comma-separated combination"
            )
        if part not in normalized:
            normalized.append(part)
    return tuple(normalized or ("gldv2", "inaturalist_2021"))


def _selected_candidate_sources(
    dataset: str,
    args: argparse.Namespace,
) -> tuple[str, ...] | None:
    if dataset == DATASET_E_VQA:
        return _evqa_selected_sources(args)
    if dataset == DATASET_INFOSEEK:
        return ("oven",)
    return None


def _validate_selected_image_config(args: argparse.Namespace) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if not args.download_selected_images:
        return blockers
    if args.subset_mode != "paper":
        blockers.append(
            {
                "reason": "selected_images_require_paper_subset",
                "remediation": "Use --subset-mode paper with --download-selected-images.",
            }
        )
    try:
        _selected_image_limits(args)
    except ValueError as exc:
        blockers.append(
            {
                "reason": "invalid_selected_image_limit",
                "error": str(exc),
                "remediation": "Use non-negative --min-free-gb and positive transfer caps.",
            }
        )
    if args.selected_image_buffer_ratio < 0:
        blockers.append(
            {
                "reason": "invalid_selected_image_buffer_ratio",
                "error": "selected-image-buffer-ratio must be non-negative",
                "remediation": "Use a non-negative selected-image buffer ratio.",
            }
        )
    if args.oven_max_shards < 0:
        blockers.append(
            {
                "reason": "invalid_oven_max_shards",
                "error": "oven-max-shards must be non-negative",
                "remediation": "Use a non-negative OVEN shard limit.",
            }
        )
    try:
        _evqa_selected_sources(args)
    except ValueError as exc:
        blockers.append(
            {
                "reason": "invalid_evqa_selected_sources",
                "error": str(exc),
                "remediation": "Use all, gldv2, inaturalist, or a comma-separated combination.",
            }
        )
    return blockers


def _invalid_selected_image_config_report(
    args: argparse.Namespace,
    datasets: Sequence[str],
    blockers: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "command": "prepare_echosight",
        "root": str(args.root),
        "datasets": list(datasets),
        "options": _b3_options(args),
        "operations": {},
        "summary": {
            "status": "invalid_config",
            "exit_code": 3,
            "blockers": [dict(blocker) for blocker in blockers],
        },
    }


def _rooted_path(root_path: Path, relative_path: str | Path | PurePosixPath) -> Path:
    relative = _normalize_relative_path(relative_path)
    target = root_path.joinpath(*relative.parts).resolve(strict=False)
    root_resolved = root_path.resolve(strict=False)
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"path must stay under root: {relative_path}") from exc
    return target


def _relative_to_root(root_path: Path, path: Path) -> str:
    return path.resolve(strict=False).relative_to(root_path.resolve(strict=False)).as_posix()


def _safe_json_report(value):
    to_report_dict = getattr(value, "to_report_dict", None)
    if callable(to_report_dict):
        return to_report_dict()
    if isinstance(value, Mapping):
        return {str(key): _safe_json_report(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_report(item) for item in value]
    if isinstance(value, (Path, PurePosixPath)):
        return str(value)
    return value


def _candidate_manifest_report(manifest: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "manifest_version",
        "dataset",
        "subset_root",
        "manifest_path",
        "seed",
        "sample_train",
        "sample_test",
        "buffer_ratio",
        "enabled_sources",
        "splits",
        "summary",
    )
    return {key: _safe_json_report(manifest[key]) for key in keys if key in manifest}


def _candidate_manifest_incomplete_blockers(
    manifest: Mapping[str, object],
) -> list[dict[str, object]]:
    summary = manifest.get("summary")
    summary_mapping = summary if isinstance(summary, Mapping) else {}
    splits = manifest.get("splits")
    split_mapping = splits if isinstance(splits, Mapping) else {}
    dataset = str(manifest.get("dataset") or "")
    enabled_sources = manifest.get("enabled_sources")
    if isinstance(enabled_sources, Sequence) and not isinstance(enabled_sources, (str, bytes)):
        report_sources = [str(source) for source in enabled_sources]
    else:
        report_sources = []

    blockers: list[dict[str, object]] = []
    for split_name, split_report in split_mapping.items():
        if not isinstance(split_report, Mapping):
            continue
        if bool(split_report.get("complete")):
            continue
        blockers.append(
            {
                "reason": "candidate_manifest_incomplete",
                "dataset": dataset,
                "split": str(split_name),
                "requested": split_report.get("requested", 0),
                "target_candidates": split_report.get("target_candidates", 0),
                "candidate_count": split_report.get("candidate_count", 0),
                "source_rows": split_report.get("source_rows", 0),
                "enabled_sources": report_sources,
                "remediation": (
                    "Broaden selected-image sources, add enough raw QA rows with "
                    "supported image ids, or lower the requested sample/buffer counts."
                ),
            }
        )

    if (
        not blockers
        and isinstance(summary, Mapping)
        and bool(summary_mapping.get("complete")) is False
    ):
        requested = 0
        target_candidates = 0
        for split_report in split_mapping.values():
            if isinstance(split_report, Mapping):
                requested += int(split_report.get("requested") or 0)
                target_candidates += int(split_report.get("target_candidates") or 0)
        blockers.append(
            {
                "reason": "candidate_manifest_incomplete",
                "dataset": dataset,
                "requested": requested,
                "target_candidates": target_candidates,
                "candidate_count": summary_mapping.get("candidate_count", 0),
                "enabled_sources": report_sources,
                "remediation": (
                    "Refresh selected-image candidate planning and ensure every "
                    "split reaches its requested buffered candidate count."
                ),
            }
        )

    return blockers


def _selected_candidate_ids(
    manifest: Mapping[str, object],
    source: str,
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for candidate in manifest.get("candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("source") != source:
            continue
        image_id = str(candidate.get("image_id") or "")
        if image_id and image_id not in seen:
            seen.add(image_id)
            selected.append(image_id)
    return selected


def _provider_blocker(
    *,
    dataset: str,
    provider: str,
    reason: str,
    remediation: str,
    local_path: str | None = None,
    url: str | None = None,
    error: str | None = None,
) -> dict[str, object]:
    blocker: dict[str, object] = {
        "dataset": dataset,
        "provider": provider,
        "reason": reason,
        "remediation": remediation,
    }
    if local_path is not None:
        blocker["local_path"] = local_path
    if url is not None:
        blocker["url"] = url
    if error is not None:
        blocker["error"] = error
    return blocker


def _blocked_provider_report(
    *,
    dataset: str,
    provider: str,
    reason: str,
    remediation: str,
    local_path: str | None = None,
    url: str | None = None,
    error: str | None = None,
) -> dict[str, object]:
    blocker = _provider_blocker(
        dataset=dataset,
        provider=provider,
        reason=reason,
        remediation=remediation,
        local_path=local_path,
        url=url,
        error=error,
    )
    return {
        "dataset": dataset,
        "provider": provider,
        "complete": False,
        "planned": 0,
        "missing": [],
        "blockers": [blocker],
        "remediation": [remediation],
        "requests": [],
        "request_dicts": [],
    }


def _collect_blockers(value) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if isinstance(value, Mapping):
        raw_blockers = value.get("blockers")
        if isinstance(raw_blockers, Sequence) and not isinstance(raw_blockers, (str, bytes)):
            for blocker in raw_blockers:
                if isinstance(blocker, Mapping):
                    blockers.append(dict(blocker))
        for child in value.values():
            blockers.extend(_collect_blockers(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            blockers.extend(_collect_blockers(child))
    return blockers


def _selected_metadata_fetch_report(
    *,
    dataset: str,
    provider: str,
    status: str,
    local_path: str,
    url: str,
    bytes_downloaded: int = 0,
    error: str | None = None,
    remediation: str | None = None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "dataset": dataset,
        "provider": provider,
        "status": status,
        "local_path": local_path,
        "url": url,
        "bytes_downloaded": bytes_downloaded,
    }
    if error is not None:
        report["error"] = error
    if remediation is not None:
        report["remediation"] = remediation
        report["blockers"] = [
            _provider_blocker(
                dataset=dataset,
                provider=provider,
                reason=f"{provider}_metadata_download_failed",
                remediation=remediation,
                local_path=local_path,
                url=url,
                error=error,
            )
        ]
    return report


def _ensure_min_free_bytes(path: Path, required_bytes: int, min_free_bytes: int) -> None:
    probe = path if path.exists() else path.parent
    probe.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(probe).free
    if free - required_bytes < min_free_bytes:
        raise OSError(
            f"free space would drop below selected-image floor: "
            f"free={free}, required={required_bytes}, min_free={min_free_bytes}"
        )


def _download_selected_metadata_file(
    root: str | Path,
    relative_path: str | Path | PurePosixPath,
    url: str,
    *,
    dataset: str,
    provider: str,
    limits: SelectedDownloadLimits,
    headers: Mapping[str, str] | None = None,
) -> dict[str, object]:
    root_path = Path(root)
    target = _rooted_path(root_path, relative_path)
    local_path = _relative_to_root(root_path, target)
    if target.exists():
        return _selected_metadata_fetch_report(
            dataset=dataset,
            provider=provider,
            status="present",
            local_path=local_path,
            url=url,
            bytes_downloaded=target.stat().st_size,
        )

    cap = min(SELECTED_METADATA_MAX_BYTES, limits.max_transfer_bytes)
    target.parent.mkdir(parents=True, exist_ok=True)
    part_path = target.with_name(f"{target.name}.part")
    safe_headers = dict(headers or {})
    try:
        request = Request(url, headers=safe_headers)
        with urlopen(request, timeout=limits.timeout_seconds) as response:
            status = int(getattr(response, "status", response.getcode() or 0))
            if status >= 400:
                raise RuntimeError(f"HTTP status {status}")
            expected_size = response.headers.get("Content-Length")
            if expected_size:
                required = int(expected_size)
                if required > cap:
                    raise RuntimeError(
                        f"metadata size {required} exceeds selected metadata cap {cap}"
                    )
                _ensure_min_free_bytes(target.parent, required, limits.min_free_bytes)
            downloaded = 0
            with part_path.open("wb") as handle:
                while True:
                    chunk = response.read(limits.chunk_size)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > cap:
                        raise RuntimeError(
                            f"metadata transfer exceeds selected metadata cap {cap}"
                        )
                    _ensure_min_free_bytes(
                        target.parent,
                        len(chunk),
                        limits.min_free_bytes,
                    )
                    handle.write(chunk)
        part_path.replace(target)
        return _selected_metadata_fetch_report(
            dataset=dataset,
            provider=provider,
            status="complete",
            local_path=local_path,
            url=url,
            bytes_downloaded=target.stat().st_size,
        )
    except (HTTPError, URLError, OSError, RuntimeError, ValueError) as exc:
        part_path.unlink(missing_ok=True)
        return _selected_metadata_fetch_report(
            dataset=dataset,
            provider=provider,
            status="blocked",
            local_path=local_path,
            url=url,
            error=f"{type(exc).__name__}: {exc}",
            remediation=(
                "Supply this metadata file locally or rerun with valid network "
                "access, token, transfer cap, and free disk."
            ),
        )


def _hf_auth_headers(token) -> dict[str, str]:
    value = getattr(token, "value", None)
    if not value:
        return {}
    return {"Authorization": f"Bearer {value}"}


def _selected_operation_skipped_subset(args: argparse.Namespace) -> dict[str, object]:
    return {
        "mode": args.subset_mode,
        "dry_run": False,
        "requested": {
            "sample_train": args.sample_train,
            "sample_test": args.sample_test,
            "seed": args.subset_seed,
        },
        "reports": [],
        "summary": {
            "status": "skipped",
            "reason": "selected_images_incomplete",
        },
        "blockers": [
            {
                "reason": "selected_images_incomplete",
                "remediation": (
                    "Resolve selected-image provider blockers, then rerun the "
                    "paper subset build."
                ),
            }
        ],
    }


def _selected_subset_image_roots(root: str | Path, dataset: str) -> list[Path]:
    root_path = Path(root)
    relative_roots = []
    if dataset == DATASET_E_VQA:
        from .selected_image_providers import GLDV2_SELECTED_REL

        relative_roots.append(GLDV2_SELECTED_REL)
    elif dataset == DATASET_INFOSEEK:
        from .selected_image_providers import OVEN_SELECTED_REL

        relative_roots.append(OVEN_SELECTED_REL)

    roots: list[Path] = []
    seen: set[str] = set()
    for relative_root in relative_roots:
        candidate = root_path / Path(str(relative_root))
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        roots.append(resolved)
    return roots


def _run_subset_mode(
    root: str | Path,
    datasets: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    report = _subset_plan(args)
    if args.subset_mode != "paper" or args.dry_run:
        return report

    from .subset_echosight import build_paper_subset

    reports: list[dict[str, object]] = []
    for dataset in datasets:
        try:
            reports.append(
                build_paper_subset(
                    root,
                    dataset,
                    sample_train=args.sample_train,
                    sample_test=args.sample_test,
                    seed=args.subset_seed,
                    image_roots=_selected_subset_image_roots(root, dataset),
                )
            )
        except Exception as exc:
            reports.append(_subset_build_failed_report(dataset, args, exc))

    report["reports"] = reports
    return report


def _run_download(
    root: str | Path,
    datasets: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    from .download_echosight import (
        build_download_plan,
        check_free_space,
        estimate_required_disk_bytes,
        execute_download_plan,
    )

    include_images = _include_images(args)
    image_mode = _image_mode(args)
    execution_subset_mode = "full" if args.download_full_images else args.subset_mode
    resume = True
    plan = build_download_plan(
        datasets,
        include_images=include_images,
        image_mode=image_mode,
    )
    disk_estimate = estimate_required_disk_bytes(
        include_images=include_images,
        subset_mode=execution_subset_mode,
    )
    operation: dict[str, object] = {
        "image_mode": image_mode,
        "include_images": include_images,
        "resume": resume,
        "skip_images": not include_images,
        "subset_mode": execution_subset_mode,
        "plan": plan.to_dict(),
        "disk_estimate": disk_estimate.to_dict(),
    }

    if args.min_free_gb is not None:
        required_bytes = int(args.min_free_gb * 1_000_000_000)
        operation["min_free_bytes"] = required_bytes
        operation["free_space"] = check_free_space(root, required_bytes)
        if not args.dry_run and not operation["free_space"]["sufficient"]:
            operation["execution"] = {
                "summary": {
                    "status": "insufficient_disk",
                    "total_items": len(plan.items),
                    "bytes_downloaded": 0,
                },
                "results": [],
            }
            return operation

    attempts = max(args.max_retries, 1)
    execution: dict[str, object] | None = None
    remaining_items = tuple(plan.items)
    all_results: list[dict[str, object]] = []

    for attempt in range(1, attempts + 1):
        if not remaining_items:
            break
        execution = execute_download_plan(
            root,
            remaining_items,
            dry_run=args.dry_run,
            include_images=include_images,
            image_mode=image_mode,
            subset_mode=execution_subset_mode,
            force_large_downloads=args.force_large_downloads,
            resume=resume,
            timeout_seconds=args.timeout_seconds,
            chunk_size=args.chunk_size,
            hf_token_env_file=args.hf_token_env_file,
        )
        for result in execution["results"]:
            result_with_attempt = dict(result)
            result_with_attempt["attempt"] = attempt
            all_results.append(result_with_attempt)
        if args.dry_run:
            break
        failed_ids = {
            result["asset_id"]
            for result in execution["results"]
            if result["status"] not in {"complete", "skipped"}
        }
        remaining_items = tuple(
            item for item in remaining_items if item.asset_id in failed_ids
        )

    if execution is None:
        execution = {
            "summary": {
                "status": "complete",
                "total_items": 0,
                "bytes_downloaded": 0,
            },
            "results": [],
        }
    else:
        execution = dict(execution)
        execution["results"] = all_results
        statuses = {result["status"] for result in all_results}
        if args.dry_run:
            status = "dry_run"
        elif not all_results or statuses <= {"complete", "skipped"}:
            status = "complete"
        else:
            status = "incomplete"
        execution["summary"] = {
            "status": status,
            "total_items": len(all_results),
            "bytes_downloaded": sum(
                int(result.get("bytes_downloaded") or 0) for result in all_results
            ),
            "attempts": attempts,
        }

    operation["execution"] = execution
    return operation


def _request_group_report(
    *,
    root: str | Path,
    requests: Sequence[object],
    limits: SelectedDownloadLimits,
    dry_run: bool,
    label: str,
) -> dict[str, object]:
    if dry_run:
        return {
            "status": "dry_run",
            "label": label,
            "summary": {
                "status": "dry_run",
                "requested": len(requests),
                "complete": 0,
                "failed": 0,
                "skipped": 0,
            },
            "request_dicts": [_safe_json_report(request) for request in requests],
            "results": [],
        }
    if not requests:
        return {
            "status": "complete",
            "label": label,
            "summary": {
                "status": "complete",
                "requested": 0,
                "complete": 0,
                "failed": 0,
                "skipped": 0,
            },
            "results": [],
        }

    report = _safe_json_report(execute_selected_downloads(root, requests, limits))
    if isinstance(report, Mapping):
        report = dict(report)
        summary = dict(report.get("summary", {}))
        summary.setdefault("status", report.get("status", "incomplete"))
        report["summary"] = summary
        report["label"] = label
    return report


def _status_from_report(report: Mapping[str, object]) -> str:
    summary = report.get("summary", {})
    if isinstance(summary, Mapping) and "status" in summary:
        return str(summary["status"])
    return str(report.get("status", "incomplete"))


def _run_selected_extraction(
    root: str | Path,
    extraction_plan: Sequence[Mapping[str, object]],
    *,
    limits: SelectedDownloadLimits,
    args: argparse.Namespace,
    archive_download: Mapping[str, object],
) -> dict[str, object]:
    if args.dry_run:
        return {
            "status": "dry_run",
            "summary": {
                "status": "dry_run",
                "planned_archives": len(extraction_plan),
                "complete": 0,
                "failed": 0,
            },
            "plans": _safe_json_report(extraction_plan),
            "results": [],
        }
    if not extraction_plan:
        return {
            "status": "complete",
            "summary": {
                "status": "complete",
                "planned_archives": 0,
                "complete": 0,
                "failed": 0,
            },
            "results": [],
        }
    if _status_from_report(archive_download) != "complete":
        return {
            "status": "skipped",
            "summary": {
                "status": "skipped",
                "planned_archives": len(extraction_plan),
                "complete": 0,
                "failed": len(extraction_plan),
            },
            "blockers": [
                {
                    "reason": "archive_download_incomplete",
                    "remediation": (
                        "Complete OVEN archive downloads before selected member extraction."
                    ),
                }
            ],
            "results": [],
        }

    root_path = Path(root)
    results: list[dict[str, object]] = []
    complete = 0
    failed = 0
    for plan in extraction_plan:
        try:
            source_archive = _rooted_path(root_path, str(plan["source_archive"]))
            output_root = _rooted_path(root_path, str(plan["output_root"]))
            member_names = [str(name) for name in plan.get("member_names", [])]
            result = _safe_json_report(
                extract_selected_tar_members(
                    source_archive,
                    output_root,
                    member_names,
                    remove_tar=not args.keep_selected_source_archives,
                    min_free_bytes=limits.min_free_bytes,
                )
            )
            if _status_from_report(result) == "complete":
                complete += 1
            else:
                failed += 1
            results.append(result)
        except Exception as exc:
            failed += 1
            results.append(
                {
                    "status": "incomplete",
                    "error": f"{type(exc).__name__}: {exc}",
                    "plan": _safe_json_report(plan),
                    "blockers": [
                        {
                            "reason": "selected_tar_extraction_failed",
                            "remediation": (
                                "Verify the downloaded archive exists and contains "
                                "the planned selected members."
                            ),
                        }
                    ],
                }
            )

    status = "complete" if failed == 0 else "incomplete"
    return {
        "status": status,
        "summary": {
            "status": status,
            "planned_archives": len(extraction_plan),
            "complete": complete,
            "failed": failed,
        },
        "results": results,
    }


def _run_evqa_selected_providers(
    root: str | Path,
    manifest: Mapping[str, object],
    args: argparse.Namespace,
    limits: SelectedDownloadLimits,
) -> tuple[dict[str, object], list[object]]:
    from .selected_image_providers import (
        GLDV2_TRAIN_METADATA_URL,
        build_gldv2_requests,
        resolve_local_inaturalist,
    )

    root_path = Path(root)
    sources = _evqa_selected_sources(args)
    providers: dict[str, object] = {}
    metadata_downloads: list[dict[str, object]] = []
    direct_requests: list[object] = []

    if "gldv2" in sources:
        metadata_path = _rooted_path(root_path, GLDV2_METADATA_REL)
        metadata_rel = _relative_to_root(root_path, metadata_path)
        can_build = metadata_path.exists()
        if not can_build and args.dry_run:
            providers["gldv2"] = _blocked_provider_report(
                dataset=DATASET_E_VQA,
                provider="gldv2",
                reason="missing_gldv2_metadata",
                remediation=(
                    "Supply GLDv2 train metadata locally or run without --dry-run "
                    "to download it before selected image planning."
                ),
                local_path=metadata_rel,
                url=GLDV2_TRAIN_METADATA_URL,
            )
        elif not can_build:
            fetch = _download_selected_metadata_file(
                root,
                GLDV2_METADATA_REL,
                GLDV2_TRAIN_METADATA_URL,
                dataset=DATASET_E_VQA,
                provider="gldv2",
                limits=limits,
            )
            metadata_downloads.append(_safe_json_report(fetch))
            can_build = metadata_path.exists()
            if not can_build:
                providers["gldv2"] = _blocked_provider_report(
                    dataset=DATASET_E_VQA,
                    provider="gldv2",
                    reason="missing_gldv2_metadata",
                    remediation=(
                        "Supply GLDv2 train metadata locally, or rerun with enough "
                        "network access, transfer cap, and free disk."
                    ),
                    local_path=metadata_rel,
                    url=GLDV2_TRAIN_METADATA_URL,
                    error=str(fetch.get("error") or ""),
                )
        if can_build:
            try:
                provider_report = build_gldv2_requests(root, manifest, metadata_path)
                direct_requests.extend(provider_report.get("requests", []))
                providers["gldv2"] = _safe_json_report(provider_report)
            except Exception as exc:
                providers["gldv2"] = _blocked_provider_report(
                    dataset=DATASET_E_VQA,
                    provider="gldv2",
                    reason="gldv2_provider_failed",
                    remediation="Refresh GLDv2 metadata and rerun selected-image planning.",
                    local_path=metadata_rel,
                    url=GLDV2_TRAIN_METADATA_URL,
                    error=f"{type(exc).__name__}: {exc}",
                )

    if "inaturalist_2021" in sources:
        try:
            providers["inaturalist_2021"] = _safe_json_report(
                resolve_local_inaturalist(root, manifest, (EVQA_INATURALIST_IMAGE_ROOT_REL,))
            )
        except Exception as exc:
            providers["inaturalist_2021"] = _blocked_provider_report(
                dataset=DATASET_E_VQA,
                provider="inaturalist_2021",
                reason="inaturalist_provider_failed",
                remediation=(
                    "Place local iNaturalist id2name mappings and image roots under "
                    "the raw E-VQA layout; full archives are not scheduled."
                ),
                error=f"{type(exc).__name__}: {exc}",
            )

    return {
        "providers": providers,
        "metadata_downloads": metadata_downloads,
    }, direct_requests


def _run_infoseek_selected_providers(
    root: str | Path,
    manifest: Mapping[str, object],
    args: argparse.Namespace,
    limits: SelectedDownloadLimits,
    token,
) -> tuple[dict[str, object], list[object], list[dict[str, object]]]:
    from .selected_image_providers import (
        OVEN_HF_REPO,
        build_oven_shard_requests,
        parse_oven_mapping,
        plan_oven_shards,
    )

    root_path = Path(root)
    providers: dict[str, object] = {}
    metadata_downloads: list[dict[str, object]] = []
    archive_requests: list[object] = []
    extraction_plan: list[dict[str, object]] = []
    mapping_path = _rooted_path(root_path, OVEN_MAPPING_REL)
    mapping_rel = _relative_to_root(root_path, mapping_path)
    mapping_url = f"{OVEN_HF_REPO}/ovenid2impath.csv"
    can_build = mapping_path.exists()

    if not can_build and args.dry_run:
        providers["oven_mapping"] = _blocked_provider_report(
            dataset=DATASET_INFOSEEK,
            provider="oven",
            reason="missing_oven_mapping",
            remediation=(
                "Supply ovenid2impath.csv locally or run without --dry-run to "
                "download the bounded OVEN mapping file."
            ),
            local_path=mapping_rel,
            url=mapping_url,
        )
    elif not can_build:
        fetch = _download_selected_metadata_file(
            root,
            OVEN_MAPPING_REL,
            mapping_url,
            dataset=DATASET_INFOSEEK,
            provider="oven",
            limits=limits,
            headers=_hf_auth_headers(token),
        )
        metadata_downloads.append(_safe_json_report(fetch))
        can_build = mapping_path.exists()
        if not can_build:
            providers["oven_mapping"] = _blocked_provider_report(
                dataset=DATASET_INFOSEEK,
                provider="oven",
                reason="missing_oven_mapping",
                remediation=(
                    "Supply ovenid2impath.csv locally or rerun with accepted "
                    "Hugging Face access, transfer cap, and free disk."
                ),
                local_path=mapping_rel,
                url=mapping_url,
                error=str(fetch.get("error") or ""),
            )

    if can_build:
        selected_ids = _selected_candidate_ids(manifest, "oven")
        try:
            mapping_report = parse_oven_mapping(mapping_path, selected_ids)
            shard_plan = plan_oven_shards(
                mapping_report,
                max_shards=args.oven_max_shards,
                max_transfer_bytes=limits.max_transfer_bytes,
            )
            request_report = build_oven_shard_requests(root, shard_plan, token)
            archive_requests.extend(request_report.get("requests", []))
            extraction_plan.extend(
                dict(plan) for plan in request_report.get("extraction_plan", [])
            )
            providers["oven_mapping"] = _safe_json_report(mapping_report)
            providers["oven_shards"] = _safe_json_report(shard_plan)
            providers["oven"] = _safe_json_report(request_report)
        except Exception as exc:
            providers["oven"] = _blocked_provider_report(
                dataset=DATASET_INFOSEEK,
                provider="oven",
                reason="oven_provider_failed",
                remediation=(
                    "Refresh OVEN mapping metadata and rerun selected-image planning."
                ),
                local_path=mapping_rel,
                url=mapping_url,
                error=f"{type(exc).__name__}: {exc}",
            )

    return {
        "providers": providers,
        "metadata_downloads": metadata_downloads,
    }, archive_requests, extraction_plan


def _run_selected_images(
    root: str | Path,
    datasets: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    from .selected_images import (
        CandidatePlanConfig,
        load_hf_token,
        plan_selected_image_candidates,
        token_report,
    )

    limits = _selected_image_limits(args)
    root_path = Path(root)
    token = None
    token_metadata: dict[str, object]
    token_blockers: list[dict[str, object]] = []
    try:
        token = load_hf_token(args.hf_token_env_file)
        token_metadata = token_report(token)
    except Exception as exc:
        token_metadata = {
            "present": False,
            "source": None,
            "redacted": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        token_blockers.append(
            {
                "reason": "hf_token_load_failed",
                "remediation": (
                    "Fix the Hugging Face token env file or provide a token via "
                    "environment variable."
                ),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    report: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "settings": {
            "subset_mode": args.subset_mode,
            "sample_train": args.sample_train,
            "sample_test": args.sample_test,
            "subset_seed": args.subset_seed,
            "selected_image_buffer_ratio": args.selected_image_buffer_ratio,
            "max_selected_image_transfer_bytes": limits.max_transfer_bytes,
            "min_free_bytes": limits.min_free_bytes,
            "timeout_seconds": limits.timeout_seconds,
            "chunk_size": limits.chunk_size,
            "max_retries": limits.max_retries,
            "selected_image_request_delay_seconds": limits.request_delay_seconds,
            "selected_image_rate_limit_cooldown_seconds": (
                limits.rate_limit_cooldown_seconds
            ),
            "rate_limit_cooldown_seconds": limits.rate_limit_cooldown_seconds,
            "hf_token_env_file": str(args.hf_token_env_file),
            "token": token_metadata,
            "evqa_selected_sources": list(_evqa_selected_sources(args)),
            "oven_max_shards": args.oven_max_shards,
            "keep_selected_source_archives": args.keep_selected_source_archives,
        },
        "datasets": [],
    }

    direct_requests: list[object] = []
    archive_requests: list[object] = []
    extraction_plan: list[dict[str, object]] = []
    dataset_reports: list[dict[str, object]] = []

    for dataset in datasets:
        dataset_report: dict[str, object] = {
            "dataset": dataset,
            "providers": {},
            "metadata_downloads": [],
        }
        try:
            config = CandidatePlanConfig(
                sample_train=args.sample_train,
                sample_test=args.sample_test,
                seed=args.subset_seed,
                buffer_ratio=args.selected_image_buffer_ratio,
                enabled_sources=_selected_candidate_sources(dataset, args),
            )
            candidate_manifest = plan_selected_image_candidates(root_path, dataset, config)
            dataset_report["candidate_manifest"] = _candidate_manifest_report(
                candidate_manifest
            )
            candidate_blockers = _candidate_manifest_incomplete_blockers(
                candidate_manifest
            )
            if candidate_blockers:
                dataset_report["blockers"] = candidate_blockers
                provider_report = {
                    "providers": {},
                    "metadata_downloads": [],
                }
            elif dataset == DATASET_E_VQA:
                provider_report, requests = _run_evqa_selected_providers(
                    root_path,
                    candidate_manifest,
                    args,
                    limits,
                )
                direct_requests.extend(requests)
            elif dataset == DATASET_INFOSEEK:
                provider_report, requests, plans = _run_infoseek_selected_providers(
                    root_path,
                    candidate_manifest,
                    args,
                    limits,
                    token,
                )
                archive_requests.extend(requests)
                extraction_plan.extend(plans)
            else:
                provider_report = {
                    "providers": {
                        "unsupported": _blocked_provider_report(
                            dataset=dataset,
                            provider="selected_images",
                            reason="unsupported_dataset",
                            remediation="Use E-VQA or InfoSeek.",
                        )
                    },
                    "metadata_downloads": [],
                }
            dataset_report["providers"] = provider_report["providers"]
            dataset_report["metadata_downloads"] = provider_report["metadata_downloads"]
        except Exception as exc:
            dataset_report["candidate_manifest"] = {}
            dataset_report["providers"] = {
                "candidate_manifest": _blocked_provider_report(
                    dataset=dataset,
                    provider="selected_images",
                    reason="candidate_planning_failed",
                    remediation=(
                        "Ensure raw qa_train.csv and qa_test.csv exist, then rerun "
                        "selected-image planning."
                    ),
                    error=f"{type(exc).__name__}: {exc}",
                )
            }
        dataset_reports.append(_safe_json_report(dataset_report))

    report["datasets"] = dataset_reports
    report["direct_download"] = _request_group_report(
        root=root_path,
        requests=direct_requests,
        limits=limits,
        dry_run=args.dry_run,
        label="direct_images",
    )
    report["archive_download"] = _request_group_report(
        root=root_path,
        requests=archive_requests,
        limits=limits,
        dry_run=args.dry_run,
        label="source_archives",
    )
    report["archive_extraction"] = _run_selected_extraction(
        root_path,
        extraction_plan,
        limits=limits,
        args=args,
        archive_download=report["archive_download"],
    )

    blockers = token_blockers + _collect_blockers(report["datasets"])
    blockers.extend(_collect_blockers(report["direct_download"]))
    blockers.extend(_collect_blockers(report["archive_download"]))
    blockers.extend(_collect_blockers(report["archive_extraction"]))
    if args.dry_run:
        status = "dry_run"
    elif (
        blockers
        or _status_from_report(report["direct_download"]) != "complete"
        or _status_from_report(report["archive_download"]) != "complete"
        or _status_from_report(report["archive_extraction"]) != "complete"
    ):
        status = "incomplete"
    else:
        status = "complete"
    report["summary"] = {
        "status": status,
        "datasets": len(datasets),
        "direct_requests": len(direct_requests),
        "archive_requests": len(archive_requests),
        "extraction_plans": len(extraction_plan),
        "blockers": blockers,
    }
    return _safe_json_report(report)


def _validation_report(
    root: str | Path,
    datasets: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    from .validate_echosight import validate_echosight_full_assets

    return validate_echosight_full_assets(
        root,
        datasets=datasets,
        require_complete_subset=args.require_complete_subset,
        require_images=_effective_require_images(args),
        subset_sample_train=args.sample_train,
        subset_sample_test=args.sample_test,
        subset_seed=args.subset_seed,
    )


def _paper_subset_name(args: argparse.Namespace) -> str:
    return f"paper_{args.sample_train}_{args.sample_test}_seed{args.subset_seed}"


def _complete_subset_path_matches(
    complete: Mapping[str, object],
    args: argparse.Namespace,
) -> bool:
    expected_name = _paper_subset_name(args)
    path_values = [
        value
        for value in (
            complete.get("subset_root"),
            complete.get("manifest_path"),
        )
        if isinstance(value, str) and value
    ]
    if path_values:
        return any(
            expected_name in path_value.replace("\\", "/").split("/")
            for path_value in path_values
        )

    seed = complete.get("seed")
    return isinstance(seed, int) and seed == args.subset_seed


def _complete_subset_report_matches(
    complete: Mapping[str, object],
    args: argparse.Namespace,
) -> bool:
    return (
        complete.get("sample_train") == args.sample_train
        and complete.get("sample_test") == args.sample_test
        and _complete_subset_path_matches(complete, args)
    )


def _complete_subset_requested(report: Mapping[str, object], args: argparse.Namespace) -> bool:
    for dataset_report in report.get("dataset_reports", []):
        subset_reports = dataset_report.get("subset_reports", {})
        complete_reports = subset_reports.get("complete_reports", [])
        if not any(_complete_subset_report_matches(complete, args) for complete in complete_reports):
            return False
    return True


def _summary_status(report: Mapping[str, object]) -> str:
    saw_dry_run = False

    def operation_status(status: object) -> str | None:
        nonlocal saw_dry_run
        if status == "complete":
            return None
        if status == "dry_run":
            saw_dry_run = True
            return None
        return "incomplete"

    selected_images = report.get("operations", {}).get("selected_images")
    if isinstance(selected_images, Mapping):
        status = operation_status(_status_from_report(selected_images))
        if status is not None:
            return status

    download = report.get("operations", {}).get("download")
    if isinstance(download, Mapping):
        execution = download.get("execution", {})
        if not isinstance(execution, Mapping):
            return "incomplete"
        status = operation_status(_status_from_report(execution))
        if status is not None:
            return status

    subset = report.get("operations", {}).get("subset")
    if isinstance(subset, Mapping) and subset.get("mode") == "paper":
        if subset.get("dry_run") is True:
            saw_dry_run = True
        summary = subset.get("summary")
        if isinstance(summary, Mapping) and "status" in summary:
            status = operation_status(summary.get("status"))
            if status is not None:
                return status
        statuses = {
            subset_report.get("summary", {}).get("status")
            for subset_report in subset.get("reports", [])
        }
        for subset_status in statuses:
            status = operation_status(subset_status)
            if status is not None:
                return status

    tar_subset = report.get("operations", {}).get("tar_subset")
    if isinstance(tar_subset, Mapping):
        status = operation_status(_status_from_report(tar_subset))
        if status is not None:
            return status

    validation = report.get("validation")
    if isinstance(validation, Mapping):
        validation_status = str(validation.get("summary", {}).get("status", "incomplete"))
        if validation_status != "complete":
            return validation_status

    if saw_dry_run:
        return "dry_run"
    return "complete"


def _b3_options(args: argparse.Namespace) -> dict[str, object]:
    return {
        "download": args.download,
        "download_full_images": args.download_full_images,
        "download_selected_images": args.download_selected_images,
        "dry_run": args.dry_run,
        "image_mode": _image_mode(args),
        "include_images": _include_images(args),
        "subset_mode": args.subset_mode,
        "sample_train": args.sample_train,
        "sample_test": args.sample_test,
        "subset_seed": args.subset_seed,
        "require_complete_subset": args.require_complete_subset,
        "require_images": _effective_require_images(args),
        "force_large_downloads": args.force_large_downloads,
        "min_free_gb": args.min_free_gb,
        "verify": args.verify,
        "timeout_seconds": args.timeout_seconds,
        "chunk_size": args.chunk_size,
        "max_retries": args.max_retries,
        "selected_image_buffer_ratio": args.selected_image_buffer_ratio,
        "max_selected_image_transfer_gb": args.max_selected_image_transfer_gb,
        "selected_image_request_delay_seconds": args.selected_image_request_delay_seconds,
        "selected_image_rate_limit_cooldown_seconds": (
            args.selected_image_rate_limit_cooldown_seconds
        ),
        "hf_token_env_file": str(args.hf_token_env_file),
        "evqa_selected_sources": args.evqa_selected_sources,
        "oven_max_shards": args.oven_max_shards,
        "keep_selected_source_archives": args.keep_selected_source_archives,
        "build_tar_subset": args.build_tar_subset,
        "build_infoseek_local_subset": args.build_infoseek_local_subset,
        "infoseek_local_subset_name": args.infoseek_local_subset_name,
        "infoseek_copy_images": args.infoseek_copy_images,
        "tar_image_root": args.tar_image_root,
        "tar_subset_name": args.tar_subset_name,
        "tar_limit_archives": args.tar_limit_archives,
        "tar_contact_sheet": args.tar_contact_sheet,
    }


def _default_infoseek_local_subset_name(args: argparse.Namespace) -> str:
    return f"paper_local_{args.sample_train}_{args.sample_test}_seed{args.subset_seed}"


def _run_infoseek_local_subset(
    root: str | Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    subset_name = (
        args.infoseek_local_subset_name
        if args.infoseek_local_subset_name is not None
        else _default_infoseek_local_subset_name(args)
    )
    try:
        return _safe_json_report(
            build_infoseek_local_image_subset(
                root=root,
                subset_name=subset_name,
                sample_train=args.sample_train,
                sample_test=args.sample_test,
                seed=args.subset_seed,
                copy_images=args.infoseek_copy_images,
            )
        )
    except Exception as exc:
        blocker = {
            "dataset": DATASET_INFOSEEK,
            "reason": "infoseek_local_subset_build_failed",
            "message": "InfoSeek local subset build failed.",
            "details": {"error": f"{type(exc).__name__}: {exc}"},
        }
        subset_root = f"{DATASETS_MM_ROOT}/{DATASET_INFOSEEK}/subsets/{subset_name}"
        return {
            "dataset": DATASET_INFOSEEK,
            "subset_root": subset_root,
            "manifest_path": f"{subset_root}/manifest.json",
            "report_path": f"{subset_root}/local_subset_report.json",
            "summary": {
                "status": "infoseek_local_subset_build_failed",
                "requested_train": args.sample_train,
                "requested_test": args.sample_test,
                "selected_train": 0,
                "selected_test": 0,
            },
            "blockers": [blocker],
        }


def _infoseek_local_subset_status_is_usable(status: object) -> bool:
    return str(status) in {"complete", "subset_incomplete"}


def _run_b3_cli(args: argparse.Namespace, datasets: Sequence[str]) -> tuple[int, dict[str, object]]:
    if args.build_tar_subset:
        tar_config_blockers = _validate_tar_subset_config(args, datasets)
        if tar_config_blockers:
            return 2, _invalid_tar_subset_config_report(
                args,
                datasets,
                tar_config_blockers,
            )

    selected_config_blockers = _validate_selected_image_config(args)
    if selected_config_blockers:
        return 3, _invalid_selected_image_config_report(
            args,
            datasets,
            selected_config_blockers,
        )

    operations: dict[str, object] = {}
    report: dict[str, object] = {
        "command": "prepare_echosight",
        "root": str(args.root),
        "datasets": list(datasets),
        "options": _b3_options(args),
        "operations": operations,
    }

    if args.download:
        operations["download"] = _run_download(args.root, datasets, args)

    if args.download_selected_images:
        operations["selected_images"] = _run_selected_images(args.root, datasets, args)

    if args.build_tar_subset:
        operations["tar_subset"] = _run_tar_subset(args)

    if args.build_infoseek_local_subset:
        operations["infoseek_local_subset"] = _run_infoseek_local_subset(args.root, args)

    if args.subset_mode != "none":
        selected_status = "complete"
        if args.download_selected_images and not args.dry_run:
            selected_status = str(
                operations["selected_images"].get("summary", {}).get(
                    "status",
                    "incomplete",
                )
            )
        if selected_status == "complete" or args.dry_run:
            operations["subset"] = _run_subset_mode(args.root, datasets, args)
        else:
            operations["subset"] = _selected_operation_skipped_subset(args)

    if args.verify or args.require_complete_subset or args.require_images:
        report["validation"] = _validation_report(args.root, datasets, args)

    status = _summary_status(report)
    validation = report.get("validation", {})
    validation_status = None
    if isinstance(validation, Mapping):
        validation_status = str(validation.get("summary", {}).get("status", "incomplete"))
    if args.require_complete_subset and not args.build_infoseek_local_subset:
        if not isinstance(validation, Mapping) or not _complete_subset_requested(validation, args):
            status = "incomplete"
    if args.build_tar_subset:
        status = _status_from_report(operations["tar_subset"])
    if args.build_infoseek_local_subset:
        infoseek_status = _status_from_report(operations["infoseek_local_subset"])
        status = infoseek_status
        if (
            (args.verify or args.require_images)
            and validation_status is not None
            and validation_status != "complete"
        ):
            status = validation_status

    exit_code = 0
    if (
        args.require_complete_subset
        and not args.build_infoseek_local_subset
        and status != "complete"
    ):
        exit_code = 2
    if args.download_selected_images and not args.dry_run and status != "complete":
        exit_code = 2
    if args.build_tar_subset and status != "complete":
        exit_code = 2
    if (
        args.build_infoseek_local_subset
        and not _infoseek_local_subset_status_is_usable(status)
    ):
        exit_code = 2
    if (
        args.require_images
        and validation_status is not None
        and validation_status != "complete"
    ):
        exit_code = 2
    report["summary"] = {
        "status": status,
        "exit_code": exit_code,
    }
    return exit_code, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    datasets = args.dataset or None
    include_optional = not args.required_only

    if _uses_b3_cli(args):
        error = _b3_cli_argument_error(args)
        if error is not None:
            print(error, file=sys.stderr)
            return 2
        selected = _selected_datasets(datasets)
        exit_code, report = _run_b3_cli(args, selected)
        print(json.dumps(report, indent=2, sort_keys=True))
        return exit_code

    if args.write_plans:
        written = write_planning_files(
            args.root,
            datasets=datasets,
            include_optional=include_optional,
        )
        print(
            json.dumps(
                {
                    dataset: {name: str(path) for name, path in files.items()}
                    for dataset, files in written.items()
                },
                indent=2,
                sort_keys=True,
            )
        )

    if args.validate:
        try:
            report = validate_required_assets(
                args.root,
                datasets=datasets,
                compute_checksums=args.compute_checksums,
            )
        except MissingRequiredAssetsError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))

    if not args.write_plans and not args.validate:
        print(
            json.dumps(
                manifest_document(datasets, include_optional=include_optional),
                indent=2,
                sort_keys=True,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
