"""Download planning for EchoSight-style assets."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Mapping, Protocol
from urllib.request import Request, urlopen

from .full_image_layout import canonical_full_image_root
from .prepare_echosight import SUPPORTED_DATASETS, AssetDefinition, asset_catalog


PROVIDER_DIRECT_FILE = "direct_file"
PROVIDER_IMAGE_SOURCE = "image_source_provider"
DOWNLOAD_MODE_REQUIRED = "required"
STATUS_COMPLETE = "complete"
STATUS_DRY_RUN = "dry_run"
STATUS_DOWNLOAD_FAILED = "download_failed"
STATUS_INSUFFICIENT_DISK = "insufficient_disk"
STATUS_SKIPPED = "skipped"
FULL_IMAGE_MIRROR_REQUIRED_BYTES = 1_500_000_000_000

OPTIONAL_NON_DOWNLOADED_ASSET_IDS = {
    "e_vqa_echosight_hard_negatives",
    "e_vqa_reranker_checkpoint",
    "e_vqa_reranker_training_artifacts",
    "e_vqa_reranker_inference_artifacts",
    "infoseek_echosight_hard_negatives",
    "infoseek_reranker_checkpoint",
    "infoseek_reranker_training_artifacts",
    "infoseek_reranker_inference_artifacts",
}

IMAGE_SOURCE_ASSET_IDS = {
    "e_vqa_image_source_inaturalist_2021",
    "e_vqa_image_source_google_landmarks_v2",
    "infoseek_image_source_oven",
}

HF_IMAGE_SOURCE_REPO_IDS = {
    "e_vqa_image_source_google_landmarks_v2": "andito/google-landmarks",
    "e_vqa_image_source_inaturalist_2021": "Artiprocher/iNaturalist2021",
    "infoseek_image_source_oven": "ychenNLP/oven",
}


def _hf_dataset_root_url(repo_id: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}"


SOURCE_URLS = {
    "e_vqa_kb_archive": (
        "https://storage.googleapis.com/encyclopedic-vqa/encyclopedic_kb_wiki.zip"
    ),
    "e_vqa_vqa_dev_csv": "https://storage.googleapis.com/encyclopedic-vqa/val.csv",
    "e_vqa_vqa_test_csv": "https://storage.googleapis.com/encyclopedic-vqa/test.csv",
    "e_vqa_image_source_inaturalist_2021": _hf_dataset_root_url(
        HF_IMAGE_SOURCE_REPO_IDS["e_vqa_image_source_inaturalist_2021"]
    ),
    "e_vqa_image_source_google_landmarks_v2": _hf_dataset_root_url(
        HF_IMAGE_SOURCE_REPO_IDS["e_vqa_image_source_google_landmarks_v2"]
    ),
    "infoseek_image_source_oven": _hf_dataset_root_url(
        HF_IMAGE_SOURCE_REPO_IDS["infoseek_image_source_oven"]
    ),
}

PUBLIC_SOURCE_PAGES = {
    "e_vqa_kb_archive": "https://github.com/google-research/google-research/tree/master/encyclopedic_vqa",
    "e_vqa_faiss_archive": "https://github.com/Go2Heart/EchoSight",
    "e_vqa_vqa_train_csv": "https://github.com/Go2Heart/EchoSight",
    "e_vqa_vqa_dev_csv": "https://github.com/google-research/google-research/tree/master/encyclopedic_vqa",
    "e_vqa_vqa_test_csv": "https://github.com/google-research/google-research/tree/master/encyclopedic_vqa",
    "e_vqa_train_id2name": "https://github.com/Go2Heart/EchoSight",
    "e_vqa_val_id2name": "https://github.com/Go2Heart/EchoSight",
    "infoseek_kb_archive": "https://github.com/Go2Heart/EchoSight",
    "infoseek_faiss_archive": "https://github.com/Go2Heart/EchoSight",
    "infoseek_vqa_train_csv": "https://open-vision-language.github.io/infoseek/",
    "infoseek_vqa_test_csv": "https://open-vision-language.github.io/infoseek/",
    "e_vqa_image_source_inaturalist_2021": "https://github.com/visipedia/inat_comp/tree/master/2021",
    "e_vqa_image_source_google_landmarks_v2": "https://github.com/cvdfoundation/google-landmark",
    "infoseek_image_source_oven": "https://huggingface.co/datasets/ychenNLP/oven",
}

IMAGE_PROVIDER_REMEDIATION = {
    "e_vqa_image_source_inaturalist_2021": (
        "iNaturalist 2021 images are a full image-source mirror. In paper "
        "subset mode, resolve selected subset images first and keep the full "
        "mirror as a larger-disk workflow. Upstream URL: "
        "https://github.com/visipedia/inat_comp/tree/master/2021. Example "
        "larger root: /path/to/evograph-mm-data or an external drive with 1.5TB+ free."
    ),
    "e_vqa_image_source_google_landmarks_v2": (
        "Google Landmarks Dataset V2 images are a full image-source mirror. In "
        "paper subset mode, resolve selected subset images first and keep the "
        "full mirror as a larger-disk workflow. Upstream URL: "
        "https://github.com/cvdfoundation/google-landmark. Example larger "
        "root: /path/to/evograph-mm-data or an external drive with 1.5TB+ free."
    ),
    "infoseek_image_source_oven": (
        "OVEN images are gated through Hugging Face. Log in with "
        "`huggingface-cli login`, accept the dataset conditions, then continue "
        "from https://huggingface.co/datasets/ychenNLP/oven. In paper subset "
        "mode, resolve selected subset images first and keep the full mirror "
        "as a larger-disk workflow from /path/to/evograph-mm-data or an "
        "external drive with 1.5TB+ free."
    ),
}


def _remediation_hint(asset: AssetDefinition) -> str:
    parts = [asset.source_note]
    source_page = PUBLIC_SOURCE_PAGES.get(asset.asset_id)
    if source_page:
        parts.append(f"Upstream source: {source_page}.")
    if asset.asset_id not in SOURCE_URLS:
        parts.append(
            "No stable public direct-download URL is bundled; place this asset "
            "manually before running validation or build steps."
        )
    return " ".join(part for part in parts if part)


@dataclass(frozen=True)
class DownloadPlanItem:
    dataset: str
    asset_id: str
    provider: str
    source_url: str | None
    local_path: str
    expected_kind: str
    download_mode: str
    remediation_hint: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DownloadPlan:
    items: tuple[DownloadPlanItem, ...]

    def to_dict(self) -> dict[str, object]:
        return {"items": [item.to_dict() for item in self.items]}


@dataclass(frozen=True)
class DownloadResult:
    dataset: str
    asset_id: str
    local_path: str
    status: str
    bytes_downloaded: int
    source_url: str | None
    remediation: str | None = None
    error: str | None = None
    required_bytes: int | None = None
    free_bytes: int | None = None
    subset_mode: str | None = None
    requested_root: str | None = None
    probed_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DiskEstimate:
    minimum_free_bytes: int
    human_recommendation: str
    ram_recommendation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class HTTPClient(Protocol):
    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ):
        ...


class UrlopenHTTPClient:
    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ):
        request = Request(url, headers=dict(headers or {}))
        return urlopen(request, timeout=timeout)


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


def _provider_for(asset: AssetDefinition) -> str:
    if asset.asset_id in IMAGE_SOURCE_ASSET_IDS:
        return PROVIDER_IMAGE_SOURCE
    return PROVIDER_DIRECT_FILE


def build_download_plan(
    datasets: Iterable[str] | str | None = None,
    *,
    include_images: bool = False,
    image_mode: str = "none",
) -> DownloadPlan:
    selected = _normalize_datasets(datasets)
    if image_mode not in {"none", "full"}:
        raise ValueError(f"unsupported image_mode: {image_mode}")
    items: list[DownloadPlanItem] = []

    for asset in asset_catalog(selected, required_only=True):
        if asset.asset_id in IMAGE_SOURCE_ASSET_IDS and not include_images:
            continue

        source_url = SOURCE_URLS.get(asset.asset_id)

        local_path = asset.local_path
        if asset.asset_id in IMAGE_SOURCE_ASSET_IDS and image_mode == "full":
            local_path = canonical_full_image_root(asset.dataset, asset.asset_id)

        items.append(
            DownloadPlanItem(
                dataset=asset.dataset,
                asset_id=asset.asset_id,
                provider=_provider_for(asset),
                source_url=source_url,
                local_path=local_path,
                expected_kind=asset.expected_kind,
                download_mode=DOWNLOAD_MODE_REQUIRED,
                remediation_hint=_remediation_hint(asset),
            )
        )

    return DownloadPlan(tuple(items))


def estimate_required_disk_bytes(*, include_images: bool, subset_mode: str) -> DiskEstimate:
    if include_images and subset_mode == "full":
        return DiskEstimate(
            minimum_free_bytes=FULL_IMAGE_MIRROR_REQUIRED_BYTES,
            human_recommendation=(
                "Reserve at least 1.5TB for full image mirrors; prefer 2TB "
                "when extracting archives."
            ),
            ram_recommendation="8-16GB",
        )

    return DiskEstimate(
        minimum_free_bytes=150_000_000_000,
        human_recommendation=(
            "Reserve 120-150GB for both paper-sized subsets plus required "
            "KB/FAISS/CSV/id2name assets."
        ),
        ram_recommendation="8-16GB",
    )


def check_free_space(root: str | Path, required_bytes: int) -> dict[str, object]:
    root_path = Path(root).resolve(strict=False)
    probe = root_path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {
        "requested_root": str(root_path),
        "probed_path": str(probe),
        "required_bytes": required_bytes,
        "free_bytes": usage.free,
        "sufficient": usage.free >= required_bytes,
    }


def _repo_relative_path(root: str | Path, path: str | Path) -> str:
    root_path = Path(root).resolve()
    path_value = Path(path)
    target_path = path_value if path_value.is_absolute() else root_path / path_value
    try:
        return target_path.resolve().relative_to(root_path).as_posix()
    except ValueError:
        return path_value.as_posix()


def _ensure_under_root(root: str | Path, path: str | Path, *, label: str) -> None:
    root_resolved = Path(root).resolve()
    path_resolved = Path(path).resolve(strict=False)
    try:
        path_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes download root: {path}") from exc


def _resolve_local_target(root: str | Path, local_path: str | Path) -> Path:
    local_text = str(local_path)
    windows_path = PureWindowsPath(local_text)
    posix_path = PurePosixPath(local_text.replace("\\", "/"))
    child_parts = tuple(part for part in posix_path.parts if part not in {"", "."})

    if not local_text:
        raise ValueError("local_path must not be empty")
    if windows_path.drive or windows_path.root or posix_path.is_absolute():
        raise ValueError(f"unsafe absolute local_path: {local_text}")
    if any(part == ".." for part in posix_path.parts):
        raise ValueError(f"unsafe parent traversal in local_path: {local_text}")
    if not child_parts:
        raise ValueError(f"local_path must name a child path under root: {local_text}")

    target_path = Path(root).joinpath(*child_parts)
    _ensure_under_root(root, target_path, label="target")
    return target_path


def _sidecar_path(target_path: str | Path) -> Path:
    target = Path(target_path)
    return target.with_suffix(target.suffix + ".download.json")


def _part_path(target_path: str | Path) -> Path:
    target = Path(target_path)
    return target.with_suffix(target.suffix + ".part")


def _write_sidecar(target_path: str | Path, result: DownloadResult) -> None:
    sidecar = _sidecar_path(target_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )
    temp_path = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(sidecar)


def _complete_sidecar_result(
    root_path: Path,
    target_path: Path,
    item: DownloadPlanItem,
) -> DownloadResult | None:
    sidecar = _sidecar_path(target_path)
    if not target_path.is_file() or not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("status") != STATUS_COMPLETE:
        return None
    if payload.get("asset_id") != item.asset_id:
        return None
    if payload.get("local_path") != item.local_path:
        return None
    return DownloadResult(
        dataset=item.dataset,
        asset_id=item.asset_id,
        local_path=_repo_relative_path(root_path, target_path),
        status=STATUS_COMPLETE,
        bytes_downloaded=target_path.stat().st_size,
        source_url=item.source_url,
    )


def _complete_directory_sidecar_result(
    root_path: Path,
    target_dir: Path,
    item: DownloadPlanItem,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
) -> DownloadResult | None:
    sidecar = target_dir / ".download.json"
    if not target_dir.is_dir() or not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("status") != STATUS_COMPLETE:
        return None
    if payload.get("asset_id") != item.asset_id:
        return None
    if payload.get("local_path") != _repo_relative_path(root_path, target_dir):
        return None
    mirror_file_count = payload.get("mirror_file_count")
    mirror_bytes = payload.get("mirror_bytes")
    if type(mirror_file_count) is not int or mirror_file_count <= 0:
        return None
    if type(mirror_bytes) is not int or mirror_bytes <= 0:
        return None
    if repo_id is not None and payload.get("mirror_repo_id") != repo_id:
        return None
    if payload.get("mirror_revision") != revision:
        return None
    actual_file_count, actual_bytes = _directory_mirror_stats(target_dir)
    if actual_file_count <= 0:
        return None
    if actual_file_count != mirror_file_count or actual_bytes != mirror_bytes:
        return None
    return DownloadResult(
        dataset=item.dataset,
        asset_id=item.asset_id,
        local_path=_repo_relative_path(root_path, target_dir),
        status=STATUS_COMPLETE,
        bytes_downloaded=mirror_bytes,
        source_url=item.source_url,
    )


def _write_directory_sidecar(target_dir: str | Path, payload: dict[str, object]) -> None:
    sidecar = Path(target_dir) / ".download.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )
    temp_path = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(sidecar)


def _directory_mirror_stats(target_dir: str | Path) -> tuple[int, int]:
    target_path = Path(target_dir)
    sidecar = target_path / ".download.json"
    temp_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
    file_count = 0
    total_bytes = 0
    for path in target_path.rglob("*"):
        if not path.is_file():
            continue
        if path in {sidecar, temp_sidecar}:
            continue
        file_count += 1
        total_bytes += path.stat().st_size
    return file_count, total_bytes


def _image_provider_remediation(
    item: DownloadPlanItem,
    *,
    status: str,
    subset_mode: str,
    required_bytes: int | None,
    free_bytes: int | None,
) -> str:
    parts = []
    if item.remediation_hint:
        parts.append(item.remediation_hint)
    parts.append(
        IMAGE_PROVIDER_REMEDIATION.get(
            item.asset_id,
            (
                "Image source provider is not mirrored automatically here. "
                "Resolve selected subset images first, then continue the full "
                "mirror from a larger-disk workflow root if needed."
            ),
        )
    )
    if status == STATUS_INSUFFICIENT_DISK:
        parts.append(
            f"Insufficient disk for full image mirror: required "
            f"{required_bytes} bytes, observed free {free_bytes} bytes."
        )
    if subset_mode == "paper":
        parts.append(
            "Subset mode paper records this sidecar instead of starting a full "
            "image mirror download."
        )
    return " ".join(parts)


def record_image_provider_remediation(
    root: str | Path,
    item: DownloadPlanItem,
    *,
    status: str = STATUS_DOWNLOAD_FAILED,
    required_bytes: int | None = None,
    free_bytes: int | None = None,
    requested_root: str | None = None,
    probed_path: str | None = None,
    subset_mode: str = "paper",
) -> DownloadResult:
    root_path = Path(root)
    target_dir = _resolve_local_target(root_path, item.local_path)
    sidecar_path = target_dir / ".download.json"
    _ensure_under_root(root_path, target_dir, label="image source")
    _ensure_under_root(root_path, sidecar_path, label="sidecar")

    remediation = _image_provider_remediation(
        item,
        status=status,
        subset_mode=subset_mode,
        required_bytes=required_bytes,
        free_bytes=free_bytes,
    )
    result = DownloadResult(
        dataset=item.dataset,
        asset_id=item.asset_id,
        local_path=_repo_relative_path(root_path, target_dir),
        status=status,
        bytes_downloaded=0,
        source_url=item.source_url or SOURCE_URLS.get(item.asset_id),
        remediation=remediation,
        required_bytes=required_bytes,
        free_bytes=free_bytes,
        subset_mode=subset_mode,
        requested_root=requested_root,
        probed_path=probed_path,
    )
    payload = result.to_dict()
    _write_directory_sidecar(target_dir, payload)
    return result


def _response_status(response) -> int:
    status = getattr(response, "status", None)
    if status is None and hasattr(response, "getcode"):
        status = response.getcode()
    return int(status or 200)


def _response_header(response, name: str) -> str | None:
    wanted = name.lower()
    headers = getattr(response, "headers", None)
    if headers is not None:
        if hasattr(headers, "get"):
            value = headers.get(name) or headers.get(name.lower())
            if value is not None:
                return str(value)
        if hasattr(headers, "items"):
            for key, value in headers.items():
                if str(key).lower() == wanted:
                    return str(value)
    if hasattr(response, "info"):
        info = response.info()
        if hasattr(info, "get"):
            value = info.get(name) or info.get(name.lower())
            if value is not None:
                return str(value)
    return None


def _response_content_type(response) -> str:
    return _response_header(response, "Content-Type") or ""


def _response_content_length(response) -> int | None:
    content_length = _response_header(response, "Content-Length")
    if content_length is None:
        return None
    try:
        return int(content_length)
    except ValueError:
        return None


def _parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    try:
        unit, range_spec = value.strip().split(None, 1)
        byte_range, total_text = range_spec.split("/", 1)
        start_text, end_text = byte_range.split("-", 1)
        if unit.lower() != "bytes" or total_text.strip() == "*":
            return None
        start = int(start_text)
        end = int(end_text)
        total = int(total_text)
    except ValueError:
        return None
    if start < 0 or end < start or total <= 0 or end >= total:
        return None
    return start, end, total


def _expected_final_size(response, status: int) -> int | None:
    if status == 206:
        content_range = _parse_content_range(
            _response_header(response, "Content-Range")
        )
        if content_range is None:
            return None
        return content_range[2]

    if status == 200:
        return _response_content_length(response)

    return None


def _resume_validation_error(response, existing_bytes: int) -> tuple[int | None, str | None]:
    content_range = _parse_content_range(_response_header(response, "Content-Range"))
    if content_range is None:
        return None, "missing or malformed Content-Range for resumed response"

    start, end, total = content_range
    if start != existing_bytes:
        return total, (
            f"resumed Content-Range starts at {start}; expected {existing_bytes}"
        )
    if end != total - 1:
        return total, (
            f"resumed Content-Range ends at {end}; expected {total - 1}"
        )

    content_length = _response_content_length(response)
    expected_length = end - start + 1
    if content_length is not None and content_length != expected_length:
        return total, (
            f"Content-Length {content_length} does not match resumed range "
            f"length {expected_length}"
        )

    return total, None


def _failure_result(
    item: DownloadPlanItem,
    *,
    target_path: Path,
    root: str | Path,
    bytes_downloaded: int = 0,
    error: str,
) -> DownloadResult:
    remediation_parts = []
    if item.remediation_hint:
        remediation_parts.append(item.remediation_hint)
    remediation_parts.append(
        "Retry with resume enabled after confirming the source URL returns a file."
    )
    return DownloadResult(
        dataset=item.dataset,
        asset_id=item.asset_id,
        local_path=_repo_relative_path(root, target_path),
        status=STATUS_DOWNLOAD_FAILED,
        bytes_downloaded=bytes_downloaded,
        source_url=item.source_url,
        remediation=" ".join(remediation_parts),
        error=error,
    )


def _results_envelope(
    status: str,
    results: Iterable[DownloadResult],
    *,
    result_history: Iterable[DownloadResult] | None = None,
) -> dict[str, object]:
    result_list = list(results)
    history_list = list(result_history or ())
    final_results = _final_item_results(result_list)
    summary_status = status
    if status != STATUS_DRY_RUN:
        summary_status = _overall_status(final_results)
    summary = {
        "status": summary_status,
        "total_items": len(final_results),
        "bytes_downloaded": sum(result.bytes_downloaded for result in final_results),
    }
    if history_list:
        summary["result_history_items"] = len(history_list)
    elif len(final_results) != len(result_list):
        summary["result_history_items"] = len(result_list)
    envelope = {
        "summary": summary,
        "results": [result.to_dict() for result in result_list],
    }
    if history_list:
        envelope["result_history"] = [result.to_dict() for result in history_list]
    return envelope


def _final_item_results(results: Iterable[DownloadResult]) -> list[DownloadResult]:
    final_by_key: dict[tuple[str, str, str], DownloadResult] = {}
    key_order: list[tuple[str, str, str]] = []
    for result in results:
        key = (result.dataset, result.asset_id, result.local_path)
        if key not in final_by_key:
            key_order.append(key)
        final_by_key[key] = result
    return [final_by_key[key] for key in key_order]


def _overall_status(results: Iterable[DownloadResult]) -> str:
    result_list = list(results)
    if any(result.status == STATUS_INSUFFICIENT_DISK for result in result_list):
        return STATUS_INSUFFICIENT_DISK
    if any(result.status == STATUS_DOWNLOAD_FAILED for result in result_list):
        return STATUS_DOWNLOAD_FAILED
    if result_list and all(result.status == STATUS_SKIPPED for result in result_list):
        return STATUS_SKIPPED
    return STATUS_COMPLETE


def _should_retry_direct_result(result: DownloadResult) -> bool:
    return result.status == STATUS_DOWNLOAD_FAILED and result.bytes_downloaded > 0


def _image_provider_failure_result(
    root: str | Path,
    item: DownloadPlanItem,
    *,
    target_dir: Path,
    error: str,
) -> DownloadResult:
    result = DownloadResult(
        dataset=item.dataset,
        asset_id=item.asset_id,
        local_path=_repo_relative_path(root, target_dir),
        status=STATUS_DOWNLOAD_FAILED,
        bytes_downloaded=0,
        source_url=item.source_url or SOURCE_URLS.get(item.asset_id),
        remediation=_image_provider_remediation(
            item,
            status=STATUS_DOWNLOAD_FAILED,
            subset_mode="full",
            required_bytes=None,
            free_bytes=None,
        ),
        error=error,
        subset_mode="full",
    )
    _write_directory_sidecar(target_dir, result.to_dict())
    return result


def download_hugging_face_image_source_mirror(
    root: str | Path,
    item: DownloadPlanItem,
    *,
    repo_id: str,
    resume: bool = True,
    hf_token_env_file: str | Path | None = None,
    revision: str | None = None,
) -> DownloadResult:
    root_path = Path(root)
    target_dir = _resolve_local_target(root_path, item.local_path)
    _ensure_under_root(root_path, target_dir, label="image source")
    _ensure_under_root(root_path, target_dir / ".download.json", label="sidecar")

    if resume:
        existing = _complete_directory_sidecar_result(
            root_path,
            target_dir,
            item,
            repo_id=repo_id,
            revision=revision,
        )
        if existing is not None:
            return existing

    try:
        from huggingface_hub import snapshot_download

        from .selected_images import load_hf_token

        token = load_hf_token(hf_token_env_file).value
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=target_dir,
            force_download=not resume,
            token=token,
            local_dir_use_symlinks=False,
            resume_download=resume,
        )
    except Exception as exc:
        return _image_provider_failure_result(
            root_path,
            item,
            target_dir=target_dir,
            error=str(exc),
        )

    file_count, mirror_bytes = _directory_mirror_stats(target_dir)
    if file_count <= 0 or mirror_bytes <= 0:
        return _image_provider_failure_result(
            root_path,
            item,
            target_dir=target_dir,
            error="mirror completed without payload files",
        )

    result = DownloadResult(
        dataset=item.dataset,
        asset_id=item.asset_id,
        local_path=_repo_relative_path(root_path, target_dir),
        status=STATUS_COMPLETE,
        bytes_downloaded=mirror_bytes,
        source_url=item.source_url or SOURCE_URLS.get(item.asset_id),
    )
    payload = result.to_dict()
    payload["mirror_file_count"] = file_count
    payload["mirror_bytes"] = mirror_bytes
    payload["mirror_repo_id"] = repo_id
    payload["mirror_revision"] = revision
    _write_directory_sidecar(target_dir, payload)
    return result


def download_image_source_provider(
    root: str | Path,
    item: DownloadPlanItem,
    *,
    resume: bool = True,
    image_mode: str = "full",
    hf_token_env_file: str | Path | None = None,
) -> DownloadResult:
    target_dir = _resolve_local_target(root, item.local_path)
    repo_id = HF_IMAGE_SOURCE_REPO_IDS.get(item.asset_id)
    if repo_id is None:
        return _image_provider_failure_result(
            root,
            item,
            target_dir=target_dir,
            error=f"unsupported image source asset id: {item.asset_id}",
        )
    if image_mode != "full":
        return record_image_provider_remediation(
            root,
            item,
            status=STATUS_SKIPPED,
            subset_mode=image_mode,
        )
    return download_hugging_face_image_source_mirror(
        root,
        item,
        repo_id=repo_id,
        resume=resume,
        hf_token_env_file=hf_token_env_file,
    )


def download_direct_file(
    root: str | Path,
    item: DownloadPlanItem,
    *,
    http_client: HTTPClient | None = None,
    resume: bool = True,
    timeout_seconds: float = 60,
    chunk_size: int = 1024 * 1024,
) -> DownloadResult:
    root_path = Path(root)
    target_path = _resolve_local_target(root_path, item.local_path)
    part_path = _part_path(target_path)
    sidecar_path = _sidecar_path(target_path)
    _ensure_under_root(root_path, part_path, label="partial")
    _ensure_under_root(root_path, sidecar_path, label="sidecar")

    complete_result = _complete_sidecar_result(root_path, target_path, item)
    if complete_result is not None:
        return complete_result

    if item.source_url is None:
        result = _failure_result(
            item,
            target_path=target_path,
            root=root_path,
            error="missing source URL",
        )
        _write_sidecar(target_path, result)
        return result

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    client = http_client or UrlopenHTTPClient()
    existing_bytes = part_path.stat().st_size if resume and part_path.exists() else 0
    headers = {"Range": f"bytes={existing_bytes}-"} if existing_bytes else {}

    try:
        with client.open(
            item.source_url,
            headers=headers,
            timeout=timeout_seconds,
        ) as response:
            status = _response_status(response)
            content_type = _response_content_type(response)
            expected_size = _expected_final_size(response, status)
            if status >= 400 or "text/html" in content_type.lower():
                reason = f"HTTP status {status}"
                if content_type:
                    reason = f"{reason}; content-type {content_type}"
                result = _failure_result(
                    item,
                    target_path=target_path,
                    root=root_path,
                    bytes_downloaded=existing_bytes,
                    error=reason,
                )
                _write_sidecar(target_path, result)
                return result

            append_part = existing_bytes > 0 and status == 206
            if existing_bytes > 0 and not append_part:
                existing_bytes = 0
            if append_part:
                expected_size, resume_error = _resume_validation_error(
                    response,
                    existing_bytes,
                )
                if resume_error:
                    result = _failure_result(
                        item,
                        target_path=target_path,
                        root=root_path,
                        bytes_downloaded=existing_bytes,
                        error=resume_error,
                    )
                    _write_sidecar(target_path, result)
                    return result

            with part_path.open("ab" if append_part else "wb") as output:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    output.write(chunk)

        part_size = part_path.stat().st_size
        if expected_size is None or part_size != expected_size:
            error = f"downloaded {part_size} bytes"
            if expected_size is None:
                error = f"{error}; missing expected response size"
            else:
                error = f"{error}; expected {expected_size} bytes"
            result = _failure_result(
                item,
                target_path=target_path,
                root=root_path,
                bytes_downloaded=part_size,
                error=error,
            )
            _write_sidecar(target_path, result)
            return result

        part_path.replace(target_path)
        bytes_downloaded = target_path.stat().st_size
        result = DownloadResult(
            dataset=item.dataset,
            asset_id=item.asset_id,
            local_path=_repo_relative_path(root_path, target_path),
            status=STATUS_COMPLETE,
            bytes_downloaded=bytes_downloaded,
            source_url=item.source_url,
        )
        _write_sidecar(target_path, result)
        return result
    except Exception as exc:
        existing = part_path.stat().st_size if part_path.exists() else 0
        result = _failure_result(
            item,
            target_path=target_path,
            root=root_path,
            bytes_downloaded=existing,
            error=str(exc),
        )
        _write_sidecar(target_path, result)
        return result


def execute_download_plan(
    root: str | Path,
    items: Iterable[DownloadPlanItem] | DownloadPlan,
    *,
    dry_run: bool = False,
    include_images: bool = False,
    image_mode: str = "none",
    subset_mode: str = "paper",
    force_large_downloads: bool = False,
    resume: bool = True,
    timeout_seconds: float = 60,
    chunk_size: int = 1024 * 1024,
    http_client: HTTPClient | None = None,
    image_provider_runner=None,
    hf_token_env_file: str | Path | None = None,
) -> dict[str, object]:
    if image_mode not in {"none", "full"}:
        raise ValueError(f"unsupported image_mode: {image_mode}")
    effective_image_mode = image_mode
    if include_images and subset_mode == "full" and image_mode == "none":
        effective_image_mode = "full"
    plan_items = items.items if isinstance(items, DownloadPlan) else tuple(items)
    provider_runner = (
        download_image_source_provider
        if image_provider_runner is None
        else image_provider_runner
    )

    if dry_run:
        results = [
            DownloadResult(
                dataset=item.dataset,
                asset_id=item.asset_id,
                local_path=item.local_path,
                status=STATUS_DRY_RUN,
                bytes_downloaded=0,
                source_url=item.source_url,
                remediation=item.remediation_hint,
            )
            for item in plan_items
        ]
        return _results_envelope(STATUS_DRY_RUN, results)

    results: list[DownloadResult] = []
    result_history: list[DownloadResult] = []
    for item in plan_items:
        if item.provider == PROVIDER_DIRECT_FILE:
            result = download_direct_file(
                root,
                item,
                http_client=http_client,
                resume=resume,
                timeout_seconds=timeout_seconds,
                chunk_size=chunk_size,
            )
            if _should_retry_direct_result(result):
                retry_result = download_direct_file(
                    root,
                    item,
                    http_client=http_client,
                    resume=resume,
                    timeout_seconds=timeout_seconds,
                    chunk_size=chunk_size,
                )
                result_history.extend([result, retry_result])
                result = retry_result
            results.append(result)
            continue

        if item.provider == PROVIDER_IMAGE_SOURCE:
            if not include_images or effective_image_mode != "full":
                results.append(
                    record_image_provider_remediation(
                        root,
                        item,
                        status=STATUS_SKIPPED,
                        subset_mode=subset_mode,
                    )
                )
                continue

            required_bytes = estimate_required_disk_bytes(
                include_images=True,
                subset_mode="full",
            ).minimum_free_bytes
            preflight: dict[str, object] | None = None
            if not force_large_downloads:
                preflight = check_free_space(root, required_bytes)
                free_bytes = int(preflight["free_bytes"])
                if not preflight["sufficient"]:
                    results.append(
                        record_image_provider_remediation(
                            root,
                            item,
                            status=STATUS_INSUFFICIENT_DISK,
                            required_bytes=required_bytes,
                            free_bytes=free_bytes,
                            requested_root=(
                                str(preflight["requested_root"])
                                if "requested_root" in preflight
                                else None
                            ),
                            probed_path=(
                                str(preflight["probed_path"])
                                if "probed_path" in preflight
                                else None
                            ),
                            subset_mode=subset_mode,
                        )
                    )
                    continue

            results.append(
                provider_runner(
                    root,
                    item,
                    resume=resume,
                    image_mode=effective_image_mode,
                    hf_token_env_file=hf_token_env_file,
                )
            )
            continue

        results.append(
            DownloadResult(
                dataset=item.dataset,
                asset_id=item.asset_id,
                local_path=item.local_path,
                status=STATUS_SKIPPED,
                bytes_downloaded=0,
                source_url=item.source_url,
                remediation=f"Unsupported provider: {item.provider}",
            )
        )

    return _results_envelope(
        _overall_status(results),
        results,
        result_history=result_history,
    )
