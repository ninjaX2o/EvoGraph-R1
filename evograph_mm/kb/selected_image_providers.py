"""Selected-image provider planners for EchoSight E-VQA sources."""

from __future__ import annotations

import csv
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from .prepare_echosight import (
    DATASET_E_VQA,
    DATASET_INFOSEEK,
    DATASETS_MM_ROOT,
    raw_dataset_root,
)
from .selected_image_download import SelectedDownloadRequest
from .selected_images import HFToken, token_report


GLDV2_TRAIN_METADATA_URL = "https://s3.amazonaws.com/google-landmark/metadata/train.csv"
GLDV2_USER_AGENT = (
    "EvoGraphR1-EchoSightSubset/0.1 "
    "(local research data preparation) Python-urllib"
)
OVEN_HF_REPO = "https://huggingface.co/datasets/ychenNLP/oven/resolve/main"

GLDV2_PROVIDER = "gldv2"
INATURALIST_PROVIDER = "inaturalist_2021"
OVEN_PROVIDER = "oven"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
GLDV2_SELECTED_REL = (
    PurePosixPath(DATASETS_MM_ROOT)
    / DATASET_E_VQA
    / "raw"
    / "images"
    / "google_landmarks_v2"
    / "selected"
)
OVEN_IMAGE_ROOT_REL = (
    PurePosixPath(DATASETS_MM_ROOT)
    / DATASET_INFOSEEK
    / "raw"
    / "images"
    / OVEN_PROVIDER
)
OVEN_SOURCE_ARCHIVES_REL = OVEN_IMAGE_ROOT_REL / "source_archives"
OVEN_SELECTED_REL = OVEN_IMAGE_ROOT_REL / "selected"
MAPPING_VALUE_KEYS = ("file_name", "filename", "path", "image_path", "image", "img")
IMAGE_ID_KEYS = ("image_id", "dataset_image_ids", "image", "img_id", "img", "id")
OVEN_MAPPING_ID_KEYS = ("oven_id", "id", "image_id")
OVEN_MAPPING_PATH_KEYS = ("image_path", "path", "impath")
OVEN_NUMBERED_SHARDS = tuple(f"shard{index:02d}.tar" for index in range(1, 9))
OVEN_SHARD_SIZE_BYTES = {
    # Hugging Face displays decimal GB sizes for these gated archive files.
    "all_wikipedia_images.tar": 32_400_000_000,
    "shard01.tar": 25_500_000_000,
    "shard02.tar": 21_800_000_000,
    "shard03.tar": 21_200_000_000,
    "shard04.tar": 30_100_000_000,
    "shard05.tar": 42_400_000_000,
    "shard06.tar": 38_800_000_000,
    "shard07.tar": 38_500_000_000,
    "shard08.tar": 42_400_000_000,
}


def build_gldv2_requests(
    root: str | Path,
    candidate_manifest: Mapping[str, object],
    metadata_csv: str | Path,
) -> dict[str, object]:
    """Map selected GLDv2 ids to concrete download requests from local metadata."""

    root_path = Path(root)
    metadata_path = _resolve_existing_file(root_path, metadata_csv, "metadata_csv")
    selected_ids, candidate_blockers = _selected_image_ids(
        candidate_manifest,
        GLDV2_PROVIDER,
        _valid_gldv2_id,
    )
    metadata, metadata_blockers = _read_gldv2_metadata(metadata_path)

    requests: list[SelectedDownloadRequest] = []
    missing: list[str] = []
    blockers = list(candidate_blockers)
    for image_id in selected_ids:
        if image_id in metadata_blockers:
            missing.append(image_id)
            blockers.extend(metadata_blockers[image_id])
            continue

        row = metadata.get(image_id)
        if row is None:
            missing.append(image_id)
            blockers.append(
                _blocker(
                    provider=GLDV2_PROVIDER,
                    image_id=image_id,
                    reason="missing_gldv2_metadata",
                    remediation=(
                        "Refresh or supply GLDv2 train metadata containing this "
                        "selected image id."
                    ),
                )
            )
            continue

        suffix = _extension_from_url(row["url"])
        target_path = _safe_relative_output_path(
            root_path,
            GLDV2_SELECTED_REL / f"{image_id}{suffix}",
        )
        requests.append(
            SelectedDownloadRequest(
                dataset=DATASET_E_VQA,
                provider=GLDV2_PROVIDER,
                image_id=image_id,
                url=row["url"],
                target_path=target_path,
                headers={"User-Agent": GLDV2_USER_AGENT},
            )
        )

    return _provider_summary(
        provider=GLDV2_PROVIDER,
        complete=not blockers,
        planned=len(requests),
        missing=missing,
        blockers=blockers,
        remediation=_gldv2_remediation(blockers),
        requests=requests,
        extra={
            "metadata_csv": str(metadata_path),
            "metadata_url": GLDV2_TRAIN_METADATA_URL,
            "request_dicts": [asdict(request) for request in requests],
        },
    )


def resolve_local_inaturalist(
    root: str | Path,
    candidate_manifest: Mapping[str, object],
    image_roots: Sequence[str | Path],
) -> dict[str, object]:
    """Resolve selected iNaturalist ids to existing local files only."""

    root_path = Path(root)
    selected_ids, candidate_blockers = _selected_image_ids(
        candidate_manifest,
        INATURALIST_PROVIDER,
        _valid_inaturalist_id,
    )
    mappings, mapping_files, mapping_blockers = _load_inaturalist_mappings(root_path)
    search_roots = _image_roots(root_path, image_roots)

    resolved: list[dict[str, object]] = []
    missing: list[str] = []
    blockers = list(candidate_blockers)
    blockers.extend(mapping_blockers)
    for image_id in selected_ids:
        mapped = mappings.get(image_id)
        if not mapped:
            missing.append(image_id)
            blockers.append(
                _blocker(
                    provider=INATURALIST_PROVIDER,
                    image_id=image_id,
                    reason="missing_inaturalist_mapping",
                    remediation=(
                        "Place the EchoSight iNaturalist id2name JSON files under "
                        "datasets_mm/E-VQA/raw/id2name and supply local image roots."
                    ),
                )
            )
            continue

        relative_candidates = _relative_image_candidates(mapped)
        if not relative_candidates:
            missing.append(image_id)
            blockers.append(
                _blocker(
                    provider=INATURALIST_PROVIDER,
                    image_id=image_id,
                    reason="unsafe_inaturalist_mapping",
                    remediation=(
                        "Fix the local id2name mapping so it names a relative "
                        ".jpg, .jpeg, or .png image path."
                    ),
                    mapping_value=_reportable_mapping_value(mapped),
                )
            )
            continue

        local_path = _resolve_existing_local_image(search_roots, relative_candidates)
        if local_path is None:
            missing.append(image_id)
            blockers.append(
                _blocker(
                    provider=INATURALIST_PROVIDER,
                    image_id=image_id,
                    reason="missing_local_inaturalist_file",
                    remediation=(
                        "Supply an image root containing the mapped iNaturalist file; "
                        "full archive downloads are intentionally not scheduled."
                    ),
                    mapping_value=_reportable_mapping_value(mapped),
                )
            )
            continue

        resolved.append(
            {
                "dataset": DATASET_E_VQA,
                "provider": INATURALIST_PROVIDER,
                "image_id": image_id,
                "local_path": str(local_path),
                "mapping_value": mapped,
            }
        )

    return _provider_summary(
        provider=INATURALIST_PROVIDER,
        complete=not blockers,
        planned=0,
        missing=missing,
        blockers=blockers,
        remediation=_inaturalist_remediation(blockers),
        requests=[],
        extra={
            "image_roots": [str(path) for path in search_roots],
            "mapping_files": [str(path) for path in mapping_files],
            "request_dicts": [],
            "resolved": resolved,
        },
    )


def parse_oven_mapping(
    mapping_csv: str | Path,
    selected_ids: Iterable[str],
) -> dict[str, object]:
    """Map selected InfoSeek OVEN ids to gated Hugging Face tar members."""

    mapping_path = Path(mapping_csv)
    selected = _normalize_selected_oven_ids(selected_ids)
    selected_set = set(selected)
    members_by_id: dict[str, dict[str, str]] = {}
    blockers: list[dict[str, object]] = []
    blocked_ids: set[str] = set()

    for image_id in selected:
        if not _valid_oven_id(image_id):
            blocked_ids.add(image_id)
            blockers.append(
                _oven_blocker(
                    image_id=image_id,
                    reason="invalid_selected_oven_id",
                    remediation=(
                        "Regenerate the selected-image candidate manifest with "
                        "InfoSeek OVEN ids shaped like oven_00000000."
                    ),
                )
            )

    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        id_key = _required_csv_column(
            reader.fieldnames,
            OVEN_MAPPING_ID_KEYS,
            "OVEN mapping id",
        )
        path_key = _required_csv_column(
            reader.fieldnames,
            OVEN_MAPPING_PATH_KEYS,
            "OVEN mapping path",
        )

        for row in reader:
            image_id = str(row.get(id_key) or "").strip()
            if image_id not in selected_set or image_id in blocked_ids:
                continue

            raw_member_path = str(row.get(path_key) or "").strip()
            try:
                member_path = _safe_oven_member_path(raw_member_path)
                shard_name = _oven_shard_name(member_path)
            except ValueError:
                members_by_id.pop(image_id, None)
                blocked_ids.add(image_id)
                blockers.append(
                    _oven_blocker(
                        image_id=image_id,
                        reason="unsafe_oven_mapping_path",
                        remediation=(
                            "Fix ovenid2impath.csv so selected ids map to "
                            "relative .jpg, .jpeg, or .png tar member paths "
                            "inside known OVEN shard directories."
                        ),
                        mapping_value=_reportable_unsafe_mapping_value(raw_member_path),
                    )
                )
                continue

            existing = members_by_id.get(image_id)
            member_name = member_path.as_posix()
            if existing is None:
                members_by_id[image_id] = {
                    "image_id": image_id,
                    "member_name": member_name,
                    "shard_name": shard_name,
                }
                continue
            if (
                existing["member_name"] == member_name
                and existing["shard_name"] == shard_name
            ):
                continue

            members_by_id.pop(image_id, None)
            blocked_ids.add(image_id)
            blockers.append(
                _oven_blocker(
                    image_id=image_id,
                    reason="conflicting_oven_mapping_path",
                    remediation=(
                        "Deduplicate ovenid2impath.csv so each selected OVEN id "
                        "maps to one tar member path."
                    ),
                )
            )

    missing: list[str] = []
    for image_id in selected:
        if image_id in blocked_ids:
            missing.append(image_id)
            continue
        if image_id not in members_by_id:
            missing.append(image_id)
            blockers.append(
                _oven_blocker(
                    image_id=image_id,
                    reason="missing_oven_mapping",
                    remediation=(
                        "Supply an ovenid2impath.csv row for this selected "
                        "InfoSeek OVEN image id."
                    ),
                )
            )

    members = [members_by_id[image_id] for image_id in selected if image_id in members_by_id]
    complete = not blockers
    return _provider_summary(
        dataset=DATASET_INFOSEEK,
        provider=OVEN_PROVIDER,
        complete=complete,
        planned=len(members) if complete else 0,
        missing=missing,
        blockers=blockers,
        remediation=_oven_mapping_remediation(blockers),
        requests=[],
        extra={
            "mapping_csv": str(mapping_path),
            "selected_ids": selected,
            "members": members,
            "shards": _oven_group_members_by_shard(members),
            "request_dicts": [],
        },
    )


def plan_oven_shards(
    mapping_report: Mapping[str, object],
    *,
    max_shards: int = 5,
    max_transfer_bytes: int = 120_000_000_000,
) -> dict[str, object]:
    """Choose the bounded OVEN Hugging Face tar archives needed for selected ids."""

    max_shards = _require_nonnegative_int("max_shards", max_shards)
    max_transfer_bytes = _require_nonnegative_int(
        "max_transfer_bytes",
        max_transfer_bytes,
    )
    blockers = [dict(blocker) for blocker in mapping_report.get("blockers", [])]
    missing = [str(image_id) for image_id in mapping_report.get("missing", [])]
    candidate_shards = _oven_shards_with_sizes(mapping_report.get("shards", []))
    required_selected_ids = _oven_required_selected_ids(mapping_report, candidate_shards)
    candidate_cover = _minimum_oven_shard_cover(
        candidate_shards,
        required_selected_ids,
    )
    estimated_transfer_bytes = sum(
        int(shard["size_bytes"]) for shard in candidate_cover
    )

    if not blockers:
        uncovered_selected_ids = sorted(
            required_selected_ids - _oven_covered_selected_ids(candidate_cover)
        )
        for image_id in uncovered_selected_ids:
            blockers.append(
                _oven_blocker(
                    image_id=image_id,
                    reason="missing_oven_shard_coverage",
                    remediation=(
                        "Regenerate the OVEN mapping report so every selected id "
                        "is covered by at least one known shard archive."
                    ),
                )
            )
    if not blockers:
        if _is_near_full_oven_mirror(candidate_cover):
            blockers.append(
                _oven_blocker(
                    image_id="",
                    reason="near_full_oven_mirror",
                    remediation=(
                        "The selected OVEN ids require a near-full OVEN mirror. "
                        "Use a smaller selected subset or add a future explicit "
                        "force flag."
                    ),
                )
            )
        if len(candidate_cover) > max_shards:
            blockers.append(
                _oven_blocker(
                    image_id="",
                    reason="max_oven_shards_exceeded",
                    remediation=(
                        "Reduce the selected InfoSeek candidate buffer or raise "
                        "max_shards only after reviewing disk and transfer impact."
                    ),
                )
            )
        if estimated_transfer_bytes > max_transfer_bytes:
            blockers.append(
                _oven_blocker(
                    image_id="",
                    reason="max_oven_transfer_exceeded",
                    remediation=(
                        "Reduce the selected InfoSeek candidate buffer or raise "
                        "max_transfer_bytes only after reviewing disk capacity."
                    ),
                )
            )

    complete = not blockers
    planned_shards = candidate_cover if complete else []
    return _provider_summary(
        dataset=DATASET_INFOSEEK,
        provider=OVEN_PROVIDER,
        complete=complete,
        planned=len(planned_shards),
        missing=missing,
        blockers=blockers,
        remediation=_oven_plan_remediation(blockers),
        requests=[],
        extra={
            "request_dicts": [],
            "shards": planned_shards,
            "candidate_shards": candidate_shards,
            "estimated_transfer_bytes": estimated_transfer_bytes,
            "max_shards": max_shards,
            "max_transfer_bytes": max_transfer_bytes,
            "extraction_output_root": OVEN_SELECTED_REL.as_posix(),
        },
    )


def build_oven_shard_requests(
    root: str | Path,
    shard_plan: Mapping[str, object],
    token: HFToken | None,
) -> dict[str, object]:
    """Build bounded OVEN archive download requests without fetching payloads."""

    root_path = Path(root)
    blockers = [dict(blocker) for blocker in shard_plan.get("blockers", [])]
    missing = [str(image_id) for image_id in shard_plan.get("missing", [])]
    requests: list[SelectedDownloadRequest] = []
    extraction_plan: list[dict[str, object]] = []

    if not blockers and bool(shard_plan.get("complete", False)):
        headers = _oven_auth_headers(token)
        for shard in shard_plan.get("shards", []):
            if not isinstance(shard, Mapping):
                blockers.append(
                    _oven_blocker(
                        image_id="",
                        reason="invalid_oven_shard_plan",
                        remediation="Regenerate the OVEN shard plan before building requests.",
                    )
                )
                continue
            shard_name = str(shard.get("shard_name") or "")
            if shard_name not in OVEN_SHARD_SIZE_BYTES:
                blockers.append(
                    _oven_blocker(
                        image_id="",
                        reason="unknown_oven_shard_archive",
                        remediation="Regenerate the OVEN shard plan from ovenid2impath.csv.",
                    )
                )
                continue

            target_path = _safe_relative_output_path(
                root_path,
                OVEN_SOURCE_ARCHIVES_REL / shard_name,
            )
            member_names, member_blockers = _validated_oven_shard_member_names(
                shard_name,
                shard.get("member_names", []),
            )
            blockers.extend(member_blockers)
            if member_blockers:
                continue
            selected_image_ids = [
                str(image_id) for image_id in shard.get("selected_image_ids", [])
            ]
            requests.append(
                SelectedDownloadRequest(
                    dataset=DATASET_INFOSEEK,
                    provider=OVEN_PROVIDER,
                    image_id=shard_name,
                    url=f"{OVEN_HF_REPO}/{shard_name}",
                    target_path=target_path,
                    headers=headers,
                    payload_kind="archive",
                )
            )
            extraction_plan.append(
                {
                    "source_archive": target_path,
                    "output_root": _safe_relative_output_path(root_path, OVEN_SELECTED_REL),
                    "member_names": member_names,
                    "selected_image_ids": selected_image_ids,
                }
            )

    complete = not blockers
    if not complete:
        requests = []
        extraction_plan = []

    return _provider_summary(
        dataset=DATASET_INFOSEEK,
        provider=OVEN_PROVIDER,
        complete=complete,
        planned=len(requests),
        missing=missing,
        blockers=blockers,
        remediation=_oven_request_remediation(blockers, token),
        requests=requests,
        extra={
            "request_dicts": [_request_dict_without_sensitive_headers(request) for request in requests],
            "extraction_plan": extraction_plan,
            "token": _oven_token_report(token),
            "hf_repo": OVEN_HF_REPO,
        },
    )


def _normalize_selected_oven_ids(selected_ids: Iterable[str]) -> list[str]:
    if isinstance(selected_ids, (str, bytes)):
        raise ValueError("selected_ids must be an iterable of OVEN image ids")
    seen: set[str] = set()
    normalized: list[str] = []
    for value in selected_ids:
        image_id = str(value).strip()
        if not image_id or image_id in seen:
            continue
        seen.add(image_id)
        normalized.append(image_id)
    return sorted(normalized)


def _valid_oven_id(image_id: str) -> bool:
    return bool(re.fullmatch(r"oven_[0-9]{8}", image_id))


def _required_csv_column(
    fieldnames: Sequence[str] | None,
    accepted: Sequence[str],
    label: str,
) -> str:
    lookup = {str(name).strip().lower(): name for name in fieldnames or []}
    for key in accepted:
        if key in lookup:
            return lookup[key]
    expected = ", ".join(accepted)
    raise ValueError(f"{label} CSV missing one of column(s): {expected}")


def _safe_oven_member_path(value: str) -> PurePosixPath:
    if _is_url_like(value):
        raise ValueError("OVEN mapping path must not be URL-like")
    member_path = _safe_relative_posix_path(value, "OVEN mapping path")
    if member_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("OVEN mapping path must name an image member")
    _oven_shard_name(member_path)
    return member_path


def _oven_shard_name(member_path: PurePosixPath) -> str:
    if not member_path.parts:
        raise ValueError("OVEN mapping path must name a shard member")
    shard_dir = member_path.parts[0]
    if re.fullmatch(r"shard0[1-8]", shard_dir):
        return f"{shard_dir}.tar"
    if shard_dir == "all_wikipedia_images":
        return "all_wikipedia_images.tar"
    raise ValueError("OVEN mapping path must be under a known OVEN shard directory")


def _reportable_unsafe_mapping_value(value: str) -> str:
    if _is_url_like(value):
        return "<redacted-url>"
    return "<unsafe-path>"


def _oven_group_members_by_shard(
    members: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for member in members:
        shard_name = str(member["shard_name"])
        shard = grouped.setdefault(
            shard_name,
            {
                "shard_name": shard_name,
                "member_names": [],
                "selected_image_ids": [],
            },
        )
        shard["member_names"].append(str(member["member_name"]))
        shard["selected_image_ids"].append(str(member["image_id"]))
    return [grouped[name] for name in sorted(grouped, key=_oven_shard_sort_key)]


def _oven_required_selected_ids(
    mapping_report: Mapping[str, object],
    candidate_shards: Sequence[Mapping[str, object]],
) -> set[str]:
    missing = _string_set_from_sequence(mapping_report.get("missing"))
    selected_ids = mapping_report.get("selected_ids")
    required = _string_set_from_sequence(selected_ids) - missing
    if required:
        return required

    members = mapping_report.get("members")
    if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
        required = {
            str(member.get("image_id"))
            for member in members
            if isinstance(member, Mapping) and str(member.get("image_id") or "")
        }
        if required:
            return required

    return _oven_covered_selected_ids(candidate_shards)


def _string_set_from_sequence(value: object) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    result: set[str] = set()
    for item in value:
        if item is None:
            continue
        normalized = str(item).strip()
        if normalized:
            result.add(normalized)
    return result


def _oven_shards_with_sizes(shards: object) -> list[dict[str, object]]:
    if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)):
        raise ValueError("OVEN mapping report shards must be a sequence")
    planned: list[dict[str, object]] = []
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("OVEN mapping report shard entries must be mappings")
        shard_name = str(shard.get("shard_name") or "")
        size_bytes = OVEN_SHARD_SIZE_BYTES.get(shard_name)
        if size_bytes is None:
            raise ValueError(f"unknown OVEN shard archive: {shard_name}")
        planned.append(
            {
                "shard_name": shard_name,
                "size_bytes": size_bytes,
                "member_names": [str(name) for name in shard.get("member_names", [])],
                "selected_image_ids": [
                    str(image_id) for image_id in shard.get("selected_image_ids", [])
                ],
            }
        )
    return sorted(planned, key=lambda item: _oven_shard_sort_key(str(item["shard_name"])))


def _minimum_oven_shard_cover(
    candidate_shards: Sequence[dict[str, object]],
    required_selected_ids: set[str],
) -> list[dict[str, object]]:
    if not candidate_shards:
        return []
    if not required_selected_ids:
        return list(candidate_shards)

    best_key: tuple[int, int, tuple[tuple[int, int | str], ...]] | None = None
    best_cover: list[dict[str, object]] = []
    for shard_count in range(1, len(candidate_shards) + 1):
        for shard_combo in combinations(candidate_shards, shard_count):
            if not required_selected_ids.issubset(
                _oven_covered_selected_ids(shard_combo)
            ):
                continue
            shard_names = tuple(
                sorted(
                    (str(shard["shard_name"]) for shard in shard_combo),
                    key=_oven_shard_sort_key,
                )
            )
            stable_name_key = tuple(_oven_shard_sort_key(name) for name in shard_names)
            cover_key = (
                sum(int(shard["size_bytes"]) for shard in shard_combo),
                shard_count,
                stable_name_key,
            )
            if best_key is None or cover_key < best_key:
                best_key = cover_key
                best_cover = list(shard_combo)

    return sorted(
        best_cover,
        key=lambda item: _oven_shard_sort_key(str(item["shard_name"])),
    )


def _oven_covered_selected_ids(
    shards: Sequence[Mapping[str, object]],
) -> set[str]:
    covered: set[str] = set()
    for shard in shards:
        selected_image_ids = shard.get("selected_image_ids", [])
        if not isinstance(selected_image_ids, Sequence) or isinstance(
            selected_image_ids,
            (str, bytes),
        ):
            continue
        covered.update(str(image_id) for image_id in selected_image_ids if str(image_id))
    return covered


def _is_near_full_oven_mirror(shards: Sequence[Mapping[str, object]]) -> bool:
    shard_names = {str(shard.get("shard_name") or "") for shard in shards}
    numbered_count = len(set(OVEN_NUMBERED_SHARDS).intersection(shard_names))
    return numbered_count == len(OVEN_NUMBERED_SHARDS) or (
        "all_wikipedia_images.tar" in shard_names and numbered_count >= 7
    )


def _oven_shard_sort_key(shard_name: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"shard0([1-8])\.tar", shard_name)
    if match:
        return (0, int(match.group(1)))
    if shard_name == "all_wikipedia_images.tar":
        return (1, shard_name)
    return (2, shard_name)


def _require_nonnegative_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _oven_auth_headers(token: HFToken | None) -> dict[str, str] | None:
    if token is None or not token.value:
        return None
    return {"Authorization": f"Bearer {token.value}"}


def _request_dict_without_sensitive_headers(
    request: SelectedDownloadRequest,
) -> dict[str, object]:
    payload = request.to_report_dict()
    payload["headers"] = payload["headers"] or None
    return payload


def _validated_oven_shard_member_names(
    shard_name: str,
    member_names: object,
) -> tuple[list[str], list[dict[str, object]]]:
    if not isinstance(member_names, Sequence) or isinstance(member_names, (str, bytes)):
        return [], [
            _oven_blocker(
                image_id="",
                reason="invalid_oven_shard_plan",
                remediation="Regenerate the OVEN shard plan before building requests.",
            )
        ]

    valid_member_names: list[str] = []
    blockers: list[dict[str, object]] = []
    for raw_member_name in member_names:
        member_name = str(raw_member_name)
        try:
            member_path = _safe_oven_member_path(member_name)
            if _oven_shard_name(member_path) != shard_name:
                raise ValueError("OVEN shard member is under a different shard root")
        except ValueError:
            blockers.append(
                _oven_blocker(
                    image_id="",
                    reason="unsafe_oven_shard_member",
                    remediation=(
                        "Regenerate the public OVEN shard plan so every member is "
                        "a relative .jpg, .jpeg, or .png path under its shard root."
                    ),
                    mapping_value=_reportable_unsafe_mapping_value(member_name),
                )
            )
            continue
        valid_member_names.append(member_path.as_posix())
    return valid_member_names, blockers


def _oven_token_report(token: HFToken | None) -> dict[str, object]:
    if token is None:
        return {"present": False, "source": None, "redacted": False}
    return token_report(token)


def _oven_blocker(
    *,
    image_id: str,
    reason: str,
    remediation: str,
    mapping_value: str | None = None,
) -> dict[str, object]:
    return _blocker(
        dataset=DATASET_INFOSEEK,
        provider=OVEN_PROVIDER,
        image_id=image_id,
        reason=reason,
        remediation=remediation,
        mapping_value=mapping_value,
    )


def _oven_mapping_remediation(blockers: Sequence[Mapping[str, object]]) -> list[str]:
    if not blockers:
        return []
    return [
        "Use the gated OVEN ovenid2impath.csv mapping from Hugging Face and "
        "ensure selected ids map to safe relative tar member paths.",
    ]


def _oven_plan_remediation(blockers: Sequence[Mapping[str, object]]) -> list[str]:
    if not blockers:
        return []
    return [
        "Reduce the InfoSeek selected-image buffer or continue with an explicit "
        "larger-disk OVEN mirror workflow; this planner only schedules bounded "
        "selected-source archives.",
    ]


def _oven_request_remediation(
    blockers: Sequence[Mapping[str, object]],
    token: HFToken | None,
) -> list[str]:
    messages: list[str] = []
    if blockers:
        messages.append("Regenerate a complete bounded OVEN shard plan before downloading.")
    if token is None or not token.value:
        messages.append(
            "OVEN is gated on Hugging Face; log in, accept the dataset conditions, "
            "and pass an HF token when executing the planned requests."
        )
    return messages


def _selected_image_ids(
    candidate_manifest: Mapping[str, object],
    provider: str,
    validator,
) -> tuple[list[str], list[dict[str, object]]]:
    dataset = str(candidate_manifest.get("dataset", DATASET_E_VQA))
    if dataset != DATASET_E_VQA:
        raise ValueError(f"selected provider only supports {DATASET_E_VQA}: {dataset}")

    candidates = candidate_manifest.get("candidates", [])
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("candidate_manifest['candidates'] must be a sequence")

    image_ids: list[str] = []
    blockers: list[dict[str, object]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate_manifest candidates must be mappings")
        if candidate.get("source") != provider:
            continue
        image_id = str(candidate.get("image_id", "")).strip()
        if not validator(image_id):
            blockers.append(
                _blocker(
                    provider=provider,
                    image_id=image_id,
                    reason="invalid_candidate_image_id",
                    remediation=(
                        "Regenerate the selected-image candidate manifest with "
                        "valid E-VQA source classification."
                    ),
                )
            )
            continue
        if image_id in seen:
            continue
        seen.add(image_id)
        image_ids.append(image_id)
    return image_ids, blockers


def _valid_gldv2_id(image_id: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{16}", image_id))


def _valid_inaturalist_id(image_id: str) -> bool:
    return image_id.isdigit()


def _read_gldv2_metadata(
    path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, object]]]]:
    rows: dict[str, dict[str, str]] = {}
    blockers: dict[str, list[dict[str, object]]] = {}
    blocked_ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = {"id", "url"} - fieldnames
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"GLDv2 metadata missing required column(s): {missing}")
        for row in reader:
            image_id = str(row.get("id") or "").strip()
            url = str(row.get("url") or "").strip()
            if not _valid_gldv2_id(image_id):
                continue
            if image_id in blocked_ids:
                continue
            if not _valid_http_url(url):
                blocked_ids.add(image_id)
                rows.pop(image_id, None)
                _add_gldv2_metadata_blocker(
                    blockers,
                    image_id,
                    "invalid_gldv2_metadata_url",
                )
                continue
            existing = rows.get(image_id)
            if existing is None:
                rows[image_id] = {"id": image_id, "url": url}
                continue
            if existing["url"] == url:
                continue
            rows.pop(image_id, None)
            blocked_ids.add(image_id)
            _add_gldv2_metadata_blocker(
                blockers,
                image_id,
                "conflicting_gldv2_metadata_url",
            )
    return rows, blockers


def _valid_http_url(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _add_gldv2_metadata_blocker(
    blockers: dict[str, list[dict[str, object]]],
    image_id: str,
    reason: str,
) -> None:
    if any(blocker.get("reason") == reason for blocker in blockers.get(image_id, [])):
        return
    blockers.setdefault(image_id, []).append(
        _blocker(
            provider=GLDV2_PROVIDER,
            image_id=image_id,
            reason=reason,
            remediation=_gldv2_metadata_remediation(reason),
        )
    )


def _gldv2_metadata_remediation(reason: str) -> str:
    if reason == "invalid_gldv2_metadata_url":
        return "Use GLDv2 metadata rows whose url column contains an http or https URL."
    if reason == "conflicting_gldv2_metadata_url":
        return (
            "Deduplicate GLDv2 metadata so each selected image id maps to one "
            "consistent URL."
        )
    return "Refresh or supply valid GLDv2 train metadata for this selected image id."


def _extension_from_url(url: str) -> str:
    suffix = PurePosixPath(urlsplit(url).path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return suffix
    return ".jpg"


def _load_inaturalist_mappings(
    root_path: Path,
) -> tuple[dict[str, str], list[Path], list[dict[str, object]]]:
    raw_rel = _safe_relative_output_path(root_path, raw_dataset_root(DATASET_E_VQA))
    raw_root = root_path.joinpath(*PurePosixPath(raw_rel).parts)
    mapping_root = raw_root / "id2name"
    mapping_files: list[Path] = []
    mappings: dict[str, str] = {}
    blockers: list[dict[str, object]] = []
    try:
        mapping_root_resolved = mapping_root.resolve(strict=True)
    except FileNotFoundError:
        return mappings, mapping_files, blockers
    if not mapping_root_resolved.is_dir():
        return mappings, mapping_files, blockers

    root_resolved = root_path.resolve(strict=False)
    raw_root_resolved = raw_root.resolve(strict=False)
    if not (
        _is_resolved_under(mapping_root_resolved, raw_root_resolved)
        and _is_resolved_under(mapping_root_resolved, root_resolved)
    ):
        blockers.append(
            _blocker(
                provider=INATURALIST_PROVIDER,
                image_id="",
                reason="unsafe_inaturalist_mapping_root",
                remediation=(
                    "Keep datasets_mm/E-VQA/raw/id2name inside the selected-image "
                    "root instead of pointing it at an external directory."
                ),
            )
        )
        return {}, [], blockers

    for path in sorted(mapping_root.glob("*.json")):
        try:
            resolved = path.resolve(strict=True)
            if not (
                _is_resolved_under(resolved, mapping_root_resolved)
                and _is_resolved_under(resolved, root_resolved)
            ):
                raise ValueError
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(f"unsafe iNaturalist mapping file path: {path}") from exc
        if not resolved.is_file():
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mapping_files.append(resolved)
        _merge_mapping_payload(mappings, payload)
    return mappings, mapping_files, blockers


def _merge_mapping_payload(target: dict[str, str], payload: object) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            mapped = _mapping_value(value)
            image_id = str(key).strip()
            if image_id and mapped:
                target[image_id] = mapped
        return
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            image_id = _first_mapping_item_value(item, IMAGE_ID_KEYS)
            mapped = _first_mapping_item_value(item, MAPPING_VALUE_KEYS)
            if image_id and mapped:
                target[image_id] = mapped


def _mapping_value(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, Mapping):
        return _first_mapping_item_value(value, MAPPING_VALUE_KEYS)
    return None


def _first_mapping_item_value(item: Mapping[str, object], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        stripped = str(value).strip()
        if stripped:
            return stripped
    return None


def _image_roots(root_path: Path, image_roots: Sequence[str | Path]) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for image_root in image_roots:
        candidate = _resolve_optional_directory(root_path, image_root, "image_root")
        if candidate is None:
            continue
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        roots.append(candidate)
    return roots


def _resolve_optional_directory(
    root_path: Path,
    path: str | Path,
    label: str,
) -> Path | None:
    candidate = _input_path(root_path, path, label)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        return None
    if not resolved.is_dir():
        return None
    return resolved


def _resolve_existing_file(root_path: Path, path: str | Path, label: str) -> Path:
    candidate = _input_path(root_path, path, label)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    return resolved


def _input_path(root_path: Path, path: str | Path, label: str) -> Path:
    text = str(path)
    if not text:
        raise ValueError(f"{label} must not be empty")
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    relative = _safe_relative_posix_path(path, label)
    target = root_path.joinpath(*relative.parts)
    root_resolved = root_path.resolve(strict=False)
    target_resolved = target.resolve(strict=False)
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes selected-image root: {path}") from exc
    return target


def _safe_relative_output_path(root_path: Path, relative_path: str | Path) -> str:
    relative = _safe_relative_posix_path(relative_path, "relative_path")
    target = root_path.joinpath(*relative.parts)
    root_resolved = root_path.resolve(strict=False)
    target_resolved = target.resolve(strict=False)
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"relative_path escapes selected-image root: {relative_path}"
        ) from exc
    return relative.as_posix()


def _safe_relative_posix_path(path: str | Path, label: str) -> PurePosixPath:
    raw = str(path).strip()
    normalized = raw.replace("\\", "/")
    windows_path = PureWindowsPath(raw)
    posix_path = PurePosixPath(normalized)
    if not raw:
        raise ValueError(f"{label} must not be empty")
    if (
        windows_path.drive
        or windows_path.root
        or posix_path.is_absolute()
        or ".." in windows_path.parts
        or ".." in posix_path.parts
    ):
        raise ValueError(f"{label} must be relative and stay under its root: {path}")
    parts = tuple(part for part in posix_path.parts if part not in {"", "."})
    if not parts:
        raise ValueError(f"{label} must name a child path: {path}")
    return PurePosixPath(*parts)


def _relative_image_candidates(name: str) -> list[PurePosixPath]:
    if _is_url_like(name):
        return []
    try:
        normalized = _safe_relative_posix_path(name, "mapping_value")
    except ValueError:
        return []
    suffix = normalized.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return [normalized]
    if suffix:
        return []
    return [PurePosixPath(f"{normalized.as_posix()}{suffix}") for suffix in sorted(IMAGE_SUFFIXES)]


def _resolve_existing_local_image(
    search_roots: Sequence[Path],
    relative_candidates: Sequence[PurePosixPath],
) -> Path | None:
    for search_root in search_roots:
        root_resolved = search_root.resolve(strict=True)
        for relative_candidate in relative_candidates:
            candidate = search_root.joinpath(*relative_candidate.parts)
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root_resolved)
            except (FileNotFoundError, ValueError):
                continue
            if resolved.is_file():
                return resolved
    return None


def _is_url_like(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value.strip()))


def _reportable_mapping_value(value: str) -> str:
    if _is_url_like(value):
        return "<redacted-url>"
    return value


def _is_resolved_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _blocker(
    *,
    dataset: str = DATASET_E_VQA,
    provider: str,
    image_id: str,
    reason: str,
    remediation: str,
    mapping_value: str | None = None,
) -> dict[str, object]:
    blocker: dict[str, object] = {
        "dataset": dataset,
        "provider": provider,
        "image_id": image_id,
        "reason": reason,
        "remediation": remediation,
    }
    if mapping_value is not None:
        blocker["mapping_value"] = mapping_value
    return blocker


def _provider_summary(
    *,
    dataset: str = DATASET_E_VQA,
    provider: str,
    complete: bool,
    planned: int,
    missing: Sequence[str],
    blockers: Sequence[Mapping[str, object]],
    remediation: Sequence[str],
    requests: Sequence[SelectedDownloadRequest],
    extra: Mapping[str, object],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "dataset": dataset,
        "provider": provider,
        "complete": complete,
        "planned": planned,
        "missing": list(missing),
        "blockers": [dict(blocker) for blocker in blockers],
        "remediation": list(remediation),
        "requests": list(requests),
    }
    summary.update(extra)
    return summary


def _gldv2_remediation(blockers: Sequence[Mapping[str, object]]) -> list[str]:
    if not blockers:
        return []
    return [
        "Download GLDv2 train metadata from GLDV2_TRAIN_METADATA_URL, then pass "
        "the local CSV path to build_gldv2_requests.",
    ]


def _inaturalist_remediation(blockers: Sequence[Mapping[str, object]]) -> list[str]:
    if not blockers:
        return []
    return [
        "Provide local iNaturalist image roots matching raw/id2name mappings; this "
        "provider does not schedule full image archives.",
    ]


__all__ = [
    "GLDV2_TRAIN_METADATA_URL",
    "OVEN_HF_REPO",
    "build_gldv2_requests",
    "build_oven_shard_requests",
    "parse_oven_mapping",
    "plan_oven_shards",
    "resolve_local_inaturalist",
]
