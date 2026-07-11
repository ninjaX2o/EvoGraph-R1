"""Canonical and compatibility roots for full-image EchoSight mirrors."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


_FULL_IMAGE_LAYOUTS: dict[tuple[str, str], tuple[PurePosixPath, tuple[PurePosixPath, ...]]] = {
    (
        "E-VQA",
        "e_vqa_image_source_google_landmarks_v2",
    ): (
        PurePosixPath("datasets_mm/E-VQA/raw/images/full/gldv2"),
        (
            PurePosixPath("datasets_mm/E-VQA/raw/images/google_landmarks_v2"),
        ),
    ),
    (
        "E-VQA",
        "e_vqa_image_source_inaturalist_2021",
    ): (
        PurePosixPath("datasets_mm/E-VQA/raw/images/full/inaturalist"),
        (
            PurePosixPath("datasets_mm/E-VQA/raw/images/inaturalist_2021"),
            PurePosixPath("datasets_mm/E-VQA/raw/images"),
        ),
    ),
    (
        "InfoSeek",
        "infoseek_image_source_oven",
    ): (
        PurePosixPath("datasets_mm/InfoSeek/raw/images/full/oven"),
        (
            PurePosixPath("datasets_mm/InfoSeek/raw/images/oven"),
        ),
    ),
}

_DATASET_FULL_IMAGE_ASSET_IDS: dict[str, tuple[str, ...]] = {
    "E-VQA": (
        "e_vqa_image_source_google_landmarks_v2",
        "e_vqa_image_source_inaturalist_2021",
    ),
    "InfoSeek": ("infoseek_image_source_oven",),
}


def _layout_for(dataset: str, asset_id: str) -> tuple[PurePosixPath, tuple[PurePosixPath, ...]]:
    try:
        return _FULL_IMAGE_LAYOUTS[(dataset, asset_id)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported full-image layout for dataset={dataset!r}, asset_id={asset_id!r}"
        ) from exc


def canonical_full_image_root(dataset: str, asset_id: str) -> str:
    canonical_root, _ = _layout_for(dataset, asset_id)
    return canonical_root.as_posix()


def candidate_full_image_roots(
    root: str | Path,
    dataset: str,
    asset_id: str,
) -> tuple[Path, ...]:
    canonical_root, compatibility_roots = _layout_for(dataset, asset_id)
    root_path = Path(root)
    relative_roots = (canonical_root, *compatibility_roots)
    return tuple(root_path.joinpath(*relative_root.parts) for relative_root in relative_roots)


def dataset_full_image_roots(root: str | Path, dataset: str) -> tuple[Path, ...]:
    try:
        asset_ids = _DATASET_FULL_IMAGE_ASSET_IDS[dataset]
    except KeyError as exc:
        raise ValueError(f"unsupported full-image dataset: {dataset!r}") from exc

    ordered: list[Path] = []
    seen: set[Path] = set()
    for asset_id in asset_ids:
        for candidate in candidate_full_image_roots(root, dataset, asset_id):
            if candidate in seen:
                continue
            seen.add(candidate)
            ordered.append(candidate)
    return tuple(ordered)


__all__ = [
    "candidate_full_image_roots",
    "canonical_full_image_root",
    "dataset_full_image_roots",
]
