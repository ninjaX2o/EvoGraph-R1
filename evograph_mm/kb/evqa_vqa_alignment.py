"""Align current E-VQA KB anchors to official raw E-VQA VQA rows."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KBAnchor:
    data_id: str
    text_doc_id: str
    image_id: str
    wikipedia_url: str
    wikipedia_title: str
    contents: str


@dataclass(frozen=True)
class ResolvedVQAImage:
    image_id: str
    path: Path
    exists: bool
    resolution: str
    blocker: str


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _norm(value: object) -> str:
    return " ".join(_text(value).casefold().split())


def _source_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    source_metadata = row.get("source_metadata", {})
    if not isinstance(source_metadata, Mapping):
        return {}
    source_row = source_metadata.get("source_row", {})
    return source_row if isinstance(source_row, Mapping) else {}


def load_kb_anchors(path: str | Path) -> list[KBAnchor]:
    anchors: list[KBAnchor] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source_row = _source_row(row)
            anchors.append(
                KBAnchor(
                    data_id=_text(row.get("data_id")),
                    text_doc_id=_text(row.get("text_doc_id")),
                    image_id=_text(row.get("image_id")),
                    wikipedia_url=_text(source_row.get("wikipedia_url")),
                    wikipedia_title=_text(source_row.get("wikipedia_title")),
                    contents=_text(row.get("contents")),
                )
            )
    return anchors


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def match_raw_rows_to_anchors(
    anchors: Sequence[KBAnchor],
    raw_csv_paths: Sequence[str | Path],
) -> list[dict[str, object]]:
    anchors_by_url: dict[str, list[KBAnchor]] = defaultdict(list)
    anchors_by_title: dict[str, list[KBAnchor]] = defaultdict(list)
    for anchor in anchors:
        normalized_url = _norm(anchor.wikipedia_url)
        normalized_title = _norm(anchor.wikipedia_title)
        if normalized_url:
            anchors_by_url[normalized_url].append(anchor)
        if normalized_title:
            anchors_by_title[normalized_title].append(anchor)

    matches: list[dict[str, object]] = []
    for csv_path in raw_csv_paths:
        path = Path(csv_path)
        for row_index, row in enumerate(_read_csv_rows(path)):
            raw_url = _norm(row.get("wikipedia_url"))
            raw_title = _norm(row.get("wikipedia_title"))
            match_key = ""
            matched_anchors: list[KBAnchor] = []
            if raw_url:
                matched_anchors = anchors_by_url.get(raw_url, [])
                if matched_anchors:
                    match_key = "wikipedia_url"
            if not matched_anchors and raw_title:
                matched_anchors = anchors_by_title.get(raw_title, [])
                if matched_anchors:
                    match_key = "wikipedia_title"

            for anchor in matched_anchors:
                matches.append(
                    {
                        "match_key": match_key,
                        "raw_csv": str(path),
                        "raw_row_index": row_index,
                        "raw_row": row,
                        "kb_anchor": asdict(anchor),
                    }
                )
    return matches


def first_dataset_image_id(value: object) -> str:
    text = _text(value)
    if not text:
        return ""
    return text.split("|", 1)[0].strip()


def resolve_raw_vqa_image_path(
    *,
    dataset_name: str,
    dataset_image_ids: str,
    inaturalist_root: str | Path,
    inaturalist_id2name: Mapping[str, str],
    gldv2_root: str | Path,
) -> ResolvedVQAImage:
    image_id = first_dataset_image_id(dataset_image_ids)
    if not image_id:
        return ResolvedVQAImage(
            image_id="",
            path=Path(""),
            exists=False,
            resolution="missing",
            blocker="missing_dataset_image_ids",
        )

    normalized_dataset = _norm(dataset_name)
    if normalized_dataset == "inaturalist":
        mapped_name = _text(inaturalist_id2name.get(image_id))
        if not mapped_name:
            return ResolvedVQAImage(
                image_id=image_id,
                path=Path(inaturalist_root) / "__missing_id2name__" / f"{image_id}.jpg",
                exists=False,
                resolution="inaturalist_id2name",
                blocker="missing_inaturalist_id2name",
            )
        path = Path(inaturalist_root) / mapped_name
        return ResolvedVQAImage(
            image_id=image_id,
            path=path,
            exists=path.is_file(),
            resolution="inaturalist_id2name",
            blocker="",
        )

    if normalized_dataset in {"landmarks", "google-landmarks", "google_landmarks"}:
        if len(image_id) < 3:
            return ResolvedVQAImage(
                image_id=image_id,
                path=Path(gldv2_root) / f"{image_id}.jpg",
                exists=False,
                resolution="gldv2_trigram",
                blocker="short_landmarks_image_id",
            )
        selected_path = _gldv2_selected_path(Path(gldv2_root), image_id)
        if selected_path.is_file():
            return ResolvedVQAImage(
                image_id=image_id,
                path=selected_path,
                exists=True,
                resolution="gldv2_selected",
                blocker="",
            )
        path = (
            Path(gldv2_root)
            / image_id[0]
            / image_id[1]
            / image_id[2]
            / f"{image_id}.jpg"
        )
        return ResolvedVQAImage(
            image_id=image_id,
            path=path,
            exists=path.is_file(),
            resolution="gldv2_trigram",
            blocker="",
        )

    return ResolvedVQAImage(
        image_id=image_id,
        path=Path(""),
        exists=False,
        resolution="unsupported",
        blocker=f"unsupported_dataset_name:{dataset_name}",
    )


def _gldv2_selected_path(gldv2_root: Path, image_id: str) -> Path:
    if gldv2_root.name == "index":
        return gldv2_root.parent / "selected" / f"{image_id}.jpg"
    if gldv2_root.name == "gldv2":
        return gldv2_root.parent.parent / "google_landmarks_v2" / "selected" / f"{image_id}.jpg"
    return gldv2_root.parent / "selected" / f"{image_id}.jpg"


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _raw_match_id(match: Mapping[str, object], record_index: int) -> tuple[str, str]:
    raw_csv = _text(match.get("raw_csv"))
    raw_row_index = _text(match.get("raw_row_index"))
    if raw_csv or raw_row_index:
        return (raw_csv, raw_row_index)
    return ("__record__", str(record_index))


def _raw_split(match: Mapping[str, object]) -> str:
    raw_row = match.get("raw_row", {})
    if isinstance(raw_row, Mapping):
        split = _text(raw_row.get("encyclopedic_vqa_split"))
        if split:
            return split

    stem = Path(_text(match.get("raw_csv"))).stem
    return {
        "qa_train": "train",
        "qa_dev": "dev",
        "qa_test": "test",
    }.get(stem, stem or "unknown")


def summarize_alignment(matches: Sequence[Mapping[str, object]]) -> dict[str, object]:
    match_key_counts: Counter[str] = Counter()
    dataset_name_counts: Counter[str] = Counter()
    question_type_counts: Counter[str] = Counter()
    raw_split_counts: Counter[str] = Counter()
    resolved_image_exists = 0
    resolved_image_missing = 0
    unique_raw_matches: dict[tuple[str, str], Mapping[str, object]] = {}

    for record_index, match in enumerate(matches):
        match_key = _text(match.get("match_key"))
        if match_key:
            match_key_counts[match_key] += 1

        raw_id = _raw_match_id(match, record_index)
        unique_raw_matches.setdefault(raw_id, match)

    for match in unique_raw_matches.values():
        raw_row = match.get("raw_row", {})
        if isinstance(raw_row, Mapping):
            dataset_name = _text(raw_row.get("dataset_name"))
            question_type = _text(raw_row.get("question_type"))
            if dataset_name:
                dataset_name_counts[dataset_name] += 1
            if question_type:
                question_type_counts[question_type] += 1

        split = _raw_split(match)
        if split:
            raw_split_counts[split] += 1

        resolved_image = match.get("resolved_image", {})
        exists = (
            bool(resolved_image.get("exists"))
            if isinstance(resolved_image, Mapping)
            else False
        )
        if exists:
            resolved_image_exists += 1
        else:
            resolved_image_missing += 1

    return {
        "matched_alignment_records": len(matches),
        "matched_raw_rows": len(unique_raw_matches),
        "match_key_counts": _counter_dict(match_key_counts),
        "dataset_name_counts": _counter_dict(dataset_name_counts),
        "question_type_counts": _counter_dict(question_type_counts),
        "raw_split_counts": _counter_dict(raw_split_counts),
        "resolved_image_exists": resolved_image_exists,
        "resolved_image_missing": resolved_image_missing,
    }


def _load_id2name(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}
    return {_text(key): _text(value) for key, value in payload.items() if _text(key)}


def _resolved_image_dict(resolved: ResolvedVQAImage) -> dict[str, object]:
    return {
        "image_id": resolved.image_id,
        "path": str(resolved.path),
        "exists": resolved.exists,
        "resolution": resolved.resolution,
        "blocker": resolved.blocker,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_alignment_audit(
    root: str | Path,
    working_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    root_path = Path(root)
    working_path = Path(working_dir)
    raw_root = root_path / "datasets_mm" / "E-VQA" / "raw"
    report_dir = (
        Path(output_dir)
        if output_dir is not None
        else root_path / "datasets_mm" / "E-VQA" / "reports"
    )

    text_documents = working_path / "graphr1_text" / "text_documents.jsonl"
    anchors = load_kb_anchors(text_documents)
    raw_csv_paths = [
        raw_root / name
        for name in ("qa_train.csv", "qa_dev.csv", "qa_test.csv")
        if (raw_root / name).is_file()
    ]

    id2name: dict[str, str] = {}
    for name in ("train_id2name.json", "val_id2name.json"):
        id2name.update(_load_id2name(raw_root / "id2name" / name))

    images_root = raw_root / "images"
    inaturalist_root = (
        images_root / "inaturalist_2021"
        if (images_root / "inaturalist_2021").exists()
        else images_root
    )
    gldv2_root = (
        images_root / "full" / "gldv2"
        if (images_root / "full" / "gldv2").exists()
        else images_root / "google_landmarks_v2" / "index"
    )

    matches = match_raw_rows_to_anchors(anchors, raw_csv_paths)
    resolved_matches: list[dict[str, object]] = []
    for match in matches:
        raw_row = match.get("raw_row", {})
        if not isinstance(raw_row, Mapping):
            raw_row = {}
        resolved = resolve_raw_vqa_image_path(
            dataset_name=_text(raw_row.get("dataset_name")),
            dataset_image_ids=_text(raw_row.get("dataset_image_ids")),
            inaturalist_root=inaturalist_root,
            inaturalist_id2name=id2name,
            gldv2_root=gldv2_root,
        )
        item = dict(match)
        item["resolved_image"] = _resolved_image_dict(resolved)
        resolved_matches.append(item)

    report_dir.mkdir(parents=True, exist_ok=True)
    summary_json = report_dir / "vqa_alignment_summary.json"
    candidates_jsonl = report_dir / "vqa_alignment_candidates.jsonl"

    summary = summarize_alignment(resolved_matches)
    report: dict[str, object] = {
        "kb_anchor_count": len(anchors),
        "unique_kb_wikipedia_url_count": len(
            {_norm(anchor.wikipedia_url) for anchor in anchors if _norm(anchor.wikipedia_url)}
        ),
        "unique_kb_wikipedia_title_count": len(
            {
                _norm(anchor.wikipedia_title)
                for anchor in anchors
                if _norm(anchor.wikipedia_title)
            }
        ),
        **summary,
        "output_files": {
            "summary_json": str(summary_json),
            "candidates_jsonl": str(candidates_jsonl),
        },
        "recommended_next_step": "download_missing_images_or_build_existing-image_subset",
    }

    summary_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(candidates_jsonl, resolved_matches)
    return report


__all__ = [
    "KBAnchor",
    "ResolvedVQAImage",
    "first_dataset_image_id",
    "load_kb_anchors",
    "match_raw_rows_to_anchors",
    "resolve_raw_vqa_image_path",
    "run_alignment_audit",
    "summarize_alignment",
]
