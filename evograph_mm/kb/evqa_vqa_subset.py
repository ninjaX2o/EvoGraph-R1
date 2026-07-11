"""Build E-VQA subsets from KB-aligned raw VQA rows with local images."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import ijson

from .evqa_vqa_alignment import first_dataset_image_id
from .subset_echosight import SUBSET_MANIFEST_VERSION


DATASET_NAME = "E-VQA"
DEFAULT_SUBSET_NAME = "paper_vqa_aligned_5120_128_seed0"
DEFAULT_EVIDENCE_SUBSET_NAME = "paper_vqa_evidence_aligned_5120_128_seed0"
REPORT_NAME = "vqa_aligned_subset_report.json"
EVIDENCE_REPORT_NAME = "vqa_evidence_aligned_subset_report.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CSV_FIELDS = [
    "wikipedia_title",
    "wikipedia_url",
    "question_original",
    "question",
    "question_type",
    "answer",
    "evidence",
    "evidence_section_id",
    "evidence_section_title",
    "dataset_name",
    "dataset_category_id",
    "wikipedia_url_used_in_train",
    "encyclopedic_vqa_split",
    "dataset_image_ids",
    "evidence_source",
    "evidence_fill_status",
    "evidence_answer_alias",
    "evidence_page_url",
]


def build_vqa_aligned_subset(
    *,
    root: str | Path,
    candidates_path: str | Path | None = None,
    subset_name: str = DEFAULT_SUBSET_NAME,
    sample_train: int = 5120,
    sample_test: int = 128,
    seed: int = 0,
) -> dict[str, object]:
    """Build a deterministic VQA subset from alignment candidates with images."""

    sample_train = _require_nonnegative_int("sample_train", sample_train)
    sample_test = _require_nonnegative_int("sample_test", sample_test)
    seed = _require_int("seed", seed)

    root_path = Path(root)
    _reject_repository_root(root_path)
    dataset_root = root_path / "datasets_mm" / DATASET_NAME
    report_dir = dataset_root / "reports"
    candidates = (
        Path(candidates_path)
        if candidates_path is not None
        else report_dir / "vqa_alignment_candidates.jsonl"
    )
    if not candidates.is_file():
        raise FileNotFoundError(f"missing alignment candidates: {candidates}")

    subsets_root = dataset_root / "subsets"
    subset_root = _require_under(
        subsets_root.resolve(),
        subsets_root / subset_name,
        "subset output must stay under dataset subsets root",
    )
    images_dir = _require_under(
        subset_root,
        subset_root / "images",
        "subset images output escaped",
    )

    all_candidates = _load_candidates(candidates)
    eligible_rows, counters = _eligible_unique_rows(all_candidates)
    split_rows = {
        "train": [row for row in eligible_rows if row.split == "train"],
        "test": [row for row in eligible_rows if row.split == "test"],
    }

    rng = random.Random(seed)
    for rows in split_rows.values():
        rng.shuffle(rows)

    train_rows = split_rows["train"][:sample_train]
    test_rows = split_rows["test"][:sample_test]

    _prepare_subset_root(subset_root, images_dir)
    copied_images: list[dict[str, object]] = []
    train_csv_rows = _writeable_rows(
        train_rows,
        split="train",
        copied_images=copied_images,
        root=root_path,
        images_dir=images_dir,
    )
    test_csv_rows = _writeable_rows(
        test_rows,
        split="test",
        copied_images=copied_images,
        root=root_path,
        images_dir=images_dir,
    )

    _write_csv(subset_root / "qa_train.csv", train_csv_rows)
    _write_csv(subset_root / "qa_test.csv", test_csv_rows)

    blockers = _build_blockers(
        requested_train=sample_train,
        requested_test=sample_test,
        train_count=len(train_rows),
        test_count=len(test_rows),
        counters=counters,
    )
    status = "complete" if not blockers else "subset_incomplete"
    summary = {
        "status": status,
        "requested_train": sample_train,
        "requested_test": sample_test,
        "complete_train": len(train_rows),
        "complete_test": len(test_rows),
        "selected_train": len(train_rows),
        "selected_test": len(test_rows),
        "candidate_records": len(all_candidates),
        "eligible_raw_rows": len(eligible_rows),
        "eligible_train": len(split_rows["train"]),
        "eligible_test": len(split_rows["test"]),
        "copied_image_records": len(copied_images),
        "extracted_images": len(copied_images),
        "unique_image_files": len({item["subset_path"] for item in copied_images}),
        "skipped_duplicate_raw_rows": counters["duplicate_raw_rows"],
        "skipped_missing_images": counters["missing_images"],
        "skipped_unsupported_splits": counters["unsupported_splits"],
        "skipped_unsafe_image_ids": counters["unsafe_image_ids"],
    }
    subset_rel = _relative_to_root(root_path, subset_root)
    manifest = {
        "manifest_version": SUBSET_MANIFEST_VERSION,
        "dataset": DATASET_NAME,
        "subset_name": subset_name,
        "subset_root": subset_rel,
        "seed": seed,
        "sample_train": sample_train,
        "sample_test": sample_test,
        "source_files": {
            "alignment_candidates": str(candidates),
        },
        "summary": summary,
        "blockers": blockers,
        "copied_images": copied_images,
    }
    (subset_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = {
        "dataset": DATASET_NAME,
        "subset_name": subset_name,
        "subset_root": subset_rel,
        "manifest_path": _relative_to_root(root_path, subset_root / "manifest.json"),
        "report_path": _relative_to_root(root_path, subset_root / REPORT_NAME),
        "summary": summary,
        "blockers": blockers,
    }
    (subset_root / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_evidence_aligned_subset(
    *,
    root: str | Path,
    wiki_kb_zip: str | Path,
    candidates_path: str | Path | None = None,
    subset_name: str = DEFAULT_EVIDENCE_SUBSET_NAME,
    sample_train: int = 5120,
    sample_test: int = 128,
    seed: int = 0,
) -> dict[str, object]:
    """Build a VQA subset whose every selected row has answer-containing evidence."""

    sample_train = _require_nonnegative_int("sample_train", sample_train)
    sample_test = _require_nonnegative_int("sample_test", sample_test)
    seed = _require_int("seed", seed)

    root_path = Path(root)
    _reject_repository_root(root_path)
    dataset_root = root_path / "datasets_mm" / DATASET_NAME
    report_dir = dataset_root / "reports"
    candidates = (
        Path(candidates_path)
        if candidates_path is not None
        else report_dir / "vqa_alignment_candidates.jsonl"
    )
    if not candidates.is_file():
        raise FileNotFoundError(f"missing alignment candidates: {candidates}")
    wiki_zip = Path(wiki_kb_zip)
    if not wiki_zip.is_file():
        raise FileNotFoundError(f"missing wiki KB zip: {wiki_zip}")

    subsets_root = dataset_root / "subsets"
    subset_root = _require_under(
        subsets_root.resolve(),
        subsets_root / subset_name,
        "subset output must stay under dataset subsets root",
    )
    images_dir = _require_under(
        subset_root,
        subset_root / "images",
        "subset images output escaped",
    )

    all_candidates = _load_candidates(candidates)
    eligible_rows, counters = _eligible_unique_rows(all_candidates)
    target_urls = {
        _text(row.raw_row.get("wikipedia_url"))
        for row in eligible_rows
        if _text(row.raw_row.get("wikipedia_url"))
    }
    wiki_pages = _load_wiki_pages(wiki_zip, target_urls)

    evidence_rows: list[_EligibleRow] = []
    for row in eligible_rows:
        page_url = _text(row.raw_row.get("wikipedia_url"))
        page = wiki_pages.get(page_url)
        if page is None:
            counters["missing_wiki_pages"] += 1
            continue
        selected = _select_answer_section(
            answer=row.raw_row.get("answer", ""),
            page=page,
        )
        if selected is None:
            counters["no_answer_section"] += 1
            continue
        evidence_rows.append(_with_evidence(row, page_url=page_url, selected=selected))

    split_rows = {
        "train": [row for row in evidence_rows if row.split == "train"],
        "test": [row for row in evidence_rows if row.split == "test"],
    }
    rng = random.Random(seed)
    for rows in split_rows.values():
        rng.shuffle(rows)

    train_rows = split_rows["train"][:sample_train]
    test_rows = split_rows["test"][:sample_test]

    _prepare_subset_root(subset_root, images_dir)
    copied_images: list[dict[str, object]] = []
    train_csv_rows = _writeable_rows(
        train_rows,
        split="train",
        copied_images=copied_images,
        root=root_path,
        images_dir=images_dir,
    )
    test_csv_rows = _writeable_rows(
        test_rows,
        split="test",
        copied_images=copied_images,
        root=root_path,
        images_dir=images_dir,
    )
    _write_csv(subset_root / "qa_train.csv", train_csv_rows)
    _write_csv(subset_root / "qa_test.csv", test_csv_rows)

    blockers = _build_evidence_blockers(
        requested_train=sample_train,
        requested_test=sample_test,
        train_count=len(train_rows),
        test_count=len(test_rows),
        counters=counters,
    )
    status = "complete" if not blockers else "subset_incomplete"
    summary = {
        "status": status,
        "requested_train": sample_train,
        "requested_test": sample_test,
        "complete_train": len(train_rows),
        "complete_test": len(test_rows),
        "selected_train": len(train_rows),
        "selected_test": len(test_rows),
        "candidate_records": len(all_candidates),
        "eligible_raw_rows": len(eligible_rows),
        "wiki_target_urls": len(target_urls),
        "wiki_pages_loaded": len(wiki_pages),
        "evidence_aligned_rows": len(evidence_rows),
        "evidence_aligned_train": len(split_rows["train"]),
        "evidence_aligned_test": len(split_rows["test"]),
        "copied_image_records": len(copied_images),
        "extracted_images": len(copied_images),
        "unique_image_files": len({item["subset_path"] for item in copied_images}),
        "skipped_duplicate_raw_rows": counters["duplicate_raw_rows"],
        "skipped_missing_images": counters["missing_images"],
        "skipped_unsupported_splits": counters["unsupported_splits"],
        "skipped_unsafe_image_ids": counters["unsafe_image_ids"],
        "skipped_missing_wiki_pages": counters["missing_wiki_pages"],
        "skipped_no_answer_section": counters["no_answer_section"],
    }
    subset_rel = _relative_to_root(root_path, subset_root)
    manifest = {
        "manifest_version": SUBSET_MANIFEST_VERSION,
        "dataset": DATASET_NAME,
        "subset_name": subset_name,
        "subset_root": subset_rel,
        "seed": seed,
        "sample_train": sample_train,
        "sample_test": sample_test,
        "source_files": {
            "alignment_candidates": str(candidates),
            "wiki_kb_zip": str(wiki_zip),
        },
        "summary": summary,
        "blockers": blockers,
        "copied_images": copied_images,
    }
    (subset_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "dataset": DATASET_NAME,
        "subset_name": subset_name,
        "subset_root": subset_rel,
        "manifest_path": _relative_to_root(root_path, subset_root / "manifest.json"),
        "report_path": _relative_to_root(root_path, subset_root / EVIDENCE_REPORT_NAME),
        "summary": summary,
        "blockers": blockers,
    }
    (subset_root / EVIDENCE_REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


class _EligibleRow:
    def __init__(
        self,
        *,
        raw_id: tuple[str, str],
        raw_row: dict[str, str],
        image_id: str,
        image_path: Path,
        split: str,
        match: Mapping[str, object],
    ) -> None:
        self.raw_id = raw_id
        self.raw_row = raw_row
        self.image_id = image_id
        self.image_path = image_path
        self.split = split
        self.match = match


class _SelectedEvidence:
    def __init__(
        self,
        *,
        section_id: int,
        section_title: str,
        evidence: str,
        answer_alias: str,
    ) -> None:
        self.section_id = section_id
        self.section_title = section_title
        self.evidence = evidence
        self.answer_alias = answer_alias


def _require_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _require_nonnegative_int(name: str, value: object) -> int:
    value = _require_int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _reject_repository_root(root: Path) -> None:
    resolved = root.resolve(strict=False)
    repo = REPOSITORY_ROOT.resolve(strict=False)
    try:
        resolved.relative_to(repo)
    except ValueError:
        return
    raise ValueError("refusing to write E-VQA subset inside the repository root")


def _require_under(root_resolved: Path, target: Path, message: str) -> Path:
    target_resolved = target.resolve(strict=False)
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{message}: {target}") from exc
    return target_resolved


def _load_candidates(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _eligible_unique_rows(
    candidates: Sequence[Mapping[str, object]],
) -> tuple[list[_EligibleRow], Counter[str]]:
    rows: list[_EligibleRow] = []
    seen_raw_ids: set[tuple[str, str]] = set()
    counters: Counter[str] = Counter()

    for index, candidate in enumerate(candidates):
        raw_id = _raw_id(candidate, index)
        if raw_id in seen_raw_ids:
            counters["duplicate_raw_rows"] += 1
            continue
        seen_raw_ids.add(raw_id)

        raw_row = _mapping_dict(candidate.get("raw_row"))
        split = _normalized_split(raw_row.get("encyclopedic_vqa_split"), candidate.get("raw_csv"))
        if split not in {"train", "test"}:
            counters["unsupported_splits"] += 1
            continue

        image_id = _candidate_image_id(candidate, raw_row)
        image_path = _candidate_image_path(candidate)
        if not image_id or image_path is None or not image_path.is_file():
            counters["missing_images"] += 1
            continue
        if not _is_safe_image_id(image_id):
            counters["unsafe_image_ids"] += 1
            continue

        rows.append(
            _EligibleRow(
                raw_id=raw_id,
                raw_row={key: str(value) for key, value in raw_row.items()},
                image_id=image_id,
                image_path=image_path,
                split=split,
                match=candidate,
            )
        )
    return rows, counters


def _raw_id(candidate: Mapping[str, object], fallback_index: int) -> tuple[str, str]:
    raw_csv = _text(candidate.get("raw_csv"))
    raw_row_index = _text(candidate.get("raw_row_index"))
    if raw_csv and raw_row_index:
        return raw_csv, raw_row_index
    if raw_csv:
        return raw_csv, f"__candidate__:{fallback_index}"
    if raw_row_index:
        return "__unknown_csv__", raw_row_index
    return "__candidate__", str(fallback_index)


def _mapping_dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _normalized_split(split_value: object, raw_csv: object) -> str:
    split = _text(split_value).casefold()
    if split:
        return {"dev": "val"}.get(split, split)
    stem = Path(_text(raw_csv)).stem.casefold()
    return {
        "qa_train": "train",
        "qa_test": "test",
        "qa_dev": "val",
    }.get(stem, stem)


def _candidate_image_id(candidate: Mapping[str, object], raw_row: Mapping[str, object]) -> str:
    resolved_image = candidate.get("resolved_image")
    if isinstance(resolved_image, Mapping):
        image_id = _text(resolved_image.get("image_id"))
        if image_id:
            return image_id
    return first_dataset_image_id(raw_row.get("dataset_image_ids"))


def _candidate_image_path(candidate: Mapping[str, object]) -> Path | None:
    resolved_image = candidate.get("resolved_image")
    if not isinstance(resolved_image, Mapping):
        return None
    if resolved_image.get("exists") is False:
        return None
    path = _text(resolved_image.get("path"))
    return Path(path) if path else None


def _load_wiki_pages(
    wiki_kb_zip: Path,
    target_urls: set[str],
) -> dict[str, dict[str, object]]:
    pages: dict[str, dict[str, object]] = {}
    if not target_urls:
        return pages
    with zipfile.ZipFile(wiki_kb_zip) as archive:
        names = archive.namelist()
        if not names:
            return pages
        with archive.open(names[0]) as handle:
            for url, page in ijson.kvitems(handle, ""):
                if url not in target_urls:
                    continue
                if isinstance(page, Mapping):
                    pages[url] = dict(page)
                if len(pages) == len(target_urls):
                    break
    return pages


def _select_answer_section(
    *,
    answer: object,
    page: Mapping[str, object],
) -> _SelectedEvidence | None:
    section_texts = page.get("section_texts", [])
    section_titles = page.get("section_titles", [])
    if not isinstance(section_texts, Sequence) or isinstance(section_texts, str):
        return None
    if not isinstance(section_titles, Sequence) or isinstance(section_titles, str):
        section_titles = []

    aliases = _answer_aliases(answer)
    for section_id, section_text in enumerate(section_texts):
        evidence = _text(section_text)
        if not evidence:
            continue
        for alias in aliases:
            if _alias_in_text(alias, evidence):
                section_title = ""
                if section_id < len(section_titles):
                    section_title = _text(section_titles[section_id])
                return _SelectedEvidence(
                    section_id=section_id,
                    section_title=section_title,
                    evidence=evidence,
                    answer_alias=alias,
                )
    return None


def _with_evidence(
    row: _EligibleRow,
    *,
    page_url: str,
    selected: _SelectedEvidence,
) -> _EligibleRow:
    raw_row = dict(row.raw_row)
    raw_row["evidence"] = selected.evidence
    raw_row["evidence_section_id"] = str(selected.section_id)
    raw_row["evidence_section_title"] = selected.section_title
    raw_row["evidence_source"] = "wiki_kb_section"
    raw_row["evidence_fill_status"] = "answer_section"
    raw_row["evidence_answer_alias"] = selected.answer_alias
    raw_row["evidence_page_url"] = page_url
    return _EligibleRow(
        raw_id=row.raw_id,
        raw_row=raw_row,
        image_id=row.image_id,
        image_path=row.image_path,
        split=row.split,
        match=row.match,
    )


def _answer_aliases(answer: object) -> list[str]:
    raw = _text(answer)
    if not raw:
        return []
    aliases: list[str] = []
    seen: set[str] = set()
    for group in raw.split("|"):
        for part in group.split("&&"):
            alias = " ".join(part.strip().split())
            key = alias.casefold()
            if alias and key not in seen:
                aliases.append(alias)
                seen.add(key)
    normalized_raw = " ".join(raw.split())
    key = normalized_raw.casefold()
    if normalized_raw and key not in seen:
        aliases.append(normalized_raw)
    return aliases


def _compact_for_match(value: object) -> str:
    return re.sub(r"[^\w]+", " ", _text(value).casefold()).strip()


def _alias_in_text(alias: str, text: str) -> bool:
    compact_alias = _compact_for_match(alias)
    if not compact_alias:
        return False
    compact_text = _compact_for_match(text)
    return re.search(
        rf"(?<!\w){re.escape(compact_alias)}(?!\w)",
        compact_text,
    ) is not None


_SAFE_IMAGE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _is_safe_image_id(image_id: str) -> bool:
    if not image_id or image_id in {".", ".."}:
        return False
    if not _SAFE_IMAGE_ID.fullmatch(image_id):
        return False
    candidate_name = f"{image_id}.jpg"
    return (
        PurePosixPath(candidate_name).name == candidate_name
        and PureWindowsPath(candidate_name).name == candidate_name
    )


def _prepare_subset_root(subset_root: Path, images_dir: Path) -> None:
    subset_root.mkdir(parents=True, exist_ok=True)
    for filename in (
        "qa_train.csv",
        "qa_test.csv",
        "manifest.json",
        REPORT_NAME,
        EVIDENCE_REPORT_NAME,
    ):
        target = _require_under(subset_root, subset_root / filename, "subset file escaped")
        if target.exists():
            target.unlink()
    images_resolved = _require_under(subset_root, images_dir, "subset images escaped")
    if images_resolved.exists():
        shutil.rmtree(images_resolved)
    images_resolved.mkdir(parents=True, exist_ok=True)


def _writeable_rows(
    rows: Sequence[_EligibleRow],
    *,
    split: str,
    copied_images: list[dict[str, object]],
    root: Path,
    images_dir: Path,
) -> list[dict[str, str]]:
    csv_rows: list[dict[str, str]] = []
    copied_paths: dict[str, Path] = {}
    for row_index, row in enumerate(rows):
        target = copied_paths.get(row.image_id)
        if target is None:
            target = _safe_image_target(
                images_dir,
                row.image_id,
                row.image_path.suffix.lower() or ".jpg",
            )
            shutil.copy2(row.image_path, target)
            copied_paths[row.image_id] = target
        csv_row = dict(row.raw_row)
        csv_row["encyclopedic_vqa_split"] = split
        csv_row["dataset_image_ids"] = row.image_id
        csv_rows.append(csv_row)
        copied_images.append(
            {
                "split": split,
                "row_index": row_index,
                "raw_csv": row.raw_id[0],
                "raw_row_index": row.raw_id[1],
                "image_id": row.image_id,
                "source_path": str(row.image_path),
                "subset_path": _relative_to_root(root, target),
                "sha256": _sha256(target),
                "match_key": _text(row.match.get("match_key")),
            }
        )
    return csv_rows


def _safe_image_target(images_dir: Path, image_id: str, suffix: str) -> Path:
    target = images_dir / f"{image_id}{suffix}"
    return _require_under(
        images_dir.resolve(strict=False),
        target,
        "subset image target escaped images directory",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_blockers(
    *,
    requested_train: int,
    requested_test: int,
    train_count: int,
    test_count: int,
    counters: Counter[str],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if train_count < requested_train:
        blockers.append(
            {
                "split": "train",
                "reason": "not_enough_existing_image_rows",
                "requested": requested_train,
                "available": train_count,
            }
        )
    if test_count < requested_test:
        blockers.append(
            {
                "split": "test",
                "reason": "not_enough_existing_image_rows",
                "requested": requested_test,
                "available": test_count,
            }
        )
    if blockers:
        blockers.append(
            {
                "reason": "candidate_filter_summary",
                "skipped_duplicate_raw_rows": counters["duplicate_raw_rows"],
                "skipped_missing_images": counters["missing_images"],
                "skipped_unsupported_splits": counters["unsupported_splits"],
                "skipped_unsafe_image_ids": counters["unsafe_image_ids"],
            }
        )
    return blockers


def _build_evidence_blockers(
    *,
    requested_train: int,
    requested_test: int,
    train_count: int,
    test_count: int,
    counters: Counter[str],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if train_count < requested_train:
        blockers.append(
            {
                "split": "train",
                "reason": "not_enough_evidence_aligned_rows",
                "requested": requested_train,
                "available": train_count,
            }
        )
    if test_count < requested_test:
        blockers.append(
            {
                "split": "test",
                "reason": "not_enough_evidence_aligned_rows",
                "requested": requested_test,
                "available": test_count,
            }
        )
    if blockers:
        blockers.append(
            {
                "reason": "candidate_filter_summary",
                "skipped_duplicate_raw_rows": counters["duplicate_raw_rows"],
                "skipped_missing_images": counters["missing_images"],
                "skipped_unsupported_splits": counters["unsupported_splits"],
                "skipped_unsafe_image_ids": counters["unsafe_image_ids"],
                "skipped_missing_wiki_pages": counters["missing_wiki_pages"],
                "skipped_no_answer_section": counters["no_answer_section"],
            }
        )
    return blockers


def _relative_to_root(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        try:
            return (
                path.resolve(strict=False)
                .relative_to(root.resolve(strict=False))
                .as_posix()
            )
        except ValueError:
            return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DEFAULT_EVIDENCE_SUBSET_NAME",
    "DEFAULT_SUBSET_NAME",
    "build_evidence_aligned_subset",
    "build_vqa_aligned_subset",
]
