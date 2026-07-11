"""Build E-VQA subsets directly from EchoSight GLDv2 tar shards."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
import tarfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


DATASET_NAME = "E-VQA"
MANIFEST_VERSION = "echosight-tar-subset-v1"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
MAX_SELECTED_IMAGE_BYTES = 50 * 1024 * 1024
MAX_TEXT_MEMBER_BYTES = 1 * 1024 * 1024
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
]


@dataclass(frozen=True)
class _TarExample:
    image_id: str
    extension: str
    archive: Path
    archive_index: int
    image_member: str
    image_size: int
    texts_member: str
    source_member: str | None
    source: str
    texts: list[object]
    selected_turn: dict[str, object]
    diversity_key: str


def build_tar_anchored_e_vqa_subset(
    *,
    root: str | Path,
    source_tar_root: str | Path,
    sample_train: int = 5120,
    sample_test: int = 128,
    seed: int = 0,
    subset_name: str | None = None,
    limit_archives: int | None = None,
    contact_sheet: bool = False,
) -> dict[str, object]:
    """Extract a deterministic E-VQA-like subset from local GLDv2 tar shards."""

    sample_train = _require_nonnegative_int("sample_train", sample_train)
    sample_test = _require_nonnegative_int("sample_test", sample_test)
    seed = _require_int("seed", seed)
    if limit_archives is not None:
        limit_archives = _require_nonnegative_int("limit_archives", limit_archives)

    root_path = Path(root)
    _reject_repository_output_root(root_path)
    source_root = Path(source_tar_root)
    subset = subset_name or f"paper_tar_{sample_train}_{sample_test}_seed{seed}"
    subsets_root = root_path / "datasets_mm" / DATASET_NAME / "subsets"
    subset_root = _require_under(
        subsets_root.resolve(),
        subsets_root / subset,
        "subset output must stay under the dataset subsets root",
    )
    images_dir = _require_under(subset_root, subset_root / "images", "images output escaped")

    archives = sorted(source_root.glob("*.tar.gz"))
    if limit_archives is not None:
        archives = archives[:limit_archives]

    blockers: list[dict[str, object]] = []
    archive_reports: list[dict[str, object]] = []
    examples: list[_TarExample] = []
    invalid_text_records = 0

    for archive_index, archive in enumerate(archives):
        archive_examples, archive_blockers, archive_report = _scan_archive(
            archive=archive,
            archive_index=archive_index,
        )
        examples.extend(archive_examples)
        blockers.extend(archive_blockers)
        archive_reports.append(archive_report)
        invalid_text_records += int(archive_report["invalid_or_missing_texts"])

    ordered = _diverse_order(examples, seed)
    requested_total = sample_train + sample_test
    selected, selection_blockers = _select_with_size_guard(
        ordered,
        requested_total=requested_total,
    )
    blockers.extend(selection_blockers)
    selected = _assign_unique_image_ids(selected)
    train_examples = selected[:sample_train]
    test_examples = selected[sample_train : sample_train + sample_test]

    _prepare_subset_root(subset_root, images_dir)
    copied_images: list[dict[str, object]] = []
    train_rows = _rows_for_split(train_examples, split="train")
    test_rows = _rows_for_split(test_examples, split="test")

    selected_by_member = {
        (example.archive, example.image_member): (split, row_index, example)
        for split, split_examples in (
            ("train", train_examples),
            ("test", test_examples),
        )
        for row_index, example in enumerate(split_examples)
    }
    extracted = _extract_selected_images(
        selected_by_member=selected_by_member,
        images_dir=images_dir,
        subset_root=subset_root,
        root_path=root_path,
    )
    copied_images.extend(extracted)

    _write_csv(subset_root / "qa_train.csv", train_rows)
    _write_csv(subset_root / "qa_test.csv", test_rows)

    if len(selected) < requested_total:
        blockers.append(
            _blocker(
                "selection",
                "not enough valid tar examples",
                {
                    "requested_total": requested_total,
                    "valid_examples": len(examples),
                    "selected_total": len(selected),
                },
            )
        )
    if invalid_text_records:
        blockers.append(
            _blocker(
                "scan",
                "invalid or missing texts.json",
                {"count": invalid_text_records},
            )
        )

    status = "complete" if len(selected) >= requested_total else "subset_incomplete"
    summary = {
        "status": status,
        "requested_train": sample_train,
        "requested_test": sample_test,
        "complete_train": len(train_examples),
        "complete_test": len(test_examples),
        "valid_examples": len(examples),
        "selected_total": len(selected),
        "extracted_images": len(copied_images),
    }
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dataset": DATASET_NAME,
        "subset_name": subset,
        "subset_root": _relative_to_root(root_path, subset_root),
        "seed": seed,
        "sample_train": sample_train,
        "sample_test": sample_test,
        "source_tar_root": str(source_root),
        "summary": summary,
        "copied_images": copied_images,
        "blockers": blockers,
    }
    report = {
        "dataset": DATASET_NAME,
        "subset_name": subset,
        "subset_root": _relative_to_root(root_path, subset_root),
        "source_tar_root": str(source_root),
        "summary": summary,
        "blockers": blockers,
        "archives_scanned": len(archives),
        "archives": archive_reports,
        "valid_pair_counts": {
            "total": len(examples),
            "by_archive": {
                item["archive"]: item["valid_pairs"] for item in archive_reports
            },
        },
        "selected_counts": {"train": len(train_examples), "test": len(test_examples)},
        "extracted_image_count": len(copied_images),
        "contact_sheet": {
            "requested": contact_sheet,
            "created": False,
            "note": "contact_sheet is report-only in Task 1",
        },
        "policy": {
            "downloads_enabled": False,
            "remote_urls_accessed": False,
            "selected_images_only": True,
        },
    }

    _write_json(subset_root / "manifest.json", manifest)
    _write_json(subset_root / "tar_subset_report.json", report)

    return {
        "dataset": DATASET_NAME,
        "subset_name": subset,
        "subset_root": _relative_to_root(root_path, subset_root),
        "manifest_path": _relative_to_root(root_path, subset_root / "manifest.json"),
        "report_path": _relative_to_root(root_path, subset_root / "tar_subset_report.json"),
        "summary": summary,
        "blockers": blockers,
    }


def _scan_archive(
    *,
    archive: Path,
    archive_index: int,
) -> tuple[list[_TarExample], list[dict[str, object]], dict[str, object]]:
    examples: list[_TarExample] = []
    blockers: list[dict[str, object]] = []
    invalid_texts = 0
    oversized_sources = 0
    records: dict[str, dict[str, object]] = {}

    try:
        with tarfile.open(archive, "r|gz") as tar:
            for member in tar:
                if not member.isfile() or not _safe_tar_member(member.name):
                    continue
                path = PurePosixPath(member.name)
                if path.parent.as_posix() != "images":
                    continue

                name = path.name
                if name.endswith(".texts.json"):
                    stem = name[: -len(".texts.json")]
                    record = records.setdefault(stem, {})
                    record["texts_member"] = member.name
                    if member.size > MAX_TEXT_MEMBER_BYTES:
                        record["texts"] = []
                        record["selected_turn"] = None
                        continue
                    texts = _read_json_member(tar, member)
                    record["texts"] = texts if isinstance(texts, list) else []
                    record["selected_turn"] = _select_turn(texts)
                elif name.endswith(".source.txt"):
                    stem = name[: -len(".source.txt")]
                    record = records.setdefault(stem, {})
                    if member.size > MAX_TEXT_MEMBER_BYTES:
                        oversized_sources += 1
                        continue
                    record["source_member"] = member.name
                    record["source"] = _read_text_member(tar, member)
                elif path.suffix.lower() in IMAGE_EXTENSIONS:
                    stem = path.stem
                    record = records.setdefault(stem, {})
                    record["image_member"] = member.name
                    record["image_size"] = int(member.size)
                    record["extension"] = path.suffix.lower()

        for stem in sorted(records):
            record = records[stem]
            image_member = record.get("image_member")
            if not isinstance(image_member, str):
                continue
            texts_member = record.get("texts_member")
            selected_turn = record.get("selected_turn")
            if not isinstance(texts_member, str) or not isinstance(selected_turn, dict):
                invalid_texts += 1
                continue
            source_member = record.get("source_member")
            source = record.get("source") or "gldv2"
            texts = record["texts"] if isinstance(record.get("texts"), list) else []
            examples.append(
                _TarExample(
                    image_id=stem,
                    extension=str(record["extension"]),
                    archive=archive,
                    archive_index=archive_index,
                    image_member=image_member,
                    image_size=int(record["image_size"]),
                    texts_member=texts_member,
                    source_member=source_member if isinstance(source_member, str) else None,
                    source=str(source).strip() or "gldv2",
                    texts=texts,
                    selected_turn=selected_turn,
                    diversity_key=_diversity_key(texts, selected_turn),
                )
            )
    except (tarfile.TarError, OSError) as exc:
        blockers.append(
            _blocker("scan", "could not scan tar archive", {"archive": str(archive), "error": str(exc)})
        )

    report = {
        "archive": str(archive),
        "valid_pairs": len(examples),
        "invalid_or_missing_texts": invalid_texts,
        "oversized_sources": oversized_sources,
    }
    return examples, blockers, report


def _select_with_size_guard(
    ordered: Sequence[_TarExample],
    *,
    requested_total: int,
) -> tuple[list[_TarExample], list[dict[str, object]]]:
    selected: list[_TarExample] = []
    blockers: list[dict[str, object]] = []
    for example in ordered:
        if len(selected) >= requested_total:
            break
        if example.image_size > MAX_SELECTED_IMAGE_BYTES:
            blockers.append(
                _blocker(
                    "selection",
                    "selected image member exceeds size limit",
                    {
                        "source_archive": str(example.archive),
                        "source_member": example.image_member,
                        "image_id": example.image_id,
                        "member_size": example.image_size,
                        "max_selected_image_bytes": MAX_SELECTED_IMAGE_BYTES,
                    },
                )
            )
            continue
        selected.append(example)
    return selected, blockers


def _assign_unique_image_ids(examples: Sequence[_TarExample]) -> list[_TarExample]:
    used: set[str] = set()
    unique: list[_TarExample] = []
    for example in examples:
        assigned = example.image_id
        suffix = 1
        while assigned in used:
            assigned = f"{example.image_id}_{suffix}"
            suffix += 1
        used.add(assigned)
        if assigned == example.image_id:
            unique.append(example)
        else:
            unique.append(replace(example, image_id=assigned))
    return unique


def _read_json_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> object:
    extracted = tar.extractfile(member)
    if extracted is None:
        return None
    try:
        with extracted:
            return json.loads(extracted.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _read_text_member(tar: tarfile.TarFile, member: tarfile.TarInfo | None) -> str | None:
    if member is None:
        return None
    extracted = tar.extractfile(member)
    if extracted is None:
        return None
    try:
        with extracted:
            return extracted.read().decode("utf-8").strip()
    except UnicodeDecodeError:
        return None


def _select_turn(texts: object) -> dict[str, object] | None:
    if not isinstance(texts, list):
        return None
    for item in texts:
        if not isinstance(item, Mapping):
            continue
        question = str(item.get("user", "")).strip()
        answer = str(item.get("assistant", "")).strip()
        if question and answer:
            selected = dict(item)
            selected["user"] = question
            selected["assistant"] = answer
            return selected
    return None


def _diversity_key(texts: object, selected_turn: Mapping[str, object]) -> str:
    selected_label = _explicit_label(selected_turn)
    if selected_label:
        return selected_label
    if isinstance(texts, list):
        for item in texts:
            if not isinstance(item, Mapping):
                continue
            label = _explicit_label(item)
            if label:
                return label
        for item in texts:
            if not isinstance(item, Mapping):
                continue
            label = _category_label_from_answer(str(item.get("assistant", "")))
            if label:
                return label
    answer = str(selected_turn.get("assistant", "")).strip()
    entity = _entity_label_from_answer(answer)
    if entity:
        return entity
    return "unknown"


def _explicit_label(turn: Mapping[str, object]) -> str | None:
    for key in (
        "category",
        "dataset_category_id",
        "location",
        "place",
        "wikipedia_title",
        "title",
    ):
        value = str(turn.get(key, "")).strip()
        if value:
            return value
    return None


def _category_label_from_answer(answer: str) -> str:
    match = re.search(r"\bcategory\s*:\s*(.+)$", answer.strip(), flags=re.IGNORECASE)
    if not match:
        return ""
    label = match.group(1).strip()
    label = label.strip(" \t\r\n\"'`“”‘’")
    label = re.sub(r"[\s.。．!！?？]+$", "", label).strip()
    return label


def _entity_label_from_answer(answer: str) -> str:
    label = answer.strip()
    for pattern in (
        r"^(?:this|it)\s+is\s+",
        r"^this\s+(?:place|landmark|location|image|picture)\s+is\s+",
        r"^(?:the\s+)?(?:place|landmark|location)\s+(?:is|shown\s+is)\s+",
    ):
        label = re.sub(pattern, "", label, count=1, flags=re.IGNORECASE).strip()
    label = label.strip(" \t\r\n\"'`“”‘’")
    label = re.sub(r"[\s.。．!！?？]+$", "", label).strip()
    return label


def _diverse_order(examples: Sequence[_TarExample], seed: int) -> list[_TarExample]:
    rng = random.Random(seed)
    grouped: dict[str, dict[int, list[_TarExample]]] = {}
    for example in examples:
        grouped.setdefault(example.diversity_key, {}).setdefault(
            example.archive_index, []
        ).append(example)
    for shard_groups in grouped.values():
        for bucket in shard_groups.values():
            rng.shuffle(bucket)

    keys = sorted(grouped)
    rng.shuffle(keys)
    shard_positions = {key: 0 for key in keys}
    ordered: list[_TarExample] = []
    while keys:
        next_keys: list[str] = []
        for key in keys:
            shard_ids = sorted(
                shard_id for shard_id, bucket in grouped[key].items() if bucket
            )
            if not shard_ids:
                continue
            position = shard_positions[key] % len(shard_ids)
            shard_id = shard_ids[position]
            shard_positions[key] += 1
            ordered.append(grouped[key][shard_id].pop(0))
            if any(grouped[key][sid] for sid in grouped[key]):
                next_keys.append(key)
        keys = next_keys
    return ordered


def _rows_for_split(examples: Sequence[_TarExample], *, split: str) -> list[dict[str, str]]:
    rows = []
    for example in examples:
        question = str(example.selected_turn["user"])
        answer = str(example.selected_turn["assistant"])
        title = str(
            example.selected_turn.get("wikipedia_title")
            or example.selected_turn.get("title")
            or example.diversity_key
        )
        rows.append(
            {
                "wikipedia_title": title,
                "wikipedia_url": str(example.selected_turn.get("wikipedia_url", "")),
                "question_original": question,
                "question": question,
                "question_type": str(example.selected_turn.get("question_type", "tar_anchored")),
                "answer": answer,
                "evidence": str(example.selected_turn.get("evidence", "")),
                "evidence_section_id": str(example.selected_turn.get("evidence_section_id", "")),
                "evidence_section_title": str(
                    example.selected_turn.get("evidence_section_title", "")
                ),
                "dataset_name": "google-landmarks",
                "dataset_category_id": example.diversity_key,
                "wikipedia_url_used_in_train": "False",
                "encyclopedic_vqa_split": split,
                "dataset_image_ids": example.image_id,
            }
        )
    return rows


def _extract_selected_images(
    *,
    selected_by_member: Mapping[tuple[Path, str], tuple[str, int, _TarExample]],
    images_dir: Path,
    subset_root: Path,
    root_path: Path,
) -> list[dict[str, object]]:
    copied: list[dict[str, object]] = []
    by_archive: dict[Path, list[tuple[str, int, _TarExample]]] = {}
    for (archive, _member), details in selected_by_member.items():
        by_archive.setdefault(archive, []).append(details)

    for archive, items in by_archive.items():
        selected_items = {
            example.image_member: (split, row_index, example)
            for split, row_index, example in items
        }
        with tarfile.open(archive, "r|gz") as tar:
            for member in tar:
                if not member.isfile() or member.name not in selected_items:
                    continue
                split, row_index, example = selected_items[member.name]
                destination = images_dir / f"{example.image_id}{example.extension}"
                _require_under(subset_root, destination, "image output escaped")
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                with extracted:
                    with destination.open("wb") as handle:
                        shutil.copyfileobj(extracted, handle)
                copied.append(
                    {
                        "image_id": example.image_id,
                        "split": split,
                        "row_index": row_index,
                        "source_path": str(archive),
                        "source_archive": str(archive),
                        "source_member": example.image_member,
                        "source_texts_member": example.texts_member,
                        "source_source_member": example.source_member,
                        "subset_path": _relative_to_root(root_path, destination),
                        "sha256": _sha256(destination),
                        "source": example.source,
                        "texts": example.texts,
                        "selected_turn": example.selected_turn,
                        "diversity_key": example.diversity_key,
                    }
                )
    split_order = {"train": 0, "test": 1}
    copied.sort(
        key=lambda item: (
            split_order.get(str(item["split"]), 99),
            int(item["row_index"]),
        )
    )
    return copied


def _prepare_subset_root(subset_root: Path, images_dir: Path) -> None:
    subset_root.mkdir(parents=True, exist_ok=True)
    for filename in ("qa_train.csv", "qa_test.csv", "manifest.json", "tar_subset_report.json"):
        target = _require_under(subset_root, subset_root / filename, "subset file escaped")
        if target.exists():
            target.unlink()
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _blocker(stage: str, message: str, details: object | None = None) -> dict[str, object]:
    blocker: dict[str, object] = {
        "dataset": DATASET_NAME,
        "stage": stage,
        "message": message,
    }
    if details is not None:
        blocker["details"] = details
    return blocker


def _reject_repository_output_root(root_path: Path) -> None:
    root_resolved = root_path.resolve()
    repo_resolved = REPOSITORY_ROOT.resolve()
    try:
        root_resolved.relative_to(repo_resolved)
    except ValueError:
        return
    raise ValueError(
        "root must not resolve inside the repository working tree; "
        f"got {root_resolved}"
    )


def _safe_tar_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def _normalize_key(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip().lower()).strip("._")
    return normalized or "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _relative_to_root(root_path: Path, target: Path) -> str:
    try:
        return target.resolve().relative_to(root_path.resolve()).as_posix()
    except ValueError:
        return str(target)


__all__ = ["build_tar_anchored_e_vqa_subset"]
