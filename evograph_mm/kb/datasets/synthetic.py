"""Synthetic metadata-only multimodal dataset adapter."""

from __future__ import annotations

import json
from collections.abc import Iterable
from collections.abc import Mapping
from pathlib import Path

from ..mm_schema import MM_SCHEMA_VERSION, build_multimodal_qa_record


SYNTHETIC_DATASET = "synthetic-mm"
SYNTHETIC_PARQUET_COLUMNS = [
    "context",
    "data_source",
    "prompt",
    "ability",
    "reward_model",
    "extra_info",
    "data_id",
    "image_id",
    "image_path",
    "image_urls",
]
SYNTHETIC_SPLITS: dict[str, tuple[dict[str, object], ...]] = {
    "train": (
        {
            "question": "What color is the synthetic object?",
            "answers": ["blue"],
            "image_id": "synthetic-train-0000",
            "image_path": "images/train/synthetic-train-0000.jpg",
            "metadata": {"fixture": "metadata-only", "color": "blue"},
        },
        {
            "question": "What shape is represented by the synthetic object?",
            "answers": ["triangle"],
            "image_id": "synthetic-train-0001",
            "image_path": "images/train/synthetic-train-0001.jpg",
            "metadata": {"fixture": "metadata-only", "shape": "triangle"},
        },
    ),
    "dev": (
        {
            "question": "Which synthetic label is attached to the image?",
            "answers": ["label-dev"],
            "image_id": "synthetic-dev-0000",
            "image_path": "images/dev/synthetic-dev-0000.jpg",
            "metadata": {"fixture": "metadata-only", "label": "label-dev"},
        },
    ),
    "test": (
        {
            "question": "Which split does this metadata-only sample belong to?",
            "answers": ["test"],
            "image_id": "synthetic-test-0000",
            "image_path": "images/test/synthetic-test-0000.jpg",
            "metadata": {"fixture": "metadata-only", "split": "test"},
        },
    ),
}


def _processed_root(output_root: str | Path, subset: str | None = None) -> Path:
    root = Path(output_root)
    if root.name == "processed":
        processed_root = root
    else:
        processed_root = root / SYNTHETIC_DATASET / "processed"
    if subset is not None:
        processed_root = processed_root / subset
    return processed_root


def _ordered_parquet_row(record: Mapping[str, object]) -> dict[str, object]:
    row = {column: record[column] for column in SYNTHETIC_PARQUET_COLUMNS}
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict) and extra_info.get("source_row") == {}:
        row["extra_info"] = {**extra_info, "source_row": None}
    return row


def iter_synthetic_records(split: str | None = None) -> Iterable[dict[str, object]]:
    """Yield deterministic metadata-only records for smoke preprocessing."""

    selected_splits = (split,) if split is not None else tuple(SYNTHETIC_SPLITS)
    for split_name in selected_splits:
        if split_name not in SYNTHETIC_SPLITS:
            raise ValueError(f"unsupported synthetic split: {split_name}")
        for index, example in enumerate(SYNTHETIC_SPLITS[split_name]):
            metadata = dict(example["metadata"])
            metadata.update({"source_dataset": SYNTHETIC_DATASET})
            yield build_multimodal_qa_record(
                data_source=SYNTHETIC_DATASET,
                split=split_name,
                index=index,
                question=str(example["question"]),
                answers=tuple(example["answers"]),
                image_id=str(example["image_id"]),
                image_path=str(example["image_path"]),
                image_missing=True,
                original_metadata=metadata,
            )


def process_synthetic_dataset(
    output_root: str | Path,
    *,
    output_format: str = "jsonl",
    subset: str | None = None,
) -> dict[str, object]:
    """Write deterministic synthetic metadata-only splits under output_root."""

    if output_format not in {"jsonl", "parquet"}:
        raise ValueError("synthetic preprocessing supports only jsonl or parquet")

    processed_root = _processed_root(output_root, subset=subset)
    processed_root.mkdir(parents=True, exist_ok=True)

    output_files: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    for split in SYNTHETIC_SPLITS:
        rows = list(iter_synthetic_records(split))
        output_path = processed_root / f"{split}.{output_format}"
        if output_format == "jsonl":
            with output_path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True))
                    handle.write("\n")
        else:
            import pandas as pd

            pd.DataFrame([_ordered_parquet_row(row) for row in rows]).to_parquet(
                output_path,
                index=False,
            )
        output_files[split] = str(output_path)
        split_counts[split] = len(rows)

    report = {
        "dataset": SYNTHETIC_DATASET,
        "mode": "synthetic_metadata_only",
        "schema_version": MM_SCHEMA_VERSION,
        "format": output_format,
        "processed_root": str(processed_root),
        "split_counts": split_counts,
        "output_files": output_files,
        "policy": {
            "metadata_only": True,
            "real_payloads_required": False,
            "text_dataset_roots_written": False,
            "text_expr_roots_written": False,
        },
    }
    if subset is not None:
        report["subset"] = subset
    return report
