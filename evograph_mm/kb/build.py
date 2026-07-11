"""Build orchestrator for multimodal KB artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .indexing import (
    DEFAULT_EMBEDDING_DIMENSION,
    FUSED_DOCUMENT_INSTRUCTION,
    GME_MODEL_REPO_ID,
    GMEQwen2VLEncoder,
    IMAGE_DOCUMENT_INSTRUCTION,
    QWEN3_VL_MODEL_REPO_ID,
    TEXT_DOCUMENT_INSTRUCTION,
    TEXT_QUERY_INSTRUCTION,
    LocalModelUnavailable,
    MockEmbeddingEncoder,
    Qwen3VLEncoder,
    write_vector_index,
)
from .layout import build_layout, find_protected_path_violations
from .mm_graph import build_mm_graph_records, write_mm_graph
from .store import (
    build_records_from_rows,
    visual_embedding_id,
    write_store,
)

SUPPORTED_EMBEDDING_BACKENDS = ("gme", "gme-qwen2-vl", "qwen3-vl")
SPLIT_ORDER = ("train", "val", "test")
FULL_BUILD_REQUIRED_SPLITS = ("train", "test")
SUBSET_BUILD_REQUIRED_SPLITS = ("train", "test")
OPTIONAL_SPLITS = ("val",)
GRAPHR1_WIKI_BATCH_TARGET_CHARS = 12000
GRAPHR1_INSERT_BATCH_SIZE = 50
GRAPHR1_INSERT_MAX_RETRIES = 50
GRAPHR1_INSERT_RETRY_DELAY_SECONDS = 10
GRAPHR1_TEXT_LLM_MODEL = "openai/gpt-4o-mini"


def _positive_limit_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--limit must be > 0")
    return parsed


def _positive_embedding_batch_size_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--embedding-batch-size must be > 0")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build multimodal KB artifacts.")
    parser.add_argument("--root", default=".", help="Repository or workspace root.")
    parser.add_argument("--dataset", required=True, help="Dataset name, for example E-VQA.")
    parser.add_argument(
        "--subset",
        default=None,
        help="Processed dataset subset name. Omit for a full dataset build.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root. Defaults to <root>/expr_mm.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Local embedding model path, recorded for mock builds too.",
    )
    parser.add_argument(
        "--mock-encoder",
        action="store_true",
        help="Use deterministic mock embeddings instead of loading the local model.",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Write deterministic mock GraphR1 text artifacts instead of calling the LLM.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIMENSION,
        help="Embedding dimension for mock or local encoder.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=SUPPORTED_EMBEDDING_BACKENDS,
        default=os.getenv("MM_EMBEDDING_BACKEND", "gme"),
        help="Embedding backend to use for indexing.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=_positive_embedding_batch_size_arg,
        default=1,
        help="Encoder batch size for local Qwen3-VL embedding.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_limit_arg,
        default=None,
        help="Optional global row cap applied after train/val/test ordering.",
    )
    return parser


def run_build(
    root: str | Path,
    dataset: str,
    subset: str | None,
    output_root: str | Path | None,
    model: str | Path,
    mock_encoder: bool = False,
    mock_llm: bool = False,
    embedding_dim: int = DEFAULT_EMBEDDING_DIMENSION,
    embedding_backend: str = "gme",
    embedding_batch_size: int = 1,
    limit: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    layout = build_layout(
        root=root,
        dataset=dataset,
        subset=subset,
        output_root=output_root,
    )
    model_path = Path(model)
    model_repo_id = _model_repo_id_for_backend(embedding_backend)
    base_report = _base_report(
        dataset=dataset,
        subset=subset,
        model_path=model_path,
        model_repo_id=model_repo_id,
        embedding_dim=embedding_dim,
    )
    protected_path_violations = list(
        find_protected_path_violations([layout.output_dir], repo_root=layout.root)
    )
    output_root_violation = _output_root_violation(layout)
    if not protected_path_violations and output_root_violation is not None:
        protected_path_violations.append(output_root_violation)
    if protected_path_violations:
        return _blocked_report(
            base_report=base_report,
            started=started,
            blocker=(
                "protected output path violation: "
                + ", ".join(protected_path_violations)
            ),
            protected_path_violations=protected_path_violations,
        )

    try:
        if embedding_backend not in SUPPORTED_EMBEDDING_BACKENDS:
            raise ValueError(f"unsupported embedding backend: {embedding_backend}")
        if embedding_batch_size <= 0:
            raise ValueError("--embedding-batch-size must be > 0")
        if limit is not None and limit <= 0:
            raise ValueError("--limit must be > 0")
        _prepare_dataset_output_dir(layout.output_dir, source_subset=layout.source_subset)
        _validate_required_processed_splits(
            layout.processed_paths,
            required_splits=_required_split_names(subset),
        )
        split_rows, source_parquet_paths = _read_processed_rows(
            layout.processed_paths,
            limit=limit,
        )
        split_rows = _normalize_build_rows(split_rows)
        rows = _flatten_split_rows(split_rows)
        split_counts = {split: len(values) for split, values in split_rows.items()}
        row_quality = _summarize_row_quality(split_rows)
        encoder = (
            MockEmbeddingEncoder(dimension=embedding_dim)
            if mock_encoder
            else _create_real_encoder(
                embedding_backend,
                model_path=model_path,
                embedding_dim=embedding_dim,
                batch_size=embedding_batch_size,
            )
        )
        encoder_mode = "mock" if mock_encoder else embedding_backend
        bundle = build_records_from_rows(rows)
        store_counts = write_store(layout, bundle)
        mm_graph = build_mm_graph_records(
            text_documents=bundle.text_documents,
            image_records=bundle.visual_records,
        )
        write_mm_graph(layout.output_dir / "mm_store" / "graph", mm_graph)
        mock_text_graph = mock_encoder or mock_llm
        text_graph_documents = _graphr1_insert_documents(
            bundle.text_documents,
            require_evidence=not mock_text_graph,
        )
        text_graph_fingerprint = _text_documents_fingerprint(text_graph_documents)
        text_graph_reused = _can_reuse_graphr1_text_artifacts(
            layout.output_dir,
            expected_text_document_count=len(text_graph_documents),
            expected_fingerprint=text_graph_fingerprint,
        )
        if not text_graph_reused:
            _remove_graphr1_text_artifacts(layout.output_dir)
            build_text_graphr1_graph(
                layout.output_dir,
                bundle.text_documents,
                mock_llm=mock_text_graph,
            )
        text_graph_index_paths = (
            _reuse_text_graphr1_indexes(
                output_dir=layout.output_dir,
                model_path=model_path,
                model_repo_id=model_repo_id,
                encoder_mode=encoder_mode,
                embedding_dim=embedding_dim,
            )
            if text_graph_reused
            else None
        )
        text_graph_indexes_reused = text_graph_index_paths is not None
        if text_graph_index_paths is None:
            text_graph_index_paths = build_text_graphr1_indexes(
                output_dir=layout.output_dir,
                encoder=encoder,
                model_path=model_path,
                model_repo_id=model_repo_id,
                encoder_mode=encoder_mode,
            )

        image_ids = [visual_embedding_id(record["image_id"]) for record in bundle.visual_records]
        image_index_summary = _reuse_mm_vector_index(
            output_dir=layout.indexing_store_root,
            name="image",
            expected_ids=image_ids,
            model_path=model_path,
            model_repo_id=model_repo_id,
            encoder_mode=encoder_mode,
            embedding_dim=embedding_dim,
            instruction=IMAGE_DOCUMENT_INSTRUCTION,
        )
        image_index_reused = image_index_summary is not None
        if image_index_summary is None:
            image_embeddings = encoder.encode_images(
                (record["image_path"] for record in bundle.visual_records),
                instruction=IMAGE_DOCUMENT_INSTRUCTION,
            )
            image_index_summary = write_vector_index(
                output_dir=layout.indexing_store_root,
                name="image",
                ids=image_ids,
                embeddings=image_embeddings,
                model_path=model_path,
                model_repo_id=model_repo_id,
                instruction=IMAGE_DOCUMENT_INSTRUCTION,
                encoder_mode=encoder_mode,
            )
        index_paths = {"image": image_index_summary}
        embedding_counts = {
            "entity": text_graph_index_paths["entity"]["vector_count"],
            "hyperedge": text_graph_index_paths["hyperedge"]["vector_count"],
            "image": len(bundle.visual_records),
        }
        metadata = {
            **base_report,
            "root": str(layout.root.resolve(strict=False)),
            "output_root": str(layout.output_root),
            "output_dir": str(layout.output_dir),
            "source_subset": layout.source_subset,
            "text_graph_fingerprint": text_graph_fingerprint,
            "split_counts": split_counts,
            "text_document_count": store_counts["text_document_count"],
            "visual_record_count": store_counts["visual_record_count"],
            "link_count": store_counts["link_count"],
            "embedding_counts": embedding_counts,
            "index_paths": index_paths,
            "text_graph_index_paths": text_graph_index_paths,
            "missing_images": row_quality["missing_images"],
            "missing_fields": row_quality["missing_fields"],
            "instructions": _instruction_metadata(),
            "source_parquet_paths": source_parquet_paths,
            "protected_path_violations": [],
            "encoder_mode": encoder_mode,
            "embedding_batch_size": embedding_batch_size,
            "text_graph_llm": GRAPHR1_TEXT_LLM_MODEL,
            "mock_llm": bool(mock_encoder or mock_llm),
            "text_graph_reused": text_graph_reused,
            "text_graph_indexes_reused": text_graph_indexes_reused,
            "image_index_reused": image_index_reused,
        }
        written_files = _success_written_files(layout, index_paths, text_graph_index_paths)

        report = {
            **base_report,
            "status": "success",
            "warnings": [],
            "written_files": written_files,
            "protected_path_violations": [],
            "output_dir": str(layout.output_dir),
            "split_counts": split_counts,
            "store_counts": store_counts,
            "embedding_counts": embedding_counts,
            "index_paths": index_paths,
            "text_graph_index_paths": text_graph_index_paths,
            "missing_images": row_quality["missing_images"],
            "missing_fields": row_quality["missing_fields"],
            "embedding_batch_size": embedding_batch_size,
            "text_graph_llm": GRAPHR1_TEXT_LLM_MODEL,
            "mock_llm": bool(mock_encoder or mock_llm),
            "text_graph_reused": text_graph_reused,
            "text_graph_indexes_reused": text_graph_indexes_reused,
            "image_index_reused": image_index_reused,
            "blockers": [],
        }
    except (LocalModelUnavailable, FileNotFoundError, ValueError) as exc:
        metadata = None
        report = {
            **base_report,
            "status": "blocked",
            "warnings": [],
            "written_files": [],
            "protected_path_violations": [],
            "split_counts": {},
            "store_counts": {},
            "embedding_counts": {},
            "index_paths": {},
            "missing_images": {},
            "missing_fields": {},
            "embedding_batch_size": embedding_batch_size,
            "blockers": [str(exc)],
        }

    report["elapsed_seconds"] = time.perf_counter() - started
    _write_reports(layout.output_dir, report, metadata)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_build(
        root=args.root,
        dataset=args.dataset,
        subset=args.subset,
        output_root=args.output_root,
        model=args.model,
        mock_encoder=args.mock_encoder,
        mock_llm=args.mock_llm,
        embedding_dim=args.embedding_dim,
        embedding_backend=args.embedding_backend,
        embedding_batch_size=args.embedding_batch_size,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] != "blocked" else 1


def _read_processed_rows(
    processed_paths: dict[str, Path],
    limit: int | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    source_parquet_paths: dict[str, str] = {}
    remaining = limit
    for split in _ordered_splits(processed_paths):
        if remaining is not None and remaining <= 0:
            break
        path = processed_paths[split]
        if not path.is_file():
            continue
        source_parquet_paths[split] = str(path)
        if remaining is None:
            frame = pd.read_parquet(path)
            rows_by_split[split] = frame.to_dict(orient="records")
        else:
            rows_by_split[split] = _read_limited_parquet_rows(path, remaining)
        if remaining is not None:
            remaining -= len(rows_by_split[split])
    if source_parquet_paths:
        return rows_by_split, source_parquet_paths
    searched_paths = ", ".join(
        str(processed_paths[split]) for split in _ordered_splits(processed_paths)
    )
    raise FileNotFoundError(f"no processed split parquet found under: {searched_paths}")


def _required_split_names(subset: str | None) -> tuple[str, ...]:
    if subset is None:
        return FULL_BUILD_REQUIRED_SPLITS
    return SUBSET_BUILD_REQUIRED_SPLITS


def _validate_required_processed_splits(
    processed_paths: dict[str, Path],
    *,
    required_splits: tuple[str, ...],
) -> None:
    missing_required = [
        split
        for split in required_splits
        if split not in processed_paths or not processed_paths[split].is_file()
    ]
    if not missing_required:
        return
    processed_root = _processed_root_from_paths(processed_paths, required_splits)
    raise FileNotFoundError(
        "missing required processed split parquet(s): "
        + ", ".join(missing_required)
        + f" under {processed_root}"
    )


def _processed_root_from_paths(
    processed_paths: dict[str, Path],
    required_splits: tuple[str, ...],
) -> Path:
    for split in (*required_splits, *_ordered_splits(processed_paths)):
        path = processed_paths.get(split)
        if path is not None:
            return path.parent
    raise ValueError("processed_paths must include at least one split path")


def _ordered_splits(paths_by_split: dict[str, Any]) -> list[str]:
    return [split for split in SPLIT_ORDER if split in paths_by_split]


def _read_limited_parquet_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    parquet_file = pq.ParquetFile(path)
    try:
        for batch in parquet_file.iter_batches(batch_size=limit):
            batch_rows = batch.to_pylist()
            remaining = limit - len(rows)
            rows.extend(batch_rows[:remaining])
            if len(rows) >= limit:
                break
    finally:
        parquet_file.close()
    return rows


def _flatten_split_rows(split_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in _ordered_splits(split_rows):
        rows.extend(split_rows[split])
    return rows


def _normalize_build_rows(
    split_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    normalized: dict[str, list[dict[str, Any]]] = {}
    for split, rows in split_rows.items():
        normalized[split] = [_normalize_build_row(row) for row in rows]
    return normalized


def _normalize_build_row(row: dict[str, Any]) -> dict[str, Any]:
    extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    normalized = dict(row)

    image_id = _string_or_none(_first_present(row.get("image_id"), extra_info.get("image_id")))
    if image_id is not None and _is_blank(normalized.get("image_id")):
        normalized["image_id"] = image_id

    image_path = _string_or_none(
        _first_present(row.get("image_path"), extra_info.get("image_path"))
    )
    if image_path is not None:
        try:
            from verl.utils.multimodal import resolve_image_path

            image_path = str(resolve_image_path(image_path))
        except Exception:
            pass
    if image_path is not None and _is_blank(normalized.get("image_path")):
        normalized["image_path"] = image_path
    elif image_path is not None:
        normalized["image_path"] = image_path

    if _is_blank(normalized.get("data_id")):
        data_id = _string_or_none(_first_present(row.get("data_id"), extra_info.get("data_id")))
        if data_id is None:
            data_id = _fallback_full_mode_data_id(normalized, extra_info)
        if data_id is not None:
            normalized["data_id"] = data_id

    return normalized


def _fallback_full_mode_data_id(
    row: dict[str, Any],
    extra_info: dict[str, Any],
) -> str | None:
    data_source = _string_or_none(_first_present(row.get("data_source"), extra_info.get("dataset")))
    split = _string_or_none(_first_present(extra_info.get("split"), row.get("split")))
    index = _string_or_none(_first_present(extra_info.get("index"), row.get("index")))
    image_id = _string_or_none(_first_present(row.get("image_id"), extra_info.get("image_id")))
    if data_source is None or split is None or index is None or image_id is None:
        return None
    return f"{data_source}:full:{split}:{index}:{image_id}"


def _base_report(
    dataset: str,
    subset: str | None,
    model_path: Path,
    model_repo_id: str,
    embedding_dim: int,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "model_path": str(model_path),
        "model_repo_id": model_repo_id,
        "embedding_dimension": embedding_dim,
    }


def _model_repo_id_for_backend(backend: str) -> str:
    normalized = (backend or "gme").lower().replace("_", "-")
    if normalized in {"gme", "gme-qwen2-vl"}:
        return GME_MODEL_REPO_ID
    if normalized == "qwen3-vl":
        return QWEN3_VL_MODEL_REPO_ID
    return GME_MODEL_REPO_ID


def _create_real_encoder(
    backend: str,
    *,
    model_path: Path,
    embedding_dim: int,
    batch_size: int,
):
    normalized = (backend or "gme").lower().replace("_", "-")
    if normalized in {"gme", "gme-qwen2-vl"}:
        return GMEQwen2VLEncoder(
            model_path,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
        )
    if normalized == "qwen3-vl":
        return Qwen3VLEncoder(
            model_path,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
        )
    raise ValueError(f"Unsupported embedding backend for Task D/E/F: {backend}")


def build_text_graphr1_graph(
    output_dir: Path,
    text_documents: list[dict[str, Any]],
    *,
    mock_llm: bool = False,
) -> None:
    insert_documents = _graphr1_insert_documents(
        text_documents,
        require_evidence=not mock_llm,
    )
    contents = [str(doc["contents"]) for doc in insert_documents]
    if mock_llm:
        _write_mock_graphr1_text_artifacts(output_dir, insert_documents)
        return

    _write_graphr1_seed_guard(output_dir)
    from graphr1.graphr1 import GraphR1

    rag = GraphR1(
        working_dir=str(output_dir),
        llm_model_name=GRAPHR1_TEXT_LLM_MODEL,
    )
    _insert_graphr1_batches(rag, contents)


def _insert_graphr1_batches(
    rag: Any,
    contents: list[str],
    *,
    batch_size: int = GRAPHR1_INSERT_BATCH_SIZE,
    max_retries: int = GRAPHR1_INSERT_MAX_RETRIES,
    retry_delay_seconds: float = GRAPHR1_INSERT_RETRY_DELAY_SECONDS,
) -> None:
    if batch_size <= 0:
        raise ValueError("GraphR1 insert batch_size must be > 0")
    for start in range(0, len(contents), batch_size):
        batch = contents[start : start + batch_size]
        retries = 0
        while True:
            try:
                rag.insert(batch)
                break
            except Exception:
                retries += 1
                if retries >= max_retries:
                    raise
                if retry_delay_seconds > 0:
                    time.sleep(retry_delay_seconds)


def _graphr1_insert_documents(
    text_documents: list[dict[str, Any]],
    *,
    require_evidence: bool,
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen_contents: set[str] = set()
    for doc in text_documents:
        contents = str(doc.get("contents", "")).strip()
        if require_evidence and not contents:
            continue
        if require_evidence and not _has_wiki_evidence(contents):
            continue
        if contents in seen_contents:
            continue
        seen_contents.add(contents)
        insert_doc = dict(doc)
        insert_doc["contents"] = contents
        documents.append(insert_doc)
    if require_evidence:
        return _graphr1_wiki_section_documents(documents)
    return documents


def _graphr1_wiki_section_documents(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    grouped_parts: dict[tuple[str, str], list[str]] = {}
    for doc in documents:
        contents = str(doc.get("contents") or "").strip()
        if not contents:
            continue
        key = _graphr1_wiki_section_key(contents)
        if key not in grouped:
            digest = hashlib.sha1("\n".join(key).encode("utf-8", errors="replace")).hexdigest()[:16]
            grouped[key] = {
                **doc,
                "text_doc_id": f"wiki_section::{digest}",
                "data_id": f"wiki_section::{digest}",
            }
            grouped_parts[key] = []
        if contents not in grouped_parts[key]:
            grouped_parts[key].append(contents)

    section_documents: list[dict[str, Any]] = []
    for key, doc in grouped.items():
        section_doc = dict(doc)
        section_doc["contents"] = "\n\n".join(grouped_parts[key])
        section_documents.append(section_doc)
    return section_documents


def _graphr1_wiki_section_key(contents: str) -> tuple[str, str]:
    title = ""
    section = ""
    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if line.startswith("Wikipedia title:"):
            title = line.removeprefix("Wikipedia title:").strip()
        elif line.startswith("Evidence section:"):
            section = line.removeprefix("Evidence section:").strip()
        elif not section and line.startswith("Section:"):
            section = line.removeprefix("Section:").strip()
    if not title:
        title = contents.splitlines()[0].strip() if contents.splitlines() else ""
    return (title, section)


def _pack_graphr1_wiki_documents(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    packed: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_chars = 0
    batch_index = 0
    for doc in documents:
        contents = str(doc.get("contents") or "").strip()
        if not contents:
            continue
        separator = "\n\n---\n\n" if current_parts else ""
        additional = len(separator) + len(contents)
        if current_parts and current_chars + additional > GRAPHR1_WIKI_BATCH_TARGET_CHARS:
            batch_id = f"wiki_batch::{batch_index:05d}"
            packed.append(
                {
                    "text_doc_id": batch_id,
                    "data_id": batch_id,
                    "contents": "\n\n---\n\n".join(current_parts),
                }
            )
            batch_index += 1
            current_parts = []
            current_chars = 0
            separator = ""
            additional = len(contents)
        current_parts.append(contents)
        current_chars += additional
    if current_parts:
        batch_id = f"wiki_batch::{batch_index:05d}"
        packed.append(
            {
                "text_doc_id": batch_id,
                "data_id": batch_id,
                "contents": "\n\n---\n\n".join(current_parts),
            }
        )
    return packed


def _has_wiki_evidence(contents: str) -> bool:
    lines = [line.strip() for line in contents.splitlines()]
    for line in lines:
        if not line.startswith("Evidence:"):
            continue
        if line.removeprefix("Evidence:").strip():
            return True
    if len(lines) >= 3 and _looks_like_title_line(lines[0]) and lines[1].startswith("Section:"):
        return any(line for line in lines[2:])
    return False


def _looks_like_title_line(line: str) -> bool:
    return len(line) >= 2 and line.startswith('"') and line.endswith('"')


def _prepare_dataset_output_dir(output_dir: Path, *, source_subset: str | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_subset:
        stale_subset_dir = output_dir / source_subset
        if stale_subset_dir.exists():
            if stale_subset_dir.is_dir() and not stale_subset_dir.is_symlink():
                shutil.rmtree(stale_subset_dir)
            else:
                stale_subset_dir.unlink()
    _remove_obsolete_mm_indexes(output_dir)


def _remove_obsolete_mm_indexes(output_dir: Path) -> None:
    indexing_dir = output_dir / "mm_store" / "indexing"
    if not indexing_dir.exists():
        return
    for pattern in ("text_*", "fused_*"):
        for path in indexing_dir.glob(pattern):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()


def _write_graphr1_seed_guard(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_file(
        output_dir / ".graphr1_seeded.json",
        {
            "seeded": False,
            "source": "evograph_mm",
            "copied": [],
            "note": "MM build disables GraphR1 expr seeding for dataset-level KB isolation.",
        },
    )


def _can_reuse_graphr1_text_artifacts(
    output_dir: Path,
    *,
    expected_text_document_count: int,
    expected_fingerprint: str,
) -> bool:
    if expected_text_document_count <= 0:
        return False
    if not _has_valid_graphr1_seed_guard(output_dir / ".graphr1_seeded.json"):
        return False
    required_files = (
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "kv_store_entities.json",
        "kv_store_hyperedges.json",
        "graph_chunk_entity_relation.graphml",
    )
    if any(not (output_dir / name).is_file() for name in required_files):
        return False
    graphml_path = output_dir / "graph_chunk_entity_relation.graphml"
    if graphml_path.stat().st_size <= 0:
        return False
    try:
        full_docs = _read_json_object(output_dir / "kv_store_full_docs.json")
        text_chunks = _read_json_object(output_dir / "kv_store_text_chunks.json")
        entity_records = _load_entity_records(output_dir / "kv_store_entities.json")
        hyperedge_records = _load_hyperedge_records(output_dir / "kv_store_hyperedges.json")
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return (
        len(full_docs) == expected_text_document_count
        and len(text_chunks) >= expected_text_document_count
        and _full_docs_fingerprint(full_docs) == expected_fingerprint
        and len(entity_records) > 0
        and len(hyperedge_records) > 0
    )


def _text_documents_fingerprint(text_documents: list[dict[str, Any]]) -> str:
    contents = [str(doc.get("contents") or "") for doc in text_documents]
    return _contents_fingerprint(contents)


def _full_docs_fingerprint(full_docs: dict[str, Any]) -> str:
    contents = [
        str(value.get("content") or "")
        for value in full_docs.values()
        if isinstance(value, dict)
    ]
    return _contents_fingerprint(contents)


def _contents_fingerprint(contents: list[str]) -> str:
    digest = hashlib.sha256()
    for content in sorted(contents):
        encoded = content.encode("utf-8", errors="replace")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _has_valid_graphr1_seed_guard(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _read_json_object(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return payload.get("seeded") is False and payload.get("source") == "evograph_mm"


def _remove_graphr1_text_artifacts(output_dir: Path) -> None:
    for name in (
        ".graphr1_seeded.json",
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "kv_store_chunks.json",
        "kv_store_entities.json",
        "kv_store_hyperedges.json",
        "kv_store_llm_response_cache.json",
        "graph_chunk_entity_relation.graphml",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


def build_text_graphr1_indexes(
    *,
    output_dir: Path,
    encoder: Any,
    model_path: Path,
    model_repo_id: str,
    encoder_mode: str,
) -> dict[str, dict[str, Any]]:
    entity_records = _load_entity_records(output_dir / "kv_store_entities.json")
    hyperedge_records = _load_hyperedge_records(output_dir / "kv_store_hyperedges.json")

    entity_embeddings = encoder.encode_texts(
        (record["content"] for record in entity_records),
        instruction=TEXT_DOCUMENT_INSTRUCTION,
    )
    hyperedge_embeddings = encoder.encode_texts(
        (record["content"] for record in hyperedge_records),
        instruction=TEXT_DOCUMENT_INSTRUCTION,
    )

    return {
        "entity": _write_root_graphr1_index(
            output_dir=output_dir,
            namespace="entity",
            ids=[record["id"] for record in entity_records],
            contents=[record["content"] for record in entity_records],
            embeddings=entity_embeddings,
            corpus_file="corpus_entity.npy",
            index_file="index_entity.bin",
            metadata_file="entity_index_metadata.json",
            model_path=model_path,
            model_repo_id=model_repo_id,
            encoder_mode=encoder_mode,
        ),
        "hyperedge": _write_root_graphr1_index(
            output_dir=output_dir,
            namespace="hyperedge",
            ids=[record["id"] for record in hyperedge_records],
            contents=[record["content"] for record in hyperedge_records],
            embeddings=hyperedge_embeddings,
            corpus_file="corpus_hyperedge.npy",
            index_file="index_hyperedge.bin",
            metadata_file="hyperedge_index_metadata.json",
            model_path=model_path,
            model_repo_id=model_repo_id,
            encoder_mode=encoder_mode,
        ),
    }


def _reuse_text_graphr1_indexes(
    *,
    output_dir: Path,
    model_path: Path,
    model_repo_id: str,
    encoder_mode: str,
    embedding_dim: int,
) -> dict[str, dict[str, Any]] | None:
    try:
        entity_count = len(_load_entity_records(output_dir / "kv_store_entities.json"))
        hyperedge_count = len(_load_hyperedge_records(output_dir / "kv_store_hyperedges.json"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    entity_index = _reuse_root_graphr1_index(
        output_dir=output_dir,
        namespace="entity",
        metadata_file="entity_index_metadata.json",
        default_corpus_file="corpus_entity.npy",
        default_index_file="index_entity.bin",
        expected_vector_count=entity_count,
        model_path=model_path,
        model_repo_id=model_repo_id,
        encoder_mode=encoder_mode,
        embedding_dim=embedding_dim,
    )
    hyperedge_index = _reuse_root_graphr1_index(
        output_dir=output_dir,
        namespace="hyperedge",
        metadata_file="hyperedge_index_metadata.json",
        default_corpus_file="corpus_hyperedge.npy",
        default_index_file="index_hyperedge.bin",
        expected_vector_count=hyperedge_count,
        model_path=model_path,
        model_repo_id=model_repo_id,
        encoder_mode=encoder_mode,
        embedding_dim=embedding_dim,
    )
    if entity_index is None or hyperedge_index is None:
        return None
    return {"entity": entity_index, "hyperedge": hyperedge_index}


def _reuse_root_graphr1_index(
    *,
    output_dir: Path,
    namespace: str,
    metadata_file: str,
    default_corpus_file: str,
    default_index_file: str,
    expected_vector_count: int,
    model_path: Path,
    model_repo_id: str,
    encoder_mode: str,
    embedding_dim: int,
) -> dict[str, Any] | None:
    metadata_path = output_dir / metadata_file
    if not metadata_path.is_file():
        return None
    try:
        metadata = _read_json_object(metadata_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    corpus_file = str(metadata.get("corpus_file") or default_corpus_file)
    index_file = str(metadata.get("index_file") or default_index_file)
    corpus_path = output_dir / corpus_file
    index_path = output_dir / index_file
    if not corpus_path.is_file() or not index_path.is_file():
        return None
    if corpus_path.stat().st_size <= 0 or index_path.stat().st_size <= 0:
        return None
    if metadata.get("namespace") != namespace:
        return None
    if int(metadata.get("vector_count") or 0) != expected_vector_count:
        return None
    if int(metadata.get("embedding_dimension") or 0) != embedding_dim:
        return None
    if metadata.get("model_repo_id") != model_repo_id:
        return None
    if metadata.get("encoder_mode") != encoder_mode:
        return None
    recorded_model = Path(str(metadata.get("model_path") or ""))
    if recorded_model.resolve(strict=False) != model_path.resolve(strict=False):
        return None
    return {
        "namespace": namespace,
        "vector_count": expected_vector_count,
        "embedding_dimension": embedding_dim,
        "index_path": str(index_path),
        "corpus_path": str(corpus_path),
        "metadata_path": str(metadata_path),
    }


def _reuse_mm_vector_index(
    *,
    output_dir: Path,
    name: str,
    expected_ids: list[str],
    model_path: Path,
    model_repo_id: str,
    encoder_mode: str,
    embedding_dim: int,
    instruction: str,
) -> dict[str, Any] | None:
    metadata_path = output_dir / f"{name}_index_metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = _read_json_object(metadata_path)
        ids = json.loads((output_dir / str(metadata.get("ids_file") or f"{name}_ids.json")).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if ids != expected_ids:
        return None
    embeddings_file = str(metadata.get("embedding_file") or f"{name}_embeddings.npy")
    ids_file = str(metadata.get("ids_file") or f"{name}_ids.json")
    index_file = str(metadata.get("index_file") or metadata.get("faiss_file") or f"{name}_index.faiss")
    embeddings_path = output_dir / embeddings_file
    ids_path = output_dir / ids_file
    index_path = output_dir / index_file
    if not embeddings_path.is_file() or not ids_path.is_file() or not index_path.is_file():
        return None
    if embeddings_path.stat().st_size <= 0 or index_path.stat().st_size <= 0:
        return None
    if metadata.get("name") != name:
        return None
    if int(metadata.get("vector_count") or 0) != len(expected_ids):
        return None
    if int(metadata.get("embedding_dimension") or 0) != embedding_dim:
        return None
    if metadata.get("model_repo_id") != model_repo_id:
        return None
    if metadata.get("encoder_mode") != encoder_mode:
        return None
    if metadata.get("instruction") != instruction:
        return None
    recorded_model = Path(str(metadata.get("model_path") or ""))
    if recorded_model.resolve(strict=False) != model_path.resolve(strict=False):
        return None
    return {
        "name": name,
        "vector_count": len(expected_ids),
        "embedding_dimension": embedding_dim,
        "index_path": str(index_path),
        "ids_path": str(ids_path),
        "embeddings_path": str(embeddings_path),
        "metadata_path": str(metadata_path),
    }


def _write_mock_graphr1_text_artifacts(
    output_dir: Path,
    text_documents: list[dict[str, Any]],
) -> None:
    import networkx as nx

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_graphr1_seed_guard(output_dir)
    full_docs: dict[str, dict[str, Any]] = {}
    text_chunks: dict[str, dict[str, Any]] = {}
    entities: dict[str, dict[str, str]] = {}
    hyperedges: dict[str, dict[str, str]] = {}
    graph = nx.Graph()

    for idx, doc in enumerate(text_documents):
        text_doc_id = str(doc.get("text_doc_id") or f"text::{idx}")
        data_id = str(doc.get("data_id") or text_doc_id)
        contents = str(doc.get("contents") or "")
        entity_name = _mock_entity_name(doc)
        entity_key = "entity::" + entity_name.strip('"')
        hyperedge_key = f"hyperedge::{data_id}"
        chunk_key = f"chunk::{text_doc_id}"

        full_docs[text_doc_id] = {"content": contents}
        text_chunks[chunk_key] = {"content": contents, "full_doc_id": text_doc_id}
        entities.setdefault(
            entity_key,
            {
                "entity_name": entity_name,
                "content": f"{entity_name} evidence from {data_id}: {contents}",
            },
        )
        hyperedges[hyperedge_key] = {
            "hyperedge_name": f"<hyperedge>{data_id}",
            "content": f"{entity_name} is grounded by text document {text_doc_id}.",
        }
        graph.add_node(entity_name, entity_type="MOCK_ENTITY")
        graph.add_node(chunk_key, entity_type="TEXT_CHUNK")
        graph.add_edge(entity_name, chunk_key, relation="MENTIONS")

    _write_json_file(output_dir / "kv_store_full_docs.json", full_docs)
    _write_json_file(output_dir / "kv_store_text_chunks.json", text_chunks)
    _write_json_file(output_dir / "kv_store_entities.json", entities)
    _write_json_file(output_dir / "kv_store_hyperedges.json", hyperedges)
    nx.write_graphml(graph, output_dir / "graph_chunk_entity_relation.graphml")


def _mock_entity_name(doc: dict[str, Any]) -> str:
    source_metadata = doc.get("source_metadata")
    source_row = (
        source_metadata.get("source_row", {})
        if isinstance(source_metadata, dict)
        and isinstance(source_metadata.get("source_row"), dict)
        else {}
    )
    for value in (
        source_row.get("wikipedia_title"),
        doc.get("answer"),
        doc.get("image_id"),
        doc.get("data_id"),
    ):
        if isinstance(value, str) and value.strip():
            return f'"{value.strip().upper()}"'
    return '"UNKNOWN"'


def _hyperedge_embedding_content(value: Any) -> str:
    content = str(value or "").strip()
    if content.startswith("<hyperedge>"):
        content = content[len("<hyperedge>") :].strip()
    return content.strip().strip('"').strip()


def _load_entity_records(path: Path) -> list[dict[str, str]]:
    data = _read_json_object(path)
    records: list[dict[str, str]] = []
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        entity_id = str(value.get("entity_name") or key)
        content = str(value.get("content") or entity_id).strip()
        if content:
            records.append({"id": entity_id, "content": content})
    if not records:
        raise ValueError(f"no entity records found in {path}")
    return records


def _load_hyperedge_records(path: Path) -> list[dict[str, str]]:
    data = _read_json_object(path)
    records: list[dict[str, str]] = []
    for key, value in data.items():
        if not isinstance(value, dict) or value.get("deleted", False):
            continue
        content = _hyperedge_embedding_content(
            value.get("content") or value.get("hyperedge_name") or ""
        )
        if content:
            records.append({"id": str(value.get("hyperedge_name") or key), "content": content})
    if not records:
        raise ValueError(f"no active hyperedge records found in {path}")
    return records


def _write_root_graphr1_index(
    *,
    output_dir: Path,
    namespace: str,
    ids: list[str],
    contents: list[str],
    embeddings: Any,
    corpus_file: str,
    index_file: str,
    metadata_file: str,
    model_path: Path,
    model_repo_id: str,
    encoder_mode: str,
) -> dict[str, Any]:
    embeddings_array = np.asarray(embeddings, dtype=np.float32)
    if embeddings_array.ndim != 2 or embeddings_array.shape[0] == 0:
        raise ValueError(f"{namespace} embeddings must be a non-empty 2D array")
    if len(ids) != embeddings_array.shape[0] or len(contents) != embeddings_array.shape[0]:
        raise ValueError(f"{namespace} ids/content counts must match embeddings")

    import faiss

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / corpus_file
    index_path = output_dir / index_file
    metadata_path = output_dir / metadata_file

    embeddings_array = np.ascontiguousarray(embeddings_array, dtype=np.float32)
    np.save(corpus_path, embeddings_array)
    index = faiss.index_factory(
        embeddings_array.shape[1],
        "Flat",
        faiss.METRIC_INNER_PRODUCT,
    )
    index.add(embeddings_array)
    faiss.write_index(index, str(index_path))

    metadata = {
        "namespace": namespace,
        "vector_count": int(embeddings_array.shape[0]),
        "embedding_dimension": int(embeddings_array.shape[1]),
        "index_type": "IndexFlatIP",
        "metric": "inner_product",
        "corpus_file": corpus_path.name,
        "index_file": index_path.name,
        "model_path": str(model_path),
        "model_repo_id": model_repo_id,
        "encoder_mode": encoder_mode,
        "ids": ids,
        "contents": contents,
    }
    _write_json_file(metadata_path, metadata)
    return {
        "namespace": namespace,
        "vector_count": int(embeddings_array.shape[0]),
        "embedding_dimension": int(embeddings_array.shape[1]),
        "index_path": str(index_path),
        "corpus_path": str(corpus_path),
        "metadata_path": str(metadata_path),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _output_root_violation(layout: Any) -> str | None:
    if layout.output_root.name == "expr_mm":
        return None
    return layout.output_dir.as_posix()


def _instruction_metadata() -> dict[str, str]:
    return {
        "text_query": TEXT_QUERY_INSTRUCTION,
        "text_document": TEXT_DOCUMENT_INSTRUCTION,
        "image": IMAGE_DOCUMENT_INSTRUCTION,
        "fused": FUSED_DOCUMENT_INSTRUCTION,
    }


def _blocked_report(
    *,
    base_report: dict[str, Any],
    started: float,
    blocker: str,
    protected_path_violations: list[str] | None = None,
) -> dict[str, Any]:
    return {
        **base_report,
        "status": "blocked",
        "warnings": [],
        "written_files": [],
        "protected_path_violations": protected_path_violations or [],
        "split_counts": {},
        "store_counts": {},
        "embedding_counts": {},
        "index_paths": {},
        "missing_images": {},
        "missing_fields": {},
        "blockers": [blocker],
        "elapsed_seconds": time.perf_counter() - started,
    }


def _summarize_row_quality(
    split_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    missing_by_split = {split: 0 for split in split_rows}
    missing_fields = {
        "source_row.wikipedia_url": 0,
        "source_row.wikipedia_title": 0,
        "question": 0,
        "image_path": 0,
    }

    for split, rows in split_rows.items():
        for row in rows:
            extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
            source_row = (
                extra_info.get("source_row")
                if isinstance(extra_info.get("source_row"), dict)
                else {}
            )
            image_path_missing = _is_blank(row.get("image_path"))
            image_marked_missing = _boolish(
                _first_present(extra_info.get("image_missing"), row.get("image_missing")),
            )
            if image_path_missing or image_marked_missing:
                missing_by_split[split] += 1
            if _is_blank(source_row.get("wikipedia_url")):
                missing_fields["source_row.wikipedia_url"] += 1
            if _is_blank(source_row.get("wikipedia_title")):
                missing_fields["source_row.wikipedia_title"] += 1
            if _is_blank(_first_present(extra_info.get("question"), row.get("question"))):
                missing_fields["question"] += 1
            if image_path_missing:
                missing_fields["image_path"] += 1

    return {
        "missing_images": {
            "total": sum(missing_by_split.values()),
            "by_split": missing_by_split,
        },
        "missing_fields": missing_fields,
    }


def _success_written_files(
    layout: Any,
    index_paths: dict[str, dict[str, Any]],
    text_graph_index_paths: dict[str, dict[str, Any]],
) -> list[str]:
    files = [
        layout.output_dir / "metadata.json",
        layout.output_dir / "build_report.json",
        layout.output_dir / "kv_store_full_docs.json",
        layout.output_dir / "kv_store_text_chunks.json",
        layout.output_dir / "kv_store_entities.json",
        layout.output_dir / "kv_store_hyperedges.json",
        layout.output_dir / "graph_chunk_entity_relation.graphml",
        layout.graphr1_text_root / "text_documents.jsonl",
        layout.graphr1_text_root / "corpus.jsonl",
        layout.graphr1_text_root / "metadata.json",
        layout.visual_store_root / "image_records.jsonl",
        layout.visual_store_root / "metadata.json",
        layout.link_store_root / "links.jsonl",
        layout.link_store_root / "metadata.json",
        layout.output_dir / "mm_store" / "graph" / "graph_mm_entity_relation.graphml",
        layout.output_dir / "mm_store" / "graph" / "image_anchor_lookup.json",
        layout.output_dir / "mm_store" / "graph" / "canonical_entity_lookup.json",
        layout.output_dir / "mm_store" / "graph" / "image_to_text_lookup.json",
        layout.output_dir / "mm_store" / "graph" / "graph_metadata.json",
    ]
    for summary in index_paths.values():
        files.extend(
            [
                summary["index_path"],
                summary["ids_path"],
                summary["embeddings_path"],
                summary["metadata_path"],
            ]
        )
    for summary in text_graph_index_paths.values():
        files.extend(
            [
                summary["index_path"],
                summary["corpus_path"],
                summary["metadata_path"],
            ]
        )
    return [str(path) for path in files]


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _string_or_none(value: Any) -> str | None:
    if _is_blank(value):
        return None
    return str(value)


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    value_type = type(value)
    if getattr(value_type, "__module__", "").startswith("pandas._libs.missing"):
        return True
    if value is pd.NA:
        return True
    return isinstance(value, str) and not value.strip()


def _write_reports(
    output_dir: Path,
    report: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if metadata is not None:
        metadata_payload = json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        (output_dir / "metadata.json").write_text(metadata_payload, encoding="utf-8")
    else:
        metadata_path = output_dir / "metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()
    report_payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (output_dir / "build_report.json").write_text(report_payload, encoding="utf-8")


__all__ = ["build_arg_parser", "run_build", "main"]
