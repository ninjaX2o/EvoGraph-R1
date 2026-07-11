"""FAISS-backed vector index writer for multimodal KB embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def write_vector_index(
    output_dir: str | Path,
    name: str,
    ids: Iterable[str],
    embeddings: Any,
    model_path: str | Path,
    model_repo_id: str,
    instruction: str,
    encoder_mode: str,
) -> dict[str, Any]:
    ids_list = list(ids)
    embeddings_array = np.asarray(embeddings, dtype=np.float32)
    if embeddings_array.ndim != 2:
        raise ValueError(f"embeddings must be a 2D array, got shape {embeddings_array.shape}")
    if embeddings_array.shape[0] == 0:
        raise ValueError("embeddings must contain at least one row")
    if not ids_list:
        raise ValueError("ids must be non-empty")
    if any(not isinstance(item, str) or not item.strip() for item in ids_list):
        raise ValueError("ids must contain only non-empty strings")
    if len(ids_list) != embeddings_array.shape[0]:
        raise ValueError(
            f"ids length {len(ids_list)} must match embedding rows {embeddings_array.shape[0]}"
        )

    import faiss

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_path / f"{name}_embeddings.npy"
    ids_path = output_path / f"{name}_ids.json"
    index_path = output_path / f"{name}_index.faiss"
    metadata_path = output_path / f"{name}_index_metadata.json"

    embeddings_array = np.ascontiguousarray(embeddings_array, dtype=np.float32)
    np.save(embeddings_path, embeddings_array)
    ids_path.write_text(
        json.dumps(ids_list, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    index = faiss.index_factory(embeddings_array.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(embeddings_array)
    faiss.write_index(index, str(index_path))

    metadata = {
        "name": name,
        "vector_count": int(embeddings_array.shape[0]),
        "embedding_dimension": int(embeddings_array.shape[1]),
        "index_type": "IndexFlatIP",
        "metric": "inner_product",
        "embedding_file": embeddings_path.name,
        "ids_file": ids_path.name,
        "index_file": index_path.name,
        "faiss_file": index_path.name,
        "model_path": str(model_path),
        "model_repo_id": model_repo_id,
        "instruction": instruction,
        "encoder_mode": encoder_mode,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "name": name,
        "vector_count": int(embeddings_array.shape[0]),
        "embedding_dimension": int(embeddings_array.shape[1]),
        "index_path": str(index_path),
        "ids_path": str(ids_path),
        "embeddings_path": str(embeddings_path),
        "metadata_path": str(metadata_path),
    }


__all__ = ["write_vector_index"]
