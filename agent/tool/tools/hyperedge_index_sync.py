"""
Utilities for keeping the hyperedge FAISS index aligned with KV state.

The search API maps FAISS result positions back to hyperedge text by iterating
kv_store_hyperedges.json, so the vector index must be rebuilt from the same
active-record order whenever CRUD mutates hyperedges.
"""

import json
import logging
import os
import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from agent.tool.tools.bge_model_manager import encode_texts_safe

logger = logging.getLogger(__name__)

HYPEREDGE_INDEX_FILE = "index_hyperedge.bin"
HYPEREDGE_CORPUS_FILE = "corpus_hyperedge.npy"
HYPEREDGE_HASHES_FILE = "corpus_hyperedge_hashes.json"
HYPEREDGE_METADATA_FILE = "hyperedge_index_metadata.json"
HYPEREDGE_KV_FILE = "kv_store_hyperedges.json"
DEFAULT_BGE_DIMENSION = 1024
DEFAULT_OPENAI_MODEL = "text-embedding-3-large"
OPENAI_EMBEDDING_DEFAULT_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}


@dataclass(frozen=True)
class HyperedgeIndexRebuildResult:
    working_dir: str
    active_count: int
    dimension: int
    provider: str
    model: str
    index_path: str
    corpus_path: str


def iter_active_hyperedge_contents(hyperedges_data: Dict) -> Iterable[str]:
    """Yield active hyperedge content in KV insertion order."""
    for hyperedge in hyperedges_data.values():
        if not isinstance(hyperedge, dict):
            continue
        if hyperedge.get("deleted", False):
            continue
        content = hyperedge.get("content") or hyperedge.get("hyperedge_name")
        if isinstance(content, str) and content.strip():
            yield content


def iter_searchable_hyperedge_contents(hyperedges_data: Dict) -> Iterable[str]:
    """Yield searchable hyperedge content in KV insertion order."""
    for hyperedge in hyperedges_data.values():
        if not isinstance(hyperedge, dict):
            continue
        content = hyperedge.get("content") or hyperedge.get("hyperedge_name")
        if isinstance(content, str) and content.strip():
            yield content


def load_hyperedges_data(working_dir: str) -> Dict:
    kv_path = os.path.join(working_dir, HYPEREDGE_KV_FILE)
    if not os.path.exists(kv_path):
        return {}
    with open(kv_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _embed_in_batches(
    texts: List[str],
    embed_func: Callable[[List[str]], np.ndarray],
    target_dimension: int,
    batch_size: int,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, target_dimension), dtype=np.float32)

    batches = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        embeddings = np.asarray(embed_func(batch), dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape != (len(batch), target_dimension):
            raise ValueError(
                "Hyperedge embedding shape mismatch: "
                f"got {embeddings.shape}, expected {(len(batch), target_dimension)}"
            )
        batches.append(embeddings)
    return np.vstack(batches)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_hyperedge_index_metadata(working_dir: str) -> Dict:
    metadata_path = os.path.join(working_dir, HYPEREDGE_METADATA_FILE)
    if not os.path.exists(metadata_path):
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata if isinstance(metadata, dict) else {}
    except Exception as e:
        logger.warning("[HYPEREDGE_INDEX_SYNC] Failed to load index metadata: %s", e)
        return {}


def write_hyperedge_index_metadata(
    working_dir: str,
    provider: str,
    dimension: int,
    model: str,
    active_count: int,
) -> str:
    metadata_path = os.path.join(working_dir, HYPEREDGE_METADATA_FILE)
    tmp_metadata_path = metadata_path + ".tmp"
    metadata = {
        "provider": provider,
        "dimension": int(dimension),
        "model": model,
        "active_count": int(active_count),
        "index_file": HYPEREDGE_INDEX_FILE,
        "corpus_file": HYPEREDGE_CORPUS_FILE,
        "hashes_file": HYPEREDGE_HASHES_FILE,
    }
    with open(tmp_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    os.replace(tmp_metadata_path, metadata_path)
    return metadata_path


def ensure_hyperedge_index_sidecars(
    working_dir: str,
    hyperedges_data: Optional[Dict] = None,
    previous_hyperedges_data: Optional[Dict] = None,
    embedding_provider: Optional[str] = None,
    target_dimension: Optional[int] = None,
    model_name: Optional[str] = None,
) -> bool:
    data = hyperedges_data if hyperedges_data is not None else load_hyperedges_data(working_dir)
    current_texts = list(iter_searchable_hyperedge_contents(data))
    previous_texts = (
        list(iter_searchable_hyperedge_contents(previous_hyperedges_data))
        if previous_hyperedges_data is not None
        else []
    )
    if not current_texts and not previous_texts:
        return False

    corpus_path = os.path.join(working_dir, HYPEREDGE_CORPUS_FILE)
    if not os.path.exists(corpus_path):
        return False

    try:
        corpus = np.load(corpus_path, mmap_mode="r")
    except Exception as e:
        logger.warning("[HYPEREDGE_INDEX_SYNC] Failed to inspect existing corpus for sidecar backfill: %s", e)
        return False

    if getattr(corpus, "ndim", 0) != 2:
        return False

    texts = None
    if previous_texts and corpus.shape[0] == len(previous_texts):
        texts = previous_texts
    elif corpus.shape[0] == len(current_texts):
        texts = current_texts
    if texts is None:
        return False

    sidecar_written = False
    hashes_path = os.path.join(working_dir, HYPEREDGE_HASHES_FILE)
    hashes = [_content_hash(text) for text in texts]
    try:
        needs_hashes = True
        if os.path.exists(hashes_path):
            with open(hashes_path, "r", encoding="utf-8") as f:
                loaded_hashes = json.load(f)
            needs_hashes = not (
                isinstance(loaded_hashes, list)
                and len(loaded_hashes) == len(hashes)
                and loaded_hashes == hashes
            )
        if needs_hashes:
            tmp_hashes_path = hashes_path + ".tmp"
            with open(tmp_hashes_path, "w", encoding="utf-8") as f:
                json.dump(hashes, f, ensure_ascii=False, indent=2)
            os.replace(tmp_hashes_path, hashes_path)
            sidecar_written = True
    except Exception as e:
        logger.warning("[HYPEREDGE_INDEX_SYNC] Failed to backfill corpus hashes: %s", e)

    try:
        metadata = load_hyperedge_index_metadata(working_dir)
        provider, resolved_dimension, resolved_model = resolve_hyperedge_index_config(
            working_dir,
            embedding_provider=embedding_provider,
            target_dimension=target_dimension,
            model_name=model_name,
        )
        expected_metadata = {
            "provider": provider,
            "dimension": int(resolved_dimension),
            "model": resolved_model,
            "active_count": int(len(texts)),
            "index_file": HYPEREDGE_INDEX_FILE,
            "corpus_file": HYPEREDGE_CORPUS_FILE,
            "hashes_file": HYPEREDGE_HASHES_FILE,
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            write_hyperedge_index_metadata(
                working_dir,
                provider=provider,
                dimension=resolved_dimension,
                model=resolved_model,
                active_count=len(texts),
            )
            sidecar_written = True
    except Exception as e:
        logger.warning("[HYPEREDGE_INDEX_SYNC] Failed to backfill index metadata: %s", e)

    return sidecar_written


def _infer_existing_dimension(working_dir: str) -> Optional[int]:
    corpus_path = os.path.join(working_dir, HYPEREDGE_CORPUS_FILE)
    if os.path.exists(corpus_path):
        try:
            corpus = np.load(corpus_path, mmap_mode="r")
            if getattr(corpus, "ndim", 0) == 2:
                return int(corpus.shape[1])
        except Exception as e:
            logger.warning("[HYPEREDGE_INDEX_SYNC] Failed to inspect corpus dimension: %s", e)

    index_path = os.path.join(working_dir, HYPEREDGE_INDEX_FILE)
    if os.path.exists(index_path):
        try:
            import faiss

            return int(faiss.read_index(index_path).d)
        except Exception as e:
            logger.warning("[HYPEREDGE_INDEX_SYNC] Failed to inspect FAISS dimension: %s", e)
    return None


def resolve_hyperedge_index_config(
    working_dir: str,
    embedding_provider: Optional[str] = None,
    target_dimension: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Tuple[str, int, str]:
    metadata = load_hyperedge_index_metadata(working_dir)

    provider = (
        embedding_provider
        or metadata.get("provider")
        or os.getenv("HYPEREDGE_INDEX_PROVIDER")
    )
    dimension = target_dimension or metadata.get("dimension") or _infer_existing_dimension(working_dir)

    if provider is None:
        provider = "bge" if dimension in (None, DEFAULT_BGE_DIMENSION) else "openai"

    provider = str(provider).lower()
    if dimension is None:
        dimension = DEFAULT_BGE_DIMENSION if provider == "bge" else 3072

    model = (
        model_name
        or metadata.get("model")
        or (
            os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_MODEL)
            if provider == "openai"
            else os.getenv("BGE_MODEL_NAME", "BAAI/bge-large-en-v1.5")
        )
    )
    return provider, int(dimension), str(model)


def _openai_model_supports_dimensions(model: str) -> bool:
    return str(model or "").startswith("text-embedding-3-")


def _build_openai_embedding_request_kwargs(
    texts: List[str],
    model: str,
    target_dimension: int,
) -> Dict:
    request_kwargs = {"model": model, "input": texts}
    configured_dimensions = os.getenv("OPENAI_EMBEDDING_DIMENSIONS")
    requested_dimension = int(configured_dimensions) if configured_dimensions else int(target_dimension)

    if int(target_dimension) != requested_dimension:
        raise ValueError(
            "OpenAI embedding dimension configuration mismatch: "
            f"target_dimension={target_dimension}, OPENAI_EMBEDDING_DIMENSIONS={requested_dimension}"
        )

    default_dimension = OPENAI_EMBEDDING_DEFAULT_DIMENSIONS.get(str(model))
    if _openai_model_supports_dimensions(model):
        request_kwargs["dimensions"] = requested_dimension
    elif default_dimension is not None and requested_dimension != default_dimension:
        raise ValueError(
            f"OpenAI embedding model {model} does not support requested dimension {requested_dimension}"
        )
    elif configured_dimensions:
        request_kwargs["dimensions"] = requested_dimension

    return request_kwargs


def _embed_openai_texts(texts: List[str], model: str, target_dimension: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, target_dimension), dtype=np.float32)

    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    request_kwargs = _build_openai_embedding_request_kwargs(texts, model, target_dimension)
    response = client.embeddings.create(**request_kwargs)
    embeddings = np.asarray([item.embedding for item in response.data], dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    return embeddings


def _load_reusable_embeddings(
    working_dir: str,
    current_texts: List[str],
    previous_hyperedges_data: Optional[Dict],
    allow_prefix_reuse: bool,
) -> Tuple[Dict[str, List[np.ndarray]], Optional[np.ndarray]]:
    corpus_path = os.path.join(working_dir, HYPEREDGE_CORPUS_FILE)
    if not os.path.exists(corpus_path):
        return {}, None

    old_embeddings = np.load(corpus_path)
    old_hashes = None

    if previous_hyperedges_data is not None:
        previous_texts = list(iter_searchable_hyperedge_contents(previous_hyperedges_data))
        if old_embeddings.shape[0] == len(previous_texts):
            old_hashes = [_content_hash(text) for text in previous_texts]

    if old_hashes is None:
        hashes_path = os.path.join(working_dir, HYPEREDGE_HASHES_FILE)
        if os.path.exists(hashes_path):
            try:
                with open(hashes_path, "r", encoding="utf-8") as f:
                    loaded_hashes = json.load(f)
                if isinstance(loaded_hashes, list) and len(loaded_hashes) == old_embeddings.shape[0]:
                    old_hashes = loaded_hashes
            except Exception as e:
                logger.warning("[HYPEREDGE_INDEX_SYNC] Failed to load hash metadata: %s", e)

    if old_hashes is None and allow_prefix_reuse and old_embeddings.shape[0] <= len(current_texts):
        old_hashes = [_content_hash(text) for text in current_texts[:old_embeddings.shape[0]]]

    if old_hashes is None or old_embeddings.shape[0] != len(old_hashes):
        return {}, old_embeddings
    if old_embeddings.ndim != 2:
        return {}, old_embeddings

    reusable: Dict[str, List[np.ndarray]] = {}
    for text_hash, vector in zip(old_hashes, old_embeddings):
        reusable.setdefault(text_hash, []).append(vector)
    return reusable, old_embeddings


def rebuild_hyperedge_vector_index(
    working_dir: str,
    hyperedges_data: Optional[Dict] = None,
    previous_hyperedges_data: Optional[Dict] = None,
    allow_prefix_reuse: bool = False,
    embed_func: Optional[Callable[[List[str]], np.ndarray]] = None,
    target_dimension: Optional[int] = None,
    embedding_provider: Optional[str] = None,
    model_name: Optional[str] = None,
    batch_size: int = 64,
) -> HyperedgeIndexRebuildResult:
    """Rebuild hyperedge FAISS index and embedding corpus from searchable KV records."""
    import faiss

    os.makedirs(working_dir, exist_ok=True)
    provider, resolved_dimension, resolved_model = resolve_hyperedge_index_config(
        working_dir,
        embedding_provider=embedding_provider,
        target_dimension=target_dimension,
        model_name=model_name,
    )
    data = hyperedges_data if hyperedges_data is not None else load_hyperedges_data(working_dir)
    texts = list(iter_searchable_hyperedge_contents(data))
    active_count = sum(
        1
        for hyperedge in data.values()
        if isinstance(hyperedge, dict) and not hyperedge.get("deleted", False)
    )
    ensure_hyperedge_index_sidecars(
        working_dir,
        hyperedges_data=data,
        previous_hyperedges_data=previous_hyperedges_data,
        embedding_provider=provider,
        target_dimension=resolved_dimension,
        model_name=resolved_model,
    )

    if embed_func is None:
        if provider == "openai":
            embed_func = lambda batch: _embed_openai_texts(
                batch,
                model=resolved_model,
                target_dimension=resolved_dimension,
            )
        elif provider == "bge":
            embed_func = lambda batch: encode_texts_safe(
                batch,
                target_dimension=resolved_dimension,
            )
        else:
            raise ValueError(f"Unsupported hyperedge index provider: {provider}")
    else:
        provider = embedding_provider or provider or "custom"
        resolved_model = model_name or resolved_model or "custom"

    reusable, _ = _load_reusable_embeddings(
        working_dir,
        current_texts=texts,
        previous_hyperedges_data=previous_hyperedges_data,
        allow_prefix_reuse=allow_prefix_reuse,
    )
    reusable = {
        key: [
            row for row in rows
            if np.asarray(row).ndim == 1 and np.asarray(row).shape[0] == resolved_dimension
        ]
        for key, rows in reusable.items()
    }
    new_texts = []
    final_rows: List[Optional[np.ndarray]] = []
    for text in texts:
        text_hash = _content_hash(text)
        reusable_rows = reusable.get(text_hash)
        if reusable_rows:
            final_rows.append(np.asarray(reusable_rows.pop(0), dtype=np.float32))
        else:
            final_rows.append(None)
            new_texts.append(text)

    new_embeddings = _embed_in_batches(new_texts, embed_func, resolved_dimension, batch_size)
    new_index = 0
    for row_index, row in enumerate(final_rows):
        if row is None:
            final_rows[row_index] = new_embeddings[new_index]
            new_index += 1

    if final_rows:
        embeddings = np.vstack(final_rows).astype(np.float32)
    else:
        embeddings = np.zeros((0, resolved_dimension), dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[1] != resolved_dimension:
        raise ValueError(
            "Hyperedge embedding corpus shape mismatch: "
            f"got {embeddings.shape}, expected (*, {resolved_dimension})"
        )

    index = faiss.index_factory(resolved_dimension, "Flat", faiss.METRIC_INNER_PRODUCT)
    if embeddings.shape[0] > 0:
        index.add(embeddings)

    index_path = os.path.join(working_dir, HYPEREDGE_INDEX_FILE)
    corpus_path = os.path.join(working_dir, HYPEREDGE_CORPUS_FILE)
    hashes_path = os.path.join(working_dir, HYPEREDGE_HASHES_FILE)
    tmp_index_path = index_path + ".tmp"
    tmp_corpus_path = corpus_path + ".tmp.npy"
    tmp_hashes_path = hashes_path + ".tmp"

    faiss.write_index(index, tmp_index_path)
    np.save(tmp_corpus_path, embeddings)
    with open(tmp_hashes_path, "w", encoding="utf-8") as f:
        json.dump([_content_hash(text) for text in texts], f, ensure_ascii=False, indent=2)
    os.replace(tmp_index_path, index_path)
    os.replace(tmp_corpus_path, corpus_path)
    os.replace(tmp_hashes_path, hashes_path)
    write_hyperedge_index_metadata(
        working_dir,
        provider=provider,
        dimension=resolved_dimension,
        model=resolved_model,
        active_count=active_count,
    )

    logger.info(
        "[HYPEREDGE_INDEX_SYNC] Rebuilt hyperedge index for %s: provider=%s, searchable=%d, active=%d, new_embeddings=%d, dim=%d",
        working_dir,
        provider,
        len(texts),
        active_count,
        len(new_texts),
        resolved_dimension,
    )
    return HyperedgeIndexRebuildResult(
        working_dir=working_dir,
        active_count=active_count,
        dimension=resolved_dimension,
        provider=provider,
        model=resolved_model,
        index_path=index_path,
        corpus_path=corpus_path,
    )
