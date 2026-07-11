"""Read-only multimodal KB retrieval loading and status helpers."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from evograph_mm.kb.indexing.encoders import (
    FUSED_DOCUMENT_INSTRUCTION,
    IMAGE_DOCUMENT_INSTRUCTION,
    MockEmbeddingEncoder,
    TEXT_DOCUMENT_INSTRUCTION,
)
from graphr1 import GraphR1, QueryParam


TEXT_QUERY_INSTRUCTION = "Retrieve multimodal evidence relevant to the question."
IMAGE_QUERY_TOKEN = "<img>"
DEFAULT_VISUAL_ENTITY_TOP_K = 5
DEFAULT_VISUAL_HYPEREDGE_TOP_K = 5
DEFAULT_VISUAL_HYPEREDGE_MIN_COHERENCE = 0.45
DEFAULT_BGE_TEXT_DIMENSION = 1024
logger = logging.getLogger(__name__)


class MMKBRequestError(ValueError):
    """Raised for invalid multimodal KB retrieval requests."""


@dataclass
class LoadedVectorIndex:
    name: str
    index: Any
    ids: list[str]
    metadata: dict[str, Any]


@dataclass
class LoadedBGEGraphIndexes:
    entity_index: Any
    entity_corpus: list[str]
    hyperedge_index: Any
    hyperedge_corpus: list[str]
    metadata: dict[str, Any]


@dataclass
class MMKBRetriever:
    working_dir: str | Path
    model_path: str | Path
    encoder_factory: Callable[..., Any] | None = None
    rag_factory: Callable[..., Any] | None = None
    dataset: str | None = None
    subset: str | None = None
    metadata: dict[str, Any] = field(init=False, default_factory=dict)
    build_report: dict[str, Any] = field(init=False, default_factory=dict)
    text_documents: list[dict[str, Any]] = field(init=False, default_factory=list)
    visual_records: list[dict[str, Any]] = field(init=False, default_factory=list)
    links: list[dict[str, Any]] = field(init=False, default_factory=list)
    image_anchor_lookup: dict[str, dict[str, Any]] = field(init=False, default_factory=dict)
    canonical_entity_lookup: dict[str, dict[str, Any]] = field(init=False, default_factory=dict)
    image_to_text_lookup: dict[str, dict[str, Any]] = field(init=False, default_factory=dict)
    graphr1_hit_source_sidecar: dict[str, Any] = field(init=False, default_factory=dict)
    indexes: dict[str, LoadedVectorIndex] = field(init=False, default_factory=dict)
    bge_graph_indexes: LoadedBGEGraphIndexes | None = field(init=False, default=None)
    bge_graph_blockers: list[str] = field(init=False, default_factory=list)
    embedding_records: dict[str, dict[str, Any]] = field(init=False, default_factory=dict)
    query_fused_vectors: Any = field(init=False, default=None)
    query_fused_lookup: dict[tuple[str, str], int] = field(init=False, default_factory=dict)
    encoder: Any = field(init=False, default=None)
    rag: Any = field(init=False, default=None)
    model_loaded: bool = field(init=False, default=False)
    blockers: list[str] = field(init=False, default_factory=list)
    last_reload_time: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.working_dir = Path(self.working_dir)
        self.model_path = Path(self.model_path)

    def load(self) -> dict[str, Any]:
        self._reset_runtime_state()
        self.last_reload_time = datetime.now(timezone.utc).isoformat()

        self.metadata = self._read_json(self.working_dir / "metadata.json", "metadata")
        self.build_report = self._read_json(
            self.working_dir / "build_report.json",
            "build report",
        )
        if self.dataset is None:
            self.dataset = _optional_string(self.metadata.get("dataset"))
        if self.subset is None:
            self.subset = _optional_string(
                self.metadata.get("source_subset") or self.metadata.get("subset")
            )

        self.text_documents = self._read_jsonl(
            self.working_dir / "graphr1_text" / "text_documents.jsonl",
            "text documents",
        )
        self.visual_records = self._read_jsonl(
            self.working_dir / "mm_store" / "visual" / "image_records.jsonl",
            "visual records",
        )
        self.links = self._read_jsonl(
            self.working_dir / "mm_store" / "links" / "links.jsonl",
            "links",
        )
        graph_dir = self.working_dir / "mm_store" / "graph"
        self.image_anchor_lookup = self._read_optional_json(
            graph_dir / "image_anchor_lookup.json",
            "image anchor lookup",
        )
        self.canonical_entity_lookup = self._read_optional_json(
            graph_dir / "canonical_entity_lookup.json",
            "canonical entity lookup",
        )
        self.image_to_text_lookup = self._read_optional_json(
            graph_dir / "image_to_text_lookup.json",
            "image-to-text lookup",
        )
        self.graphr1_hit_source_sidecar = self._read_optional_json(
            graph_dir / "graphr1_hit_source_sidecar.json",
            "GraphR1 hit source sidecar",
        )
        self._load_query_fused_cache()

        for name in ("entity", "hyperedge"):
            loaded = self._load_root_text_index(name)
            if loaded is not None:
                self.indexes[name] = loaded
                self._register_index_metadata_records(loaded)
        for name in ("image", "text", "fused"):
            loaded = self._load_vector_index(name, required=name == "image")
            if loaded is not None:
                self.indexes[name] = loaded

        self.embedding_records = self._build_embedding_record_map()
        self.bge_graph_indexes = self._load_or_build_bge_graph_indexes()
        if not self.blockers and self._runtime_encoder_enabled():
            self.encoder = self._load_encoder()
            self.model_loaded = self.encoder is not None

        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "status": "blocked" if self.blockers else "ready",
            "dataset": self.dataset,
            "subset": self.subset,
            "working_dir": str(self.working_dir),
            "model_path": str(self.model_path),
            "model_loaded": self.model_loaded,
            "runtime_encoder_enabled": self._runtime_encoder_enabled(),
            "indexes": {
                name: {
                    "loaded": name in self.indexes,
                    "vector_count": self._index_vector_count(name),
                }
                for name in ("entity", "hyperedge", "image", "text", "fused")
            },
            "bge_graph_index": {
                "loaded": self.bge_graph_indexes is not None,
                "entity_count": 0
                if self.bge_graph_indexes is None
                else len(self.bge_graph_indexes.entity_corpus),
                "hyperedge_count": 0
                if self.bge_graph_indexes is None
                else len(self.bge_graph_indexes.hyperedge_corpus),
                "blockers": list(self.bge_graph_blockers),
            },
            "ready": not self.blockers,
            "required_indexes": ["entity", "hyperedge", "image"],
            "counts": {
                "text_documents": len(self.text_documents),
                "visual_records": len(self.visual_records),
                "links": len(self.links),
            },
            "blockers": list(self.blockers),
            "last_reload_time": self.last_reload_time,
        }

    def search(
        self,
        queries: list[str] | None = None,
        context_queries: list[str] | None = None,
        image_ids: list[str] | None = None,
        image_paths: list[str] | None = None,
        rag_top_k: int = 10,
        image_top_k: int = 5,
        fused_top_k: int = 5,
        entity_top_k: int = 5,
        hyperedge_top_k: int = 5,
        entity_fusion_top_k: int = 0,
        entity_fusion_candidate_top_k: int = 20,
        visual_entity_top_k: int = DEFAULT_VISUAL_ENTITY_TOP_K,
        visual_entity_candidate_top_k: int = 20,
        visual_hyperedge_top_k: int = DEFAULT_VISUAL_HYPEREDGE_TOP_K,
        visual_hyperedge_candidate_top_k: int = 1000,
        visual_hyperedge_min_coherence: float = DEFAULT_VISUAL_HYPEREDGE_MIN_COHERENCE,
    ) -> list[dict[str, Any]]:
        try:
            rows = normalize_query_rows(
                queries=queries,
                image_ids=image_ids,
                image_paths=image_paths,
            )
        except MMKBRequestError as exc:
            message = str(exc)
            return [{"error": message, "blockers": [message]}]
        if self.blockers:
            return [
                {
                    "error": "multimodal KB is blocked",
                    "blockers": list(self.blockers),
                }
                for _ in rows
            ]
        payloads = []
        image_path_by_id = {
            record.get("image_id"): record.get("image_path")
            for record in self.visual_records
            if isinstance(record.get("image_id"), str)
            and isinstance(record.get("image_path"), str)
        }
        context_query_values = _broadcast(
            _string_rows(context_queries),
            len(rows),
            "context_queries",
        )

        for row_index, row in enumerate(rows):
            query = row.get("query", "")
            context_query = context_query_values[row_index]
            image_id = row.get("image_id", "")
            image_path = row.get("image_path", "")
            is_visual_entity_query = query.strip() == IMAGE_QUERY_TOKEN
            if image_id:
                if image_id in self.image_to_text_lookup and not is_visual_entity_query:
                    payloads.append({"results": self._image_to_text_results(image_id)})
                    continue
                stored_image_path = image_path_by_id.get(image_id, "")
                if stored_image_path:
                    image_path = stored_image_path
                elif not image_path or not is_visual_entity_query:
                    message = f"unknown image_id: {image_id}"
                    payloads.append({"error": message, "blockers": [message]})
                    continue
            resolved_image_path = self._resolve_image_path(image_path) if image_path else None
            if resolved_image_path is not None and not resolved_image_path.is_file():
                if self._allows_missing_image_files():
                    encoded_image_path = image_path
                else:
                    message = f"missing image path: {image_path}"
                    payloads.append(
                        {
                            "error": message,
                            "blockers": [message],
                        }
                    )
                    continue
            elif resolved_image_path is not None:
                encoded_image_path = str(resolved_image_path)
            else:
                encoded_image_path = ""
            resolved_image_id = image_id or self._image_id_for_path(image_path) or ""

            results = []
            entity_hits: list[dict[str, Any]] = []
            hyperedge_hits: list[dict[str, Any]] = []
            image_hits: list[dict[str, Any]] = []
            if is_visual_entity_query:
                if not image_path:
                    message = "image search requires image_id or image_path context"
                    payloads.append({"error": message, "blockers": [message]})
                    continue
                candidate_top_k = max(visual_entity_candidate_top_k, visual_entity_top_k)
                hyperedge_query_vector = self._visual_query_vector(
                    context_query=context_query,
                    encoded_image_path=encoded_image_path,
                    image_id=resolved_image_id,
                )
                if hyperedge_query_vector is None:
                    message = (
                        "image search requires a local indexed image_id, cached fused query vector, or encodable image context"
                    )
                    payloads.append({"error": message, "blockers": [message]})
                    continue
                visual_entity_hits: list[dict[str, Any]] = []
                visual_hyperedge_hits: list[dict[str, Any]] = []
                image_hits = self._search_index(
                    "image",
                    hyperedge_query_vector,
                    candidate_top_k,
                )
                if "entity" in self.indexes:
                    visual_entity_hits = self._search_index(
                        "entity",
                        hyperedge_query_vector,
                        candidate_top_k,
                    )
                if "hyperedge" in self.indexes:
                    visual_hyperedge_hits = self._search_index(
                        "hyperedge",
                        hyperedge_query_vector,
                        max(
                            candidate_top_k,
                            visual_hyperedge_top_k,
                            visual_hyperedge_candidate_top_k,
                        ),
                    )
                results.extend(
                    self._visual_entity_results(
                        image_hits=image_hits,
                        entity_hits=visual_entity_hits,
                        hyperedge_hits=visual_hyperedge_hits,
                        top_k=visual_entity_top_k or image_top_k,
                        hyperedge_query_vector=hyperedge_query_vector,
                        hyperedge_top_k=visual_hyperedge_top_k,
                        hyperedge_candidate_top_k=visual_hyperedge_candidate_top_k,
                        hyperedge_min_coherence=visual_hyperedge_min_coherence,
                    )
                )
                payloads.append({"results": results})
                continue

            if query and (
                rag_top_k > 0
                or entity_top_k > 0
                or hyperedge_top_k > 0
                or entity_fusion_top_k > 0
            ):
                bge_hits = self._search_bge_graph_indexes(
                    query,
                    entity_top_k=entity_top_k,
                    hyperedge_top_k=hyperedge_top_k,
                    rag_top_k=rag_top_k,
                )
                if bge_hits:
                    results.extend(bge_hits)
                else:
                    if self.encoder is None:
                        message = (
                            "text search requires a ready BGE graph FAISS index; "
                            "runtime GME encoder fallback is disabled"
                        )
                        payloads.append(
                            {
                                "error": message,
                                "blockers": [message, *self.bge_graph_blockers],
                            }
                        )
                        continue
                    text_vector = self.encoder.encode_texts(
                        [query],
                        instruction=TEXT_QUERY_INSTRUCTION,
                    )[0]
                    if "entity" in self.indexes or "hyperedge" in self.indexes:
                        entity_search_k = max(
                            entity_top_k,
                            entity_fusion_candidate_top_k if entity_fusion_top_k > 0 else 0,
                        )
                        hyperedge_search_k = max(
                            hyperedge_top_k,
                            entity_fusion_candidate_top_k if entity_fusion_top_k > 0 else 0,
                        )
                        entity_hits = self._search_index(
                            "entity",
                            text_vector,
                            entity_search_k,
                        )
                        hyperedge_hits = self._search_index(
                            "hyperedge",
                            text_vector,
                            hyperedge_search_k,
                        )
                        graph_context = (
                            self._graphr1_context_results(
                                query=query,
                                entity_hits=entity_hits[:entity_top_k],
                                hyperedge_hits=hyperedge_hits[:hyperedge_top_k],
                                rag_top_k=rag_top_k,
                            )
                            if rag_top_k > 0
                            else []
                        )
                        if graph_context:
                            results.extend(graph_context)
                        else:
                            results.extend(entity_hits[:entity_top_k])
                            results.extend(hyperedge_hits[:hyperedge_top_k])
                    else:
                        results.extend(self._search_index("text", text_vector, rag_top_k))
            if image_path and (image_top_k > 0 or entity_fusion_top_k > 0):
                direct_image_id = self._image_id_for_path(image_path)
                if direct_image_id is not None:
                    results.extend(self._image_to_text_results(direct_image_id))
                else:
                    if self.encoder is None:
                        message = (
                            "image search requires a local indexed image_id or cached query vector; "
                            "runtime GME encoder fallback is disabled"
                        )
                        payloads.append({"error": message, "blockers": [message]})
                        continue
                    image_vector = self.encoder.encode_images(
                        [encoded_image_path],
                        instruction=IMAGE_DOCUMENT_INSTRUCTION,
                    )[0]
                    image_search_k = max(
                        image_top_k,
                        entity_fusion_candidate_top_k if entity_fusion_top_k > 0 else 0,
                    )
                    image_hits = self._search_index("image", image_vector, image_search_k)
                    results.extend(self._expand_image_hits(image_hits[:image_top_k]))
            if entity_fusion_top_k > 0 and query and image_path:
                results.extend(
                    self._entity_fusion_results(
                        image_hits=image_hits,
                        entity_hits=entity_hits,
                        hyperedge_hits=hyperedge_hits,
                        top_k=entity_fusion_top_k,
                    )
                )
            if query and image_path and fused_top_k > 0 and "fused" in self.indexes:
                if self.encoder is None:
                    message = (
                        "fused search requires a cached fused vector or runtime encoder; "
                        "runtime GME encoder fallback is disabled"
                    )
                    payloads.append({"error": message, "blockers": [message]})
                    continue
                vector = self.encoder.encode_fused(
                    [{"text": query, "image": encoded_image_path}],
                    instruction=FUSED_DOCUMENT_INSTRUCTION,
                )[0]
                results.extend(self._search_index("fused", vector, fused_top_k))

            payloads.append({"results": results})
        return payloads

    def _reset_runtime_state(self) -> None:
        self.metadata = {}
        self.build_report = {}
        self.text_documents = []
        self.visual_records = []
        self.links = []
        self.image_anchor_lookup = {}
        self.canonical_entity_lookup = {}
        self.image_to_text_lookup = {}
        self.graphr1_hit_source_sidecar = {}
        self.indexes = {}
        self.bge_graph_indexes = None
        self.bge_graph_blockers = []
        self.embedding_records = {}
        self.query_fused_vectors = None
        self.query_fused_lookup = {}
        self.encoder = None
        self.rag = None
        self.model_loaded = False
        self.blockers = []
        self.last_reload_time = None

    def _read_json(self, path: Path, label: str) -> dict[str, Any]:
        try:
            if not path.is_file():
                self.blockers.append(f"missing {label}: {path.name}")
                return {}
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                self.blockers.append(f"invalid {label}: {path.name} is not a JSON object")
                return {}
            return payload
        except Exception as exc:
            self.blockers.append(f"failed to read {label} {path.name}: {exc}")
            return {}

    def _read_jsonl(self, path: Path, label: str) -> list[dict[str, Any]]:
        try:
            if not path.is_file():
                self.blockers.append(f"missing {label}: {path.name}")
                return []
            records = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    self.blockers.append(
                        f"invalid {label}: {path.name}:{line_number} is not a JSON object"
                    )
                    return []
                records.append(record)
            return records
        except Exception as exc:
            self.blockers.append(f"failed to read {label} {path.name}: {exc}")
            return []

    def _read_optional_json(self, path: Path, label: str) -> dict[str, Any]:
        try:
            if not path.is_file():
                return {}
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                self.blockers.append(f"invalid {label}: {path.name} is not a JSON object")
                return {}
            return payload
        except Exception as exc:
            self.blockers.append(f"failed to read {label} {path.name}: {exc}")
            return {}

    def _load_query_fused_cache(self) -> None:
        cache_dir = self.working_dir / "mm_store" / "query_fused"
        records_path = cache_dir / "qa_fused_records.jsonl"
        embeddings_path = cache_dir / "qa_fused_embeddings.npy"
        if not records_path.is_file() or not embeddings_path.is_file():
            return

        try:
            records = []
            for line in records_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if isinstance(record, dict):
                    records.append(record)
            vectors = np.load(embeddings_path, mmap_mode="r")
            if vectors.ndim != 2:
                return

            lookup: dict[tuple[str, str], int] = {}
            for record_position, record in enumerate(records):
                image_id = _optional_string(record.get("image_id"))
                question = _normalize_query_cache_text(record.get("question"))
                raw_index = record.get("embedding_index", record_position)
                try:
                    vector_index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if (
                    image_id
                    and question
                    and 0 <= vector_index < int(vectors.shape[0])
                ):
                    lookup.setdefault((image_id, question), vector_index)

            if lookup:
                self.query_fused_vectors = vectors
                self.query_fused_lookup = lookup
        except Exception:
            self.query_fused_vectors = None
            self.query_fused_lookup = {}

    def _load_vector_index(self, name: str, required: bool = True) -> LoadedVectorIndex | None:
        index_dir = self.working_dir / "mm_store" / "indexing"
        index_path = index_dir / f"{name}_index.faiss"
        ids_path = index_dir / f"{name}_ids.json"
        metadata_path = index_dir / f"{name}_index_metadata.json"
        missing = [path.name for path in (index_path, ids_path, metadata_path) if not path.is_file()]
        if missing:
            if required:
                self.blockers.append(f"missing {name} index artifact(s): {', '.join(missing)}")
            return None

        try:
            import faiss

            index = faiss.read_index(str(index_path))
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
                self.blockers.append(f"invalid {name} ids: {ids_path.name}")
                return None
            if not isinstance(metadata, dict):
                self.blockers.append(f"invalid {name} metadata: {metadata_path.name}")
                return None
            if int(index.ntotal) != len(ids):
                self.blockers.append(
                    f"invalid {name} index: vector count {index.ntotal} "
                    f"does not match ids count {len(ids)}"
                )
                return None
            return LoadedVectorIndex(name=name, index=index, ids=ids, metadata=metadata)
        except Exception as exc:
            self.blockers.append(f"failed to load {name} index {index_path.name}: {exc}")
            return None

    def _load_root_text_index(self, name: str) -> LoadedVectorIndex | None:
        index_path = self.working_dir / f"index_{name}.bin"
        corpus_path = self.working_dir / f"corpus_{name}.npy"
        metadata_path = self.working_dir / f"{name}_index_metadata.json"
        missing = [path.name for path in (index_path, metadata_path) if not path.is_file()]
        if missing:
            return None
        try:
            import faiss

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                logger.warning("[MM_KB] Invalid %s root metadata: %s", name, metadata_path.name)
                return None
            ids = self._root_text_index_ids(name, metadata)
            if not ids:
                logger.warning("[MM_KB] Missing %s root ids in %s; disabling root index", name, metadata_path.name)
                return None

            try:
                index = faiss.read_index(str(index_path))
            except Exception as exc:
                repaired = self._repair_root_text_index_from_corpus(
                    name=name,
                    index_path=index_path,
                    corpus_path=corpus_path,
                    metadata=metadata,
                    ids=ids,
                    reason=str(exc),
                )
                if repaired is not None:
                    return repaired
                logger.warning(
                    "[MM_KB] Disabled corrupt %s root index %s after failed repair: %s",
                    name,
                    index_path.name,
                    exc,
                )
                return None
            if int(index.ntotal) != len(ids):
                repaired = self._repair_root_text_index_from_corpus(
                    name=name,
                    index_path=index_path,
                    corpus_path=corpus_path,
                    metadata=metadata,
                    ids=ids,
                    reason=(
                        f"vector count {index.ntotal} does not match ids count {len(ids)}"
                    ),
                )
                if repaired is not None:
                    return repaired
                logger.warning(
                    "[MM_KB] Disabled stale %s root index %s: vector count %s does not match ids count %s",
                    name,
                    index_path.name,
                    index.ntotal,
                    len(ids),
                )
                return None
            return LoadedVectorIndex(name=name, index=index, ids=ids, metadata=metadata)
        except Exception as exc:
            logger.warning(
                "[MM_KB] Disabled %s root index %s after load error: %s",
                name,
                index_path.name,
                exc,
            )
            return None

    def _root_text_index_ids(self, name: str, metadata: dict[str, Any]) -> list[str]:
        ids = metadata.get("ids", [])
        if isinstance(ids, list) and all(isinstance(item, str) for item in ids):
            return ids
        if name == "entity":
            entities = self._read_optional_json(
                self.working_dir / "kv_store_entities.json",
                "root entity ids",
            )
            if isinstance(entities, dict):
                return [
                    entity_name
                    for entity in entities.values()
                    if isinstance(entity, dict)
                    and isinstance((entity_name := entity.get("entity_name")), str)
                    and entity_name.strip()
                ]
        if name == "hyperedge":
            hyperedges = self._read_optional_json(
                self.working_dir / "kv_store_hyperedges.json",
                "root hyperedge ids",
            )
            if isinstance(hyperedges, dict):
                values = []
                for hyperedge in hyperedges.values():
                    if not isinstance(hyperedge, dict):
                        continue
                    content = hyperedge.get("content") or hyperedge.get("hyperedge_name")
                    if isinstance(content, str) and content.strip():
                        values.append(content)
                return values
        return []

    def _repair_root_text_index_from_corpus(
        self,
        *,
        name: str,
        index_path: Path,
        corpus_path: Path,
        metadata: dict[str, Any],
        ids: list[str],
        reason: str,
    ) -> LoadedVectorIndex | None:
        if not corpus_path.is_file():
            logger.warning(
                "[MM_KB] Cannot repair %s root index without corpus %s: %s",
                name,
                corpus_path.name,
                reason,
            )
            return None
        lock_path = index_path.with_suffix(index_path.suffix + ".repair.lock")
        with self._root_index_file_lock(lock_path):
            try:
                import faiss

                try:
                    index = faiss.read_index(str(index_path))
                    if int(index.ntotal) == len(ids):
                        return LoadedVectorIndex(name=name, index=index, ids=ids, metadata=metadata)
                except Exception:
                    pass

                corpus = np.load(corpus_path, mmap_mode="r")
                if getattr(corpus, "ndim", 0) != 2:
                    logger.warning("[MM_KB] Cannot repair %s root index: corpus is not 2-D", name)
                    return None
                if int(corpus.shape[0]) != len(ids):
                    logger.warning(
                        "[MM_KB] Cannot repair %s root index: corpus rows %s do not match ids %s",
                        name,
                        corpus.shape[0],
                        len(ids),
                    )
                    return None

                dimension = int(corpus.shape[1])
                repaired = faiss.index_factory(dimension, "Flat", faiss.METRIC_INNER_PRODUCT)
                if corpus.shape[0] > 0:
                    repaired.add(np.asarray(corpus, dtype=np.float32))
                tmp_index_path = index_path.with_suffix(index_path.suffix + f".tmp.{os.getpid()}")
                faiss.write_index(repaired, str(tmp_index_path))
                os.replace(tmp_index_path, index_path)
                logger.warning(
                    "[MM_KB] Rebuilt %s root FAISS index from %s after load failure: %s",
                    name,
                    corpus_path.name,
                    reason,
                )
                return LoadedVectorIndex(name=name, index=repaired, ids=ids, metadata=metadata)
            except Exception as exc:
                logger.warning("[MM_KB] Failed to repair %s root index: %s", name, exc)
                return None

    @contextmanager
    def _root_index_file_lock(self, lock_path: Path):
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass

    def _load_or_build_bge_graph_indexes(self) -> LoadedBGEGraphIndexes | None:
        if os.getenv("EVOGRAPH_MM_ENABLE_BGE_TEXT", "1").strip() in {"0", "false", "False"}:
            return None

        graph_dir = self.working_dir / "mm_store" / "bge_graph"
        paths = {
            "entity_index": graph_dir / "index_entity.bin",
            "entity_corpus": graph_dir / "corpus_entity.npy",
            "hyperedge_index": graph_dir / "index_hyperedge.bin",
            "hyperedge_corpus": graph_dir / "corpus_hyperedge.npy",
            "metadata": graph_dir / "metadata.json",
        }
        entity_corpus, entity_embedding_texts, hyperedge_corpus = self._bge_graph_corpora()
        if not entity_corpus and not hyperedge_corpus:
            self.bge_graph_blockers.append("no KV entity/hyperedge records available for BGE graph index")
            return None

        if all(path.is_file() for path in paths.values()):
            loaded = self._load_bge_graph_indexes(paths, entity_corpus, hyperedge_corpus)
            if loaded is not None:
                return loaded

        try:
            return self._build_bge_graph_indexes(
                paths,
                entity_corpus,
                entity_embedding_texts,
                hyperedge_corpus,
            )
        except Exception as exc:
            self.bge_graph_blockers.append(f"failed to build BGE graph indexes: {exc}")
            return None

    def _load_bge_graph_indexes(
        self,
        paths: dict[str, Path],
        entity_corpus: list[str],
        hyperedge_corpus: list[str],
    ) -> LoadedBGEGraphIndexes | None:
        try:
            import faiss

            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                self.bge_graph_blockers.append("invalid BGE graph metadata")
                return None
            entity_index = faiss.read_index(str(paths["entity_index"]))
            hyperedge_index = faiss.read_index(str(paths["hyperedge_index"]))
            entity_vectors = np.load(paths["entity_corpus"], mmap_mode="r")
            hyperedge_vectors = np.load(paths["hyperedge_corpus"], mmap_mode="r")
            if (
                int(entity_index.ntotal) != len(entity_corpus)
                or entity_vectors.shape[0] != len(entity_corpus)
                or metadata.get("entity_count") != len(entity_corpus)
            ):
                self.bge_graph_blockers.append("stale BGE entity graph index; rebuilding")
                return None
            if (
                int(hyperedge_index.ntotal) != len(hyperedge_corpus)
                or hyperedge_vectors.shape[0] != len(hyperedge_corpus)
                or metadata.get("hyperedge_count") != len(hyperedge_corpus)
            ):
                self.bge_graph_blockers.append("stale BGE hyperedge graph index; rebuilding")
                return None
            return LoadedBGEGraphIndexes(
                entity_index=entity_index,
                entity_corpus=entity_corpus,
                hyperedge_index=hyperedge_index,
                hyperedge_corpus=hyperedge_corpus,
                metadata=metadata,
            )
        except Exception as exc:
            self.bge_graph_blockers.append(f"failed to load BGE graph indexes: {exc}")
            return None

    def _build_bge_graph_indexes(
        self,
        paths: dict[str, Path],
        entity_corpus: list[str],
        entity_embedding_texts: list[str],
        hyperedge_corpus: list[str],
    ) -> LoadedBGEGraphIndexes:
        from agent.tool.tools.bge_model_manager import encode_texts_safe

        import faiss

        paths["metadata"].parent.mkdir(parents=True, exist_ok=True)
        batch_size = int(os.getenv("EVOGRAPH_MM_BGE_TEXT_BATCH_SIZE", "4"))
        entity_vectors = self._encode_bge_corpus(
            entity_embedding_texts,
            encode_texts_safe,
            batch_size,
        )
        hyperedge_vectors = self._encode_bge_corpus(hyperedge_corpus, encode_texts_safe, batch_size)

        entity_index = faiss.index_factory(
            DEFAULT_BGE_TEXT_DIMENSION,
            "Flat",
            faiss.METRIC_INNER_PRODUCT,
        )
        hyperedge_index = faiss.index_factory(
            DEFAULT_BGE_TEXT_DIMENSION,
            "Flat",
            faiss.METRIC_INNER_PRODUCT,
        )
        if entity_vectors.shape[0] > 0:
            entity_index.add(entity_vectors)
        if hyperedge_vectors.shape[0] > 0:
            hyperedge_index.add(hyperedge_vectors)

        faiss.write_index(entity_index, str(paths["entity_index"]))
        faiss.write_index(hyperedge_index, str(paths["hyperedge_index"]))
        np.save(paths["entity_corpus"], entity_vectors)
        np.save(paths["hyperedge_corpus"], hyperedge_vectors)
        metadata = {
            "provider": "bge",
            "dimension": DEFAULT_BGE_TEXT_DIMENSION,
            "model": os.getenv("BGE_MODEL_PATH") or os.getenv("BGE_MODEL_NAME") or "BAAI/bge-large-en-v1.5",
            "entity_count": len(entity_corpus),
            "hyperedge_count": len(hyperedge_corpus),
            "entity_source": "kv_store_entities.json[*].content",
            "hyperedge_source": "active kv_store_hyperedges.json[*].content",
            "layout": "2wiki_compatible_sidecar",
        }
        paths["metadata"].write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return LoadedBGEGraphIndexes(
            entity_index=entity_index,
            entity_corpus=entity_corpus,
            hyperedge_index=hyperedge_index,
            hyperedge_corpus=hyperedge_corpus,
            metadata=metadata,
        )

    def _encode_bge_corpus(
        self,
        texts: list[str],
        encode_texts_safe: Callable[..., np.ndarray],
        batch_size: int,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, DEFAULT_BGE_TEXT_DIMENSION), dtype=np.float32)
        vectors = []
        for start in range(0, len(texts), max(1, batch_size)):
            vectors.append(
                encode_texts_safe(
                    texts[start : start + max(1, batch_size)],
                    target_dimension=DEFAULT_BGE_TEXT_DIMENSION,
                )
            )
        matrix = np.vstack(vectors).astype(np.float32, copy=False)
        return matrix

    def _bge_graph_corpora(self) -> tuple[list[str], list[str], list[str]]:
        entities = self._read_optional_json(self.working_dir / "kv_store_entities.json", "BGE entities")
        hyperedges = self._read_optional_json(
            self.working_dir / "kv_store_hyperedges.json",
            "BGE hyperedges",
        )
        entity_corpus = []
        entity_embedding_texts = []
        if isinstance(entities, dict):
            for entity in entities.values():
                if not isinstance(entity, dict):
                    continue
                entity_name = entity.get("entity_name")
                content = entity.get("content")
                if isinstance(entity_name, str) and entity_name.strip():
                    entity_corpus.append(entity_name)
                    entity_embedding_texts.append(
                        content
                        if isinstance(content, str) and content.strip()
                        else entity_name
                    )
        hyperedge_corpus = []
        if isinstance(hyperedges, dict):
            for hyperedge in hyperedges.values():
                if not isinstance(hyperedge, dict) or hyperedge.get("deleted", False):
                    continue
                content = hyperedge.get("content") or hyperedge.get("hyperedge_name")
                if isinstance(content, str) and content.strip():
                    hyperedge_corpus.append(content)
        return entity_corpus, entity_embedding_texts, hyperedge_corpus

    def _register_index_metadata_records(self, loaded: LoadedVectorIndex) -> None:
        contents = loaded.metadata.get("contents", [])
        if not isinstance(contents, list):
            contents = []
        metadata_records = loaded.metadata.get("records", [])
        if not isinstance(metadata_records, list):
            metadata_records = []
        for index, embedding_id in enumerate(loaded.ids):
            content = contents[index] if index < len(contents) else embedding_id
            record = (
                dict(metadata_records[index])
                if index < len(metadata_records)
                and isinstance(metadata_records[index], dict)
                else {}
            )
            record["contents"] = str(
                record.get("contents") or record.get("content") or content
            )
            source_metadata = record.get("source_metadata")
            record["source_metadata"] = (
                dict(source_metadata) if isinstance(source_metadata, dict) else {}
            )
            record["modality"] = loaded.name
            self.embedding_records[embedding_id] = record

    def _build_embedding_record_map(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = dict(self.embedding_records)
        text_by_id = {
            doc.get("text_doc_id"): doc
            for doc in self.text_documents
            if isinstance(doc.get("text_doc_id"), str)
        }
        visual_by_image_id = {
            record.get("image_id"): record
            for record in self.visual_records
            if isinstance(record.get("image_id"), str)
        }
        visual_by_id = {
            record.get("visual_record_id"): record
            for record in self.visual_records
            if isinstance(record.get("visual_record_id"), str)
        }
        text_by_data_id = {
            doc.get("data_id"): doc
            for doc in self.text_documents
            if isinstance(doc.get("data_id"), str)
        }

        for doc_id, doc in text_by_id.items():
            records[_text_embedding_id(doc_id)] = doc
        for image_id, record in visual_by_image_id.items():
            records[_visual_embedding_id(image_id)] = record
        for data_id, doc in text_by_data_id.items():
            records[_fused_embedding_id(data_id)] = doc

        for link in self.links:
            source_id = link.get("source_id")
            target_id = link.get("target_id")
            relation = link.get("relation")
            if not isinstance(target_id, str):
                continue
            if relation == "has_text_embedding" and source_id in text_by_id:
                records[target_id] = text_by_id[source_id]
            elif relation == "has_visual_embedding":
                visual_record = visual_by_id.get(source_id)
                if visual_record is not None:
                    records[target_id] = visual_record
            elif relation == "has_fused_embedding" and link.get("data_id") in text_by_data_id:
                records[target_id] = text_by_data_id[link["data_id"]]
        return records

    def _load_encoder(self) -> Any:
        try:
            dimension = self._runtime_embedding_dimension()
            if self.encoder_factory is not None:
                try:
                    encoder = self.encoder_factory(
                        self.model_path,
                        embedding_dim=dimension,
                    )
                except TypeError:
                    encoder = self.encoder_factory(self.model_path)
            else:
                return None
            if encoder is None:
                self.blockers.append("failed to load encoder: encoder factory returned None")
            return encoder
        except Exception as exc:
            self.blockers.append(f"failed to load encoder: {exc}")
            return None

    def _runtime_encoder_enabled(self) -> bool:
        if self.encoder_factory is not None:
            return True
        return os.getenv("EVOGRAPH_MM_ENABLE_RUNTIME_ENCODER", "0").strip() in {
            "1",
            "true",
            "True",
        }

    def _get_rag(self) -> Any | None:
        if self.rag is not None:
            return self.rag
        if self.rag_factory is None and not self._has_graphr1_context_artifacts():
            return None
        try:
            if self.rag_factory is not None:
                try:
                    self.rag = self.rag_factory(self.working_dir)
                except TypeError:
                    self.rag = self.rag_factory()
            else:
                self.rag = GraphR1(working_dir=str(self.working_dir))
            return self.rag
        except Exception:
            return None

    def _has_graphr1_context_artifacts(self) -> bool:
        required = (
            "graph_chunk_entity_relation.graphml",
            "kv_store_text_chunks.json",
            "kv_store_entities.json",
            "kv_store_hyperedges.json",
        )
        return all((self.working_dir / name).is_file() for name in required)

    def _graphr1_context_results(
        self,
        *,
        query: str,
        entity_hits: list[dict[str, Any]],
        hyperedge_hits: list[dict[str, Any]],
        rag_top_k: int,
    ) -> list[dict[str, Any]]:
        rag = self._get_rag()
        if rag is None:
            return []
        entity_match = [
            value
            for hit in entity_hits
            if (value := _optional_string(hit.get("id"))) is not None
        ]
        hyperedge_match = [
            value
            for hit in hyperedge_hits
            if (value := _optional_string(hit.get("id"))) is not None
        ]
        matched_hyperedge_knowledge = [
            value
            for hit in hyperedge_hits
            if (value := _optional_string(hit.get("<knowledge>"))) is not None
        ]
        try:
            result = _run_async(
                rag.aquery(
                    query,
                    param=QueryParam(
                        only_need_context=True,
                        top_k=rag_top_k,
                        max_token_for_text_unit=16000,
                        max_token_for_global_context=16000,
                        max_token_for_local_context=16000,
                    ),
                    entity_match=entity_match,
                    hyperedge_match=hyperedge_match,
                )
            )
        except Exception:
            return []
        return _merge_graphr1_context_results(result, matched_hyperedge_knowledge)

    def _runtime_embedding_dimension(self) -> int:
        dimensions: set[int] = set()
        for loaded in self.indexes.values():
            dimension = _embedding_dimension(loaded.metadata, default=0)
            if dimension > 0:
                dimensions.add(dimension)
            index_dimension = getattr(loaded.index, "d", None)
            if isinstance(index_dimension, int) and index_dimension > 0:
                dimensions.add(index_dimension)
        if len(dimensions) == 1:
            return next(iter(dimensions))
        return _embedding_dimension(self.metadata)

    def _index_vector_count(self, name: str) -> int:
        loaded = self.indexes.get(name)
        if loaded is None:
            return 0
        return int(getattr(loaded.index, "ntotal", len(loaded.ids)))

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        root = _optional_string(self.metadata.get("root"))
        bases: list[Path] = []
        root_path = Path(root) if root else None
        if root_path is not None and root_path.is_absolute():
            bases.append(root_path)
        elif root_path is not None:
            bases.extend(base / root_path for base in self._stable_workspace_roots())
            bases.append(Path.cwd() / root_path)
        else:
            bases.append(Path.cwd())
        bases.append(self.working_dir)

        seen: set[str] = set()
        for base in bases:
            key = str(base.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            candidate = base / path
            if candidate.is_file():
                return candidate
        return self.working_dir / path

    def _image_id_for_path(self, image_path: str) -> str | None:
        normalized = str(Path(image_path).resolve(strict=False))
        for image_id, payload in self.image_anchor_lookup.items():
            candidate = payload.get("image_path") if isinstance(payload, dict) else None
            if isinstance(candidate, str) and str(Path(candidate).resolve(strict=False)) == normalized:
                return image_id
        return None

    def _allows_missing_image_files(self) -> bool:
        dataset = _optional_string(self.metadata.get("dataset")) or self.dataset
        return dataset == "synthetic-mm" and isinstance(self.encoder, MockEmbeddingEncoder)

    def _stable_workspace_roots(self) -> list[Path]:
        roots: list[Path] = []
        dataset = _optional_string(self.metadata.get("dataset")) or self.dataset
        subset = _optional_string(self.metadata.get("subset")) or self.subset
        output_root = _optional_string(self.metadata.get("output_root"))
        working_dir = self.working_dir.resolve(strict=False)

        if dataset and subset:
            if output_root:
                output_path = Path(output_root)
                if output_path.is_absolute():
                    expected = (output_path / dataset / subset).resolve(strict=False)
                    if expected == working_dir:
                        roots.append(output_path.parent)
                else:
                    suffix = output_path.parts + (dataset, subset)
                    if _has_path_suffix(working_dir, suffix):
                        roots.append(working_dir.parents[len(suffix) - 1])
            if (
                working_dir.name == subset
                and working_dir.parent.name == dataset
                and working_dir.parent.parent.name == "expr_mm"
            ):
                roots.append(working_dir.parent.parent.parent)

        deduped: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(root)
        return deduped

    def _search_index(
        self,
        name: str,
        vector: Any,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        loaded = self.indexes.get(name)
        if loaded is None:
            return []

        query_vector = np.asarray([vector], dtype=np.float32)
        scores, hits = loaded.index.search(query_vector, int(top_k))
        results = []
        for rank, (hit, score) in enumerate(zip(hits[0].tolist(), scores[0].tolist())):
            if hit == -1:
                continue
            if hit < 0 or hit >= len(loaded.ids):
                continue
            results.append(
                self._format_hit(
                    modality=name,
                    embedding_id=loaded.ids[hit],
                    score=float(score),
                    rank=rank,
                )
            )
        if name == "hyperedge":
            results = _sort_hyperedge_hits_by_search_contract(results)
        return results

    def _search_bge_graph_indexes(
        self,
        query: str,
        *,
        entity_top_k: int,
        hyperedge_top_k: int,
        rag_top_k: int,
    ) -> list[dict[str, Any]]:
        if self.bge_graph_indexes is None:
            return []
        try:
            from agent.tool.tools.bge_model_manager import encode_texts_safe

            query_vector = encode_texts_safe(
                [query],
                target_dimension=DEFAULT_BGE_TEXT_DIMENSION,
            ).astype(np.float32, copy=False)
            entity_scores, entity_hits = self.bge_graph_indexes.entity_index.search(
                query_vector,
                int(max(0, entity_top_k)),
            )
            hyperedge_scores, hyperedge_hits = self.bge_graph_indexes.hyperedge_index.search(
                query_vector,
                int(max(0, hyperedge_top_k)),
            )
        except Exception as exc:
            self.bge_graph_blockers.append(f"BGE graph query failed: {exc}")
            return []

        entity_results = self._format_bge_graph_hits(
            modality="bge_entity",
            hits=entity_hits[0].tolist() if len(entity_hits) else [],
            scores=entity_scores[0].tolist() if len(entity_scores) else [],
            corpus=self.bge_graph_indexes.entity_corpus,
        )
        hyperedge_results = self._format_bge_graph_hits(
            modality="bge_hyperedge",
            hits=hyperedge_hits[0].tolist() if len(hyperedge_hits) else [],
            scores=hyperedge_scores[0].tolist() if len(hyperedge_scores) else [],
            corpus=self.bge_graph_indexes.hyperedge_corpus,
        )
        if rag_top_k > 0:
            graph_context = self._graphr1_context_results(
                query=query,
                entity_hits=entity_results,
                hyperedge_hits=hyperedge_results,
                rag_top_k=rag_top_k,
            )
            if graph_context:
                return graph_context
        return [*entity_results, *hyperedge_results]

    def _format_bge_graph_hits(
        self,
        *,
        modality: str,
        hits: list[int],
        scores: list[float],
        corpus: list[str],
    ) -> list[dict[str, Any]]:
        results = []
        for rank, (hit, score) in enumerate(zip(hits, scores)):
            if hit == -1 or hit < 0 or hit >= len(corpus):
                continue
            text = corpus[hit]
            results.append(
                {
                    "id": text,
                    "modality": modality,
                    "<knowledge>": text,
                    "<coherence>": float(score),
                    "score": float(score),
                    "provenance": {
                        "dataset": self.dataset,
                        "subset": self.subset,
                        "working_dir": str(self.working_dir),
                        "index": modality,
                        "rank": rank,
                        "embedding_id": text,
                        "store_path": (
                            "kv_store_entities.json"
                            if modality == "bge_entity"
                            else "kv_store_hyperedges.json"
                        ),
                    },
                }
            )
        return results

    def _expand_image_hits(self, image_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        for hit in image_hits:
            image_id = _optional_string(hit.get("image_id"))
            if image_id is None:
                image_id = _image_id_from_embedding_id(str(hit.get("id") or ""))
            if image_id in self.image_to_text_lookup:
                expanded.extend(
                    self._image_to_text_results(
                        image_id,
                        score=float(hit.get("score", 0.0)),
                        rank=int(hit.get("provenance", {}).get("rank", 0))
                        if isinstance(hit.get("provenance"), dict)
                        else 0,
                    )
                )
            else:
                expanded.append(hit)
        return expanded

    def _image_to_text_results(
        self,
        image_id: str,
        score: float = 1.0,
        rank: int = 0,
    ) -> list[dict[str, Any]]:
        lookup = self.image_to_text_lookup.get(image_id)
        if not isinstance(lookup, dict):
            return []
        anchor = self.image_anchor_lookup.get(image_id, {})
        image_path = anchor.get("image_path") if isinstance(anchor, dict) else None
        knowledge = str(lookup.get("fallback_text") or "")
        text_doc_ids = lookup.get("text_doc_ids", [])
        data_ids = lookup.get("data_ids", [])
        first_text_doc_id = text_doc_ids[0] if isinstance(text_doc_ids, list) and text_doc_ids else None
        text_doc = self._text_document_by_id(first_text_doc_id)
        source_metadata = text_doc.get("source_metadata") if isinstance(text_doc, dict) else {}
        if not isinstance(source_metadata, dict):
            source_metadata = {}
        return [
            {
                "<knowledge>": knowledge,
                "<coherence>": float(score),
                "modality": "image_to_text",
                "score": float(score),
                "data_id": data_ids[0] if isinstance(data_ids, list) and data_ids else None,
                "image_id": image_id,
                "image_path": image_path,
                "canonical_entity": lookup.get("canonical_entity"),
                "text_doc_id": first_text_doc_id,
                "text_doc_ids": text_doc_ids,
                "data_ids": data_ids,
                "source_metadata": source_metadata,
                "provenance": {
                    "dataset": self.dataset,
                    "source_subset": self.subset,
                    "graph": "graph_mm_entity_relation.graphml",
                    "rank": rank,
                },
            }
        ]

    def _entity_fusion_results(
        self,
        *,
        image_hits: list[dict[str, Any]],
        entity_hits: list[dict[str, Any]],
        hyperedge_hits: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        image_entities = self._aggregate_image_hits_by_entity(image_hits)
        entity_text_entities = self._aggregate_graphr1_hits_by_entity("entity", entity_hits)
        hyperedge_text_entities = self._aggregate_graphr1_hits_by_entity(
            "hyperedge",
            hyperedge_hits,
        )
        ranked = _merge_entity_scores(
            [
                (image_entities, 0.2),
                (entity_text_entities, 0.55),
                (hyperedge_text_entities, 0.25),
            ]
        )
        results = []
        for rank, item in enumerate(ranked[:top_k]):
            results.append(
                {
                    "id": f"entity_fusion::{item['wikipedia_url']}",
                    "modality": "entity_fusion",
                    "<knowledge>": item.get("wikipedia_title") or item["wikipedia_url"],
                    "<coherence>": item["score"],
                    "score": item["score"],
                    "canonical_entity": item.get("wikipedia_title"),
                    "wikipedia_url": item["wikipedia_url"],
                    "components": item.get("components", {}),
                    "source_metadata": {
                        "wikipedia_title": item.get("wikipedia_title"),
                        "wikipedia_url": item["wikipedia_url"],
                        "fusion_weights": {
                            "image": 0.2,
                            "entity": 0.55,
                            "hyperedge": 0.25,
                        },
                    },
                    "provenance": {
                        "dataset": self.dataset,
                        "subset": self.subset,
                        "working_dir": str(self.working_dir),
                        "index": "entity_fusion",
                        "rank": rank,
                        "candidate_sources": item.get("candidate_sources", []),
                    },
                }
            )
        return results

    def _visual_entity_results(
        self,
        *,
        image_hits: list[dict[str, Any]],
        entity_hits: list[dict[str, Any]] | None = None,
        hyperedge_hits: list[dict[str, Any]] | None = None,
        top_k: int,
        hyperedge_query_vector: Any | None = None,
        hyperedge_top_k: int = DEFAULT_VISUAL_HYPEREDGE_TOP_K,
        hyperedge_candidate_top_k: int = 1000,
        hyperedge_min_coherence: float = DEFAULT_VISUAL_HYPEREDGE_MIN_COHERENCE,
    ) -> list[dict[str, Any]]:
        hyperedge_hits = list(hyperedge_hits or [])
        if (
            hyperedge_query_vector is not None
            and hyperedge_candidate_top_k > len(hyperedge_hits)
            and "hyperedge" in self.indexes
        ):
            hyperedge_hits = self._search_index(
                "hyperedge",
                hyperedge_query_vector,
                max(hyperedge_top_k, hyperedge_candidate_top_k),
            )
        if entity_hits or hyperedge_hits:
            image_entities = self._aggregate_image_hits_by_entity(image_hits)
            entity_text_entities = self._aggregate_graphr1_hits_by_entity(
                    "entity",
                    entity_hits or [],
                )
            hyperedge_text_entities = self._aggregate_graphr1_hits_by_entity(
                    "hyperedge",
                    hyperedge_hits or [],
                )
            ranked = _merge_entity_scores(
                [
                    (image_entities, 1.0),
                    (entity_text_entities, 1.0),
                    (hyperedge_text_entities, 1.0),
                ]
            )
        else:
            ranked = self._aggregate_image_hits_by_entity(image_hits)
        results = []
        top_score = float(ranked[0].get("score", 0.0)) if ranked else 0.0
        related_hyperedges, global_hyperedges = self._related_hyperedges_by_wikipedia_url(
            hyperedge_hits,
            hyperedge_top_k,
            hyperedge_min_coherence,
        )
        for rank, item in enumerate(ranked[:top_k]):
            title = item.get("wikipedia_title") or item["wikipedia_url"]
            raw_score = float(item.get("score", 0.0))
            if top_score > 0:
                confidence = raw_score / top_score
            else:
                confidence = raw_score
            confidence = max(0.0, min(1.0, confidence))
            description = self._entity_description(item["wikipedia_url"])
            results.append(
                {
                    "<coherence>": confidence,
                    "entity": title,
                    "image_path": item.get("image_path") or "",
                    "image_url": item.get("image_url") or "",
                    "description": description,
                    "related_hyperedges": _merge_related_hyperedges(
                        related_hyperedges.get(item["wikipedia_url"], []),
                        global_hyperedges,
                        hyperedge_top_k,
                    ),
                }
            )
        return results

    def _related_hyperedges_by_wikipedia_url(
        self,
        hyperedge_hits: list[dict[str, Any]],
        hyperedge_top_k: int,
        hyperedge_min_coherence: float,
    ) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
        if hyperedge_top_k <= 0:
            return {}, []
        sidecar = self.graphr1_hit_source_sidecar.get("hyperedge", {})
        if not isinstance(sidecar, dict):
            sidecar = {}
        by_url: dict[str, list[dict[str, Any]]] = {}
        seen_by_url: dict[str, set[str]] = {}
        global_entries: list[dict[str, Any]] = []
        global_seen: set[str] = set()
        for hit in hyperedge_hits:
            hit_id = _optional_string(hit.get("id"))
            if hit_id is None:
                continue
            text = _clean_hyperedge_text(str(hit.get("<knowledge>", "")))
            if not text:
                continue
            score = float(hit.get("score", 0.0))
            if score < hyperedge_min_coherence:
                continue
            entry = {
                "text": text,
                "<coherence>": score,
            }
            if len(global_entries) < hyperedge_top_k and text not in global_seen:
                global_seen.add(text)
                global_entries.append(entry)
            payload = sidecar.get(hit_id, {})
            if not isinstance(payload, dict):
                continue
            urls = payload.get("wikipedia_urls", [])
            if not isinstance(urls, list):
                continue
            for url_value in urls:
                url = _optional_string(url_value)
                if url is None:
                    continue
                seen = seen_by_url.setdefault(url, set())
                if text in seen:
                    continue
                entries = by_url.setdefault(url, [])
                if len(entries) >= hyperedge_top_k:
                    continue
                seen.add(text)
                entries.append(entry)
        return by_url, global_entries

    def _visual_query_vector(
        self,
        *,
        context_query: str,
        encoded_image_path: str,
        image_id: str = "",
    ) -> np.ndarray | None:
        if context_query:
            cached_vector = self._cached_query_fused_vector(image_id, context_query)
            if cached_vector is not None:
                return cached_vector
        if image_id:
            indexed_vector = self._indexed_vector("image", _visual_embedding_id(image_id))
            if indexed_vector is not None:
                return indexed_vector
        if not encoded_image_path:
            return None
        if context_query:
            try:
                return self.encoder.encode_fused(
                    [{"text": context_query, "image": encoded_image_path}],
                    instruction=FUSED_DOCUMENT_INSTRUCTION,
                )[0]
            except Exception:
                pass
        try:
            return self.encoder.encode_images(
                [encoded_image_path],
                instruction=IMAGE_DOCUMENT_INSTRUCTION,
            )[0]
        except Exception:
            return None

    def _indexed_vector(self, name: str, embedding_id: str) -> np.ndarray | None:
        loaded = self.indexes.get(name)
        if loaded is None:
            return None
        try:
            index_position = loaded.ids.index(embedding_id)
        except ValueError:
            return None
        try:
            return np.asarray(loaded.index.reconstruct(index_position), dtype=np.float32)
        except Exception:
            return None

    def _cached_query_fused_vector(
        self,
        image_id: str,
        context_query: str,
    ) -> np.ndarray | None:
        if self.query_fused_vectors is None or not image_id:
            return None
        vector_index = self.query_fused_lookup.get(
            (image_id, _normalize_query_cache_text(context_query))
        )
        if vector_index is None:
            return None
        try:
            return np.asarray(self.query_fused_vectors[vector_index], dtype=np.float32)
        except Exception:
            return None

    def _entity_description(self, wikipedia_url: str) -> str:
        fallback = ""
        for doc in self.text_documents:
            source_metadata = doc.get("source_metadata")
            if not isinstance(source_metadata, dict):
                continue
            url = _metadata_wikipedia_url(source_metadata)
            if url != wikipedia_url:
                continue
            description = _description_from_text_document(doc)
            if not description:
                continue
            if str(source_metadata.get("evidence_section_id")) == "0":
                return _short_summary(description, max_chars=260) or ""
            if not fallback:
                fallback = description
        return _short_summary(fallback, max_chars=260) or ""

    def _hyperedges_for_entity(
        self,
        wikipedia_url: str,
        query_vector: Any | None,
        *,
        top_k: int,
        candidate_top_k: int,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or query_vector is None:
            return []
        loaded = self.indexes.get("hyperedge")
        if loaded is None:
            return []

        query = np.asarray([query_vector], dtype=np.float32)
        scores, hits = loaded.index.search(query, int(max(candidate_top_k, top_k)))
        results = []
        seen: set[str] = set()
        for hit, score in zip(hits[0].tolist(), scores[0].tolist()):
            if hit == -1 or hit < 0 or hit >= len(loaded.ids):
                continue
            hyperedge_id = loaded.ids[hit]
            if hyperedge_id in seen:
                continue
            payload = self.graphr1_hit_source_sidecar.get("hyperedge", {}).get(
                hyperedge_id,
                {},
            )
            if not isinstance(payload, dict):
                continue
            urls = payload.get("wikipedia_urls", [])
            if not isinstance(urls, list) or wikipedia_url not in urls:
                continue
            text = self._hyperedge_text(hyperedge_id, hit)
            if not text:
                continue
            seen.add(hyperedge_id)
            results.append(
                {
                    "text": text,
                    "confidence": max(0.0, min(1.0, float(score))),
                }
            )
            if len(results) >= top_k:
                break
        return results

    def _hyperedge_text(self, hyperedge_id: str, index_position: int) -> str:
        metadata = self.indexes.get("hyperedge").metadata if "hyperedge" in self.indexes else {}
        contents = metadata.get("contents", [])
        if isinstance(contents, list) and index_position < len(contents):
            value = contents[index_position]
            if isinstance(value, str) and value.strip():
                return _clean_hyperedge_text(value)
        return _clean_hyperedge_text(hyperedge_id)

    def _aggregate_image_hits_by_entity(
        self,
        hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        scores: dict[str, dict[str, Any]] = {}
        for hit in hits:
            url = _optional_string(hit.get("wikipedia_url"))
            if url is None:
                continue
            score = float(hit.get("score", 0.0))
            item = scores.get(url)
            if item is None or score > item["score"]:
                embedding_id = _optional_string(hit.get("id"))
                source_record = (
                    self.embedding_records.get(embedding_id, {})
                    if embedding_id is not None
                    else {}
                )
                scores[url] = {
                    "wikipedia_url": url,
                    "wikipedia_title": _optional_string(hit.get("canonical_entity")) or "",
                    "score": score,
                    "source": "image",
                    "image_path": _optional_string(source_record.get("image_path")) or "",
                    "image_url": _optional_string(source_record.get("image_url")) or "",
                    "source_record": source_record,
                    "candidate_sources": [
                        _optional_string(hit.get("candidate_source")) or "unknown"
                    ],
                }
        return _sorted_entity_scores(scores)

    def _aggregate_graphr1_hits_by_entity(
        self,
        modality: str,
        hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        sidecar = self.graphr1_hit_source_sidecar.get(modality, {})
        if not isinstance(sidecar, dict):
            return []
        scores: dict[str, dict[str, Any]] = {}
        for hit in hits:
            hit_id = _optional_string(hit.get("id"))
            if hit_id is None:
                continue
            payload = sidecar.get(hit_id, {})
            if not isinstance(payload, dict):
                continue
            urls = payload.get("wikipedia_urls", [])
            titles = payload.get("wikipedia_titles", [])
            if not isinstance(urls, list):
                continue
            score = float(hit.get("score", 0.0))
            for index, url_value in enumerate(urls):
                url = _optional_string(url_value)
                if url is None:
                    continue
                title = (
                    _optional_string(titles[index])
                    if isinstance(titles, list) and index < len(titles)
                    else None
                )
                item = scores.get(url)
                if item is None or score > item["score"]:
                    scores[url] = {
                        "wikipedia_url": url,
                        "wikipedia_title": title or "",
                        "score": score,
                        "source": modality,
                        "candidate_sources": [modality],
                    }
        return _sorted_entity_scores(scores)

    def _text_document_by_id(self, text_doc_id: str | None) -> dict[str, Any]:
        if text_doc_id is None:
            return {}
        for doc in self.text_documents:
            if doc.get("text_doc_id") == text_doc_id:
                return doc
        return {}

    def _format_hit(
        self,
        modality: str,
        embedding_id: str,
        score: float,
        rank: int,
    ) -> dict[str, Any]:
        record = self.embedding_records.get(embedding_id, {})
        source_metadata = record.get("source_metadata")
        if not isinstance(source_metadata, dict):
            source_metadata = {}

        return {
            "id": embedding_id,
            "modality": modality,
            "<knowledge>": _knowledge_from_record(record, embedding_id),
            "<coherence>": score,
            "score": score,
            "deleted": bool(record.get("deleted", False)),
            "active": bool(record.get("active", not record.get("deleted", False))),
            "weight": _coerce_float(record.get("weight"), 1.0),
            "data_id": _optional_string(record.get("data_id")),
            "image_id": _optional_string(record.get("image_id")),
            "image_path": _optional_string(record.get("image_path")),
            "canonical_entity": _optional_string(
                record.get("canonical_entity")
                or source_metadata.get("canonical_entity")
                or _metadata_wikipedia_title(source_metadata)
            ),
            "wikipedia_url": _optional_string(
                record.get("wikipedia_url")
                or _metadata_wikipedia_url(source_metadata)
            ),
            "candidate_source": _optional_string(
                record.get("candidate_source")
                or source_metadata.get("candidate_source")
            ),
            "text_doc_id": _optional_string(record.get("text_doc_id")),
            "source_metadata": source_metadata,
            "provenance": {
                "dataset": self.dataset,
                "subset": self.subset,
                "working_dir": str(self.working_dir),
                "index": modality,
                "rank": rank,
                "embedding_id": embedding_id,
                "store_path": _store_path_for_modality(modality),
            },
        }


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _normalize_query_cache_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _knowledge_from_record(record: dict[str, Any], fallback: str) -> str:
    for key in ("contents", "content", "fallback_text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return fallback


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sort_hyperedge_hits_by_search_contract(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for index, item in enumerate(results):
        active = bool(item.get("active", not item.get("deleted", False)))
        score = _coerce_float(item.get("score"), 0.0)
        weight = _coerce_float(item.get("weight"), 1.0)
        scored.append(((1 if active else 0, score, weight, -index), item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored]


def _short_summary(value: Any, max_chars: int = 180) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _metadata_wikipedia_url(source_metadata: dict[str, Any]) -> str | None:
    url = _optional_string(source_metadata.get("wikipedia_url"))
    if url is not None:
        return url
    source_row = source_metadata.get("source_row")
    if isinstance(source_row, dict):
        url = _optional_string(
            source_row.get("wikipedia_url")
            or source_row.get("evidence_page_url")
        )
        if url is not None:
            return url
    original_metadata = source_metadata.get("original_metadata")
    if isinstance(original_metadata, dict):
        source_row = original_metadata.get("source_row")
        if isinstance(source_row, dict):
            url = _optional_string(
                source_row.get("wikipedia_url")
                or source_row.get("evidence_page_url")
            )
            if url is not None:
                return url
    source_rows = source_metadata.get("source_rows")
    if isinstance(source_rows, list) and source_rows:
        first_row = source_rows[0]
        if isinstance(first_row, dict):
            return _optional_string(first_row.get("wikipedia_url"))
    return None


def _metadata_wikipedia_title(source_metadata: dict[str, Any]) -> str | None:
    title = _optional_string(source_metadata.get("wikipedia_title"))
    if title is not None:
        return title
    source_row = source_metadata.get("source_row")
    if isinstance(source_row, dict):
        title = _optional_string(source_row.get("wikipedia_title"))
        if title is not None:
            return title
    original_metadata = source_metadata.get("original_metadata")
    if isinstance(original_metadata, dict):
        source_row = original_metadata.get("source_row")
        if isinstance(source_row, dict):
            title = _optional_string(source_row.get("wikipedia_title"))
            if title is not None:
                return title
    source_rows = source_metadata.get("source_rows")
    if isinstance(source_rows, list) and source_rows:
        first_row = source_rows[0]
        if isinstance(first_row, dict):
            return _optional_string(first_row.get("wikipedia_title"))
    return None


def _description_from_text_document(doc: dict[str, Any]) -> str:
    contents = doc.get("contents")
    if not isinstance(contents, str):
        return ""
    lines = [line.strip() for line in contents.splitlines() if line.strip()]
    if len(lines) > 2:
        return " ".join(lines[2:])
    return " ".join(lines)


def _clean_hyperedge_text(value: str) -> str:
    text = value.strip()
    if text.startswith("<hyperedge>"):
        text = text[len("<hyperedge>") :].strip()
    return text


def _merge_related_hyperedges(
    primary: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    candidates = sorted(
        [*primary, *fallback],
        key=lambda item: float(item.get("<coherence>", 0.0)),
        reverse=True,
    )
    for item in candidates:
        text = _optional_string(item.get("text"))
        if text is None or text in seen:
            continue
        seen.add(text)
        merged.append(item)
        if len(merged) >= top_k:
            break
    return merged


def _run_async(coro: Any) -> Any:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        temp_loop = asyncio.new_event_loop()
        try:
            return temp_loop.run_until_complete(coro)
        finally:
            temp_loop.close()
    return loop.run_until_complete(coro)


def _merge_graphr1_context_results(
    results: Any,
    matched_hyperedges: list[str],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in matched_hyperedges:
        normalized = _clean_hyperedge_text(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append({"<knowledge>": normalized, "<coherence>": 1.0})

    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict) or "<knowledge>" not in item:
                continue
            knowledge = _clean_hyperedge_text(str(item.get("<knowledge>", "")))
            if not knowledge or knowledge in seen:
                continue
            seen.add(knowledge)
            coherence = item.get("<coherence>", 0.5)
            try:
                coherence = float(coherence)
            except (TypeError, ValueError):
                coherence = 0.5
            merged.append({"<knowledge>": knowledge, "<coherence>": coherence})
    return merged


def _merge_entity_scores(
    scored_groups: list[tuple[list[dict[str, Any]], float]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group, weight in scored_groups:
        for item in group:
            url = _optional_string(item.get("wikipedia_url"))
            if url is None:
                continue
            target = merged.setdefault(
                url,
                {
                    "wikipedia_url": url,
                    "wikipedia_title": _optional_string(item.get("wikipedia_title")) or "",
                    "score": 0.0,
                    "components": {},
                    "candidate_sources": [],
                    "image_path": "",
                    "image_url": "",
                },
            )
            source = _optional_string(item.get("source")) or "unknown"
            raw_score = float(item.get("score", 0.0))
            target["score"] += raw_score * float(weight)
            target["components"][source] = max(
                target["components"].get(source, float("-inf")),
                raw_score,
            )
            for candidate_source in item.get("candidate_sources", []):
                if candidate_source and candidate_source not in target["candidate_sources"]:
                    target["candidate_sources"].append(candidate_source)
            if not target["wikipedia_title"]:
                target["wikipedia_title"] = _optional_string(item.get("wikipedia_title")) or ""
            if not target.get("image_path"):
                target["image_path"] = _optional_string(item.get("image_path")) or ""
            if not target.get("image_url"):
                target["image_url"] = _optional_string(item.get("image_url")) or ""
    return _sorted_entity_scores(merged)


def _sorted_entity_scores(scores: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        scores.values(),
        key=lambda item: (-float(item.get("score", 0.0)), str(item.get("wikipedia_url", ""))),
    )


def _image_id_from_embedding_id(embedding_id: str) -> str | None:
    prefix = "visual_embedding::"
    if embedding_id.startswith(prefix):
        return embedding_id[len(prefix) :]
    return None


def _has_path_suffix(path: Path, suffix: tuple[str, ...]) -> bool:
    if not suffix:
        return False
    return tuple(path.parts[-len(suffix) :]) == suffix


def normalize_query_rows(
    queries: list[str] | None = None,
    image_ids: list[str] | None = None,
    image_paths: list[str] | None = None,
) -> list[dict[str, str]]:
    query_values = _string_rows(queries)
    image_id_values = _string_rows(image_ids)
    image_path_values = _string_rows(image_paths)
    effective_count = max(
        _effective_length(query_values),
        _effective_length(image_id_values),
        _effective_length(image_path_values),
        0,
    )
    if effective_count == 0:
        if query_values or image_id_values or image_path_values:
            raise MMKBRequestError(
                "at least one non-blank query, image_id, or image_path is required"
            )
        raise MMKBRequestError(
            "at least one query, image_id, or image_path is required"
        )

    query_values = _broadcast(query_values, effective_count, "queries")
    image_id_values = _broadcast(image_id_values, effective_count, "image_ids")
    image_path_values = _broadcast(image_path_values, effective_count, "image_paths")
    rows = [
        {
            "query": query_values[index],
            "image_id": image_id_values[index],
            "image_path": image_path_values[index],
        }
        for index in range(effective_count)
    ]
    if any(not (row["query"] or row["image_id"] or row["image_path"]) for row in rows):
        raise MMKBRequestError(
            "at least one non-blank query, image_id, or image_path is required"
        )
    return rows


def _broadcast(values: list[str], effective_count: int, name: str) -> list[str]:
    if len(values) == 0 or _effective_length(values) == 0:
        return [""] * effective_count
    if len(values) == 1:
        return values * effective_count
    if len(values) == effective_count:
        return values
    raise MMKBRequestError(
        f"{name} length must be 1 or match effective query count"
    )


def _effective_length(values: list[str]) -> int:
    return len(values) if any(values) else 0


def _string_rows(values: list[str] | None) -> list[str]:
    if values is None:
        return []
    return [str(value).strip() for value in values]


def _store_path_for_modality(modality: str) -> str:
    if modality == "image":
        return "mm_store/visual/image_records.jsonl"
    return "graphr1_text/text_documents.jsonl"


def _embedding_dimension(metadata: dict[str, Any], default: int = 2048) -> int:
    value = metadata.get("embedding_dimension", default)
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _text_embedding_id(text_doc_id: str) -> str:
    return f"text_embedding::{text_doc_id}"


def _visual_embedding_id(image_id: str) -> str:
    return f"visual_embedding::{image_id}"


def _fused_embedding_id(data_id: str) -> str:
    return f"fused_embedding::{data_id}"


__all__ = [
    "FUSED_DOCUMENT_INSTRUCTION",
    "IMAGE_DOCUMENT_INSTRUCTION",
    "TEXT_DOCUMENT_INSTRUCTION",
    "TEXT_QUERY_INSTRUCTION",
    "MMKBRequestError",
    "LoadedVectorIndex",
    "MMKBRetriever",
    "normalize_query_rows",
]
