import json
from pathlib import Path

import numpy as np
import pytest

from evograph_mm.kb.retrieval import MMKBRetriever


faiss = pytest.importorskip("faiss")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_faiss(path: Path, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(matrix.shape[1])
    if matrix.shape[0] > 0:
        index.add(matrix)
    faiss.write_index(index, str(path))


def _build_minimal_kb(root: Path) -> None:
    _write_json(root / "metadata.json", {"dataset": "E-VQA", "subset": "unit"})
    _write_json(root / "build_report.json", {})
    _write_json(
        root / "kv_store_entities.json",
        {"e1": {"entity_name": "Target Entity", "content": "target entity text"}},
    )
    _write_json(
        root / "kv_store_hyperedges.json",
        {"h1": {"content": "target hyperedge text"}},
    )
    _write_jsonl(root / "graphr1_text" / "text_documents.jsonl", [])
    _write_jsonl(
        root / "mm_store" / "visual" / "image_records.jsonl",
        [
            {
                "visual_record_id": "vr1",
                "image_id": "img1",
                "image_path": "images/img1.jpg",
                "canonical_entity": "Target Entity",
                "wikipedia_url": "https://example.com/target",
            }
        ],
    )
    _write_jsonl(root / "mm_store" / "links" / "links.jsonl", [])
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "images" / "img1.jpg").write_bytes(b"not a real image")
    _write_json(
        root / "mm_store" / "graph" / "image_to_text_lookup.json",
        {},
    )
    _write_json(
        root / "mm_store" / "graph" / "image_anchor_lookup.json",
        {"img1": {"image_path": "images/img1.jpg"}},
    )
    _write_json(
        root / "mm_store" / "graph" / "canonical_entity_lookup.json",
        {},
    )
    _write_json(
        root / "mm_store" / "graph" / "graphr1_hit_source_sidecar.json",
        {
            "entity": {
                "entity_hit": {
                    "wikipedia_urls": ["https://example.com/target"],
                    "wikipedia_titles": ["Target Entity"],
                }
            },
            "hyperedge": {},
        },
    )

    root_vector = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    _write_faiss(root / "index_entity.bin", root_vector)
    _write_json(
        root / "entity_index_metadata.json",
        {
            "ids": ["entity_hit"],
            "contents": ["Target Entity"],
            "records": [
                {
                    "content": "Target Entity",
                    "source_metadata": {
                        "wikipedia_url": "https://example.com/target",
                        "wikipedia_title": "Target Entity",
                    },
                }
            ],
        },
    )
    _write_faiss(root / "index_hyperedge.bin", root_vector)
    _write_json(
        root / "hyperedge_index_metadata.json",
        {"ids": ["hyperedge_hit"], "contents": ["target hyperedge text"], "records": []},
    )

    image_dir = root / "mm_store" / "indexing"
    _write_faiss(image_dir / "image_index.faiss", root_vector)
    _write_json(image_dir / "image_ids.json", ["visual_embedding::img1"])
    _write_json(image_dir / "image_index_metadata.json", {"embedding_dimension": 4})

    bge_dir = root / "mm_store" / "bge_graph"
    bge_vector = np.zeros((1, 1024), dtype=np.float32)
    bge_vector[0, 0] = 1.0
    _write_faiss(bge_dir / "index_entity.bin", bge_vector)
    _write_faiss(bge_dir / "index_hyperedge.bin", bge_vector)
    np.save(bge_dir / "corpus_entity.npy", bge_vector)
    np.save(bge_dir / "corpus_hyperedge.npy", bge_vector)
    _write_json(
        bge_dir / "metadata.json",
        {
            "provider": "bge",
            "dimension": 1024,
            "entity_count": 1,
            "hyperedge_count": 1,
        },
    )


def test_retriever_uses_bge_and_faiss_without_runtime_gme(tmp_path, monkeypatch):
    _build_minimal_kb(tmp_path)

    def fake_bge_encode(texts, target_dimension=1024):
        vectors = np.zeros((len(texts), target_dimension), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors

    monkeypatch.setattr(
        "agent.tool.tools.bge_model_manager.encode_texts_safe",
        fake_bge_encode,
    )

    retriever = MMKBRetriever(
        working_dir=tmp_path,
        model_path=tmp_path / "missing-gme",
        dataset="E-VQA",
        subset="unit",
    )
    status = retriever.load()

    assert status["status"] == "ready"
    assert status["model_loaded"] is False
    assert status["runtime_encoder_enabled"] is False

    text_payload = retriever.search(
        queries=["target"],
        rag_top_k=0,
        entity_top_k=1,
        hyperedge_top_k=1,
    )[0]
    assert "error" not in text_payload
    assert {item["modality"] for item in text_payload["results"]} == {
        "bge_entity",
        "bge_hyperedge",
    }

    image_payload = retriever.search(
        queries=["<img>"],
        image_ids=["img1"],
        context_queries=["target"],
        visual_entity_top_k=1,
        visual_hyperedge_top_k=0,
    )[0]
    assert "error" not in image_payload
    assert image_payload["results"][0]["entity"] == "Target Entity"
