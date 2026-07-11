"""Cross-modal graph and lookup writers for multimodal KB artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx


@dataclass(frozen=True)
class MMGraphRecords:
    graph: nx.DiGraph
    image_anchor_lookup: dict[str, dict[str, Any]]
    canonical_entity_lookup: dict[str, dict[str, Any]]
    image_to_text_lookup: dict[str, dict[str, Any]]


def canonical_entity_label(record: dict[str, object]) -> str:
    source_row = _source_row(record)
    candidates = [
        source_row.get("wikipedia_title"),
        _first_non_url(record.get("context", [])),
        source_row.get("dataset_category_id"),
        record.get("answer"),
        record.get("image_id"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return "UNKNOWN"


def build_mm_graph_records(
    *,
    text_documents: list[dict[str, Any]],
    image_records: list[dict[str, Any]],
) -> MMGraphRecords:
    graph = nx.DiGraph()
    image_records_by_id = {
        str(record.get("image_id")): record
        for record in image_records
        if str(record.get("image_id") or "").strip()
    }
    image_anchor_lookup: dict[str, dict[str, Any]] = {}
    canonical_entity_lookup: dict[str, dict[str, Any]] = {}
    image_to_text_lookup: dict[str, dict[str, Any]] = {}

    for doc in text_documents:
        image_id = str(doc.get("image_id") or "").strip()
        data_id = str(doc.get("data_id") or "").strip()
        text_doc_id = str(doc.get("text_doc_id") or "").strip()
        if not image_id or not data_id or not text_doc_id:
            continue

        image_record = image_records_by_id.get(image_id, {})
        merged_record = {**image_record, **doc}
        canonical_entity = canonical_entity_label(merged_record)
        canonical_node_id = f'"{canonical_entity}"'
        image_anchor_id = f"image::{image_id}"
        text_node_id = f"textdoc::{text_doc_id}"
        sample_node_id = f"mm_hyperedge::{data_id}"
        image_path = str(
            doc.get("image_path") or image_record.get("image_path") or ""
        )

        graph.add_node(image_anchor_id, role="image_anchor", mod="vis", image_id=image_id)
        graph.add_node(
            canonical_node_id,
            role="entity",
            entity_type="CANONICAL_ENTITY",
            mod="cross_modal",
        )
        graph.add_node(
            text_node_id,
            role="text_evidence",
            mod="text",
            source_id=data_id,
            text_doc_id=text_doc_id,
        )
        graph.add_node(
            sample_node_id,
            role="sample_hyperedge",
            mod="cross_modal",
            source_id=data_id,
            image_id=image_id,
            canonical_entity=canonical_entity,
        )
        graph.add_edge(sample_node_id, image_anchor_id, relation="grounds_image")
        graph.add_edge(sample_node_id, canonical_node_id, relation="grounds_entity")
        graph.add_edge(sample_node_id, text_node_id, relation="grounds_text")

        image_lookup = image_anchor_lookup.setdefault(
            image_id,
            {
                "image_anchor_id": image_anchor_id,
                "canonical_entity": canonical_entity,
                "data_ids": [],
                "image_path": image_path,
            },
        )
        _append_unique(image_lookup["data_ids"], data_id)
        if not image_lookup.get("image_path") and image_path:
            image_lookup["image_path"] = image_path

        entity_lookup = canonical_entity_lookup.setdefault(
            canonical_entity,
            {
                "image_ids": [],
                "data_ids": [],
                "graph_entity_name": canonical_node_id,
                "text_doc_ids": [],
            },
        )
        _append_unique(entity_lookup["image_ids"], image_id)
        _append_unique(entity_lookup["data_ids"], data_id)
        _append_unique(entity_lookup["text_doc_ids"], text_doc_id)

        text_lookup = image_to_text_lookup.setdefault(
            image_id,
            {
                "canonical_entity": canonical_entity,
                "text_doc_ids": [],
                "data_ids": [],
                "fallback_text": "",
            },
        )
        _append_unique(text_lookup["text_doc_ids"], text_doc_id)
        _append_unique(text_lookup["data_ids"], data_id)
        fallback_text = str(doc.get("contents") or "")
        if not text_lookup["fallback_text"] and fallback_text:
            text_lookup["fallback_text"] = fallback_text

    return MMGraphRecords(
        graph=graph,
        image_anchor_lookup=image_anchor_lookup,
        canonical_entity_lookup=canonical_entity_lookup,
        image_to_text_lookup=image_to_text_lookup,
    )


def write_mm_graph(output_dir: str | Path, records: MMGraphRecords) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(records.graph, output_path / "graph_mm_entity_relation.graphml")
    _write_json(output_path / "image_anchor_lookup.json", records.image_anchor_lookup)
    _write_json(output_path / "canonical_entity_lookup.json", records.canonical_entity_lookup)
    _write_json(output_path / "image_to_text_lookup.json", records.image_to_text_lookup)
    _write_json(
        output_path / "graph_metadata.json",
        {
            "node_count": records.graph.number_of_nodes(),
            "edge_count": records.graph.number_of_edges(),
            "image_anchor_count": len(records.image_anchor_lookup),
            "canonical_entity_count": len(records.canonical_entity_lookup),
        },
    )


def _source_row(record: dict[str, object]) -> dict[str, object]:
    metadata = record.get("source_metadata")
    if isinstance(metadata, dict):
        source_row = metadata.get("source_row")
        if isinstance(source_row, dict):
            return source_row
        original_metadata = metadata.get("original_metadata")
        if isinstance(original_metadata, dict):
            nested_source_row = original_metadata.get("source_row")
            if isinstance(nested_source_row, dict):
                return nested_source_row
    direct_source_row = record.get("source_row")
    return direct_source_row if isinstance(direct_source_row, dict) else {}


def _first_non_url(value: object) -> str | None:
    values = value if isinstance(value, list) else [value]
    for item in values:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if candidate and not candidate.lower().startswith(("http://", "https://")):
            return candidate
    return None


def _append_unique(values: list[Any], value: Any) -> None:
    if value not in values:
        values.append(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "MMGraphRecords",
    "build_mm_graph_records",
    "canonical_entity_label",
    "write_mm_graph",
]
