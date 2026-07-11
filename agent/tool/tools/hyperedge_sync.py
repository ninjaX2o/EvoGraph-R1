from datetime import datetime
import logging
from typing import Any, Dict, Optional

from graphr1.graphr1 import always_get_an_event_loop
from graphr1.operate import _add_ai_metadata_to_hyperedge

logger = logging.getLogger(__name__)

HYPEREDGE_PREFIX = '<hyperedge>"'
HYPEREDGE_SUFFIX = '"'


def normalize_hyperedge_content(raw: str) -> str:
    if raw is None:
        return ""
    content = str(raw).strip()
    if content.startswith(HYPEREDGE_PREFIX) and content.endswith(HYPEREDGE_SUFFIX):
        return content[len(HYPEREDGE_PREFIX):-len(HYPEREDGE_SUFFIX)]
    if content.startswith("<hyperedge>"):
        content = content[len("<hyperedge>"):].strip()
    return content.strip().strip('"')


def format_hyperedge_content(raw: str) -> str:
    return f'{HYPEREDGE_PREFIX}{normalize_hyperedge_content(raw)}{HYPEREDGE_SUFFIX}'


def coerce_weight(weight_value: Any, default_weight: float = 1.0) -> float:
    if isinstance(weight_value, str):
        try:
            return float(weight_value)
        except (ValueError, TypeError):
            return default_weight
    if isinstance(weight_value, (int, float)):
        return float(weight_value)
    return default_weight


def get_hyperedge_name(hyperedge: Dict[str, Any]) -> str:
    return format_hyperedge_content(
        hyperedge.get("hyperedge_name") or hyperedge.get("content") or ""
    )


async def get_graph_weight(
    graphr1,
    hyperedge_name: str,
    fallback_weight: float = 1.0,
) -> float:
    if graphr1 is None or not hyperedge_name:
        return fallback_weight

    graph_storage = graphr1.chunk_entity_relation_graph
    node_data = await graph_storage.get_node(hyperedge_name)
    if node_data is not None:
        return coerce_weight(node_data.get("weight"), fallback_weight)

    node_edges = await graph_storage.get_node_edges(hyperedge_name)
    if node_edges:
        source_id, target_id = node_edges[0]
        edge_data = await graph_storage.get_edge(source_id, target_id)
        if edge_data is not None:
            return coerce_weight(edge_data.get("weight"), fallback_weight)

    return fallback_weight


def prepare_update_record(
    hyperedge: Dict[str, Any],
    new_content: str,
    global_config: Dict[str, Any],
    tool_name: str,
) -> Dict[str, Any]:
    old_hyperedge_name = get_hyperedge_name(hyperedge)
    new_hyperedge_name = format_hyperedge_content(new_content)

    if hyperedge.get("deleted", False):
        hyperedge["deleted"] = False
        hyperedge.pop("deleted_at", None)
        original_weight = hyperedge.pop("original_weight", None)
        if original_weight is not None:
            hyperedge["weight"] = coerce_weight(original_weight, 1.0)
            hyperedge.pop("weight_demoted_at", None)

    hyperedge["content"] = new_hyperedge_name
    hyperedge["hyperedge_name"] = new_hyperedge_name
    updated_hyperedge = _add_ai_metadata_to_hyperedge(
        hyperedge,
        action="update",
        tool_name=tool_name,
        global_config=global_config,
    )
    return {
        "hyperedge": updated_hyperedge,
        "old_hyperedge_name": old_hyperedge_name,
        "new_hyperedge_name": new_hyperedge_name,
    }


def prepare_delete_record(
    hyperedge: Dict[str, Any],
    current_weight: float,
    global_config: Dict[str, Any],
    tool_name: str,
) -> Dict[str, Any]:
    hyperedge["deleted"] = True
    hyperedge["deleted_at"] = datetime.now().isoformat()
    if "original_weight" not in hyperedge:
        hyperedge["original_weight"] = current_weight
    demoted_weight = current_weight * 0.5 if current_weight > 0 else 0.05
    hyperedge["weight"] = demoted_weight
    hyperedge["weight_demoted_at"] = datetime.now().isoformat()
    updated_hyperedge = _add_ai_metadata_to_hyperedge(
        hyperedge,
        action="delete",
        tool_name=tool_name,
        global_config=global_config,
    )
    return {
        "hyperedge": updated_hyperedge,
        "hyperedge_name": get_hyperedge_name(updated_hyperedge),
        "demoted_weight": demoted_weight,
    }


def build_graph_node_data(
    hyperedge: Dict[str, Any],
    existing_node_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    node_data = dict(existing_node_data or {})
    for key, value in hyperedge.items():
        if key in {"content", "hyperedge_name"}:
            continue
        node_data[key] = value
    node_data["role"] = "hyperedge"
    node_data["source_id"] = hyperedge.get("source_id", node_data.get("source_id", ""))
    node_data["weight"] = coerce_weight(
        hyperedge.get("weight", node_data.get("weight", 1.0)),
        1.0,
    )
    return node_data


async def _rename_networkx_node(graph_storage, old_name: str, new_name: str) -> None:
    graph = getattr(graph_storage, "_graph", None)
    if graph is None:
        logger.warning(
            "Graph backend %s does not expose NetworkX _graph; "
            "skipping hyperedge node rename from %r to %r",
            type(graph_storage).__name__,
            old_name,
            new_name,
        )
        return
    if old_name == new_name or not graph.has_node(old_name):
        return

    node_data = dict(graph.nodes[old_name])
    incident_edges = [
        (source_id, target_id, dict(edge_data))
        for source_id, target_id, edge_data in graph.edges(old_name, data=True)
    ]

    graph.remove_node(old_name)

    if graph.has_node(new_name):
        merged_node_data = dict(graph.nodes[new_name])
        merged_node_data.update(node_data)
        graph.nodes[new_name].update(merged_node_data)
    else:
        graph.add_node(new_name, **node_data)

    for source_id, target_id, edge_data in incident_edges:
        new_source = new_name if source_id == old_name else source_id
        new_target = new_name if target_id == old_name else target_id
        if graph.has_edge(new_source, new_target):
            merged_edge_data = dict(graph.edges[(new_source, new_target)])
            merged_edge_data.update(edge_data)
            graph.edges[(new_source, new_target)].update(merged_edge_data)
        else:
            graph.add_edge(new_source, new_target, **edge_data)


async def sync_hyperedge_graph(
    graphr1,
    old_hyperedge_name: str,
    new_hyperedge_name: str,
    hyperedge: Dict[str, Any],
    incident_edge_weight: Optional[float] = None,
) -> None:
    if graphr1 is None:
        return

    graph_storage = graphr1.chunk_entity_relation_graph
    existing_node_data = await graph_storage.get_node(old_hyperedge_name)
    if existing_node_data is None and old_hyperedge_name != new_hyperedge_name:
        existing_node_data = await graph_storage.get_node(new_hyperedge_name)

    if old_hyperedge_name and new_hyperedge_name and old_hyperedge_name != new_hyperedge_name:
        await _rename_networkx_node(graph_storage, old_hyperedge_name, new_hyperedge_name)

    await graph_storage.upsert_node(
        new_hyperedge_name,
        build_graph_node_data(hyperedge, existing_node_data),
    )

    if incident_edge_weight is not None:
        node_edges = await graph_storage.get_node_edges(new_hyperedge_name)
        if node_edges:
            for source_id, target_id in node_edges:
                edge_data = await graph_storage.get_edge(source_id, target_id)
                if edge_data is None:
                    continue
                updated_edge_data = dict(edge_data)
                updated_edge_data["weight"] = incident_edge_weight
                await graph_storage.upsert_edge(source_id, target_id, updated_edge_data)


async def persist_hyperedge_graph(graphr1) -> None:
    if graphr1 is None:
        return
    await graphr1.chunk_entity_relation_graph.index_done_callback()


def run_async(coro):
    loop = always_get_an_event_loop()
    return loop.run_until_complete(coro)
