"""
GraphR1 Knowledge Insertion Tool - Supports adding knowledge to GraphR1
"""

import json
import os
import logging
import re
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Any
from agent.tool.tools.graphr1_base_tool import GraphR1BaseTool
from agent.tool.tools.hyperedge_state_sync import append_recent_hyperedge_mutations
from agent.tool.tools.hyperedge_sync import (
    format_hyperedge_content,
    run_async,
    sync_hyperedge_graph,
)
from agent.tool.tools.update_types import UpdateType
from graphr1.operate import _add_ai_metadata_to_hyperedge

logger = logging.getLogger(__name__)

# 设置日志记录器
logger = logging.getLogger(__name__)


DEFAULT_INSERT_BATCH_SIZE = 10


def _get_insert_batch_size() -> int:
    try:
        batch_size = int(os.getenv("TOOL_INSERT_BATCH_SIZE", str(DEFAULT_INSERT_BATCH_SIZE)))
    except ValueError:
        logger.warning(
            f"[INSERT] Invalid TOOL_INSERT_BATCH_SIZE, using {DEFAULT_INSERT_BATCH_SIZE}"
        )
        return DEFAULT_INSERT_BATCH_SIZE
    if batch_size <= 0:
        logger.warning(
            f"[INSERT] TOOL_INSERT_BATCH_SIZE must be > 0, using {DEFAULT_INSERT_BATCH_SIZE}"
        )
        return DEFAULT_INSERT_BATCH_SIZE
    return batch_size


def _chunks(items: List[str], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


class GraphR1InsertTool(GraphR1BaseTool):
    def __init__(self):
        name = "insert"
        description = "Insert new knowledge not in the KB. Provide the full content to add."
        parameters = {
            "type": "object",
            "properties": {
                "content": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ],
                    "description": "Full text (or list of texts) to add to the KB."
                }
            },
            "required": ["content"]
        }
        super().__init__(name, description, parameters)

    def execute(self, args: Dict) -> str:
        """Execute knowledge insertion operation.

        Accepts either a single content string or a list of content strings.
        In deferred execution mode (TOOL_USE_DEFERRED_EXECUTION=true), skips
        LLM entity extraction and writes directly to hyperedges cache for speed.
        """
        try:
            # Validate parameters
            is_valid, error_msg = self.validate_args(args)
            if not is_valid:
                return self._format_response(False, f"Parameter validation failed: {error_msg}")

            # Support both single string and list of strings
            raw_content = args.get("content", "")
            if isinstance(raw_content, list):
                contents = raw_content
            else:
                contents = [raw_content]

            # Clean each content string
            cleaned = []
            for c in contents:
                c = c.strip()
                c = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', c)
                c = c.strip()
                if c:
                    cleaned.append(c)

            if not cleaned:
                return self._format_response(False, "Knowledge content cannot be empty")

            # Check if deferred execution mode is enabled (training mode)
            use_deferred = os.getenv('TOOL_USE_DEFERRED_EXECUTION', 'false').lower() == 'true'

            if use_deferred:
                # Deferred mode: skip LLM entity extraction, write directly to cache
                return self._execute_deferred(cleaned)
            else:
                # Immediate mode: full LLM-based insertion
                return self._execute_immediate(cleaned)

        except Exception as e:
            logger.error(f"Error in GraphR1InsertTool: {e}")
            return self._format_response(False, f"Insertion failed: {str(e)}")

    def _execute_immediate(self, cleaned: List[str]) -> str:
        """Full insertion with LLM entity extraction (inference/non-training mode)."""
        from agent.tool.tools.entity_index_sync import load_entities_data

        graphr1 = self._get_or_create_graphr1()
        if graphr1 is None:
            return self._format_response(False, "Unable to create GraphR1 instance")

        initial_stats = self._get_knowledge_base_stats()
        previous_hyperedges_data = deepcopy(self._get_hyperedges_data(graphr1))
        previous_entities_data = load_entities_data(graphr1.working_dir)

        logger.debug(f"Inserting knowledge content into GraphR1...")
        logger.debug(f"Content count: {len(cleaned)}, total length: {sum(len(c) for c in cleaned)} characters")
        insert_batch_size = _get_insert_batch_size()
        for content_batch in _chunks(cleaned, insert_batch_size):
            graphr1.insert(content_batch)
        append_recent_hyperedge_mutations(
            graphr1.working_dir,
            [
                {
                    "hyperedge_id": "",
                    "action": "insert",
                    "content": content,
                    "active": True,
                    "searchable": True,
                }
                for content in cleaned
            ],
        )
        self.invalidate_hyperedges_cache(graphr1.working_dir)
        self.mark_hyperedge_index_dirty(graphr1.working_dir)
        self.mark_entity_index_dirty(graphr1.working_dir)
        hyperedge_index_rebuilt = self.rebuild_hyperedge_index(
            graphr1.working_dir,
            previous_hyperedges_data=previous_hyperedges_data,
            allow_prefix_reuse=True,
        )
        entity_index_rebuilt = self.rebuild_entity_index(
            graphr1.working_dir,
            previous_entities_data=previous_entities_data,
            allow_prefix_reuse=True,
        )

        final_stats = self._get_knowledge_base_stats()
        entities_added = final_stats.get("entities", 0) - initial_stats.get("entities", 0)
        hyperedges_added = final_stats.get("hyperedges", 0) - initial_stats.get("hyperedges", 0)
        response_data = {
            "entities_added": entities_added,
            "hyperedges_added": hyperedges_added,
            "total_entities": final_stats.get("entities", 0),
            "total_hyperedges": final_stats.get("hyperedges", 0),
            "hyperedge_index_rebuilt": hyperedge_index_rebuilt,
            "entity_index_rebuilt": entity_index_rebuilt,
            "insert_batch_size": insert_batch_size,
        }

        if not hyperedge_index_rebuilt or not entity_index_rebuilt:
            response_data.update({
                "partial_success": True,
                "kv_graph_updated": True,
            })
            return self._format_response(
                False,
                "Knowledge inserted but vector index rebuild failed",
                response_data,
            )

        return self._format_response(True, "Successfully inserted knowledge", response_data)

    def _execute_deferred(self, cleaned: List[str]) -> str:
        """Lightweight insertion for training: write to hyperedges cache, skip LLM."""
        import uuid

        graphr1 = self._get_or_create_graphr1()
        if graphr1 is None:
            return self._format_response(False, "Unable to create GraphR1 instance")

        hyperedges_data = self._get_hyperedges_data(graphr1)
        previous_hyperedges_data = deepcopy(hyperedges_data)
        global_config = self._get_global_config_for_ai_metadata(graphr1)
        added = 0

        for content_str in cleaned:
            he_id = f"rel-insert-{uuid.uuid4().hex[:12]}"
            hyperedge_name = format_hyperedge_content(content_str)
            he = {
                "content": hyperedge_name,
                "hyperedge_name": hyperedge_name,
                "weight": 1.0,
                "source_id": "agent_insert",
            }
            he = _add_ai_metadata_to_hyperedge(
                he, action="insert",
                tool_name="graphr1_insert_tool",
                global_config=global_config,
            )
            hyperedges_data[he_id] = he
            run_async(
                sync_hyperedge_graph(
                    graphr1,
                    old_hyperedge_name=hyperedge_name,
                    new_hyperedge_name=hyperedge_name,
                    hyperedge=he,
                )
            )
            added += 1

        self.remember_previous_hyperedges_data(graphr1, previous_hyperedges_data)
        self._save_hyperedges_data(graphr1, hyperedges_data)
        self.mark_graph_dirty(graphr1)
        append_recent_hyperedge_mutations(
            graphr1.working_dir,
            [
                {
                    "hyperedge_id": "",
                    "action": "insert",
                    "content": content,
                    "active": True,
                    "searchable": True,
                }
                for content in cleaned
            ],
        )
        logger.debug(f"[DEFERRED_INSERT] Added {added} hyperedges to cache and graph (no LLM)")

        return self._format_response(True, "Successfully inserted knowledge", {
            "entities_added": 0,
            "hyperedges_added": added,
        })

    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """Execute multiple insertions in batch"""
        return super().batch_execute(args_list, UpdateType.INSERT)

    def time_batch_submit(self, args: Dict, train_step: int, env_step: int) -> str:
        """Submit task to unified batch queue"""
        return super().time_batch_submit(args, train_step, env_step, UpdateType.INSERT)

    def calculate_reward(self, args: Dict, result: str) -> float:
        """Calculate reward for the insertion operation"""
        return super().calculate_reward(args, result, reward_on_success=0.1, reward_on_failure=-0.1)
