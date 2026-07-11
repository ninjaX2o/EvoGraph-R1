"""
Hyperedge Update Tool - Update hyperedge content in GraphR1 knowledge graph
"""

import logging
from copy import deepcopy
from typing import Any, Dict, List

from agent.tool.tools.batch.unified_batch_queue import find_hyperedge_by_content
from agent.tool.tools.graphr1_base_tool import GraphR1BaseTool
from agent.tool.tools.hyperedge_state_sync import (
    append_recent_hyperedge_mutations,
    load_hyperedge_lookup,
)
from agent.tool.tools.hyperedge_sync import (
    get_hyperedge_name,
    prepare_update_record,
    run_async,
    sync_hyperedge_graph,
)
from agent.tool.tools.update_types import UpdateType

logger = logging.getLogger(__name__)


class HyperedgeUpdateTool(GraphR1BaseTool):
    def __init__(self):
        name = "update"
        description = "Update knowledge item content in knowledge base."
        parameters = {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content of the knowledge item to update",
                },
                "new_content": {
                    "type": "string",
                    "description": "The new content for the knowledge item",
                },
            },
            "required": ["content", "new_content"],
        }
        super().__init__(name, description, parameters)

    def execute(self, args: Dict) -> str:
        logger.info(f"[{self.name}] Starting update")

        try:
            is_valid, error_msg = self.validate_args(args)
            if not is_valid:
                logger.error(f"[{self.name}] Parameter validation failed: {error_msg}")
                return self._format_response(False, f"Parameter validation failed: {error_msg}")

            content = args.get("content", "").strip()
            new_content = args.get("new_content", "").strip()

            if not content:
                return self._format_response(False, "Content cannot be empty")
            if not new_content:
                return self._format_response(False, "New content cannot be empty")

            graphr1 = self._get_or_create_graphr1()
            if graphr1 is None:
                return self._format_response(False, "GraphR1 instance not available")

            hyperedges_data = self._get_hyperedges_data(graphr1)
            lookup = load_hyperedge_lookup(graphr1.working_dir, skip_deleted=False)
            found_hyperedge_id = find_hyperedge_by_content(
                hyperedges_data,
                content,
                skip_deleted=False,
                lookup=lookup,
            )
            if not found_hyperedge_id:
                return self._format_response(
                    False,
                    f"No knowledge item found with content containing: '{content}'",
                )

            try:
                previous_hyperedges_data = deepcopy(hyperedges_data)
                hyperedge = hyperedges_data[found_hyperedge_id]
                best_match_content = get_hyperedge_name(hyperedge)
                was_deleted = bool(hyperedge.get("deleted", False))
                prepared = prepare_update_record(
                    hyperedge,
                    new_content=new_content,
                    global_config=self._get_global_config_for_ai_metadata(graphr1),
                    tool_name="hyperedge_update_tool",
                )
                updated_hyperedge = prepared["hyperedge"]
                hyperedges_data[found_hyperedge_id] = updated_hyperedge

                incident_edge_weight = None
                if was_deleted and "weight" in updated_hyperedge:
                    incident_edge_weight = updated_hyperedge["weight"]

                run_async(
                    sync_hyperedge_graph(
                        graphr1,
                        old_hyperedge_name=prepared["old_hyperedge_name"],
                        new_hyperedge_name=prepared["new_hyperedge_name"],
                        hyperedge=updated_hyperedge,
                        incident_edge_weight=incident_edge_weight,
                    )
                )
                self.mark_graph_dirty(graphr1)
                self._save_hyperedges_data_now(graphr1, hyperedges_data)
                append_recent_hyperedge_mutations(
                    graphr1.working_dir,
                    [
                        {
                            "hyperedge_id": found_hyperedge_id,
                            "action": "update",
                            "content": prepared["new_hyperedge_name"],
                            "active": True,
                            "searchable": True,
                        }
                    ],
                )
                index_rebuilt = self.rebuild_hyperedge_index(
                    graphr1.working_dir,
                    hyperedges_data=hyperedges_data,
                    previous_hyperedges_data=previous_hyperedges_data,
                )
                response_data = {
                    "updated_id": found_hyperedge_id,
                    "old_content": best_match_content[:100] + "..."
                    if len(best_match_content) > 100
                    else best_match_content,
                    "new_content": prepared["new_hyperedge_name"][:100] + "..."
                    if len(prepared["new_hyperedge_name"]) > 100
                    else prepared["new_hyperedge_name"],
                    "hyperedge_index_rebuilt": index_rebuilt,
                    "graph_persist_deferred": True,
                }

                if not index_rebuilt:
                    response_data.update({
                        "partial_success": True,
                        "kv_graph_updated": True,
                    })
                    return self._format_response(
                        False,
                        "Knowledge item updated but hyperedge vector index rebuild failed",
                        response_data,
                    )

                return self._format_response(
                    True,
                    "Successfully updated knowledge item",
                    response_data,
                )
            except Exception as e:
                logger.error(f"[{self.name}] Error updating hyperedge: {e}")
                return self._format_response(False, f"Error updating hyperedge: {str(e)}")

        except Exception as e:
            logger.error(f"[{self.name}] Error in HyperedgeUpdateTool: {e}")
            return self._format_response(False, f"Update failed: {str(e)}")

    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        return super().batch_execute(args_list, UpdateType.UPDATE)

    def time_batch_submit(self, args: Dict, train_step: int, env_step: int) -> str:
        return super().time_batch_submit(args, train_step, env_step, UpdateType.UPDATE)

    def calculate_reward(self, args: Dict[str, Any], result: str) -> float:
        return super().calculate_reward(args, result, reward_on_success=0.1, reward_on_failure=-0.1)
