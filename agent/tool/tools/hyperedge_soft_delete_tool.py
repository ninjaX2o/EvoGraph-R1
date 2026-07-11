"""
Hyperedge Soft Delete Tool - Soft delete hyperedges in GraphR1
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
    get_graph_weight,
    get_hyperedge_name,
    prepare_delete_record,
    run_async,
    sync_hyperedge_graph,
)
from agent.tool.tools.update_types import UpdateType

logger = logging.getLogger(__name__)


class HyperedgeSoftDeleteTool(GraphR1BaseTool):
    def __init__(self):
        name = "delete"
        description = "Soft delete knowledge item from knowledge base."
        parameters = {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The original content of the knowledge item from knowledge base",
                }
            },
            "required": ["content"],
        }
        super().__init__(name, description, parameters)

    def execute(self, args: Dict) -> str:
        logger.info(f"[{self.name}] Starting delete")

        try:
            is_valid, error_msg = self.validate_args(args)
            if not is_valid:
                return self._format_response(False, f"Parameter validation failed: {error_msg}")

            content = args.get("content", "").strip()
            if not content:
                return self._format_response(False, "Content cannot be empty")

            graphr1 = self._get_or_create_graphr1()
            if graphr1 is None:
                return self._format_response(False, "GraphR1 instance not available")

            hyperedges_data = self._get_hyperedges_data(graphr1)
            lookup = load_hyperedge_lookup(graphr1.working_dir, skip_deleted=True)
            found_hyperedge_id = find_hyperedge_by_content(
                hyperedges_data,
                content,
                skip_deleted=True,
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
                hyperedge_name = get_hyperedge_name(hyperedge)
                current_weight = run_async(
                    get_graph_weight(
                        graphr1,
                        hyperedge_name,
                        hyperedge.get("weight", 1.0),
                    )
                )
                prepared = prepare_delete_record(
                    hyperedge,
                    current_weight=current_weight,
                    global_config=self._get_global_config_for_ai_metadata(graphr1),
                    tool_name="hyperedge_soft_delete_tool",
                )
                deleted_hyperedge = prepared["hyperedge"]
                hyperedges_data[found_hyperedge_id] = deleted_hyperedge

                run_async(
                    sync_hyperedge_graph(
                        graphr1,
                        old_hyperedge_name=prepared["hyperedge_name"],
                        new_hyperedge_name=prepared["hyperedge_name"],
                        hyperedge=deleted_hyperedge,
                        incident_edge_weight=prepared["demoted_weight"],
                    )
                )
                self.mark_graph_dirty(graphr1)
                self._save_hyperedges_data_now(graphr1, hyperedges_data)
                append_recent_hyperedge_mutations(
                    graphr1.working_dir,
                    [
                        {
                            "hyperedge_id": found_hyperedge_id,
                            "action": "delete",
                            "content": prepared["hyperedge_name"],
                            "active": False,
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
                    "deleted_id": found_hyperedge_id,
                    "content": prepared["hyperedge_name"][:100] + "..."
                    if len(prepared["hyperedge_name"]) > 100
                    else prepared["hyperedge_name"],
                    "original_weight": deleted_hyperedge.get("original_weight", 1.0),
                    "demoted_weight": prepared["demoted_weight"],
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
                        "Knowledge item soft deleted but hyperedge vector index rebuild failed",
                        response_data,
                    )

                return self._format_response(
                    True,
                    "Successfully soft deleted knowledge item",
                    response_data,
                )
            except Exception as e:
                logger.error(f"[{self.name}] Error soft deleting hyperedge: {e}")
                return self._format_response(False, f"Error soft deleting hyperedge: {str(e)}")

        except Exception as e:
            logger.error(f"[{self.name}] Error in HyperedgeSoftDeleteTool: {e}")
            return self._format_response(False, f"Delete failed: {str(e)}")

    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        return super().batch_execute(args_list, UpdateType.DELETE)

    def time_batch_submit(self, args: Dict, train_step: int, env_step: int) -> str:
        return super().time_batch_submit(args, train_step, env_step, UpdateType.DELETE)

    def calculate_reward(self, args: Dict[str, Any], result: str) -> float:
        return super().calculate_reward(args, result, reward_on_success=0.1, reward_on_failure=-0.1)
