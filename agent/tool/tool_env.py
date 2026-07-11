"""
Generic tool environment implementation, usable with any set of tools
"""

import re
import json
import random
import traceback
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy

from agent.tool.tool_base import Tool
import logging
logger = logging.getLogger(__name__)
# KnowledgePipeline已移除，GraphR1工具直接操作图文件

DEFAULT_DEFERRED_QUEUE_HINT_SECONDS = 1800
DEFERRED_QUEUE_HINT_ENV = "TOOL_DEFERRED_QUEUE_HINT_SECONDS"
LEGACY_DEFERRED_QUEUE_HINT_ENV = "TOOL_TIME_BATCH_INTERVAL"
TOOL_NAME_ALIASES = {
    "knowledge-base": "kb_search",
    "knowledge": "kb_search",
    "knowledge_search": "kb_search",
    "search": "kb_search",
    "metadata": "kb_search",
    "GraphRetrieval": "kb_search",
    "kg_search": "kb_search",
    "kd_search": "kb_search",
    "ka_search": "kb_search",
    "image_search": "websearch",
    "wbsearch": "websearch",
}


def _normalize_tool_name(tool_name: Any) -> Any:
    if not isinstance(tool_name, str):
        return tool_name
    return TOOL_NAME_ALIASES.get(tool_name, tool_name)


def _normalize_tool_call(tool_name: Any, tool_args: Any) -> Dict[str, Any]:
    normalized_name = _normalize_tool_name(tool_name)
    if not isinstance(normalized_name, str) or not isinstance(tool_args, dict):
        return {"tool": normalized_name, "args": tool_args, "invalid_tool_call": True}
    if (
        tool_name == "metadata"
        and normalized_name == "kb_search"
        and isinstance(tool_args, dict)
        and "query" not in tool_args
        and "keyword" in tool_args
    ):
        tool_args = {"query": str(tool_args["keyword"])}
    if normalized_name == "update" and isinstance(tool_args, dict) and "new_content" not in tool_args:
        for alias in ("updated_content", "replacement", "revised_content", "new_value", "new_text"):
            if alias in tool_args:
                tool_args = dict(tool_args)
                tool_args["new_content"] = tool_args[alias]
                break
    return {"tool": normalized_name, "args": tool_args}


def _format_unknown_tool_error(tool_name: Any, available_tools: List[str]) -> str:
    available = ", ".join(available_tools)
    if not isinstance(tool_name, str):
        return f"Unknown tool '{tool_name}'. Available tools: {available}"
    return (
        f"Unknown tool '{tool_name}'. Available tools: {available}. "
        "For graph edits, use the exact tool names insert, update, or delete; "
        "do not use graph_edit, kb_update, web_search, or text_search."
    )


def _format_tool_validation_error(tool_name: Any, error_msg: str) -> str:
    base = f"Invalid arguments for tool '{tool_name}': {error_msg}"
    if tool_name == "insert":
        return (
            f"{base}\n"
            "Repair format exactly as:\n"
            '<tool_call>{"tool":"insert","args":{"content":"<new KB fact from websearch or verified evidence>"}}</tool_call>'
        )
    if tool_name == "update":
        return (
            f"{base}\n"
            "Repair format exactly as:\n"
            '<tool_call>{"tool":"update","args":{"content":"<exact old KB sentence to replace>","new_content":"<corrected KB sentence>"}}</tool_call>\n'
            "Use update only when you can copy the exact old KB content. If the fact is missing, use insert instead."
        )
    if tool_name == "delete":
        return (
            f"{base}\n"
            "Repair format exactly as:\n"
            '<tool_call>{"tool":"delete","args":{"content":"<exact obsolete or false KB sentence to remove>"}}</tool_call>'
        )
    if tool_name == "kb_search":
        return (
            f"{base}\n"
            "Repair format exactly as:\n"
            '<tool_call>{"tool":"kb_search","args":{"query":"<concrete image or text query>"}}</tool_call>'
        )
    if tool_name == "websearch":
        return (
            f"{base}\n"
            "Repair format exactly as:\n"
            '<tool_call>{"tool":"websearch","args":{"query":"<concrete web query>"}}</tool_call>'
        )
    return base


def _is_malformed_tool_call(action: Any) -> bool:
    return (
        not isinstance(action, dict)
        or action.get("invalid_tool_call") is True
        or not isinstance(action.get("tool"), str)
        or not isinstance(action.get("args"), dict)
    )


def _get_deferred_queue_hint_seconds() -> int:
    raw_value = os.getenv(DEFERRED_QUEUE_HINT_ENV)
    source_name = DEFERRED_QUEUE_HINT_ENV

    if raw_value is None:
        raw_value = os.getenv(LEGACY_DEFERRED_QUEUE_HINT_ENV)
        source_name = LEGACY_DEFERRED_QUEUE_HINT_ENV
        if raw_value is not None:
            logger.warning(
                f"[DEPRECATED] {LEGACY_DEFERRED_QUEUE_HINT_ENV} is deprecated; "
                f"use {DEFERRED_QUEUE_HINT_ENV} instead"
            )

    if raw_value is None:
        return DEFAULT_DEFERRED_QUEUE_HINT_SECONDS

    try:
        hint_seconds = int(raw_value)
    except ValueError:
        hint_seconds = DEFAULT_DEFERRED_QUEUE_HINT_SECONDS
        logger.warning(
            f"[WARNING] Invalid {source_name} value, using default: {hint_seconds}s"
        )

    if hint_seconds < 0:
        logger.warning(f"[WARNING] {source_name} must be >= 0, using default")
        return DEFAULT_DEFERRED_QUEUE_HINT_SECONDS

    return hint_seconds

def _write_tool_execution_result(env, tool_name: str, tool_args: Dict, result: str, step_number: int = None):
    """写入单个工具执行结果到tool_results目录"""
    if os.getenv('TOOL_WRITE_RESULTS', 'false').lower() != 'true':
        return
    try:
        tool_args = _public_tool_args(tool_args)
        # 获取工作目录
        working_dir = getattr(env, 'working_dir', os.getcwd())
        tool_results_dir = os.path.join(working_dir, "tool_results")
        os.makedirs(tool_results_dir, exist_ok=True)
        
        # 生成结果文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(tool_results_dir, f"tool_execution_{timestamp}.json")
        
        # 解析结果以获取详细信息
        parsed_result = _parse_tool_result(result)
        
        # 获取GraphR1统计信息（如果是知识管理工具）
        graphr1_stats = {}
        if tool_name in ['insert', 'update', 'delete', 'search']:
            graphr1_stats = _get_graphr1_stats_for_tool(working_dir)
        
        # 准备写入的数据
        result_data = {
            "success": parsed_result.get("success", False),
            "message": parsed_result.get("message", result),
            "tool_name": tool_name,
            "tool_args": tool_args,
            "result": result,
            "step_number": step_number,
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "working_dir": working_dir,
            "result_type": "single_execution"
        }
        
        # 如果是知识管理工具，添加详细的统计信息
        if tool_name in ['insert', 'update', 'delete', 'search']:
            result_data.update({
                "statistics": graphr1_stats.get("statistics", {}),
                "inserted_statistics": graphr1_stats.get("inserted_statistics", {}),
                "inserted_entities": graphr1_stats.get("inserted_entities", []),
                "inserted_hyperedges": graphr1_stats.get("inserted_hyperedges", []),
                "sample_entities": graphr1_stats.get("sample_entities", []),
                "sample_hyperedges": graphr1_stats.get("sample_hyperedges", [])
            })
            
            # 如果是更新操作，添加更新详情
            if tool_name == 'update' and parsed_result.get("success"):
                result_data.update({
                    "hyperedge_id": parsed_result.get("updated_id"),
                    "search_content": tool_args.get("content", ""),
                    "old_content": parsed_result.get("old_content", ""),
                    "new_content": parsed_result.get("new_content", "")
                })
        
        # 写入文件
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"[TOOL_RESULT] Tool execution result written to: {result_file}")
        
    except Exception as e:
        logger.debug(f"[TOOL_RESULT] Failed to write tool execution result: {e}")


def _inject_tool_context(env, tool_name: str, tool_args: Dict) -> Dict:
    if not isinstance(tool_name, str) or not isinstance(tool_args, dict):
        return tool_args
    if tool_name not in {"websearch", "kb_search"}:
        return tool_args
    context = getattr(env, "tool_context", {}) or {}
    if not context:
        return tool_args

    contextualized_args = dict(tool_args)
    if tool_name == "kb_search":
        if str(contextualized_args.get("query", "")).strip() != "<img>":
            return contextualized_args
        image_id = context.get("image_id")
        image_path = context.get("image_path")
        question = context.get("question")
        if image_id and "image_id" not in contextualized_args:
            contextualized_args["image_id"] = image_id
        if image_path and "image_path" not in contextualized_args:
            contextualized_args["image_path"] = image_path
        if question and "context_query" not in contextualized_args:
            contextualized_args["context_query"] = question
        return contextualized_args

    data_source = context.get("data_source")
    dataset = context.get("dataset")
    if data_source and "__data_source" not in contextualized_args:
        contextualized_args["__data_source"] = data_source
    if dataset and "__dataset" not in contextualized_args:
        contextualized_args["__dataset"] = dataset
    return contextualized_args


def _public_tool_args(tool_args: Dict) -> Dict:
    return {
        key: value
        for key, value in tool_args.items()
        if key != "knowledge" and not str(key).startswith("__")
    }

def _parse_tool_result(result: str) -> Dict[str, Any]:
    """解析工具执行结果，提取结构化信息"""
    try:
        # 尝试解析JSON结果
        if result.startswith('{') and result.endswith('}'):
            return json.loads(result)
        
        # 如果不是JSON，尝试从文本中提取信息
        parsed = {"success": False, "message": result}
        
        # 检查成功关键词
        success_keywords = ["successfully", "success", "completed", "inserted", "updated", "deleted"]
        if any(keyword in result.lower() for keyword in success_keywords):
            parsed["success"] = True
        
        return parsed
        
    except Exception as e:
        return {"success": False, "message": result, "parse_error": str(e)}

def _get_graphr1_stats_for_tool(working_dir: str) -> Dict[str, Any]:
    """为工具执行获取GraphR1统计信息"""
    try:
        # 尝试导入GraphR1
        from graphr1.graphr1 import GraphR1
        
        # 创建GraphR1实例
        graphr1 = GraphR1(working_dir=working_dir)
        
        # 获取基础统计信息
        stats = _get_graphr1_statistics(graphr1)
        
        # 获取实体和超边数据
        entities_data = _get_entities_data(graphr1)
        hyperedges_data = _get_hyperedges_data(graphr1)
        
        # 获取示例数据
        sample_entities = _get_sample_entities(entities_data, limit=5)
        sample_hyperedges = _get_sample_hyperedges(hyperedges_data, limit=5)
        
        return {
            "statistics": stats,
            "inserted_statistics": {
                "new_entities_count": 0,  # 这里需要根据实际插入情况计算
                "new_hyperedges_count": 0,
                "new_graph_nodes_count": 0,
                "new_graph_edges_count": 0
            },
            "inserted_entities": [],  # 这里需要根据实际插入情况填充
            "inserted_hyperedges": [],  # 这里需要根据实际插入情况填充
            "sample_entities": sample_entities,
            "sample_hyperedges": sample_hyperedges
        }
        
    except Exception as e:
        logger.debug(f"[TOOL_RESULT] Failed to get GraphR1 stats: {e}")
        return {
            "statistics": {},
            "inserted_statistics": {},
            "inserted_entities": [],
            "inserted_hyperedges": [],
            "sample_entities": [],
            "sample_hyperedges": []
        }

def _get_graphr1_statistics(graphr1) -> Dict[str, Any]:
    """获取GraphR1基础统计信息"""
    try:
        # 获取实体数量
        entities_count = 0
        if hasattr(graphr1.entities_vdb, '_data') and graphr1.entities_vdb._data:
            entities_count = len(graphr1.entities_vdb._data)
        elif hasattr(graphr1.entities_vdb, '_storage') and graphr1.entities_vdb._storage:
            entities_count = len(graphr1.entities_vdb._storage)
        
        # 获取超边数量
        hyperedges_count = 0
        if hasattr(graphr1.hyperedges_vdb, '_data') and graphr1.hyperedges_vdb._data:
            hyperedges_count = len(graphr1.hyperedges_vdb._data)
        elif hasattr(graphr1.hyperedges_vdb, '_storage') and graphr1.hyperedges_vdb._storage:
            hyperedges_count = len(graphr1.hyperedges_vdb._storage)
        
        # 获取图节点数量
        graph_nodes_count = 0
        if hasattr(graphr1.chunk_entity_relation_graph, '_graph') and graphr1.chunk_entity_relation_graph._graph:
            graph_nodes_count = graphr1.chunk_entity_relation_graph._graph.number_of_nodes()
        elif hasattr(graphr1.chunk_entity_relation_graph, 'number_of_nodes'):
            graph_nodes_count = graphr1.chunk_entity_relation_graph.number_of_nodes()
        
        # 获取图边数量
        graph_edges_count = 0
        if hasattr(graphr1.chunk_entity_relation_graph, '_graph') and graphr1.chunk_entity_relation_graph._graph:
            graph_edges_count = graphr1.chunk_entity_relation_graph._graph.number_of_edges()
        elif hasattr(graphr1.chunk_entity_relation_graph, 'number_of_edges'):
            graph_edges_count = graphr1.chunk_entity_relation_graph.number_of_edges()
        
        return {
            "entities_count": entities_count,
            "hyperedges_count": hyperedges_count,
            "graph_nodes_count": graph_nodes_count,
            "graph_edges_count": graph_edges_count
        }
    except Exception as e:
        logger.debug(f"[TOOL_RESULT] Failed to get GraphR1 statistics: {e}")
        return {}

def _get_entities_data(graphr1) -> Dict:
    """获取实体数据"""
    try:
        if hasattr(graphr1.entities_vdb, '_data') and graphr1.entities_vdb._data:
            return graphr1.entities_vdb._data
        elif hasattr(graphr1.entities_vdb, '_storage') and graphr1.entities_vdb._storage:
            return graphr1.entities_vdb._storage
        return {}
    except Exception as e:
        logger.debug(f"[TOOL_RESULT] Failed to get entities data: {e}")
        return {}

def _get_hyperedges_data(graphr1) -> Dict:
    """获取超边数据"""
    try:
        if hasattr(graphr1.hyperedges_vdb, '_data') and graphr1.hyperedges_vdb._data:
            return graphr1.hyperedges_vdb._data
        elif hasattr(graphr1.hyperedges_vdb, '_storage') and graphr1.hyperedges_vdb._storage:
            return graphr1.hyperedges_vdb._storage
        return {}
    except Exception as e:
        logger.debug(f"[TOOL_RESULT] Failed to get hyperedges data: {e}")
        return {}

def _get_sample_entities(entities_data: Dict, limit: int = 5) -> List[str]:
    """获取示例实体"""
    try:
        if not entities_data:
            return []
        
        sample_entities = []
        for entity_id, entity_data in list(entities_data.items())[:limit]:
            if isinstance(entity_data, dict):
                entity_name = entity_data.get("entity_name", entity_id)
                sample_entities.append(entity_name)
            else:
                sample_entities.append(str(entity_data))
        
        return sample_entities
    except Exception as e:
        logger.debug(f"[TOOL_RESULT] Failed to get sample entities: {e}")
        return []

def _get_sample_hyperedges(hyperedges_data: Dict, limit: int = 5) -> List[str]:
    """获取示例超边"""
    try:
        if not hyperedges_data:
            return []
        
        sample_hyperedges = []
        for hyperedge_id, hyperedge_data in list(hyperedges_data.items())[:limit]:
            if isinstance(hyperedge_data, dict):
                content = hyperedge_data.get("content", hyperedge_id)
                # 截断过长的内容
                if len(content) > 100:
                    content = content[:100] + "..."
                sample_hyperedges.append(content)
            else:
                sample_hyperedges.append(str(hyperedge_data))
        
        return sample_hyperedges
    except Exception as e:
        logger.debug(f"[TOOL_RESULT] Failed to get sample hyperedges: {e}")
        return []

# Independent step function
def step(env: 'ToolEnv', action_text: str, step_number: int = None):
    """
    Execute one step of environment interaction with step batch processing support
    
    Args:
        env: The tool environment
        action_text: Text generated by LLM
        step_number: Current training step number (for batch processing)
        
    Returns:
        (observation, reward, done, info)
    """
    env.steps_taken += 1
    logger.debug(f"[STEP_DEBUG] Processing action_text: {action_text[:200]}...")
    action = env.extract_tool_call(action_text)
    logger.debug(f"[STEP_DEBUG] Extracted action: {action}")
    
    if action == env.INVALID_ACTION:
        logger.debug(f"[STEP_DEBUG] Action is INVALID_ACTION")
        result = env._get_specific_error_message(action_text)
        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=False,
            action_is_effective=False,
            reward=0
        )
        return result, env.PENALTY_FOR_INVALID, False, {"action_is_valid": False, "action_is_effective": False}

    if _is_malformed_tool_call(action):
        logger.debug(f"[STEP_DEBUG] Malformed tool call: {action}")
        result = env._get_specific_error_message(action_text)
        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=False,
            action_is_effective=False,
            reward=0
        )
        return result, env.PENALTY_FOR_INVALID, False, {"action_is_valid": False, "action_is_effective": False}
    
    tool_name = action["tool"]
    tool_args = _inject_tool_context(env, tool_name, action["args"])
    logger.debug(f"[STEP_DEBUG] Tool name: {tool_name}, args: {tool_args}")
    
    # Check if tool exists and validate parameters before execution
    available_tools = [tool.name for tool in env.tools]
    logger.debug(f"[STEP_DEBUG] Available tools: {available_tools}")
    if tool_name not in available_tools:
        result = _format_unknown_tool_error(tool_name, available_tools)
        logger.debug(f"[STEP_DEBUG] Tool not found: {result}")
        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=False,
            action_is_effective=False,
            reward=0
        )
        return result, env.PENALTY_FOR_INVALID, False, {"action_is_valid": False, "action_is_effective": False}
    
    # Validate if the tool exists
    if tool_name not in env.tool_map:
        result = _format_unknown_tool_error(tool_name, available_tools)
        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=True,
            action_is_effective=False,
            reward=0
        )
        return result, env.PENALTY_FOR_INVALID, False, {"action_is_valid": True, "action_is_effective": False}
    
    # Get tool instance
    tool = env.tool_map[tool_name]
    
    # Validate tool arguments
    is_valid, error_msg = tool.validate_args(tool_args)
    if not is_valid:
        result = _format_tool_validation_error(tool_name, error_msg)
        
        # 写入参数验证失败结果到tool_results目录
        _write_tool_execution_result(env, tool_name, tool_args, result, step_number)
        
        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=True,
            action_is_effective=False,
            reward=0
        )
        return result, env.PENALTY_FOR_INVALID, False, {"action_is_valid": True, "action_is_effective": False}
    
    # GraphR1工具直接操作图文件，不需要knowledge pipeline
    
    # Execute tool - 根据配置选择执行模式
    # 延迟执行模式：知识管理操作提交到批处理队列，在后续 queue flush 时统一执行
    # 即时执行模式：立即执行操作（默认行为）
    try:
        # 检查是否启用延迟执行模式（添加更详细的检查）
        use_deferred_execution = os.getenv('TOOL_USE_DEFERRED_EXECUTION', 'false').lower() == 'true'
        
        deferred_queue_hint_seconds = _get_deferred_queue_hint_seconds()

        if use_deferred_execution:
            logger.debug(
                "[DEFERRED_MODE] Deferred execution enabled; queue flush is triggered "
                f"by TOOL_BATCH_SIZE, explicit flush(), shutdown(), or ToolEnv.close() "
                f"(queue hint: {deferred_queue_hint_seconds}s)"
            )
        else:
            logger.debug(f"[IMMEDIATE_MODE] Immediate execution mode enabled")
        
        # 知识管理工具列表（需要批处理的工具）
        knowledge_tools = ['GraphR1InsertTool', 'HyperedgeUpdateTool', 'HyperedgeSoftDeleteTool']
        tool_class_name = tool.__class__.__name__
        
        # 检查工具名称而不是类名，更可靠
        knowledge_tool_names = ['insert', 'update', 'delete']
        is_knowledge_tool = tool_name in knowledge_tool_names
        logger.debug(f"[STEP_DEBUG] use_deferred_execution: {use_deferred_execution}, is_knowledge_tool: {is_knowledge_tool}, hasattr(tool, 'time_batch_submit'): {hasattr(tool, 'time_batch_submit')}")
        
        if use_deferred_execution and is_knowledge_tool and hasattr(tool, 'time_batch_submit'):
            # 延迟执行模式：提交任务到批处理队列
            logger.debug(f"[DEFERRED_MODE] Attempting to submit {tool_name} task to batch queue")
            try:
                result = tool.time_batch_submit(tool_args, step_number, env.steps_taken)
                logger.debug(
                    f"[DEFERRED_MODE] {tool_name} task submitted to batch queue "
                    "(execution happens on queue flush)"
                )
            except Exception as e:
                logger.warning(f"[DEFERRED_MODE_ERROR] Failed to submit {tool_name} task: {e}")
                # 回退到即时执行
                result = tool.execute(tool_args)
                logger.warning(f"[FALLBACK] Executed {tool_name} immediately due to batch submission failure")
        else:
            # 即时执行模式：立即执行（默认行为）
            if use_deferred_execution and is_knowledge_tool:
                logger.warning(f"[DEFERRED_MODE_WARNING] {tool_name} tool does not support deferred execution, executing immediately")
            logger.debug(f"[IMMEDIATE_MODE] Executing {tool_name} immediately")
            result = tool.execute(tool_args)
        
        logger.debug(f"[TOOL_EXECUTION] Tool {tool_name} executed successfully")
        logger.debug(f"[TOOL_EXECUTION] Result: {str(result)[:200]}...")
        
        reward = tool.calculate_reward(tool_args, result)
        logger.debug(f"[REWARD_CALCULATION] Tool: {tool_name}, Reward: {reward}")
        
        # 写入单个工具执行结果到tool_results目录
        _write_tool_execution_result(env, tool_name, tool_args, result, step_number)
        
        # GraphR1工具直接操作图文件，不需要更新knowledge pipeline
        
        # Check if deferred queue flush hooks should run
        # 使用线程安全的兼容触发方法；当前统一队列在队列满时自动 flush
        logger.debug(f"[BATCH_CHECK] Checking if batch processing should be triggered for {tool_name}")
        ToolEnv._try_trigger_batch_processing(tool, step_number)
        
        # Record tool call history (without auto-injected knowledge parameter)
        clean_args = _public_tool_args(tool_args)
        env.tool_history.append({
            "tool": tool_name,
            "args": clean_args,
            "result": result
        })
        
        # Check if max turns reached
        done = env.steps_taken >= env.max_turns
        
        logger.debug(f"[TRACKING_UPDATE] Updating tracking variables for {tool_name}")
        logger.debug(f"[TRACKING_UPDATE] action_is_valid: True, action_is_effective: True, reward: {reward}")
        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=True,
            action_is_effective=True,
            reward=reward
        )
        
        return result, reward, done, {"action_is_valid": True, "action_is_effective": True}
    except Exception as e:
        error_trace = traceback.format_exc()
        result = f"Error executing tool '{tool_name}': {str(e)}"
        logger.error(f"[TOOL_ERROR] Error executing {tool_name}: {str(e)}")
        logger.error(f"[TOOL_ERROR] Traceback: {error_trace}")
        
        # 写入错误结果到tool_results目录
        _write_tool_execution_result(env, tool_name, tool_args, result, step_number)
        
        logger.debug(f"[TRACKING_UPDATE] Updating tracking variables for failed {tool_name}")
        logger.debug(f"[TRACKING_UPDATE] action_is_valid: True, action_is_effective: False, reward: 0")
        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=True,
            action_is_effective=False,
            reward=0
        )
        
        return result, env.PENALTY_FOR_INVALID, False, {"action_is_valid": True, "action_is_effective": False}

# Batch step function
def step_batch(envs: List['ToolEnv'], action_texts: List[str], step_numbers: List[int] = None):
    """
    Execute batch steps of environment interaction with step batch processing support
    
    Args:
        envs: List of tool environments
        action_texts: List of texts generated by LLM
        step_numbers: List of current training step numbers (for batch processing)
        
    Returns:
        List of (observation, reward, done, info) tuples
    """
    assert len(envs) == len(action_texts), "Number of environments and actions must match"
    
    # Group actions by tool name and environment
    tool_groups = {}
    tool_indices = {}
    env_indices = {}
    action_map = {}
    results = [None] * len(envs)
    
    logger.debug(f"[BATCH_STEP] Processing batch of {len(envs)} environments")
    
    # First pass: extract tool calls and group by tool name
    for i, (env, action_text) in enumerate(zip(envs, action_texts)):
        logger.debug(f"[BATCH_STEP] Processing environment {i+1}/{len(envs)}")
        logger.debug(f"[BATCH_STEP] Action text: {action_text[:100]}...")
        
        # Extract the tool call
        action = env.extract_tool_call(action_text)
        logger.debug(f"[BATCH_STEP] Extracted action: {action}")
        action_map[i] = (env, action, action_text)
        
        # Handle invalid actions
        if action == env.INVALID_ACTION:
            logger.debug(f"[BATCH_STEP] Invalid action for environment {i+1}")
            result = env._get_specific_error_message(action_text)
            env.steps_taken += 1
            env._update_tracking_variables(
                response=action_text,
                action=action,
                action_is_valid=False,
                action_is_effective=False,
                reward=0
            )
            results[i] = (result, env.PENALTY_FOR_INVALID, False, {"action_is_valid": False, "action_is_effective": False})
            continue

        if _is_malformed_tool_call(action):
            logger.debug(f"[BATCH_STEP] Malformed tool call for environment {i+1}: {action}")
            result = env._get_specific_error_message(action_text)
            env.steps_taken += 1
            env._update_tracking_variables(
                response=action_text,
                action=action,
                action_is_valid=False,
                action_is_effective=False,
                reward=0
            )
            results[i] = (result, env.PENALTY_FOR_INVALID, False, {"action_is_valid": False, "action_is_effective": False})
            continue
            
        tool_name = action["tool"]
        tool_args = _inject_tool_context(env, tool_name, action["args"])
        
        # Handle unknown tools
        if tool_name not in env.tool_map:
            result = _format_unknown_tool_error(tool_name, [tool.name for tool in env.tools])
            env.steps_taken += 1
            env._update_tracking_variables(
                response=action_text,
                action=action,
                action_is_valid=True,
                action_is_effective=False,
                reward=0
            )
            results[i] = (result, env.PENALTY_FOR_INVALID, False, {"action_is_valid": True, "action_is_effective": False})
            logger.error(f"[ERROR] Unknown tool: {result}")
            continue
            
        # Get tool instance
        tool = env.tool_map[tool_name]
        
        # Validate tool arguments
        is_valid, error_msg = tool.validate_args(tool_args)
        if not is_valid:
            result = _format_tool_validation_error(tool_name, error_msg)
            env.steps_taken += 1
            env._update_tracking_variables(
                response=action_text,
                action=action,
                action_is_valid=True,
                action_is_effective=False,
                reward=0
            )
            results[i] = (result, env.PENALTY_FOR_INVALID, False, {"action_is_valid": True, "action_is_effective": False})
            logger.error(f"[ERROR] Invalid arguments for tool: {result}")
            continue
            
        # Group by tool name
        if tool_name not in tool_groups:
            tool_groups[tool_name] = []
            tool_indices[tool_name] = []
            env_indices[tool_name] = []
            
        tool_groups[tool_name].append(tool_args)
        tool_indices[tool_name].append(i)
        env_indices[tool_name].append(env)
    
    # Second pass: execute tools in batch where possible
    for tool_name, args_list in tool_groups.items():
        indices = tool_indices[tool_name]
        envs_list = env_indices[tool_name]
        
        # All environments share the same tool instances, so we can use the first one
        tool = envs_list[0].tool_map[tool_name]
        
        # Check if we should use deferred execution mode
        import os
        use_deferred_execution = os.getenv('TOOL_USE_DEFERRED_EXECUTION', 'false').lower() == 'true'
        knowledge_tool_names = ['insert', 'update', 'delete']
        is_knowledge_tool = tool_name in knowledge_tool_names
        
        logger.debug(f"[BATCH_STEP_DEBUG] Tool: {tool_name}, deferred_execution: {use_deferred_execution}, is_knowledge_tool: {is_knowledge_tool}")
        
        # Check if tool supports time_batch_submit
        if use_deferred_execution and is_knowledge_tool and hasattr(tool, 'time_batch_submit'):
            logger.debug(f"[BATCH_STEP_DEFERRED] Submitting {len(args_list)} {tool_name} tasks to batch queue")
            # Deferred execution mode: submit tasks to batch queue
            batch_results = []
            for idx, (env_idx, env, args) in enumerate(zip(indices, envs_list, args_list)):
                step_num = step_numbers[env_idx] if step_numbers and env_idx < len(step_numbers) else 0
                env_step = env.steps_taken
                try:
                    result = tool.time_batch_submit(args, step_num, env_step)
                    batch_results.append(result)
                    logger.debug(f"[BATCH_STEP_DEFERRED] Task {idx+1}/{len(args_list)} submitted: train_step={step_num}, env_step={env_step}")
                except Exception as e:
                    logger.warning(f"[BATCH_STEP_DEFERRED_ERROR] Failed to submit task {idx+1}: {e}")
                    # Fallback to immediate execution
                    result = tool.execute(args)
                    batch_results.append(result)
        else:
            # Immediate execution mode: execute all tasks now
            if use_deferred_execution and is_knowledge_tool:
                logger.warning(f"[BATCH_STEP_WARNING] Tool {tool_name} does not support deferred execution, executing immediately")
            logger.debug(f"[BATCH_STEP_IMMEDIATE] Executing {len(args_list)} {tool_name} tasks immediately")
            # Try regular batch execution (批处理管理器会在工具内部处理)
            try:
                batch_results = tool.batch_execute(args_list)
            except Exception as e:
                logger.warning(f"[BATCH_STEP_FALLBACK] batch_execute failed for {tool_name}, falling back to individual execution: {e}")
                batch_results = []
                for args in args_list:
                    try:
                        batch_results.append(tool.execute(args))
                    except Exception as sub_e:
                        batch_results.append(f'{{"success": false, "message": "Error: {str(sub_e)}"}}')

        # Process results
        for idx, env, result, args in zip(indices, envs_list, batch_results, args_list):
            env.steps_taken += 1
            reward = tool.calculate_reward(args, result)

            # Check if deferred queue flush hooks should run
            # 使用线程安全的兼容触发方法；当前统一队列在队列满时自动 flush
            if step_numbers and idx < len(step_numbers):
                ToolEnv._try_trigger_batch_processing(tool, step_numbers[idx])

            # Record tool call history (without auto-injected knowledge parameter)
            clean_args = _public_tool_args(args)
            env.tool_history.append({
                "tool": tool_name,
                "args": clean_args,
                "result": result
            })

            # Check if max turns reached
            done = env.steps_taken >= env.max_turns

            # Update tracking variables
            action_text = action_texts[idx]
            action = action_map[idx][1]
            env._update_tracking_variables(
                response=action_text,
                action=action,
                action_is_valid=True,
                action_is_effective=True,
                reward=reward
            )

            results[idx] = (result, reward, done, {"action_is_valid": True, "action_is_effective": True})

    return results

class ToolEnv:
    """
    Generic tool environment class, handling tool calls, history tracking, and state
    """
    INVALID_ACTION = {"tool": "invalid", "args": {}}
    PENALTY_FOR_INVALID = 0.0

    @classmethod
    def _try_trigger_batch_processing(cls, tool, step_number: int) -> None:
        """
        尝试触发批处理。统一队列在队列满时自动 flush，
        此方法保留签名以兼容调用方，仅在启用延迟执行时有效。
        """
        use_deferred = os.getenv('TOOL_USE_DEFERRED_EXECUTION', 'false').lower() == 'true'
        if not use_deferred:
            return
        # 队列满时自动 flush，此处无需额外触发

    def __init__(self, tools: List[Tool] = None, max_turns: int = 10):
        """
        Initialize the tool environment

        Args:
            tools: List of available tools
            max_turns: Maximum number of interaction turns
        """
        logger.debug(f"[ENV_INIT] Initializing ToolEnv with {len(tools or [])} tools")
        logger.debug(f"[ENV_INIT] Tools: {[tool.name for tool in (tools or [])]}")
        logger.debug(f"[ENV_INIT] max_turns: {max_turns}")

        self.tools = tools or []
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.tool_desc = [tool.get_description() for tool in self.tools]
        self.max_turns = max_turns
        self.tool_context = {}
        # GraphR1工具直接操作图文件，不需要knowledge pipeline
        self.reset_tracking_variables()

    def set_tool_context(self, **context):
        self.tool_context.update(
            {
                key: str(value)
                for key, value in context.items()
                if value is not None and str(value).strip()
            }
        )
        return self

        logger.debug(f"[ENV_INIT] Environment initialized successfully")

    def close(self):
        """显式关闭环境，flush 批量队列中的剩余任务。"""
        try:
            self.flush_deferred_tools()
            from agent.tool.tools.graphr1_base_tool import GraphR1BaseTool
            from agent.tool.tools.batch.unified_batch_queue import UnifiedBatchQueue
            UnifiedBatchQueue.get_instance().shutdown()
            GraphR1BaseTool.flush_hyperedges_cache()
            GraphR1BaseTool.flush_dirty_graphs()
            GraphR1BaseTool.flush_dirty_hyperedge_indexes()
            GraphR1BaseTool.flush_dirty_entity_indexes()
        except Exception as e:
            logger.warning(f"[ToolEnv] Error during close: {e}")

    def flush_deferred_tools(self):
        """Flush tool-local deferred queues without shutting down global text queues."""
        for tool in self.tools:
            flush_deferred = getattr(tool, "flush_deferred", None)
            if callable(flush_deferred):
                try:
                    flush_deferred()
                except Exception as e:
                    logger.warning(
                        f"[ToolEnv] Error flushing deferred tool {tool.name}: {e}"
                    )

    def tools_format_func(self) -> str:
        import json as _json
        tool_defs = [_t for _t in self.tool_desc]
        tools_json = _json.dumps(tool_defs, ensure_ascii=False, indent=2)
        
        # 检查可用的工具类型
        available_tools = [tool.name for tool in self.tools]
        has_knowledge_tools = any(tool in available_tools for tool in ['insert', 'update', 'delete'])
        
        # 基础模板 - 强化推理要求
        template = (
            "You may call one or more functions to solve the task.\n\n"
            "CRITICAL FORMAT REQUIREMENT - ALWAYS FOLLOW THIS ORDER:\n"
            "1. FIRST: Always start with `<think>`...`</think>` to think through the problem\n"
            "2. THEN: If tools are needed, include <tool_call>...</tool_call> AFTER your reasoning\n"
            "3. FINALLY: When you have the final answer, use <answer>...</answer>\n\n"
            "IMPORTANT: NEVER call tools without reasoning first. Every tool call must be preceded by reasoning in `<think>` tags.\n\n"
            "Available functions are provided within <tools></tools> as JSON definitions:\n"
            "<tools>\n" + tools_json + "\n</tools>\n\n"
            "When calling a function, first close </think>, then respond with a single <tool_call>...</tool_call> block:\n"
            '<tool_call>{"tool": "<name>", "args": { ... }}</tool_call>\n\n'
            "When you have the final answer (no further tool calls are needed), use:\n"
            "`<think>`\nBrief reasoning and evidence\n`</think>`\n<answer>\nClear, direct answer\n</answer>\n\n"
        )
        
        # 根据可用工具动态添加优先级提示
        if has_knowledge_tools:
            # 完整版本：包含知识管理工具
            template += (
                "Retrieval and tool priority (strict):\n"
                '1) Visual grounding: Start with kb_search({"query":"<img>"}) to retrieve candidate entities. Take the top-ranked visual entity as the anchor.\n'
                "2) Factual lookup: Form a text query using the anchoring entity name and question cues, then call kb_search. Prefer multiple rounds of refined text kb_search to gather sufficient evidence.\n"
                "3) Fallback: Use websearch only when refined kb_search attempts remain clearly insufficient, irrelevant, missing key knowledge, or conflicting.\n"
                "4) Knowledge maintenance: After websearch, resolve conflicts with the KB and integrate new information by applying graph edit operations (insert, update, delete) as needed.\n"
            )
        else:
            # Ablation版本：仅包含搜索工具
            template += (
                "Retrieval and tool priority (strict):\n"
                '1) Visual grounding: Start with kb_search({"query":"<img>"}) to retrieve candidate entities. Take the top-ranked visual entity as the anchor.\n'
                "2) Factual lookup: Form a text query using the anchoring entity name and question cues, then call kb_search. Prefer multiple rounds of refined text kb_search to gather sufficient evidence.\n"
                "3) Fallback: Use websearch only when refined kb_search attempts remain clearly insufficient, irrelevant, missing key knowledge, or conflicting.\n"
                "4) Focus on providing accurate answers based on available information.\n"
            )
        
        return template
        
    def reset_tracking_variables(self):
        """Reset tracking variables"""
        self.reward = 0
        self.tool_history = []  # Record tool call history
        self.steps_taken = 0
        self._actions = []  # All actions (including all LLM responses)
        self._actions_valid = []  # Correctly formatted actions
        self._actions_effective = []  # Effectively executed actions
        # 重置知识管道（每次查询开始时清空）
        # GraphR1工具直接操作图文件，不需要清理knowledge pipeline
    
    def get_tracking_variables(self) -> Dict:
        """Get statistics of tracking variables"""
        return {
            "reward": self.reward,
            "steps_taken": self.steps_taken,
            "tool_history": self.tool_history,
            "actions": self._actions,
            "actions_valid": self._actions_valid,
            "actions_effective": self._actions_effective,
        }
    
    def _update_tracking_variables(
            self, 
            response: str,
            action: Any, 
            action_is_valid: bool,
            action_is_effective: bool,
            reward: float,
        ):
        """
        Update tracking variables
        
        Args:
            response: Raw LLM response
            action: Parsed action
            action_is_valid: Whether the action format is valid
            action_is_effective: Whether the action executed successfully
            reward: Reward for the current step
        """
        self._actions.append(response)
        if action_is_valid:
            self._actions_valid.append(action)
        else:
            self._actions_valid.append(None)
        if action_is_effective:
            self._actions_effective.append(action)
        else:
            self._actions_effective.append(None)
        
        self.reward += reward if action_is_valid else (reward + self.PENALTY_FOR_INVALID)
    
    def _get_specific_error_message(self, text: str) -> str:
        """
        Generate specific error message based on the type of format error
        
        Args:
            text: The invalid tool call text
            
        Returns:
            Specific error message for the detected issue
        """
        def _tool_call_hint(tool: str | None = None) -> str:
            if tool in {"kb_search", "websearch"}:
                return (
                    "Close </think> before the tool call. Then output exactly one "
                    f"<tool_call>...</tool_call> block for {tool} with valid JSON and a concrete query."
                )
            return (
                "Close </think> before the tool call. Then output exactly one "
                "<tool_call>...</tool_call> block with valid JSON."
            )

        # Check if tool_call tags are missing
        if '<tool_call>' not in text and '<query>' not in text:
            return "Missing tool call tags. " + _tool_call_hint()
        
        # Check if it's a query format (old format)
        if '<query>' in text and '<tool_call>' not in text:
            return "Old format detected. " + _tool_call_hint()
        
        # Check if closing tag is missing
        if '<tool_call>' in text and '</tool_call>' not in text:
            return "Missing closing </tool_call> tag. " + _tool_call_hint()

        # Try to extract tool name from the text
        tool_name = None
        if '<tool_call>' in text:
            try:
                # Extract content between tool_call tags
                tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
                if tool_call_match:
                    tool_call_json = tool_call_match.group(1).strip()
                    tool_call_data = json.loads(tool_call_json)
                    
                    # Handle both dictionary and list formats
                    if isinstance(tool_call_data, dict) and "tool" in tool_call_data:
                        tool_name = tool_call_data["tool"]
                    elif isinstance(tool_call_data, list) and len(tool_call_data) > 0:
                        first_call = tool_call_data[0]
                        if isinstance(first_call, dict) and "tool" in first_call:
                            tool_name = first_call["tool"]
            except:
                pass
        
        # Generate specific error message based on tool name
        tool_name = _normalize_tool_name(tool_name)

        if tool_name == "insert":
            return "Invalid insert format. Please use: <tool_call>{\"tool\": \"insert\", \"args\": {\"content\": \"your content\"}}</tool_call>"
        elif tool_name == "update":
            return "Invalid update format. Please use: <tool_call>{\"tool\": \"update\", \"args\": {\"content\": \"original content\", \"new_content\": \"updated content\"}}</tool_call>"
        elif tool_name == "delete":
            return "Invalid delete format. Please use: <tool_call>{\"tool\": \"delete\", \"args\": {\"content\": \"content to delete\"}}</tool_call>"
        elif tool_name == "kb_search":
            return "Invalid search format. " + _tool_call_hint("kb_search")
        elif tool_name == "websearch":
            return "Invalid websearch format. " + _tool_call_hint("websearch")
        elif isinstance(tool_name, str) and tool_name:
            return f"Unknown tool '{tool_name}'. Available tools: insert, update, delete, kb_search, websearch"
        else:
            return "Invalid JSON format. " + _tool_call_hint()

    def extract_tool_call(self, text: str) -> Dict:
        """
        Extract tool call from LLM output
        
        Args:
            text: Text generated by LLM
            
        Returns:
            Dictionary containing tool name and parameters
        """
        # 添加调试日志
        logger.debug(f"[TOOL_CALL_DEBUG] Extracting tool call from text: {text[:200]}...")
        
        # 首先尝试新的工具调用格式 <tool_call>{"tool": "...", "args": {...}}</tool_call>
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        
        tool_call_match = re.search(tool_call_pattern, text, re.DOTALL)
        
        if not tool_call_match:
            logger.debug(f"[TOOL_CALL_DEBUG] No <tool_call> pattern found, trying <query> pattern")
            # 兼容旧的查询格式 <query>{"query": "..."}</query>
            query_pattern = r'<query>(.*?)</query>'
            query_match = re.search(query_pattern, text, re.DOTALL)
            
            if not query_match:
                logger.debug(f"[TOOL_CALL_DEBUG] No <query> pattern found either, returning INVALID_ACTION")
                return self.INVALID_ACTION
            
            try:
                query_json = query_match.group(1).strip()
                query_data = json.loads(query_json)
                
                if "query" in query_data:
                    statement = {"query": str(query_data["query"])}
                    return {"tool": "kb_search", "args": statement}
                else:
                    return self.INVALID_ACTION
                    
            except json.JSONDecodeError:
                return self.INVALID_ACTION
            except Exception:
                return self.INVALID_ACTION
        
        try:
            tool_call_json = tool_call_match.group(1).strip()
            # 容错：移除字面 \n，修复多余转义
            tool_call_json = tool_call_json.replace('\\n', '').replace('\\t', '')
            if '\\"' in tool_call_json and '"tool"' not in tool_call_json:
                tool_call_json = tool_call_json.encode().decode('unicode_escape')
            logger.debug(f"[TOOL_CALL_DEBUG] Found tool_call pattern, JSON: {tool_call_json}")
            tool_call_data = json.loads(tool_call_json)
            logger.debug(f"[TOOL_CALL_DEBUG] Parsed tool_call_data: {tool_call_data}")
            
            # Handle both dictionary and list formats
            if isinstance(tool_call_data, dict):
                # 检查是否是新的工具调用格式
                if "tool" in tool_call_data and "args" in tool_call_data:
                    result = _normalize_tool_call(tool_call_data["tool"], tool_call_data["args"])
                    if _is_malformed_tool_call(result):
                        return self.INVALID_ACTION
                    logger.debug(f"[TOOL_CALL_DEBUG] Successfully extracted tool call: {result}")
                    return result
                else:
                    logger.debug(f"[TOOL_CALL_DEBUG] Tool call data missing 'tool' or 'args' fields")
                    return self.INVALID_ACTION
            elif isinstance(tool_call_data, list) and len(tool_call_data) > 0:
                # If it's a list, take the first element
                first_call = tool_call_data[0]
                if isinstance(first_call, dict) and "tool" in first_call and "args" in first_call:
                    result = _normalize_tool_call(first_call["tool"], first_call["args"])
                    if _is_malformed_tool_call(result):
                        return self.INVALID_ACTION
                    return result
                else:
                    return self.INVALID_ACTION
            else:
                return self.INVALID_ACTION
                
        except json.JSONDecodeError as e:
            logger.debug(f"[TOOL_CALL_DEBUG] JSON decode error: {e}")
            return self.INVALID_ACTION
        except Exception as e:
            logger.debug(f"[TOOL_CALL_DEBUG] Unexpected error: {e}")
            return self.INVALID_ACTION
    
    def get_tool_history_context(self) -> str:
        """
        Generate tool call history context
        
        Returns:
            Formatted tool call history
        """
        if not self.tool_history:
            return "No tool call history yet."
        
        context = "Tool call history:\n"
        for i, call in enumerate(self.tool_history):
            context += f"{i+1}. Tool: {call['tool']}\n"
            context += f"   Arguments: {json.dumps(call['args'], ensure_ascii=False)}\n"
            context += f"   Result: {call['result']}\n\n"
        
        return context
    
    def get_available_tools_description(self) -> str:
        """
        Get description of available tools
        
        Returns:
            Formatted tool descriptions
        """
        if not self.tools:
            return "No tools available."
            
        descriptions = ["Available tools:"]
        for tool in self.tools:
            descriptions.append(tool.get_simple_description())
            
        return "\n\n".join(descriptions)
    
    def copy(self):
        """
        Copy the tool environment
        """
        env = ToolEnv(tools=self.tools, max_turns=self.max_turns)
        env.tool_history = deepcopy(self.tool_history)
        env.reward = self.reward
        env.steps_taken = self.steps_taken
        env._actions = deepcopy(self._actions)
        env._actions_valid = deepcopy(self._actions_valid)
        env._actions_effective = deepcopy(self._actions_effective)
        env.tool_context = deepcopy(self.tool_context)
        # GraphR1工具直接操作图文件，不需要复制knowledge pipeline
        return env
    
    # GraphR1工具直接操作图文件，不需要knowledge pipeline相关方法
    
    def step(self, action_text: str):
        """
        Execute one step of environment interaction
        
        Args:
            action_text: Text generated by LLM
            
        Returns:
            (observation, reward, done, info)
        """
        return step(self, action_text)
