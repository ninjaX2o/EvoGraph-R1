"""
Unified batch queue for CRUD operations.
"""

import atexit
import json
import logging
import os
import queue as queue_module
import threading
import time
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.tool.tools.graphr1_base_tool import GraphR1BaseTool, _create_graphr1_instance
from agent.tool.tools.hyperedge_state_sync import (
    append_recent_hyperedge_mutations,
    build_hyperedge_content_lookup as build_hyperedge_content_lookup_payload,
    ensure_hyperedge_state_sidecars,
    load_hyperedge_lookup,
    normalize_lookup_content,
)
from agent.tool.tools.hyperedge_sync import (
    get_graph_weight,
    get_hyperedge_name,
    prepare_delete_record,
    prepare_update_record,
    run_async,
    sync_hyperedge_graph,
)
from agent.tool.tools.update_types import UpdateType

logger = logging.getLogger(__name__)


def _write_json_with_filelock(path: str, data: dict) -> None:
    try:
        from filelock import FileLock

        lock_path = path + ".lock"
        with FileLock(lock_path, timeout=30):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    except ImportError:
        logger.warning("[FILELOCK] filelock not installed, writing without lock")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


_queue_init_lock = threading.Lock()
MIN_SUBSTRING_MATCH_LEN = 10
MIN_SUBSTRING_RATIO = 0.5
DEFAULT_INSERT_BATCH_SIZE = 10


def _positive_int(value: Any, default: int, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        logger.warning(f"[UnifiedBatchQueue] Invalid {name}={value!r}, using {default}")
        return default
    if parsed <= 0:
        logger.warning(f"[UnifiedBatchQueue] {name} must be > 0, using {default}")
        return default
    return parsed


def _chunks(items: List[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


@dataclass
class BatchTask:
    task_id: str
    update_type: UpdateType
    working_dir: str
    data: Dict[str, Any]
    timestamp: float
    train_step: Optional[int] = None
    env_step: Optional[int] = None
    result: Optional[Dict[str, Any]] = None


def _normalize_content(raw: str) -> str:
    return normalize_lookup_content(raw)


def _set_task_result(
    task: BatchTask,
    status: str,
    message: str = "",
    **extra: Any,
) -> None:
    result = {
        "task_id": task.task_id,
        "status": status,
    }
    if message:
        result["message"] = message
    result.update(extra)
    task.result = result


def build_hyperedge_content_lookup(
    hyperedges_data: Dict,
    skip_deleted: bool = True,
) -> Dict[str, str]:
    """Build exact normalized_content -> hyperedge_id lookup for batch matching."""
    return build_hyperedge_content_lookup_payload(
        hyperedges_data,
        skip_deleted=skip_deleted,
    )


def find_hyperedge_by_content(
    hyperedges_data: Dict,
    content: str,
    skip_deleted: bool = True,
    lookup: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    if not content or not content.strip():
        return None

    normalized_input = _normalize_content(content)
    if not normalized_input:
        return None

    if lookup is not None:
        found_id = lookup.get(normalized_input)
        if found_id:
            return found_id

    for hyperedge_id, hyperedge_data in hyperedges_data.items():
        if not isinstance(hyperedge_data, dict):
            continue
        if skip_deleted and hyperedge_data.get("deleted", False):
            continue
        normalized_content = _normalize_content(hyperedge_data.get("content", ""))
        if normalized_content == normalized_input:
            return hyperedge_id

    if len(normalized_input) < MIN_SUBSTRING_MATCH_LEN:
        return None

    best_hyperedge_id = None
    best_ratio = 0.0
    for hyperedge_id, hyperedge_data in hyperedges_data.items():
        if not isinstance(hyperedge_data, dict):
            continue
        if skip_deleted and hyperedge_data.get("deleted", False):
            continue
        normalized_content = _normalize_content(hyperedge_data.get("content", ""))
        if not normalized_content or normalized_input not in normalized_content:
            continue
        ratio = len(normalized_input) / len(normalized_content)
        if ratio >= MIN_SUBSTRING_RATIO and ratio > best_ratio:
            best_ratio = ratio
            best_hyperedge_id = hyperedge_id
    return best_hyperedge_id


def batch_demote_graphml(working_dir: str, demoted_edges: List[tuple]) -> None:
    if demoted_edges:
        logger.info(
            "[BATCH_DEMOTE] GraphML text patch path is retired; graph persistence now uses chunk_entity_relation_graph"
        )


class UnifiedBatchQueue:
    _instance: Optional["UnifiedBatchQueue"] = None

    @classmethod
    def get_instance(cls, max_batch_size: int = None) -> "UnifiedBatchQueue":
        if cls._instance is None:
            with _queue_init_lock:
                if cls._instance is None:
                    cls._instance = cls(max_batch_size=max_batch_size)
        return cls._instance

    def __init__(self, max_batch_size: int = None):
        if max_batch_size is None:
            max_batch_size = _positive_int(os.getenv("TOOL_BATCH_SIZE", "50"), 50, "TOOL_BATCH_SIZE")
        else:
            max_batch_size = _positive_int(max_batch_size, 50, "max_batch_size")
        self.max_batch_size = max_batch_size
        self.insert_batch_size = _positive_int(
            os.getenv("TOOL_INSERT_BATCH_SIZE", str(DEFAULT_INSERT_BATCH_SIZE)),
            DEFAULT_INSERT_BATCH_SIZE,
            "TOOL_INSERT_BATCH_SIZE",
        )
        self._queue: deque[BatchTask] = deque()
        self._lock = threading.Lock()
        self._stats = {
            "total_submitted": 0,
            "total_flushed": 0,
            "total_failed": 0,
            "total_succeeded": 0,
            "total_skipped": 0,
            "flush_count": 0,
        }
        self._last_errors: List[str] = []
        self._last_task_results: List[Dict[str, Any]] = []
        self._shutdown_flag = False
        self._flush_queue: queue_module.Queue[Optional[List[BatchTask]]] = queue_module.Queue()
        self._flush_thread = threading.Thread(
            target=self._flush_worker,
            name="UnifiedBatchQueue-flush-worker",
            daemon=True,
        )
        self._flush_thread.start()
        atexit.register(self.shutdown)
        logger.info(
            f"[UnifiedBatchQueue] Initialized: max_batch_size={self.max_batch_size}, "
            f"insert_batch_size={self.insert_batch_size}"
        )

    def submit(self, task: BatchTask) -> bool:
        with self._lock:
            self._queue.append(task)
            self._stats["total_submitted"] += 1
            if len(self._queue) >= self.max_batch_size:
                self._flush_locked(flush_all=False)
        return True

    def submit_many(self, tasks: List[BatchTask]) -> List[bool]:
        results = []
        with self._lock:
            for task in tasks:
                self._queue.append(task)
                self._stats["total_submitted"] += 1
                results.append(True)
                if len(self._queue) >= self.max_batch_size:
                    self._flush_locked(flush_all=False)
        return results

    def flush(self) -> Dict[str, Any]:
        with self._lock:
            result = self._flush_locked(flush_all=True)
        self._flush_queue.join()
        return result

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "pending": len(self._queue),
                "max_batch_size": self.max_batch_size,
                "insert_batch_size": self.insert_batch_size,
                "last_errors": list(self._last_errors),
                "last_task_results": list(self._last_task_results),
            }

    def shutdown(self) -> Dict[str, Any]:
        if self._shutdown_flag:
            try:
                GraphR1BaseTool.flush_hyperedges_cache()
                GraphR1BaseTool.flush_dirty_graphs()
                GraphR1BaseTool.flush_dirty_hyperedge_indexes()
                GraphR1BaseTool.flush_dirty_entity_indexes()
            except Exception as e:
                logger.error(f"[UnifiedBatchQueue] Error flushing deferred state after repeated shutdown: {e}")
            return {"status": "already_shutdown"}
        self._shutdown_flag = True
        logger.info("[UnifiedBatchQueue] Shutting down, flushing remaining tasks...")
        result = self.flush()
        self._flush_queue.put(None)
        self._flush_thread.join(timeout=60)
        try:
            GraphR1BaseTool.flush_hyperedges_cache()
            GraphR1BaseTool.flush_dirty_graphs()
            GraphR1BaseTool.flush_dirty_hyperedge_indexes()
            GraphR1BaseTool.flush_dirty_entity_indexes()
        except Exception as e:
            logger.error(f"[UnifiedBatchQueue] Error flushing deferred state during shutdown: {e}")
        logger.info(f"[UnifiedBatchQueue] Shutdown complete. Stats: {self.get_stats()}")
        return result

    def _flush_locked(self, flush_all: bool = False) -> Dict[str, Any]:
        if not self._queue:
            return {"status": "empty", "processed": 0}

        submitted_batches = 0
        submitted_tasks = 0
        while self._queue and (flush_all or len(self._queue) >= self.max_batch_size):
            chunk_size = min(len(self._queue), self.max_batch_size)
            tasks = [self._queue.popleft() for _ in range(chunk_size)]
            self._stats["flush_count"] += 1
            self._flush_queue.put(tasks)
            submitted_batches += 1
            submitted_tasks += len(tasks)
            if not flush_all:
                break

        if submitted_tasks == 0:
            return {"status": "pending", "processed": 0}
        return {
            "status": "submitted_to_worker",
            "task_count": submitted_tasks,
            "batch_count": submitted_batches,
        }

    def _flush_worker(self) -> None:
        while True:
            tasks = self._flush_queue.get()
            if tasks is None:
                self._flush_queue.task_done()
                break
            try:
                self._execute_flush(tasks)
            except Exception as e:
                logger.error(f"[UnifiedBatchQueue] Flush worker error: {e}")
            finally:
                self._flush_queue.task_done()

    def _execute_flush(self, tasks: List[BatchTask]) -> None:
        task_working_dirs = {task.working_dir for task in tasks if task.working_dir}
        type_groups: Dict[UpdateType, List[BatchTask]] = defaultdict(list)
        for task in tasks:
            type_groups[task.update_type].append(task)

        total_processed = 0
        total_failed = 0
        total_succeeded = 0
        total_skipped = 0
        errors = []
        index_failure_reported_dirs = set()
        for update_type, typed_tasks in type_groups.items():
            dir_groups: Dict[str, List[BatchTask]] = defaultdict(list)
            for task in typed_tasks:
                dir_groups[task.working_dir].append(task)

            for working_dir, dir_tasks in dir_groups.items():
                try:
                    if update_type == UpdateType.INSERT:
                        self._process_insert_batch(dir_tasks, working_dir)
                    elif update_type == UpdateType.UPDATE:
                        self._process_update_batch(dir_tasks, working_dir)
                    elif update_type == UpdateType.DELETE:
                        self._process_delete_batch(dir_tasks, working_dir)
                    total_processed += len(dir_tasks)
                    for task in dir_tasks:
                        status = (task.result or {}).get("status")
                        if status == "succeeded":
                            total_succeeded += 1
                        elif status == "skipped":
                            total_skipped += 1
                        elif status == "failed":
                            total_failed += 1
                except Exception as e:
                    failed_dir_tasks = [
                        task for task in dir_tasks
                        if (task.result or {}).get("status") != "succeeded"
                    ]
                    for task in failed_dir_tasks:
                        if task.result is None:
                            _set_task_result(task, "failed", str(e), working_dir=working_dir)
                    total_failed += len(failed_dir_tasks)
                    if "vector index rebuild failed" in str(e):
                        index_failure_reported_dirs.add(working_dir)
                    error_message = (
                        f"{update_type.value} batch failed for {working_dir}: {e}"
                    )
                    errors.append(error_message)
                    logger.error(
                        f"[UnifiedBatchQueue] Error processing {update_type.value} batch for {working_dir}: {e}"
                    )

        with self._lock:
            self._stats["total_flushed"] += total_processed
            self._stats["total_failed"] += total_failed
            self._stats["total_succeeded"] += total_succeeded
            self._stats["total_skipped"] += total_skipped
            if errors:
                self._last_errors = (self._last_errors + errors)[-20:]
            task_results = [task.result for task in tasks if task.result is not None]
            if task_results:
                self._last_task_results = (self._last_task_results + task_results)[-50:]

        try:
            flushed_dirs = GraphR1BaseTool.flush_hyperedges_cache()
            if flushed_dirs:
                logger.info(f"[UnifiedBatchQueue] Flushed hyperedges cache for {flushed_dirs} dir(s)")
        except Exception as e:
            logger.error(f"[UnifiedBatchQueue] Error flushing hyperedges cache: {e}")

        try:
            hyperedge_flush_result = {
                "flushed": 0,
                "failed": 0,
                "failed_dirs": [],
            }
            for working_dir in task_working_dirs:
                dir_flush_result = GraphR1BaseTool.flush_dirty_hyperedge_indexes_detailed(working_dir)
                hyperedge_flush_result["flushed"] += dir_flush_result.get("flushed", 0)
                hyperedge_flush_result["failed"] += dir_flush_result.get("failed", 0)
                hyperedge_flush_result["failed_dirs"].extend(dir_flush_result.get("failed_dirs", []))
            flushed_indexes = hyperedge_flush_result.get("flushed", 0)
            failed_indexes = hyperedge_flush_result.get("failed", 0)
            if flushed_indexes:
                logger.info(f"[UnifiedBatchQueue] Rebuilt hyperedge index for {flushed_indexes} dir(s)")
            if failed_indexes:
                failed_dirs = hyperedge_flush_result.get("failed_dirs", [])
                new_failed_dirs = [
                    wd for wd in failed_dirs
                    if wd not in index_failure_reported_dirs
                ]
                with self._lock:
                    self._stats["total_failed"] += len(new_failed_dirs)
                    if new_failed_dirs:
                        self._last_errors = (
                            self._last_errors
                            + [f"hyperedge index rebuild failed for {wd}" for wd in new_failed_dirs]
                        )[-20:]
                logger.error(
                    f"[UnifiedBatchQueue] Hyperedge index rebuild failed for {failed_indexes} dir(s): {failed_dirs}"
                )
        except Exception as e:
            logger.error(f"[UnifiedBatchQueue] Error rebuilding hyperedge index: {e}")

        try:
            entity_flush_result = {
                "flushed": 0,
                "failed": 0,
                "failed_dirs": [],
            }
            for working_dir in task_working_dirs:
                dir_flush_result = GraphR1BaseTool.flush_dirty_entity_indexes_detailed(working_dir)
                entity_flush_result["flushed"] += dir_flush_result.get("flushed", 0)
                entity_flush_result["failed"] += dir_flush_result.get("failed", 0)
                entity_flush_result["failed_dirs"].extend(dir_flush_result.get("failed_dirs", []))
            flushed_indexes = entity_flush_result.get("flushed", 0)
            failed_indexes = entity_flush_result.get("failed", 0)
            if flushed_indexes:
                logger.info(f"[UnifiedBatchQueue] Rebuilt entity index for {flushed_indexes} dir(s)")
            if failed_indexes:
                failed_dirs = entity_flush_result.get("failed_dirs", [])
                new_failed_dirs = [
                    wd for wd in failed_dirs
                    if wd not in index_failure_reported_dirs
                ]
                with self._lock:
                    self._stats["total_failed"] += len(new_failed_dirs)
                    if new_failed_dirs:
                        self._last_errors = (
                            self._last_errors
                            + [f"entity index rebuild failed for {wd}" for wd in new_failed_dirs]
                        )[-20:]
                logger.error(
                    f"[UnifiedBatchQueue] Entity index rebuild failed for {failed_indexes} dir(s): {failed_dirs}"
                )
        except Exception as e:
            logger.error(f"[UnifiedBatchQueue] Error rebuilding entity index: {e}")

        logger.info(f"[UnifiedBatchQueue] Flushed {total_processed} tasks, failed {total_failed} tasks")

    def _process_insert_batch(self, tasks: List[BatchTask], working_dir: str) -> None:
        from agent.tool.tools.entity_index_sync import load_entities_data
        from agent.tool.tools.hyperedge_index_sync import load_hyperedges_data

        graphr1 = _create_graphr1_instance(working_dir)
        previous_hyperedges_data = load_hyperedges_data(working_dir)
        previous_entities_data = load_entities_data(working_dir)
        contents = []
        for task in tasks:
            raw_content = task.data.get("content", "")
            raw_items = raw_content if isinstance(raw_content, list) else [raw_content]
            for item in raw_items:
                if not isinstance(item, str):
                    continue
                content = item.strip()
                if content:
                    contents.append(content)
        if contents:
            for content_batch in _chunks(contents, self.insert_batch_size):
                graphr1.insert(content_batch)
            GraphR1BaseTool.invalidate_hyperedges_cache(working_dir)
            GraphR1BaseTool.mark_hyperedge_index_dirty(working_dir)
            GraphR1BaseTool.mark_entity_index_dirty(working_dir)
            append_recent_hyperedge_mutations(
                working_dir,
                [
                    {
                        "hyperedge_id": "",
                        "action": "insert",
                        "content": content,
                        "active": True,
                        "searchable": True,
                    }
                    for content in contents
                ],
            )
            hyperedge_index_rebuilt = GraphR1BaseTool.rebuild_hyperedge_index(
                working_dir,
                previous_hyperedges_data=previous_hyperedges_data,
                allow_prefix_reuse=True,
            )
            entity_index_rebuilt = GraphR1BaseTool.rebuild_entity_index(
                working_dir,
                previous_entities_data=previous_entities_data,
                allow_prefix_reuse=True,
            )
            if not hyperedge_index_rebuilt or not entity_index_rebuilt:
                raise RuntimeError(
                    "Knowledge inserted but vector index rebuild failed"
                )
        logger.info(
            f"[INSERT_BATCH] Inserted {len(contents)} items into {working_dir} "
            f"in chunks of <= {self.insert_batch_size}"
        )

    def _process_update_batch(self, tasks: List[BatchTask], working_dir: str) -> None:
        kv_path = os.path.join(working_dir, "kv_store_hyperedges.json")
        if not os.path.exists(kv_path):
            logger.warning(f"[UPDATE_BATCH] kv_store_hyperedges.json not found in {working_dir}")
            for task in tasks:
                _set_task_result(
                    task,
                    "failed",
                    f"kv_store_hyperedges.json not found in {working_dir}",
                    working_dir=working_dir,
                )
            return

        with open(kv_path, "r", encoding="utf-8") as f:
            hyperedges_data = json.load(f)
        previous_hyperedges_data = deepcopy(hyperedges_data)
        graphr1 = None

        updated_count = 0
        skipped_count = 0
        lookup = load_hyperedge_lookup(working_dir, skip_deleted=False) or build_hyperedge_content_lookup(
            hyperedges_data,
            skip_deleted=False,
        )
        recent_mutations = []
        for task in tasks:
            raw_content = task.data.get("content", "")
            raw_new_content = task.data.get("new_content", "")
            content = raw_content.strip() if isinstance(raw_content, str) else ""
            new_content = raw_new_content.strip() if isinstance(raw_new_content, str) else ""
            if not content or not new_content:
                skipped_count += 1
                _set_task_result(
                    task,
                    "skipped",
                    "content and new_content are required",
                    working_dir=working_dir,
                )
                continue

            found_id = find_hyperedge_by_content(
                hyperedges_data,
                content,
                skip_deleted=False,
                lookup=lookup,
            )
            if not found_id:
                skipped_count += 1
                _set_task_result(
                    task,
                    "skipped",
                    f"No knowledge item found with content containing: '{content}'",
                    working_dir=working_dir,
                )
                continue

            if graphr1 is None:
                graphr1 = _create_graphr1_instance(working_dir)
            was_deleted = bool(hyperedges_data[found_id].get("deleted", False))
            prepared = prepare_update_record(
                hyperedges_data[found_id],
                new_content=new_content,
                global_config=self._get_global_config(),
                tool_name="unified_batch_queue",
            )
            hyperedges_data[found_id] = prepared["hyperedge"]

            incident_edge_weight = None
            if was_deleted and "weight" in prepared["hyperedge"]:
                incident_edge_weight = prepared["hyperedge"]["weight"]

            run_async(
                sync_hyperedge_graph(
                    graphr1,
                    old_hyperedge_name=prepared["old_hyperedge_name"],
                    new_hyperedge_name=prepared["new_hyperedge_name"],
                    hyperedge=prepared["hyperedge"],
                    incident_edge_weight=incident_edge_weight,
                )
            )
            updated_count += 1
            lookup.pop(_normalize_content(content), None)
            lookup[_normalize_content(prepared["new_hyperedge_name"])] = found_id
            recent_mutations.append(
                {
                    "hyperedge_id": found_id,
                    "action": "update",
                    "content": prepared["new_hyperedge_name"],
                    "active": True,
                    "searchable": True,
                }
            )
            _set_task_result(
                task,
                "succeeded",
                "Knowledge item updated",
                working_dir=working_dir,
                updated_id=found_id,
            )

        if updated_count:
            _write_json_with_filelock(kv_path, hyperedges_data)
            ensure_hyperedge_state_sidecars(working_dir, hyperedges_data)
            GraphR1BaseTool.set_hyperedges_cache(working_dir, hyperedges_data, dirty=False)
            GraphR1BaseTool.mark_hyperedge_index_dirty(working_dir)
            GraphR1BaseTool.mark_graph_dirty(graphr1)
            append_recent_hyperedge_mutations(working_dir, recent_mutations)
            index_rebuilt = GraphR1BaseTool.rebuild_hyperedge_index(
                working_dir,
                hyperedges_data=hyperedges_data,
                previous_hyperedges_data=previous_hyperedges_data,
            )
            if not index_rebuilt:
                for task in tasks:
                    if (task.result or {}).get("status") == "succeeded":
                        task.result.update({
                            "status": "failed",
                            "partial_success": True,
                            "kv_graph_updated": True,
                            "hyperedge_index_rebuilt": False,
                            "message": "Knowledge item updated but hyperedge vector index rebuild failed",
                        })
                raise RuntimeError(
                    "Knowledge items updated but hyperedge vector index rebuild failed"
                )
        logger.info(
            f"[UPDATE_BATCH] Updated {updated_count}/{len(tasks)} items, "
            f"skipped {skipped_count} in {working_dir}"
        )

    def _process_delete_batch(self, tasks: List[BatchTask], working_dir: str) -> None:
        kv_path = os.path.join(working_dir, "kv_store_hyperedges.json")
        if not os.path.exists(kv_path):
            logger.warning(f"[DELETE_BATCH] kv_store_hyperedges.json not found in {working_dir}")
            for task in tasks:
                _set_task_result(
                    task,
                    "failed",
                    f"kv_store_hyperedges.json not found in {working_dir}",
                    working_dir=working_dir,
                )
            return

        with open(kv_path, "r", encoding="utf-8") as f:
            hyperedges_data = json.load(f)
        previous_hyperedges_data = deepcopy(hyperedges_data)
        graphr1 = None

        deleted_count = 0
        skipped_count = 0
        lookup = load_hyperedge_lookup(working_dir, skip_deleted=True) or build_hyperedge_content_lookup(
            hyperedges_data,
            skip_deleted=True,
        )
        recent_mutations = []
        for task in tasks:
            raw_content = task.data.get("content", "")
            content = raw_content.strip() if isinstance(raw_content, str) else ""
            if not content:
                skipped_count += 1
                _set_task_result(
                    task,
                    "skipped",
                    "content is required",
                    working_dir=working_dir,
                )
                continue

            found_id = find_hyperedge_by_content(
                hyperedges_data,
                content,
                skip_deleted=True,
                lookup=lookup,
            )
            if not found_id:
                skipped_count += 1
                _set_task_result(
                    task,
                    "skipped",
                    f"No active knowledge item found with content containing: '{content}'",
                    working_dir=working_dir,
                )
                continue

            if graphr1 is None:
                graphr1 = _create_graphr1_instance(working_dir)
            hyperedge = hyperedges_data[found_id]
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
                global_config=self._get_global_config(),
                tool_name="unified_batch_queue",
            )
            hyperedges_data[found_id] = prepared["hyperedge"]
            run_async(
                sync_hyperedge_graph(
                    graphr1,
                    old_hyperedge_name=prepared["hyperedge_name"],
                    new_hyperedge_name=prepared["hyperedge_name"],
                    hyperedge=prepared["hyperedge"],
                    incident_edge_weight=prepared["demoted_weight"],
                )
            )
            deleted_count += 1
            lookup.pop(_normalize_content(content), None)
            lookup.pop(_normalize_content(hyperedge_name), None)
            recent_mutations.append(
                {
                    "hyperedge_id": found_id,
                    "action": "delete",
                    "content": prepared["hyperedge_name"],
                    "active": False,
                    "searchable": True,
                }
            )
            _set_task_result(
                task,
                "succeeded",
                "Knowledge item soft deleted",
                working_dir=working_dir,
                deleted_id=found_id,
            )

        if deleted_count:
            _write_json_with_filelock(kv_path, hyperedges_data)
            ensure_hyperedge_state_sidecars(working_dir, hyperedges_data)
            GraphR1BaseTool.set_hyperedges_cache(working_dir, hyperedges_data, dirty=False)
            GraphR1BaseTool.mark_hyperedge_index_dirty(working_dir)
            GraphR1BaseTool.mark_graph_dirty(graphr1)
            append_recent_hyperedge_mutations(working_dir, recent_mutations)
            index_rebuilt = GraphR1BaseTool.rebuild_hyperedge_index(
                working_dir,
                hyperedges_data=hyperedges_data,
                previous_hyperedges_data=previous_hyperedges_data,
            )
            if not index_rebuilt:
                for task in tasks:
                    if (task.result or {}).get("status") == "succeeded":
                        task.result.update({
                            "status": "failed",
                            "partial_success": True,
                            "kv_graph_updated": True,
                            "hyperedge_index_rebuilt": False,
                            "message": "Knowledge item soft deleted but hyperedge vector index rebuild failed",
                        })
                raise RuntimeError(
                    "Knowledge items deleted but hyperedge vector index rebuild failed"
                )
        logger.info(
            f"[DELETE_BATCH] Deleted {deleted_count}/{len(tasks)} items, "
            f"skipped {skipped_count} in {working_dir}"
        )

    @staticmethod
    def _get_global_config() -> Dict:
        model_name = (
            os.getenv("OPENAI_MODEL")
            or os.getenv("SILICONFLOW_MODEL")
            or os.getenv("ZHIPU_MODEL")
            or os.getenv("AI_MODEL_NAME")
            or "unknown"
        )
        return {"llm_model_name": model_name}
