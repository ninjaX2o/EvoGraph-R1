"""Multimodal graph edit tools."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Dict
from urllib.parse import urlsplit, urlunsplit

import requests

from agent.tool.tool_base import Tool
from agent.tool.tools.batch.unified_batch_queue import BatchTask
from agent.tool.tools.edit_result_contract import build_edit_task_response
from agent.tool.tools.update_types import UpdateType


class _MMEditTool(Tool):
    endpoint_path = ""
    batch_endpoint_path = ""
    update_type: UpdateType | None = None

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        super().__init__(name, description, parameters)
        self.search_api_url = os.getenv(
            "MM_SEARCH_API_URL",
            "http://127.0.0.1:8003/search",
        )
        self.timeout = float(os.getenv("MM_SEARCH_TIMEOUT", "120"))
        self.max_batch_size = self._positive_int(
            os.getenv("TOOL_BATCH_SIZE", os.getenv("MM_TOOL_BATCH_SIZE", "50")),
            default=50,
        )
        self._deferred_lock = threading.Lock()
        self._deferred_tasks: list[BatchTask] = []
        self._deferred_stats: dict[str, Any] = {
            "total_submitted": 0,
            "total_flushed": 0,
            "total_failed": 0,
            "flush_count": 0,
            "last_errors": [],
            "last_task_results": [],
        }

    def execute(self, args: Dict) -> str:
        is_valid, error_msg = self.validate_args(args)
        if not is_valid:
            return json.dumps(
                {
                    "success": False,
                    "message": f"Parameter validation failed: {error_msg}",
                    "blockers": [error_msg],
                }
            )

        payload = {
            key: value
            for key, value in args.items()
            if key in self.parameters.get("properties", {})
        }
        try:
            data = self._post_json(self.endpoint_path, payload)
            if isinstance(data, dict):
                return json.dumps(data, ensure_ascii=False)
            return json.dumps(
                {
                    "success": False,
                    "message": "invalid MM graph edit response",
                    "blockers": ["invalid response"],
                }
            )
        except Exception as exc:
            return json.dumps(
                {
                    "success": False,
                    "message": str(exc),
                    "blockers": [str(exc)],
                }
            )

    def batch_execute(self, args_list: list[Dict]) -> list[str]:
        results: list[dict[str, Any] | None] = [None] * len(args_list)
        payload_items: list[dict[str, Any]] = []
        payload_positions: list[int] = []
        tasks: list[BatchTask] = []
        for index, args in enumerate(args_list):
            is_valid, error_msg = self.validate_args(args)
            if not is_valid:
                results[index] = self._validation_error(error_msg)
                continue
            payload_positions.append(index)
            payload = self._filter_payload(args)
            payload_items.append(payload)
            tasks.append(
                self._new_batch_task(
                    payload,
                    suffix=f"batch{index}",
                )
            )
        if not payload_items:
            return [json.dumps(item, ensure_ascii=False) for item in results if item is not None]
        try:
            data = self._post_json(self.batch_endpoint_path, {"items": payload_items})
        except Exception as exc:
            failure = {"success": False, "message": str(exc), "blockers": [str(exc)]}
            for offset, position in enumerate(payload_positions):
                results[position] = self._format_task_response(tasks[offset], failure)
            return [
                json.dumps(item or self._invalid_batch_response(), ensure_ascii=False)
                for item in results
            ]
        api_results = data.get("results") if isinstance(data, dict) else None
        if isinstance(api_results, list):
            for offset, position in enumerate(payload_positions):
                if offset < len(api_results) and isinstance(api_results[offset], dict):
                    results[position] = self._format_task_response(
                        tasks[offset],
                        api_results[offset],
                    )
                else:
                    results[position] = self._format_task_response(
                        tasks[offset],
                        self._invalid_batch_response(),
                    )
        else:
            failure = data if isinstance(data, dict) and data.get("success") is False else self._invalid_batch_response()
            for offset, position in enumerate(payload_positions):
                results[position] = self._format_task_response(tasks[offset], dict(failure))
        return [
            json.dumps(item or self._invalid_batch_response(), ensure_ascii=False)
            for item in results
        ]

    def time_batch_submit(self, args: Dict, train_step: int, env_step: int) -> str:
        is_valid, error_msg = self.validate_args(args)
        if not is_valid:
            return json.dumps(
                self._validation_error(error_msg),
                ensure_ascii=False,
            )
        payload = self._filter_payload(args)
        task = self._new_batch_task(
            payload,
            suffix=f"train{train_step}_env{env_step}",
            train_step=train_step,
            env_step=env_step,
        )
        try:
            with self._deferred_lock:
                self._deferred_tasks.append(task)
                self._deferred_stats["total_submitted"] += 1
                pending = len(self._deferred_tasks)
            flush_result = None
            if pending >= self.max_batch_size:
                flush_result = self._flush_deferred(flush_all=False)
            pending_after_flush = self.get_time_batch_stats()["pending"]
            auto_flushed = flush_result is not None
            flush_success = True
            task_status = "queued"
            flush_errors: list[str] = []
            task_result: dict[str, Any] = {
                "success": True,
                "deferred": True,
                "queued": True,
                "pending": pending_after_flush,
                "auto_flushed": auto_flushed,
            }
            if auto_flushed:
                flushed_task = self._find_flushed_task_response(flush_result, task.task_id)
                if flushed_task is not None:
                    flush_success = bool(flushed_task.get("success", False))
                    task_status = str(
                        flushed_task.get("task_status")
                        or flushed_task.get("status")
                        or ("succeeded" if flush_success else "failed")
                    )
                    raw_task_result = flushed_task.get("task_result")
                    task_result = (
                        dict(raw_task_result)
                        if isinstance(raw_task_result, dict)
                        else {
                            "success": flush_success,
                            "message": flushed_task.get("message"),
                        }
                    )
                    task_result.setdefault("success", flush_success)
                    if flushed_task.get("message"):
                        task_result.setdefault("message", flushed_task.get("message"))
                else:
                    flush_success = bool(flush_result.get("success", False))
                    task_status = "succeeded" if flush_success else "failed"
                if not flush_success:
                    flush_errors = self._flush_failure_messages(
                        {"task_results": [flushed_task]} if flushed_task is not None else flush_result
                    )
            task_result.update(
                {
                    "deferred": True,
                    "queued": True,
                    "pending": pending_after_flush,
                    "auto_flushed": auto_flushed,
                }
            )
            if flush_errors:
                task_result["message"] = flush_errors[0]
            if flush_success:
                return json.dumps({"success": True}, ensure_ascii=False)
            return json.dumps(
                build_edit_task_response(
                    success=flush_success,
                    update_type=task.update_type.value,
                    task_result=task_result,
                    task_id=task.task_id,
                    task_status=task_status,
                    train_step=train_step,
                    env_step=env_step,
                    errors=flush_errors,
                    blockers=flush_errors,
                    flush_result=flush_result,
                    deferred=True,
                    queued=True,
                    pending=pending_after_flush,
                    auto_flushed=auto_flushed,
                ),
                ensure_ascii=False,
            )
        except Exception as exc:
            return json.dumps(
                {
                    "success": False,
                    "deferred": True,
                    "message": str(exc),
                    "blockers": [str(exc)],
                },
                ensure_ascii=False,
            )

    def process_time_batch(self, train_step: int) -> Dict[str, Any]:
        del train_step
        return self.flush_deferred()

    def flush_deferred(self) -> Dict[str, Any]:
        return self._flush_deferred(flush_all=True)

    def get_time_batch_stats(self) -> Dict[str, Any]:
        with self._deferred_lock:
            return {
                **self._deferred_stats,
                "pending": len(self._deferred_tasks),
                "max_batch_size": self.max_batch_size,
            }

    def _post_json(self, endpoint_path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            self._endpoint_url(endpoint_path),
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            return {
                "success": False,
                "message": f"MM graph edit API returned {response.status_code}",
                "blockers": [str(response.status_code)],
            }
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _endpoint_url(self, endpoint_path: str | None = None) -> str:
        parsed = urlsplit(self.search_api_url)
        base_path = parsed.path.rsplit("/", 1)[0] if parsed.path else ""
        path = endpoint_path if endpoint_path is not None else self.endpoint_path
        endpoint = f"{base_path}/{path.lstrip('/')}"
        return urlunsplit((parsed.scheme, parsed.netloc, endpoint, "", ""))

    def _flush_deferred(self, *, flush_all: bool) -> Dict[str, Any]:
        tasks = self._pop_deferred_tasks(flush_all=flush_all)
        if not tasks:
            return {
                "success": True,
                "status": "idle",
                "flushed": 0,
                "pending": self.get_time_batch_stats()["pending"],
            }
        payload_items = [task.data for task in tasks]
        try:
            data = self._post_json(self.batch_endpoint_path, {"items": payload_items})
            api_results = data.get("results") if isinstance(data, dict) else None
            if not isinstance(api_results, list):
                api_results = [
                    data
                    if isinstance(data, dict) and data.get("success") is False
                    else self._invalid_batch_response()
                    for _ in tasks
                ]
        except Exception as exc:
            api_results = [
                {
                    "success": False,
                    "message": str(exc),
                    "blockers": [str(exc)],
                }
                for _ in tasks
            ]
        task_results: list[dict[str, Any]] = []
        failed = 0
        for index, task in enumerate(tasks):
            result = (
                api_results[index]
                if index < len(api_results) and isinstance(api_results[index], dict)
                else self._invalid_batch_response()
            )
            if not result.get("success", False):
                failed += 1
            task_response = self._format_task_response(task, result)
            task.result = {
                "task_id": task.task_id,
                "status": task_response["task_status"],
                "working_dir": task.working_dir,
                "update_type": task.update_type.value,
                "task_result": result,
            }
            task_results.append(task_response)
        with self._deferred_lock:
            self._deferred_stats["total_flushed"] += len(tasks)
            self._deferred_stats["total_failed"] += failed
            self._deferred_stats["flush_count"] += 1
            if failed:
                errors = [
                    str(
                        item.get("task_result", {}).get(
                            "message",
                            "MM graph edit batch failed",
                        )
                    )
                    for item in task_results
                    if item.get("status") == "failed"
                ]
                self._deferred_stats["last_errors"] = (
                    self._deferred_stats["last_errors"] + errors
                )[-20:]
            self._deferred_stats["last_task_results"] = (
                self._deferred_stats["last_task_results"] + task_results
            )[-50:]
            pending = len(self._deferred_tasks)
        return {
            "success": failed == 0,
            "status": "succeeded" if failed == 0 else "partial_failed",
            "flushed": len(tasks),
            "failed": failed,
            "pending": pending,
            "task_results": task_results,
        }

    def _pop_deferred_tasks(self, *, flush_all: bool) -> list[BatchTask]:
        with self._deferred_lock:
            if not self._deferred_tasks:
                return []
            if not flush_all and len(self._deferred_tasks) < self.max_batch_size:
                return []
            count = len(self._deferred_tasks) if flush_all else self.max_batch_size
            tasks = self._deferred_tasks[:count]
            del self._deferred_tasks[:count]
            return tasks

    def _filter_payload(self, args: Dict) -> dict[str, Any]:
        return {
            key: value
            for key, value in args.items()
            if key in self.parameters.get("properties", {})
        }

    def _new_batch_task(
        self,
        payload: dict[str, Any],
        *,
        suffix: str,
        train_step: int | None = None,
        env_step: int | None = None,
    ) -> BatchTask:
        if self.update_type is None:
            raise ValueError(f"{self.name} does not define update_type")
        task_id = f"{self.update_type.value}_{suffix}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        return BatchTask(
            task_id=task_id,
            update_type=self.update_type,
            working_dir=self._mm_batch_working_dir(),
            data=dict(payload),
            timestamp=time.time(),
            train_step=train_step,
            env_step=env_step,
        )

    def _format_task_response(
        self,
        task: BatchTask,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        success = bool(result.get("success", False))
        return build_edit_task_response(
            success=success,
            update_type=task.update_type.value,
            task_result=result,
            task_id=task.task_id,
            train_step=task.train_step,
            env_step=task.env_step,
        )

    def _mm_batch_working_dir(self) -> str:
        return f"mm_api:{self._endpoint_url(self.batch_endpoint_path)}"

    @staticmethod
    def _positive_int(raw: Any, *, default: int) -> int:
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    def _validation_error(self, error_msg: str) -> dict[str, Any]:
        result = {
            "success": False,
            "message": f"Parameter validation failed: {error_msg}",
            "blockers": [error_msg],
        }
        update_type = self.update_type.value if self.update_type is not None else None
        return build_edit_task_response(
            success=False,
            update_type=update_type,
            task_result=result,
            errors=[error_msg],
            blockers=[error_msg],
        )

    @staticmethod
    def _invalid_batch_response() -> dict[str, Any]:
        return {
            "success": False,
            "message": "invalid MM graph edit batch response",
            "blockers": ["invalid response"],
        }

    @staticmethod
    def _find_flushed_task_response(
        flush_result: dict[str, Any],
        task_id: str,
    ) -> dict[str, Any] | None:
        for item in flush_result.get("task_results", []):
            if isinstance(item, dict) and item.get("task_id") == task_id:
                return item
        return None

    @staticmethod
    def _flush_failure_messages(flush_result: dict[str, Any]) -> list[str]:
        messages: list[str] = []
        for item in flush_result.get("task_results", []):
            if not isinstance(item, dict):
                continue
            if item.get("success", False) or item.get("status") == "succeeded":
                continue
            task_result = item.get("task_result")
            if isinstance(task_result, dict):
                message = task_result.get("message")
                if message:
                    messages.append(str(message))
                    continue
            message = item.get("message")
            if message:
                messages.append(str(message))
        if not messages:
            message = flush_result.get("message")
            if message:
                messages.append(str(message))
        if not messages:
            messages.append("MM graph edit auto-flush failed")
        return messages

    def calculate_reward(self, args: Dict, result: str) -> float:
        del args
        try:
            payload = json.loads(result)
        except Exception:
            return -0.1
        if isinstance(payload, dict) and payload.get("success", False):
            return 0.1
        return -0.1


class MMGraphR1InsertTool(_MMEditTool):
    """Insert a multimodal graph knowledge fragment."""

    endpoint_path = "/insert"
    batch_endpoint_path = "/batch/insert"
    update_type = UpdateType.INSERT

    def __init__(self):
        parameters = {
            "type": "object",
            "properties": {
                "content": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Full text (or list of texts) to add to the KB.",
                },
            },
            "required": ["content"],
        }
        super().__init__(
            "insert",
            "Insert new knowledge not in the KB. Provide the full content to add.",
            parameters,
        )


class MMHyperedgeUpdateTool(_MMEditTool):
    """Update a multimodal graph hyperedge."""

    endpoint_path = "/hyperedge/update"
    batch_endpoint_path = "/batch/hyperedge/update"
    update_type = UpdateType.UPDATE

    def __init__(self):
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
        super().__init__(
            "update",
            "Update knowledge item content in knowledge base.",
            parameters,
        )


class MMHyperedgeSoftDeleteTool(_MMEditTool):
    """Soft-delete a multimodal graph hyperedge."""

    endpoint_path = "/hyperedge/delete"
    batch_endpoint_path = "/batch/hyperedge/delete"
    update_type = UpdateType.DELETE

    def __init__(self):
        parameters = {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The original content of the knowledge item from knowledge base",
                },
            },
            "required": ["content"],
        }
        super().__init__(
            "delete",
            "Soft delete knowledge item from knowledge base.",
            parameters,
        )


__all__ = [
    "MMGraphR1InsertTool",
    "MMHyperedgeUpdateTool",
    "MMHyperedgeSoftDeleteTool",
]
