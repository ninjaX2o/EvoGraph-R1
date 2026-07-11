"""Shared response helpers for graph edit tool batch/deferred results."""

from __future__ import annotations

from typing import Any


def build_edit_task_response(
    *,
    success: bool,
    update_type: str | None,
    task_result: dict[str, Any] | None = None,
    task_id: str | None = None,
    task_status: str | None = None,
    train_step: int | None = None,
    env_step: int | None = None,
    message: str | None = None,
    errors: list[str] | None = None,
    blockers: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a stable edit task response consumed by text/MM tool runners."""
    result = dict(task_result or {})
    if message is not None:
        result.setdefault("message", message)
    result.setdefault("success", bool(success))

    resolved_message = str(result.get("message") or "")
    resolved_errors = _normalize_messages(errors if errors is not None else result.get("errors"))
    resolved_blockers = _normalize_messages(
        blockers if blockers is not None else result.get("blockers")
    )
    if not success and resolved_message:
        if not resolved_errors:
            resolved_errors = [resolved_message]
        if not resolved_blockers:
            resolved_blockers = [resolved_message]

    resolved_status = task_status or ("succeeded" if success else "failed")
    response = {
        **result,
        **extra,
        "success": bool(success),
        "status": resolved_status,
        "task_status": resolved_status,
        "task_id": task_id,
        "update_type": update_type,
        "train_step": train_step,
        "env_step": env_step,
        "errors": resolved_errors,
        "blockers": resolved_blockers,
        "task_result": result,
    }
    if resolved_message:
        response["message"] = resolved_message
    return response


def _normalize_messages(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]
