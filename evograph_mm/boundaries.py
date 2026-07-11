"""Multimodal EvoGraph-R1 boundary constants.

This module records path and tool-name conventions only. It does not create
directories, mutate defaults, inspect EchoSight assets, or define a dataset
schema.
"""

from __future__ import annotations

from pathlib import PurePosixPath

DATASETS_MM_ROOT = "datasets_mm"
EXPR_MM_ROOT = "expr_mm"
KB_PACKAGE = "evograph_mm.kb"
MM_KB_SOURCE_PATH = "evograph_mm/kb"

MULTIMODAL_DATASET_PATH_PATTERN = "datasets_mm/<dataset>"
MULTIMODAL_EXPR_PATH_PATTERN = "expr_mm/<dataset>"

PROMPT_TOOL_NAMES = ("kb_search", "websearch", "insert", "update", "delete")
TOOL_NAME_ALIASES = {
    "GraphRetrieval": "kb_search",
    "knowledge-base": "kb_search",
    "kg_search": "kb_search",
    "metadata": "kb_search",
    "search": "kb_search",
}

ROOT_SCRIPT_POLICY = (
    "Root script_*_mm.py files should remain thin wrappers that parse CLI "
    "arguments and delegate to evograph_mm.kb."
)

OPTIONAL_MULTIMODAL_WORK = (
    "EchoSight concrete schema",
    "EchoSight asset download or audit",
    "multimodal adapters",
    "multimodal API/tool implementation",
    "Qwen2.5-VL pixel rollout",
)


def _validate_dataset_name(dataset: str) -> str:
    if not dataset or dataset != dataset.strip():
        raise ValueError("dataset must be a non-empty single directory name")

    normalized = dataset.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError("dataset must be a single relative directory name")

    return dataset


def dataset_workspace(dataset: str) -> str:
    """Return the documented multimodal dataset workspace path."""

    return str(PurePosixPath(DATASETS_MM_ROOT) / _validate_dataset_name(dataset))


def expr_workspace(dataset: str) -> str:
    """Return the documented multimodal runtime artifact workspace path."""

    return str(PurePosixPath(EXPR_MM_ROOT) / _validate_dataset_name(dataset))
