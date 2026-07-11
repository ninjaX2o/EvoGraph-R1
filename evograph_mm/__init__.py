"""Isolated multimodal EvoGraph-R1 branch package.

The text EvoGraph-R1 branch remains under the existing package and root
scripts. This package only defines the multimodal boundary for later work.
"""

from .boundaries import (
    DATASETS_MM_ROOT,
    EXPR_MM_ROOT,
    KB_PACKAGE,
    MULTIMODAL_DATASET_PATH_PATTERN,
    MULTIMODAL_EXPR_PATH_PATTERN,
    PROMPT_TOOL_NAMES,
    ROOT_SCRIPT_POLICY,
    TOOL_NAME_ALIASES,
    dataset_workspace,
    expr_workspace,
)

__all__ = [
    "DATASETS_MM_ROOT",
    "EXPR_MM_ROOT",
    "KB_PACKAGE",
    "MULTIMODAL_DATASET_PATH_PATTERN",
    "MULTIMODAL_EXPR_PATH_PATTERN",
    "PROMPT_TOOL_NAMES",
    "ROOT_SCRIPT_POLICY",
    "TOOL_NAME_ALIASES",
    "dataset_workspace",
    "expr_workspace",
]
