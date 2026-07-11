"""Layout helpers for guarded multimodal KB build outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROTECTED_EXACT = frozenset(
    {
        "script_process.py",
        "script_build.py",
        "script_api.py",
        "MULTI_AGENT_CODEX_PROMPTS.md",
        "MULTI_AGENT_CODEX_PROMPTS.zh-CN.md",
    }
)
PROTECTED_PREFIXES = (
    "expr/",
    "graphr1/",
    "agent/",
    "verl/",
    "plugins/codex-agent-team/",
)


@dataclass(frozen=True)
class MMKBLayout:
    root: Path
    dataset: str
    subset: str | None
    source_subset: str | None
    output_root: Path
    output_dir: Path
    graphr1_text_root: Path
    visual_store_root: Path
    link_store_root: Path
    indexing_store_root: Path
    processed_paths: dict[str, Path]

    @property
    def workspace_root(self) -> Path:
        return self.output_dir

    @property
    def processed_train_path(self) -> Path:
        return self.processed_paths["train"]

    @property
    def processed_test_path(self) -> Path:
        return self.processed_paths["test"]

    def ensure_directories(self) -> None:
        """Create build output directories without touching parquet files."""

        for directory in (
            self.graphr1_text_root,
            self.visual_store_root,
            self.link_store_root,
            self.indexing_store_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def build_layout(
    root: str | Path,
    dataset: str,
    subset: str | None,
    output_root: str | Path | None = None,
) -> MMKBLayout:
    root_path = Path(root)
    output_path = Path(output_root) if output_root is not None else root_path / "expr_mm"
    output_dir = output_path / dataset
    mm_store_root = output_dir / "mm_store"
    processed_root = root_path / "datasets_mm" / dataset / "processed"
    if subset is not None:
        processed_root = processed_root / subset

    return MMKBLayout(
        root=root_path,
        dataset=dataset,
        subset=subset,
        source_subset=subset,
        output_root=output_path,
        output_dir=output_dir,
        graphr1_text_root=output_dir / "graphr1_text",
        visual_store_root=mm_store_root / "visual",
        link_store_root=mm_store_root / "links",
        indexing_store_root=mm_store_root / "indexing",
        processed_paths={
            "train": processed_root / "train.parquet",
            "val": processed_root / "val.parquet",
            "test": processed_root / "test.parquet",
        },
    )


def find_protected_path_violations(
    paths: Iterable[str | Path],
    repo_root: str | Path = ".",
) -> tuple[str, ...]:
    repo_root_path = Path(repo_root).resolve()
    violations: list[str] = []

    for raw_path in paths:
        normalized = _repo_relative_posix(raw_path, repo_root_path)
        if normalized in PROTECTED_EXACT or any(
            normalized == prefix.rstrip("/") or normalized.startswith(prefix)
            for prefix in PROTECTED_PREFIXES
        ):
            violations.append(normalized)

    return tuple(violations)


def _repo_relative_posix(path: str | Path, repo_root: Path) -> str:
    path_obj = Path(path)
    comparison_path = path_obj if path_obj.is_absolute() else repo_root / path_obj
    try:
        path_obj = comparison_path.resolve().relative_to(repo_root)
    except ValueError:
        return comparison_path.resolve().as_posix()
    value = path_obj.as_posix()
    if value.startswith("./"):
        value = value[2:]
    return value
