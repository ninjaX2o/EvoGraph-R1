"""FastAPI wrapper for multimodal KB retrieval and GraphR1-native graph edits."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from threading import RLock
from typing import Any, Callable

from fastapi import FastAPI
from pydantic import BaseModel, Field

from evograph_mm.kb import graph_edit
from evograph_mm.kb.retrieval import (
    DEFAULT_VISUAL_ENTITY_TOP_K,
    DEFAULT_VISUAL_HYPEREDGE_TOP_K,
    DEFAULT_VISUAL_HYPEREDGE_MIN_COHERENCE,
    MMKBRetriever,
)


class MMSearchRequest(BaseModel):
    queries: list[str] = Field(default_factory=list)
    context_queries: list[str] = Field(default_factory=list)
    entity_top_k: int = 5
    hyperedge_top_k: int = 5
    rag_top_k: int = 10
    image_ids: list[str] = Field(default_factory=list)
    image_paths: list[str] = Field(default_factory=list)
    image_top_k: int = 5
    entity_fusion_top_k: int = 0
    entity_fusion_candidate_top_k: int = 20
    visual_entity_top_k: int = DEFAULT_VISUAL_ENTITY_TOP_K
    visual_entity_candidate_top_k: int = 20
    visual_hyperedge_top_k: int = DEFAULT_VISUAL_HYPEREDGE_TOP_K
    visual_hyperedge_candidate_top_k: int = 1000
    visual_hyperedge_min_coherence: float = DEFAULT_VISUAL_HYPEREDGE_MIN_COHERENCE
    fused_top_k: int = 0
    dataset: str | None = None
    subset: str | None = None


class ReloadRequest(BaseModel):
    force: bool = False
    dataset: str | None = None
    subset: str | None = None
    working_dir: str | None = None


class MMEditScope(BaseModel):
    dataset: str | None = None
    subset: str | None = None
    working_dir: str | None = None
    image_id: str | None = None
    image_path: str | None = None
    data_id: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class MMInsertRequest(MMEditScope):
    content: str | list[str]


class MMHyperedgeUpdateRequest(MMEditScope):
    content: str
    new_content: str


class MMHyperedgeDeleteRequest(MMEditScope):
    content: str


class MMBatchInsertRequest(BaseModel):
    items: list[MMInsertRequest]


class MMBatchHyperedgeUpdateRequest(BaseModel):
    items: list[MMHyperedgeUpdateRequest]


class MMBatchHyperedgeDeleteRequest(BaseModel):
    items: list[MMHyperedgeDeleteRequest]


class MMAPIState:
    def __init__(
        self,
        working_dir: str | Path | None = None,
        model_path: str | Path = Path("runtime_encoder_disabled"),
        output_root: str | Path = "expr_mm",
        dataset: str | None = "E-VQA",
        subset: str | None = "paper_tar_5120_128_seed0",
        encoder_factory: Callable[..., Any] | None = None,
        rag_factory: Callable[..., Any] | None = None,
        reload_interval: int = 300,
        check_interval: int = 60,
    ) -> None:
        self.model_path = Path(model_path)
        self.output_root = Path(output_root)
        self.dataset = dataset
        self.subset = subset
        self.reload_interval = reload_interval
        self.check_interval = check_interval
        self.edit_reload_policy = self._resolve_edit_reload_policy()
        self.working_dir = (
            Path(working_dir)
            if working_dir is not None
            else self._default_working_dir(dataset, subset)
        )
        self.base_working_dir = self.working_dir
        self.encoder_factory = encoder_factory
        self.rag_factory = rag_factory
        self.retriever: MMKBRetriever | None = None
        self.last_check_time = 0.0
        self.last_reload_mtime = 0.0
        self.lock = RLock()
        self.reload(
            working_dir=self.working_dir,
            dataset=self.dataset,
            subset=self.subset,
        )

    def reload(
        self,
        force: bool = False,
        dataset: str | None = None,
        subset: str | None = None,
        working_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        del force
        with self.lock:
            next_dataset = dataset if dataset is not None else self.dataset
            next_subset = subset if subset is not None else self.subset
            explicit_working_dir = working_dir is not None
            if explicit_working_dir:
                next_working_dir = Path(working_dir)
                self.base_working_dir = next_working_dir
            elif next_dataset != self.dataset or next_subset != self.subset:
                next_working_dir = self._default_working_dir(next_dataset, next_subset)
                self.base_working_dir = next_working_dir
            elif self.working_dir is not None:
                next_working_dir = self.working_dir
            else:
                next_working_dir = self._default_working_dir(next_dataset, next_subset)
                self.base_working_dir = next_working_dir

            self.working_dir = next_working_dir
            self.dataset = next_dataset
            self.subset = next_subset
            self.retriever = MMKBRetriever(
                working_dir=self.working_dir,
                model_path=self.model_path,
                encoder_factory=self.encoder_factory,
                rag_factory=self.rag_factory,
                dataset=self.dataset,
                subset=self.subset,
            )
            status = self.retriever.load()
            self.base_working_dir = self._loaded_base_working_dir(next_working_dir)
            self.last_reload_mtime = self._max_watched_mtime(self.working_dir)
            return status

    def status(self) -> dict[str, Any]:
        with self.lock:
            if self.retriever is None:
                return {
                    "status": "blocked",
                    "dataset": self.dataset,
                    "subset": self.subset,
                    "working_dir": str(self.working_dir) if self.working_dir else None,
                    "base_working_dir": str(self.base_working_dir)
                    if self.base_working_dir
                    else None,
                    "model_path": str(self.model_path),
                    "model_loaded": False,
                    "indexes": {},
                    "counts": {},
                    "blockers": ["retriever is not loaded"],
                    "last_reload_time": None,
                    **self._reload_config_status(),
                }
            status = self.retriever.status()
            status.update(self._reload_config_status())
            status["base_working_dir"] = (
                str(self.base_working_dir) if self.base_working_dir else None
            )
            return status

    def _reload_config_status(self) -> dict[str, Any]:
        return {
            "auto_reload_enabled": self.reload_interval > 0,
            "reload_interval": self.reload_interval,
            "check_interval": self.check_interval,
            "last_reload_mtime": self.last_reload_mtime,
            "edit_reload_policy": self.edit_reload_policy,
        }

    def _resolve_edit_reload_policy(self) -> str:
        policy = os.getenv("MM_EDIT_RELOAD_POLICY", "immediate").strip().lower()
        if policy in {"immediate", "periodic"}:
            return policy
        return "immediate"

    def reload_if_changed(self, *, force: bool = False) -> dict[str, Any] | None:
        now = time.time()
        if not force and now - self.last_check_time < self.check_interval:
            return None
        self.last_check_time = now
        with self.lock:
            current_mtime = self._max_watched_mtime(self.working_dir)
            if not force and current_mtime <= self.last_reload_mtime:
                return None
            return self.reload(
                force=True,
                dataset=self.dataset,
                subset=self.subset,
                working_dir=self.working_dir,
            )

    def _watched_paths(self, working_dir: str | Path | None = None) -> list[Path]:
        if working_dir is None:
            working_dir = self.working_dir
        if working_dir is None:
            return []
        base = Path(working_dir)
        names = [
            "metadata.json",
            "kv_store_entities.json",
            "kv_store_hyperedges.json",
            "kv_store_text_chunks.json",
            "kv_store_chunks.json",
            "graph_chunk_entity_relation.graphml",
            "index.bin",
            "corpus.npy",
            "index_entity.bin",
            "corpus_entity.npy",
            "entity_index_metadata.json",
            "index_hyperedge.bin",
            "corpus_hyperedge.npy",
            "hyperedge_index_metadata.json",
            "hyperedge_content_lookup.json",
            "hyperedge_recent_mutations.json",
            "image_records.jsonl",
            "image_index.faiss",
            "image_embeddings.npy",
        ]
        paths = [base / name for name in names]
        paths.extend((base / "graphr1_text").glob("*"))
        return paths

    def _max_watched_mtime(self, working_dir: str | Path | None = None) -> float:
        max_mtime = 0.0
        for path in self._watched_paths(working_dir):
            try:
                if path.is_file():
                    max_mtime = max(max_mtime, os.path.getmtime(path))
            except OSError:
                continue
        return max_mtime

    def _default_working_dir(
        self,
        dataset: str | None,
        subset: str | None,
    ) -> Path:
        del subset
        if dataset:
            return self.output_root / dataset
        return self.output_root

    def _loaded_base_working_dir(self, working_dir: Path) -> Path:
        if self.retriever is None:
            return working_dir
        base_output_dir = self.retriever.metadata.get("base_output_dir")
        if isinstance(base_output_dir, str) and base_output_dir:
            return Path(base_output_dir)
        return working_dir

    def insert(self, request: MMInsertRequest) -> dict[str, Any]:
        contents = self._clean_contents(request.content)
        if not contents:
            message = "Knowledge content cannot be empty"
            return self._edit_response(
                success=False,
                action="insert",
                message=message,
                blockers=[message],
            )
        return self._apply_edit("insert", request, contents=contents)

    def update(self, request: MMHyperedgeUpdateRequest) -> dict[str, Any]:
        content = self._clean_content(request.content)
        new_content = self._clean_content(request.new_content)
        if not content:
            message = "Content cannot be empty"
            return self._edit_response(
                success=False,
                action="update",
                message=message,
                blockers=[message],
            )
        if not new_content:
            message = "New content cannot be empty"
            return self._edit_response(
                success=False,
                action="update",
                message=message,
                blockers=[message],
            )
        return self._apply_edit(
            "update",
            request,
            contents=[content],
            new_content=new_content,
        )

    def delete(self, request: MMHyperedgeDeleteRequest) -> dict[str, Any]:
        content = self._clean_content(request.content)
        if not content:
            message = "Content cannot be empty"
            return self._edit_response(
                success=False,
                action="delete",
                message=message,
                blockers=[message],
            )
        return self._apply_edit(
            "delete",
            request,
            contents=[content],
        )

    def insert_batch(self, request: MMBatchInsertRequest) -> dict[str, Any]:
        return self._apply_batch_edit("insert", request.items)

    def update_batch(self, request: MMBatchHyperedgeUpdateRequest) -> dict[str, Any]:
        return self._apply_batch_edit("update", request.items)

    def delete_batch(self, request: MMBatchHyperedgeDeleteRequest) -> dict[str, Any]:
        return self._apply_batch_edit("delete", request.items)

    @staticmethod
    def _clean_content(content: str) -> str:
        return content.strip()

    def _clean_contents(self, content: str | list[str]) -> list[str]:
        raw_contents = content if isinstance(content, list) else [content]
        return [cleaned for item in raw_contents if (cleaned := self._clean_content(item))]

    def _apply_edit(
        self,
        action: str,
        request: MMEditScope,
        *,
        contents: list[str],
        new_content: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            return self._apply_edit_unlocked(
                action,
                request,
                contents=contents,
                new_content=new_content,
            )

    def _apply_edit_unlocked(
        self,
        action: str,
        request: MMEditScope,
        *,
        contents: list[str],
        new_content: str | None = None,
    ) -> dict[str, Any]:
        scope_error = self._edit_scope_error(request)
        if scope_error:
            return self._edit_response(
                success=False,
                action=action,
                message=scope_error,
                blockers=[scope_error],
            )
        try:
            edit_working_dir = graph_edit.ensure_edit_working_dir(
                self.base_working_dir,
                self.working_dir,
            )
        except Exception as exc:
            return self._edit_response(
                success=False,
                action=action,
                message=str(exc),
                blockers=[str(exc)],
            )
        if self.retriever is None or self.retriever.blockers:
            message = "multimodal KB retriever is not ready for graph edits"
            blockers = []
            if self.retriever is not None:
                blockers = list(self.retriever.blockers)
            return self._edit_response(
                success=False,
                action=action,
                message=message,
                blockers=blockers or [message],
            )

        checkpoint: dict[str, bytes | None] | None = None
        try:
            checkpoint = graph_edit.create_edit_checkpoint(edit_working_dir)
            edit_result = self._run_graph_edit(
                action,
                edit_working_dir,
                request,
                contents=contents,
                new_content=new_content,
            )
            text_indexes = graph_edit.rebuild_graph_text_indexes(
                working_dir=edit_working_dir,
                include_entities=action == "insert",
                include_hyperedges=True,
            )
            if self.edit_reload_policy == "immediate":
                self._reload_after_edit(edit_working_dir, checkpoint)
        except Exception as exc:
            if checkpoint is not None:
                graph_edit.restore_edit_checkpoint(edit_working_dir, checkpoint)
            return self._edit_response(
                success=False,
                action=action,
                message=str(exc),
                blockers=[str(exc)],
            )

        ids = [str(item) for item in edit_result.get("ids", []) if item]
        extra: dict[str, Any] = {
            **edit_result,
            "ids": ids,
            "native_edit": True,
            "native_graph_edit": True,
            "copied": False,
            "base_working_dir": str(self.base_working_dir) if self.base_working_dir else None,
            "text_indexes": text_indexes,
        }
        if action == "update" and ids:
            extra["updated_id"] = ids[-1]
        if action == "delete" and ids:
            extra["deleted_id"] = ids[-1]
        return self._edit_response(
            success=True,
            action=action,
            message=f"MM graph {action} applied",
            searchable=action != "delete",
            extra=extra,
        )

    def _apply_batch_edit(
        self,
        action: str,
        items: list[MMEditScope],
    ) -> dict[str, Any]:
        with self.lock:
            try:
                edit_working_dir = graph_edit.ensure_edit_working_dir(
                    self.base_working_dir,
                    self.working_dir,
                )
            except Exception as exc:
                return self._batch_response(
                    success=False,
                    action=action,
                    message=str(exc),
                    results=[],
                    blockers=[str(exc)],
                )
            if self.retriever is None or self.retriever.blockers:
                message = "multimodal KB retriever is not ready for graph edits"
                blockers = []
                if self.retriever is not None:
                    blockers = list(self.retriever.blockers)
                return self._batch_response(
                    success=False,
                    action=action,
                    message=message,
                    results=[],
                    blockers=blockers or [message],
                )

            checkpoint: dict[str, bytes | None] | None = None
            results: list[dict[str, Any]] = []
            any_applied = False
            try:
                checkpoint = graph_edit.create_edit_checkpoint(edit_working_dir)
                if action == "insert":
                    insert_results: list[dict[str, Any] | None] = [None] * len(items)
                    clean_items: list[tuple[int, MMInsertRequest, list[str]]] = []
                    for index, item in enumerate(items):
                        scope_error = self._edit_scope_error(item)
                        if scope_error:
                            insert_results[index] = {"success": False, "message": scope_error}
                            continue
                        contents = self._clean_contents(item.content)  # type: ignore[attr-defined]
                        if not contents:
                            insert_results[index] = {
                                "success": False,
                                "message": "Knowledge content cannot be empty",
                            }
                            continue
                        clean_items.append((index, item, contents))  # type: ignore[arg-type]
                    for index, item, contents in clean_items:
                        try:
                            edit_result = graph_edit.apply_graph_insert(
                                working_dir=edit_working_dir,
                                contents=contents,
                                image_id=item.image_id,
                                image_path=item.image_path,
                                data_id=item.data_id,
                                source_metadata=item.source_metadata,
                            )
                            any_applied = True
                            insert_results[index] = {
                                "success": True,
                                "message": "MM graph insert applied",
                                "native_graph_insert": True,
                                **edit_result,
                            }
                        except Exception as exc:
                            insert_results[index] = {"success": False, "message": str(exc)}
                            results.extend(
                                result if result is not None else {"success": False, "message": "MM graph insert was not processed"}
                                for result in insert_results
                            )
                            raise
                    results.extend(
                        result if result is not None else {"success": False, "message": "MM graph insert was not processed"}
                        for result in insert_results
                    )
                else:
                    for item in items:
                        scope_error = self._edit_scope_error(item)
                        if scope_error:
                            results.append({"success": False, "message": scope_error})
                            continue
                        try:
                            if action == "update":
                                content = self._clean_content(item.content)  # type: ignore[attr-defined]
                                new_content = self._clean_content(item.new_content)  # type: ignore[attr-defined]
                                if not content:
                                    results.append(
                                        {
                                            "success": False,
                                            "message": "Content cannot be empty",
                                        }
                                    )
                                    continue
                                if not new_content:
                                    results.append(
                                        {
                                            "success": False,
                                            "message": "New content cannot be empty",
                                        }
                                    )
                                    continue
                                result = graph_edit.apply_hyperedge_update(
                                    working_dir=edit_working_dir,
                                    content=content,
                                    new_content=new_content,
                                    image_id=item.image_id,
                                    image_path=item.image_path,
                                    data_id=item.data_id,
                                    source_metadata=item.source_metadata,
                                )
                            elif action == "delete":
                                result = graph_edit.apply_hyperedge_soft_delete(
                                    working_dir=edit_working_dir,
                                    content=self._clean_content(item.content),  # type: ignore[attr-defined]
                                )
                            else:
                                raise ValueError(f"unsupported MM graph edit action: {action}")
                            any_applied = True
                            results.append({"success": True, "message": f"MM graph {action} applied", **result})
                        except Exception as exc:
                            results.append({"success": False, "message": str(exc)})
                            raise
                text_indexes = None
                if any_applied:
                    text_indexes = graph_edit.rebuild_graph_text_indexes(
                        working_dir=edit_working_dir,
                        include_entities=action == "insert",
                        include_hyperedges=True,
                    )
                    if self.edit_reload_policy == "immediate":
                        self._reload_after_edit(edit_working_dir, checkpoint)
            except Exception as exc:
                if checkpoint is not None:
                    graph_edit.restore_edit_checkpoint(edit_working_dir, checkpoint)
                results = self._mark_batch_results_rolled_back(results, str(exc))
                return self._batch_response(
                    success=False,
                    action=action,
                    message=str(exc),
                    results=results,
                    blockers=[str(exc)],
                )
            return self._batch_response(
                success=all(result.get("success") for result in results) if results else False,
                action=action,
                message=f"MM graph {action} batch applied",
                results=results,
                extra={
                    "copied": False,
                    "native_graph_edit": True,
                    "text_indexes": text_indexes,
                },
            )

    def _run_graph_edit(
        self,
        action: str,
        edit_working_dir: Path,
        request: MMEditScope,
        *,
        contents: list[str],
        new_content: str | None = None,
    ) -> dict[str, Any]:
        if action == "insert":
            return graph_edit.apply_graph_insert(
                working_dir=edit_working_dir,
                contents=contents,
                image_id=request.image_id,
                image_path=request.image_path,
                data_id=request.data_id,
                source_metadata=request.source_metadata,
            )
        if action == "update":
            if new_content is None:
                raise ValueError("new_content is required for update")
            return graph_edit.apply_hyperedge_update(
                working_dir=edit_working_dir,
                content=contents[0],
                new_content=new_content,
                image_id=request.image_id,
                image_path=request.image_path,
                data_id=request.data_id,
                source_metadata=request.source_metadata,
            )
        if action == "delete":
            return graph_edit.apply_hyperedge_soft_delete(
                working_dir=edit_working_dir,
                content=contents[0],
            )
        raise ValueError(f"unsupported MM graph edit action: {action}")

    def _reload_after_edit(
        self,
        edit_working_dir: Path,
        checkpoint: dict[str, bytes | None],
    ) -> None:
        next_retriever = MMKBRetriever(
            working_dir=edit_working_dir,
            model_path=self.model_path,
            encoder_factory=self.encoder_factory,
            rag_factory=self.rag_factory,
            dataset=self.dataset,
            subset=self.subset,
        )
        reload_status = next_retriever.load()
        if reload_status.get("status") != "ready":
            graph_edit.restore_edit_checkpoint(edit_working_dir, checkpoint)
            blockers = list(reload_status.get("blockers", []))
            raise RuntimeError(
                "GraphR1-native edit applied but edited MM KB reload failed: "
                + "; ".join(blockers)
            )
        self.working_dir = edit_working_dir
        self.retriever = next_retriever
        self.last_reload_mtime = self._max_watched_mtime(edit_working_dir)

    def _edit_scope_error(self, request: MMEditScope) -> str:
        requested_dataset = request.dataset
        current_dataset = self.dataset
        raw_metadata_dataset = None
        metadata_subset = None
        metadata_output_dir = None
        loaded_subsets = {value for value in (self.subset,) if value is not None}
        if self.retriever is not None:
            raw_metadata_dataset = self.retriever.metadata.get("dataset")
            metadata_subset = (
                self.retriever.metadata.get("source_subset")
                or self.retriever.metadata.get("subset")
            )
            metadata_output_dir = self.retriever.metadata.get("output_dir")
            loaded_subsets.update(
                value
                for value in (
                    self.retriever.subset,
                    metadata_subset,
                )
                if value is not None
            )
        if requested_dataset is not None and requested_dataset != "E-VQA":
            return "Task H multimodal graph edits are restricted to E-VQA"
        if current_dataset is not None and current_dataset != "E-VQA":
            return "Task H multimodal graph edits are restricted to E-VQA"
        if raw_metadata_dataset is not None and raw_metadata_dataset != "E-VQA":
            return "Task H multimodal graph edits are restricted to E-VQA"
        if metadata_subset is not None:
            configured_subsets = [
                value
                for value in (
                    self.subset,
                    self.retriever.subset if self.retriever is not None else None,
                )
                if value is not None
            ]
            if any(value != metadata_subset for value in configured_subsets):
                return "configured subset does not match the loaded E-VQA KB metadata"
        if (
            request.subset is not None
            and metadata_subset is not None
            and request.subset != metadata_subset
        ):
            return "requested subset does not match the loaded E-VQA KB"
        if (
            request.subset is not None
            and metadata_subset is None
            and loaded_subsets
            and request.subset not in loaded_subsets
        ):
            return "requested subset does not match the loaded E-VQA KB"
        if metadata_output_dir is not None and not self._same_working_dir(
            metadata_output_dir,
            self.working_dir,
        ):
            return "metadata output_dir does not match the loaded working_dir"
        if request.working_dir is not None and not self._same_working_dir(
            request.working_dir,
            self.working_dir,
        ):
            return "working_dir edit overrides require /reload before editing"
        return ""

    @staticmethod
    def _same_working_dir(left: str | Path, right: str | Path | None) -> bool:
        if right is None:
            return False
        left_path = Path(left)
        right_path = Path(right)
        try:
            return left_path.resolve() == right_path.resolve()
        except OSError:
            return left_path == right_path

    def _edit_response(
        self,
        *,
        success: bool,
        action: str,
        message: str,
        searchable: bool = False,
        blockers: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "success": success,
            "action": action,
            "message": message,
            "searchable": searchable,
            "full_rebuild": False,
            "image_rebuild": False,
            "dataset": self.dataset,
            "subset": self.subset,
            "working_dir": str(self.working_dir) if self.working_dir else None,
            "base_working_dir": str(self.base_working_dir)
            if self.base_working_dir
            else None,
            "blockers": blockers or [],
        }
        if extra:
            payload.update(extra)
        return payload

    @staticmethod
    def _mark_batch_results_rolled_back(
        results: list[dict[str, Any]],
        message: str,
    ) -> list[dict[str, Any]]:
        rolled_back: list[dict[str, Any]] = []
        for result in results:
            item = dict(result)
            if item.get("success", False):
                item["success"] = False
                item["rolled_back"] = True
                item["message"] = message
                blockers = item.get("blockers", [])
                if not isinstance(blockers, list):
                    blockers = [str(blockers)]
                item["blockers"] = [*blockers, message] if message not in blockers else blockers
            else:
                item.setdefault("rolled_back", False)
            rolled_back.append(item)
        return rolled_back

    def _batch_response(
        self,
        *,
        success: bool,
        action: str,
        message: str,
        results: list[dict[str, Any]],
        blockers: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "success": success,
            "action": action,
            "message": message,
            "batch": True,
            "results": results,
            "full_rebuild": False,
            "image_rebuild": False,
            "dataset": self.dataset,
            "subset": self.subset,
            "working_dir": str(self.working_dir) if self.working_dir else None,
            "base_working_dir": str(self.base_working_dir)
            if self.base_working_dir
            else None,
            "blockers": blockers or [],
        }
        if extra:
            payload.update(extra)
        return payload


def create_app(
    working_dir: str | Path | None = None,
    model_path: str | Path = Path("runtime_encoder_disabled"),
    output_root: str | Path = "expr_mm",
    dataset: str | None = "E-VQA",
    subset: str | None = "paper_tar_5120_128_seed0",
    encoder_factory: Callable[..., Any] | None = None,
    rag_factory: Callable[..., Any] | None = None,
    reload_interval: int = 300,
    check_interval: int = 60,
) -> FastAPI:
    app = FastAPI()
    state = MMAPIState(
        working_dir=working_dir,
        model_path=model_path,
        output_root=output_root,
        dataset=dataset,
        subset=subset,
        encoder_factory=encoder_factory,
        rag_factory=rag_factory,
        reload_interval=reload_interval,
        check_interval=check_interval,
    )
    app.state.mm_api = state
    if reload_interval > 0:
        def _auto_reload_worker() -> None:
            while True:
                time.sleep(reload_interval)
                try:
                    state.reload_if_changed()
                except Exception:
                    # Keep the API process alive; explicit /status or API logs expose blockers.
                    continue

        threading.Thread(target=_auto_reload_worker, daemon=True).start()

    @app.post("/search")
    def search(request: MMSearchRequest) -> list[str]:
        with state.lock:
            if state.retriever is None:
                retriever = None
            else:
                retriever = state.retriever
        if retriever is None:
            payloads = [
                {
                    "error": "multimodal KB is blocked",
                    "blockers": ["retriever is not loaded"],
                }
            ]
        else:
            payloads = retriever.search(
                queries=request.queries,
                context_queries=request.context_queries,
                image_ids=request.image_ids,
                image_paths=request.image_paths,
                rag_top_k=request.rag_top_k,
                image_top_k=request.image_top_k,
                entity_fusion_top_k=request.entity_fusion_top_k,
                entity_fusion_candidate_top_k=request.entity_fusion_candidate_top_k,
                visual_entity_top_k=request.visual_entity_top_k,
                visual_entity_candidate_top_k=request.visual_entity_candidate_top_k,
                visual_hyperedge_top_k=request.visual_hyperedge_top_k,
                visual_hyperedge_candidate_top_k=request.visual_hyperedge_candidate_top_k,
                visual_hyperedge_min_coherence=request.visual_hyperedge_min_coherence,
                fused_top_k=request.fused_top_k,
                entity_top_k=request.entity_top_k,
                hyperedge_top_k=request.hyperedge_top_k,
            )
        return [json.dumps(payload, ensure_ascii=False) for payload in payloads]

    @app.post("/reload")
    def reload(request: ReloadRequest) -> dict[str, Any]:
        status = state.reload(
            force=request.force,
            dataset=request.dataset,
            subset=request.subset,
            working_dir=request.working_dir,
        )
        blockers = list(status.get("blockers", []))
        status_text = str(status.get("status", "blocked"))
        ready = status_text == "ready"
        return {
            "success": ready,
            "message": "multimodal KB reloaded" if ready else "multimodal KB blocked",
            "reloaded": True,
            "status": status_text,
            "details": status,
            "blockers": blockers,
        }

    @app.post("/insert")
    def insert(request: MMInsertRequest) -> dict[str, Any]:
        return state.insert(request)

    @app.post("/hyperedge/update")
    def hyperedge_update(request: MMHyperedgeUpdateRequest) -> dict[str, Any]:
        return state.update(request)

    @app.post("/hyperedge/delete")
    def hyperedge_delete(request: MMHyperedgeDeleteRequest) -> dict[str, Any]:
        return state.delete(request)

    @app.post("/batch/insert")
    def batch_insert(request: MMBatchInsertRequest) -> dict[str, Any]:
        return state.insert_batch(request)

    @app.post("/batch/hyperedge/update")
    def batch_hyperedge_update(request: MMBatchHyperedgeUpdateRequest) -> dict[str, Any]:
        return state.update_batch(request)

    @app.post("/batch/hyperedge/delete")
    def batch_hyperedge_delete(request: MMBatchHyperedgeDeleteRequest) -> dict[str, Any]:
        return state.delete_batch(request)

    @app.get("/status")
    def status() -> dict[str, Any]:
        return state.status()

    @app.post("/test_params")
    def test_params(request: MMSearchRequest) -> dict[str, Any]:
        return {
            "received_params": {
                "entity_top_k": request.entity_top_k,
                "hyperedge_top_k": request.hyperedge_top_k,
                "rag_top_k": request.rag_top_k,
                "image_top_k": request.image_top_k,
                "entity_fusion_top_k": request.entity_fusion_top_k,
                "entity_fusion_candidate_top_k": request.entity_fusion_candidate_top_k,
                "visual_entity_top_k": request.visual_entity_top_k,
                "visual_entity_candidate_top_k": request.visual_entity_candidate_top_k,
                "visual_hyperedge_top_k": request.visual_hyperedge_top_k,
                "visual_hyperedge_candidate_top_k": request.visual_hyperedge_candidate_top_k,
                "visual_hyperedge_min_coherence": request.visual_hyperedge_min_coherence,
                "fused_top_k": request.fused_top_k,
                "dataset": request.dataset,
                "subset": request.subset,
            },
            "queries": request.queries,
            "context_queries": request.context_queries,
            "image_ids": request.image_ids,
            "image_paths": request.image_paths,
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        status_payload = state.status()
        blockers = list(status_payload.get("blockers", []))
        if status_payload.get("status") == "ready":
            return {"status": "healthy", "blockers": blockers}
        return {"status": "blocked", "blockers": blockers}

    return app


__all__ = [
    "MMSearchRequest",
    "ReloadRequest",
    "MMInsertRequest",
    "MMHyperedgeUpdateRequest",
    "MMHyperedgeDeleteRequest",
    "MMBatchInsertRequest",
    "MMBatchHyperedgeUpdateRequest",
    "MMBatchHyperedgeDeleteRequest",
    "MMAPIState",
    "create_app",
]
