"""Multimodal knowledge base search tool."""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import requests

from agent.tool.tool_base import Tool


TOP_K_FIELDS = (
    "entity_top_k",
    "hyperedge_top_k",
    "image_top_k",
    "rag_top_k",
    "entity_fusion_top_k",
    "entity_fusion_candidate_top_k",
    "visual_entity_top_k",
    "visual_entity_candidate_top_k",
    "visual_hyperedge_top_k",
    "visual_hyperedge_candidate_top_k",
    "visual_hyperedge_min_coherence",
)
TEXT_TOP_K_FIELDS = ("entity_top_k", "hyperedge_top_k", "rag_top_k")
IMAGE_QUERY_TOKEN = "<img>"
DEFAULT_VISUAL_ENTITY_TOP_K = 3


class MMKBSearchTool(Tool):
    """Tool for calling the multimodal KB search API."""

    def __init__(self):
        name = "kb_search"
        description = (
            "Search the internal multimodal knowledge base and return matched results. "
            "Supports two query modes: use the literal query '<img>' to retrieve "
            "candidate visual entities related to the current image; use a text query "
            "to retrieve relevant information"
        )
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query. Use exactly '<img>' for visual entity lookup when "
                        "you need candidate entities from the current question image. "
                        "Use a natural-language text query for knowledge search when "
                        "you need relevant information from the KB."
                    ),
                },
            },
            "required": ["query"],
        }
        super().__init__(name, description, parameters)
        self.search_api_url = os.getenv(
            "MM_SEARCH_API_URL",
            "http://127.0.0.1:8003/search",
        )
        self.text_search_api_url = (
            os.getenv("TEXT_SEARCH_API_URL")
            or os.getenv("MM_TEXT_SEARCH_API_URL")
            or self.search_api_url
        )
        self.timeout = float(os.getenv("MM_SEARCH_TIMEOUT", "120"))

    def execute(self, args: Dict) -> str:
        query = self._clean_string(args.get("query"))
        if not query:
            return self._empty_result()

        image_query = query == IMAGE_QUERY_TOKEN
        payload = (
            self._build_single_payload(args, query)
            if image_query
            else self._build_text_payload([query], args)
        )
        api_url = self._api_url_for_query(query)
        try:
            resp = requests.post(api_url, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                return self._empty_result()
            return self._coerce_single_response(
                resp.json(),
                image_query=image_query,
            )
        except Exception as exc:
            return self._empty_result(error=str(exc))

    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        if not args_list:
            return []

        results = [self._empty_result() for _ in args_list]
        grouped_indices = self._group_indices_by_route_and_top_k_options(args_list)
        for group_key, indices in grouped_indices.items():
            route = dict(group_key).get("route")
            top_k_options = {
                key: value
                for key, value in dict(group_key).items()
                if key != "route"
            }
            image_query = route == "image"
            payload = (
                self._build_batch_payload(args_list, indices, top_k_options)
                if image_query
                else self._build_text_payload(
                    [self._clean_string(args_list[index].get("query")) for index in indices],
                    top_k_options,
                )
            )
            api_url = self.search_api_url if image_query else self.text_search_api_url
            try:
                resp = requests.post(api_url, json=payload, timeout=self.timeout)
                if resp.status_code != 200:
                    for index in indices:
                        results[index] = self._empty_result()
                    continue

                response_items = resp.json()
                if not isinstance(response_items, list) or len(response_items) != len(indices):
                    for index in indices:
                        results[index] = self._empty_result(
                            error="malformed multimodal search response length"
                        )
                    continue

                for result_index, response_item in zip(indices, response_items):
                    query = self._clean_string(args_list[result_index].get("query"))
                    results[result_index] = self._coerce_response_item(
                        response_item,
                        image_query=image_query,
                    )
            except Exception as exc:
                error = str(exc)
                for index in indices:
                    results[index] = self._empty_result(error=error)
        return results

    def _build_single_payload(self, args: Dict, query: str) -> dict:
        payload = {"queries": [query]}
        image_id, image_path, context_query = self._query_context(args, query)
        if image_id:
            payload["image_ids"] = [image_id]
        if image_path:
            payload["image_paths"] = [image_path]
        if context_query:
            payload["context_queries"] = [context_query]
        if query == IMAGE_QUERY_TOKEN and args.get("visual_entity_top_k") is None:
            payload["visual_entity_top_k"] = DEFAULT_VISUAL_ENTITY_TOP_K
        for field in TOP_K_FIELDS:
            if args.get(field) is not None:
                payload[field] = args.get(field)
        return payload

    def _build_text_payload(self, queries: List[str], args: Dict) -> dict:
        payload = {"queries": queries}
        for field in TEXT_TOP_K_FIELDS:
            if args.get(field) is not None:
                payload[field] = args.get(field)
        return payload

    def _build_batch_payload(
        self,
        args_list: List[Dict],
        indices: list[int],
        top_k_options: dict,
    ) -> dict:
        queries = [self._clean_string(args_list[index].get("query")) for index in indices]
        contexts = [
            self._query_context(args_list[index], query)
            for index, query in zip(indices, queries)
        ]
        image_ids = [context[0] for context in contexts]
        image_paths = [context[1] for context in contexts]
        context_queries = [context[2] for context in contexts]

        payload = {"queries": queries}
        if any(image_ids):
            payload["image_ids"] = image_ids
        if any(image_paths):
            payload["image_paths"] = image_paths
        if any(context_queries):
            payload["context_queries"] = context_queries
        if any(query == IMAGE_QUERY_TOKEN for query in queries):
            top_k = top_k_options.get("visual_entity_top_k")
            payload["visual_entity_top_k"] = (
                top_k if top_k is not None else DEFAULT_VISUAL_ENTITY_TOP_K
            )
        for field, value in top_k_options.items():
            if value is not None:
                payload[field] = value
        return payload

    @staticmethod
    def _group_indices_by_route_and_top_k_options(args_list: List[Dict]) -> dict:
        grouped = defaultdict(list)
        for index, args in enumerate(args_list):
            query = MMKBSearchTool._clean_string(args.get("query"))
            image_query = query == IMAGE_QUERY_TOKEN
            fields = TOP_K_FIELDS if image_query else TEXT_TOP_K_FIELDS
            route = "image" if image_query else "text"
            key = (("route", route), *((field, args.get(field)) for field in fields))
            grouped[key].append(index)
        return dict(grouped)

    def _api_url_for_query(self, query: str) -> str:
        return self.search_api_url if query == IMAGE_QUERY_TOKEN else self.text_search_api_url

    @staticmethod
    def _coerce_single_response(data, *, image_query: bool = False) -> str:
        if isinstance(data, list) and data:
            return MMKBSearchTool._coerce_response_item(
                data[0],
                image_query=image_query,
            )
        return MMKBSearchTool._empty_result()

    @staticmethod
    def _coerce_response_item(item, *, image_query: bool = False) -> str:
        if isinstance(item, str):
            try:
                parsed = json.loads(item)
            except Exception:
                return MMKBSearchTool._empty_result(error="invalid multimodal search JSON")
            if image_query:
                return MMKBSearchTool._compact_image_response(parsed)
            return item
        if isinstance(item, dict):
            if image_query:
                return MMKBSearchTool._compact_image_response(item)
            return json.dumps(item)
        return MMKBSearchTool._empty_result(error="invalid multimodal search response item")

    @staticmethod
    def _compact_image_response(item) -> str:
        if not isinstance(item, dict):
            return MMKBSearchTool._empty_result(error="invalid multimodal image response item")

        results = []
        seen = set()
        for result in item.get("results", []):
            if isinstance(result, dict):
                name = MMKBSearchTool._clean_string(result.get("entity"))
                image_path = MMKBSearchTool._normalize_image_path(
                    MMKBSearchTool._clean_string(result.get("image_path"))
                )
                image_url = MMKBSearchTool._clean_string(result.get("image_url"))
            else:
                name = MMKBSearchTool._clean_string(result)
                image_path = ""
                image_url = ""
            if not name or name in seen:
                continue
            compact_result = {"entity": name}
            if image_path:
                compact_result["image_path"] = image_path
            if image_url:
                compact_result["image_url"] = image_url
            results.append(compact_result)
            seen.add(name)
            if len(results) >= DEFAULT_VISUAL_ENTITY_TOP_K:
                break
        return json.dumps({"results": results})

    @staticmethod
    def _clean_string(value) -> str:
        return str(value).strip() if value is not None else ""

    @classmethod
    def _query_context(cls, args: Dict, query: str) -> tuple[str, str, str]:
        explicit_image_id = cls._clean_string(args.get("image_id"))
        explicit_image_path = cls._clean_string(args.get("image_path"))
        explicit_context_query = cls._clean_string(args.get("context_query"))
        if query == IMAGE_QUERY_TOKEN:
            image_path = cls._normalize_image_path(
                explicit_image_path
                or cls._first_env(
                    "EVOGRAPH_MM_CURRENT_IMAGE_PATH",
                    "MM_CURRENT_IMAGE_PATH",
                )
            )
            image_id = ""
            if not image_path:
                image_id = explicit_image_id or cls._first_env(
                    "EVOGRAPH_MM_CURRENT_IMAGE_ID",
                    "MM_CURRENT_IMAGE_ID",
                )
            context_query = explicit_context_query or cls._first_env(
                "EVOGRAPH_MM_CURRENT_QUESTION",
                "MM_CURRENT_QUESTION",
            )
            return image_id, image_path, context_query
        return "", "", ""

    @staticmethod
    def _normalize_image_path(image_path: str) -> str:
        if not image_path:
            return ""
        candidate = Path(image_path)
        if candidate.is_file():
            return str(candidate)

        normalized = image_path.replace("\\", "/")
        marker = "datasets_mm/"
        if marker in normalized:
            relative = normalized[normalized.index(marker) :]
            return relative
        return image_path

    @classmethod
    def _hidden_image_id(cls, args: Dict) -> str:
        return cls._clean_string(args.get("image_id")) or cls._first_env(
            "EVOGRAPH_MM_CURRENT_IMAGE_ID",
            "MM_CURRENT_IMAGE_ID",
        )

    @classmethod
    def _hidden_image_path(cls, args: Dict) -> str:
        return cls._clean_string(args.get("image_path")) or cls._first_env(
            "EVOGRAPH_MM_CURRENT_IMAGE_PATH",
            "MM_CURRENT_IMAGE_PATH",
        )

    @classmethod
    def _hidden_context_query(cls, args: Dict) -> str:
        return cls._clean_string(args.get("context_query")) or cls._first_env(
            "EVOGRAPH_MM_CURRENT_QUESTION",
            "MM_CURRENT_QUESTION",
        )

    @staticmethod
    def _first_env(*names: str) -> str:
        for name in names:
            value = os.getenv(name)
            if value and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _empty_result(error: str | None = None) -> str:
        payload = {"results": []}
        if error:
            payload["error"] = error
        return json.dumps(payload)
