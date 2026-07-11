"""
Knowledge Base Search tool implementation for searching knowledge graphs
"""

import json
import os
from typing import Dict, List
from urllib.parse import urlsplit, urlunsplit

import requests

from agent.tool.tool_base import Tool
from agent.tool.tools.hyperedge_state_sync import HYPEREDGE_RECENT_MUTATIONS_FILE


class KBSearchTool(Tool):
    """
    Tool for searching knowledge graphs using the local search API.
    """

    def __init__(self):
        name = "kb_search"
        description = "Search internal knowledge base for relevant information. Returns a JSON with results."
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
            },
            "required": ["query"],
        }

        super().__init__(name, description, parameters)
        self.search_api_url = os.getenv("SEARCH_API_URL", "http://127.0.0.1:8001/search")
        self.reload_api_url = os.getenv(
            "SEARCH_API_RELOAD_URL",
            self._derive_reload_url(self.search_api_url),
        )
        self.auto_reload = os.getenv("KB_SEARCH_AUTO_RELOAD", "true").lower() == "true"
        self.reload_policy = self._resolve_reload_policy()
        self._mutation_marker_mtimes = {}

    def _resolve_reload_policy(self) -> str:
        policy = (os.getenv("KB_SEARCH_RELOAD_POLICY") or "").strip().lower()
        if policy in {"always", "on_mutation", "off"}:
            return policy
        return "always" if self.auto_reload else "off"

    @staticmethod
    def _derive_reload_url(search_api_url: str) -> str:
        parts = urlsplit(search_api_url)
        path = parts.path.rstrip("/")
        if path.endswith("/search"):
            path = path[: -len("/search")] + "/reload"
        else:
            path = path + "/reload"
        return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))

    @staticmethod
    def _get_mutation_marker_path() -> str:
        working_dir = os.getenv("GRAPHR1_WORKING_DIR", "").strip()
        if not working_dir:
            return ""
        return os.path.join(working_dir, HYPEREDGE_RECENT_MUTATIONS_FILE)

    @staticmethod
    def _get_marker_mtime(marker_path: str):
        if not marker_path or not os.path.exists(marker_path):
            return None
        try:
            return os.path.getmtime(marker_path)
        except OSError:
            return None

    def _get_on_mutation_reload_target(self):
        marker_path = self._get_mutation_marker_path()
        if not marker_path:
            return "", None, True

        current_mtime = self._get_marker_mtime(marker_path)
        sentinel = object()
        previous_mtime = self._mutation_marker_mtimes.get(marker_path, sentinel)

        if previous_mtime is sentinel:
            return marker_path, current_mtime, current_mtime is not None

        if previous_mtime is None:
            return marker_path, current_mtime, current_mtime is not None

        if current_mtime is None:
            self._mutation_marker_mtimes[marker_path] = None
            return marker_path, current_mtime, False

        return marker_path, current_mtime, current_mtime > previous_mtime

    def _reload_search_api(self) -> None:
        if self.reload_policy == "off":
            return

        marker_path = ""
        current_mtime = None
        if self.reload_policy == "on_mutation":
            marker_path, current_mtime, should_reload = self._get_on_mutation_reload_target()
            if not should_reload:
                return

        if not self.reload_api_url:
            return

        try:
            response = requests.post(
                self.reload_api_url,
                json={"force": True, "load_graph": False},
                timeout=10,
            )
            if self.reload_policy == "on_mutation" and marker_path:
                success = response.status_code == 200
                if success:
                    try:
                        payload = response.json()
                    except Exception:
                        payload = {}
                    if isinstance(payload, dict) and payload.get("success") is False:
                        success = False
                if success:
                    self._mutation_marker_mtimes[marker_path] = current_mtime
        except Exception:
            # Search remains best-effort if the backend has no reload endpoint.
            pass

    def execute(self, args: Dict) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return json.dumps({"results": []})

        try:
            self._reload_search_api()
            resp = requests.post(self.search_api_url, json={"queries": [query]}, timeout=30)
            if resp.status_code != 200:
                return json.dumps({"results": []})
            data = resp.json()
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
                try:
                    parsed0 = json.loads(data[0])
                    if isinstance(parsed0, dict) and "results" in parsed0:
                        items = []
                        for result in parsed0.get("results", []):
                            if isinstance(result, dict) and "<knowledge>" in result:
                                items.append(
                                    {
                                        "<knowledge>": result.get("<knowledge>", ""),
                                        "<coherence>": float(result.get("<coherence>", 0.5)),
                                    }
                                )
                            else:
                                items.append({"<knowledge>": str(result), "<coherence>": 0.5})
                        return json.dumps({"results": items})
                except Exception:
                    pass
                return json.dumps({"results": []})
            if isinstance(data, dict) and "query_results" in data:
                try:
                    first = data["query_results"][0]
                    items = []
                    for result in first.get("results", []):
                        items.append(
                            {
                                "<knowledge>": result.get("document", ""),
                                "<coherence>": float(result.get("score", 0.5)),
                            }
                        )
                    return json.dumps({"results": items})
                except Exception:
                    return json.dumps({"results": []})
            return json.dumps({"results": []})
        except Exception:
            return json.dumps({"results": []})

    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        queries = [item.get("query", "").strip() for item in args_list]
        try:
            self._reload_search_api()
            resp = requests.post(self.search_api_url, json={"queries": queries}, timeout=30)
            if resp.status_code != 200:
                return [json.dumps({"results": []}) for _ in args_list]
            results_str = resp.json()
            if isinstance(results_str, list) and len(results_str) == len(args_list):
                return [item if isinstance(item, str) else json.dumps(item) for item in results_str]
            return [json.dumps({"results": []}) for _ in args_list]
        except Exception:
            results = []
            for args in args_list:
                try:
                    results.append(self.execute(args))
                except Exception:
                    results.append(json.dumps({"results": []}))
            return results

    def calculate_reward(self, args: Dict, result: str) -> float:
        if "results" in result:
            return 0.0
        return -0.1
