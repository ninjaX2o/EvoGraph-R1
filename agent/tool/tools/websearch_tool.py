"""
Web search tool implementation backed by the Jina AI search API.
"""

import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import requests

from agent.tool.tool_base import Tool

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    name = "websearch"
    description = "Search the internet for external information."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string",
            }
        },
        "required": ["query"],
    }

    def __init__(self):
        super().__init__(self.name, self.description, self.parameters)
        self.cache_enabled = self._read_cache_enabled()
        self.cache_path = self._resolve_cache_path()
        self.fuzzy_enabled = self._read_fuzzy_enabled()
        self.fuzzy_threshold = self._read_fuzzy_threshold()
        # In-memory indexes keyed by sqlite cache path: normalized_query -> token set
        self._token_indexes: Dict[str, Dict[str, set]] = {}
        self._verified_cache_paths: set[str] = set()
        if self.cache_enabled:
            cache_ready = self._ensure_cache_db(self.cache_path)
            if cache_ready and self.fuzzy_enabled:
                self._load_token_index(self.cache_path)

    def execute(self, args: Dict) -> str:
        """
        Execute a web search query using the Jina AI API.

        Args:
            args: Tool parameters containing a required "query" string.

        Returns:
            Title and description from search results, limited to the first
            five results, or an error payload as JSON.
        """
        query = args.get("query", "").strip()
        if not query:
            return json.dumps({"error": "Query cannot be empty"}, ensure_ascii=False)

        try:
            if self.cache_enabled:
                return self._search_with_cache(query, self._dataset_from_args(args))
            return self._jina_search(query)
        except sqlite3.Error as e:
            error_msg = f"Websearch cache failed before Jina fallback: {str(e)}"
            return json.dumps({"error": error_msg}, ensure_ascii=False)
        except Exception as e:
            error_msg = f"Jina search failed: {str(e)}"
            return json.dumps({"error": error_msg}, ensure_ascii=False)

    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """
        Execute multiple web search queries.

        Args:
            args_list: List of tool parameter objects.

        Returns:
            List of JSON strings with search results.
        """
        return [self.execute(args) for args in args_list]

    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for a web search action.

        Args:
            args: Tool parameters.
            result: Tool execution result.

        Returns:
            Reward value.
        """
        try:
            result_dict = json.loads(result) if isinstance(result, str) else result
            if "error" not in result_dict:
                return 0.0
            return -0.1
        except Exception:
            return -0.1

    def _search_with_cache(self, query: str, dataset: str = "") -> str:
        normalized_query = self._normalize_query(query)
        cache_path = self._cache_path_for_dataset(dataset)
        cache_ready = self._ensure_cache_db(cache_path)
        if not cache_ready:
            logger.warning("[WEBSEARCH_CACHE] Cache unavailable for %s; using Jina directly", cache_path)
            return self._jina_search(query)
        if self.fuzzy_enabled and cache_path not in self._token_indexes:
            self._load_token_index(cache_path)

        # 1. Exact match
        cached_result = None
        try:
            cached_result = self._cache_get(normalized_query, cache_path)
        except sqlite3.Error as exc:
            self._handle_cache_sqlite_error(cache_path, exc, operation="read")
        if cached_result is not None:
            return cached_result

        # 2. Fuzzy match
        if self.fuzzy_enabled:
            fuzzy_result = None
            try:
                fuzzy_result = self._fuzzy_cache_get(normalized_query, cache_path)
            except sqlite3.Error as exc:
                self._handle_cache_sqlite_error(cache_path, exc, operation="fuzzy read")
            if fuzzy_result is not None:
                return fuzzy_result

        # 3. API call
        result = self._jina_search(query)
        try:
            self._cache_set(normalized_query, query, result, cache_path)
        except sqlite3.Error as exc:
            self._handle_cache_sqlite_error(cache_path, exc, operation="write")
            logger.warning(
                "[WEBSEARCH_CACHE] Returning live Jina result after cache write failure for %s",
                cache_path,
            )
        # Update in-memory token index
        if self.fuzzy_enabled:
            self._token_indexes.setdefault(cache_path, {})[normalized_query] = set(
                normalized_query.split()
            )
        return result

    def _read_cache_enabled(self) -> bool:
        raw_value = os.getenv("WEBSEARCH_CACHE_ENABLED", "true").strip().lower()
        return raw_value not in {"0", "false", "no", "off"}

    def _read_fuzzy_enabled(self) -> bool:
        raw_value = os.getenv("WEBSEARCH_FUZZY_ENABLED", "true").strip().lower()
        return raw_value not in {"0", "false", "no", "off"}

    def _read_fuzzy_threshold(self) -> float:
        raw_value = os.getenv("WEBSEARCH_FUZZY_THRESHOLD", "0.80").strip()
        try:
            return float(raw_value)
        except ValueError:
            return 0.80

    def _load_token_index(self, cache_path: str) -> None:
        """Load all cached query keys into memory as token sets."""
        try:
            with self._connect_cache(cache_path) as connection:
                rows = connection.execute(
                    "SELECT query_key FROM websearch_cache"
                ).fetchall()
            self._token_indexes[cache_path] = {
                key: set(key.split()) for (key,) in rows
            }
        except sqlite3.Error as exc:
            self._handle_cache_sqlite_error(cache_path, exc, operation="token index load")
            self._token_indexes[cache_path] = {}
        except Exception:
            self._token_indexes[cache_path] = {}

    def _jaccard(self, set_a: set, set_b: set) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union

    def _fuzzy_cache_get(self, normalized_query: str, cache_path: str) -> Optional[str]:
        """Find the best fuzzy match in the token index."""
        query_tokens = set(normalized_query.split())
        best_key = None
        best_score = 0.0

        for cached_key, cached_tokens in self._token_indexes.get(cache_path, {}).items():
            score = self._jaccard(query_tokens, cached_tokens)
            if score > best_score:
                best_score = score
                best_key = cached_key

        if best_score >= self.fuzzy_threshold and best_key is not None:
            return self._cache_get(best_key, cache_path)
        return None

    def _resolve_cache_path(self) -> str:
        configured_path = os.getenv("WEBSEARCH_CACHE_PATH", "").strip()
        if configured_path:
            return os.path.abspath(os.path.expanduser(configured_path))

        repo_root = Path(__file__).resolve().parents[3]
        return str(repo_root / "websearch_cache.sqlite")

    def _cache_path_for_dataset(self, dataset: str = "") -> str:
        dataset = self._sanitize_dataset_name(dataset)
        if not dataset:
            return self.cache_path

        configured_dir = os.getenv("WEBSEARCH_CACHE_DIR", "").strip()
        if configured_dir:
            cache_dir = Path(os.path.expanduser(configured_dir)).resolve()
            return str(cache_dir / f"{dataset}.sqlite")

        configured_path = os.getenv("WEBSEARCH_CACHE_PATH", "").strip()
        if configured_path:
            base_path = Path(os.path.expanduser(configured_path)).resolve()
            suffix = base_path.suffix or ".sqlite"
            return str(base_path.with_name(f"{base_path.stem}.{dataset}{suffix}"))

        repo_root = Path(__file__).resolve().parents[3]
        return str(repo_root / "websearch_cache" / f"{dataset}.sqlite")

    def _dataset_from_args(self, args: Dict) -> str:
        return self._sanitize_dataset_name(
            args.get("__dataset")
            or args.get("dataset")
            or args.get("__data_source")
            or args.get("data_source")
            or os.getenv("WEBSEARCH_CACHE_DATASET", "")
        )

    def _sanitize_dataset_name(self, dataset: Any) -> str:
        if dataset is None:
            return ""
        dataset = str(dataset).strip()
        if not dataset:
            return ""
        dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset)
        dataset = dataset.strip("._-")
        return dataset or ""

    def _normalize_query(self, query: str) -> str:
        return re.sub(r"\s+", " ", query).strip().lower()

    def _connect_cache(self, cache_path: str) -> sqlite3.Connection:
        connection = sqlite3.connect(cache_path, timeout=5.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _connect_cache_plain(self, cache_path: str) -> sqlite3.Connection:
        connection = sqlite3.connect(cache_path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _ensure_cache_db(self, cache_path: str) -> bool:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        try:
            self._create_cache_schema(cache_path)
            self._verify_cache_integrity_once(cache_path)
            return True
        except sqlite3.Error as exc:
            logger.warning(
                "[WEBSEARCH_CACHE] Cache validation failed for %s: %s",
                cache_path,
                exc,
            )
            if self._recover_cache_db(cache_path, exc):
                return True
            logger.error("[WEBSEARCH_CACHE] Cache recovery failed for %s", cache_path)
            return False

    def _create_cache_schema(self, cache_path: str) -> None:
        with self._connect_cache(cache_path) as connection:
            self._create_cache_schema_on_connection(connection)

    def _create_cache_schema_on_connection(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS websearch_cache (
                query_key TEXT PRIMARY KEY,
                original_query TEXT NOT NULL,
                result_text TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 1
            )
            """
        )

    def _verify_cache_integrity_once(self, cache_path: str) -> None:
        if cache_path in self._verified_cache_paths:
            return
        with self._connect_cache(cache_path) as connection:
            rows = connection.execute("PRAGMA integrity_check").fetchall()
        messages = [str(row[0]) for row in rows if row and str(row[0]).lower() != "ok"]
        if messages:
            raise sqlite3.DatabaseError("; ".join(messages[:5]))
        self._verified_cache_paths.add(cache_path)

    def _handle_cache_sqlite_error(
        self,
        cache_path: str,
        exc: sqlite3.Error,
        *,
        operation: str,
    ) -> None:
        logger.warning(
            "[WEBSEARCH_CACHE] Cache %s failed for %s: %s",
            operation,
            cache_path,
            exc,
        )
        self._recover_cache_db(cache_path, exc)
        self._token_indexes.pop(cache_path, None)
        self._verified_cache_paths.discard(cache_path)

    def _recover_cache_db(self, cache_path: str, exc: Exception | None = None) -> bool:
        del exc
        lock_path = cache_path + ".repair.lock"
        with self._cache_file_lock(lock_path):
            try:
                self._create_cache_schema(cache_path)
                with self._connect_cache(cache_path) as connection:
                    rows = connection.execute("PRAGMA integrity_check").fetchall()
                if all(row and str(row[0]).lower() == "ok" for row in rows):
                    self._verified_cache_paths.add(cache_path)
                    return True
            except sqlite3.Error:
                pass

            if self._try_reindex_cache(cache_path):
                self._verified_cache_paths.add(cache_path)
                return True

            salvaged_rows = self._read_salvageable_cache_rows(cache_path)
            backup_path = self._isolate_cache_files(cache_path)
            try:
                self._create_cache_schema(cache_path)
                if salvaged_rows:
                    self._restore_salvaged_cache_rows(cache_path, salvaged_rows)
                self._verified_cache_paths.discard(cache_path)
                logger.warning(
                    "[WEBSEARCH_CACHE] Rebuilt cache %s after corruption; backup=%s salvaged_rows=%d",
                    cache_path,
                    backup_path,
                    len(salvaged_rows),
                )
                return True
            except sqlite3.Error as create_exc:
                logger.error(
                    "[WEBSEARCH_CACHE] Failed to rebuild cache %s: %s",
                    cache_path,
                    create_exc,
                )
                return False

    def _try_reindex_cache(self, cache_path: str) -> bool:
        try:
            with self._connect_cache(cache_path) as connection:
                connection.execute("REINDEX")
                connection.execute("PRAGMA optimize")
                rows = connection.execute("PRAGMA integrity_check").fetchall()
            if all(row and str(row[0]).lower() == "ok" for row in rows):
                logger.warning("[WEBSEARCH_CACHE] Repaired sqlite indexes for %s with REINDEX", cache_path)
                return True
        except sqlite3.Error as exc:
            logger.warning("[WEBSEARCH_CACHE] REINDEX repair failed for %s: %s", cache_path, exc)
        return False

    def _read_salvageable_cache_rows(self, cache_path: str) -> list[tuple]:
        if not os.path.exists(cache_path):
            return []
        try:
            rows = []
            with self._connect_cache_plain(cache_path) as connection:
                cursor = connection.execute(
                    """
                    SELECT query_key, original_query, result_text, created_at, updated_at, hit_count
                    FROM websearch_cache
                    """
                )
                for row in cursor:
                    rows.append(row)
            return rows
        except sqlite3.Error as exc:
            logger.warning("[WEBSEARCH_CACHE] Could not salvage rows from %s: %s", cache_path, exc)
            return []

    def _restore_salvaged_cache_rows(self, cache_path: str, rows: list[tuple]) -> None:
        with self._connect_cache(cache_path) as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO websearch_cache (
                    query_key,
                    original_query,
                    result_text,
                    created_at,
                    updated_at,
                    hit_count
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def _isolate_cache_files(self, cache_path: str) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = f"{cache_path}.corrupt.{timestamp}.{os.getpid()}"
        for suffix in ("", "-wal", "-shm"):
            source = cache_path + suffix
            if not os.path.exists(source):
                continue
            target = backup_path + suffix
            try:
                os.replace(source, target)
            except OSError:
                shutil.copy2(source, target)
                os.remove(source)
        return backup_path

    @contextmanager
    def _cache_file_lock(self, lock_path: str):
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass

    def _cache_get(self, normalized_query: str, cache_path: str) -> Optional[str]:
        with self._connect_cache(cache_path) as connection:
            row = connection.execute(
                "SELECT result_text FROM websearch_cache WHERE query_key = ?",
                (normalized_query,),
            ).fetchone()
            if row is None:
                return None

            connection.execute(
                """
                UPDATE websearch_cache
                SET hit_count = hit_count + 1, updated_at = ?
                WHERE query_key = ?
                """,
                (time.time(), normalized_query),
            )
            return row[0]

    def _cache_set(
        self,
        normalized_query: str,
        original_query: str,
        result: str,
        cache_path: str,
    ) -> None:
        now = time.time()
        with self._connect_cache(cache_path) as connection:
            connection.execute(
                """
                INSERT INTO websearch_cache (
                    query_key,
                    original_query,
                    result_text,
                    created_at,
                    updated_at,
                    hit_count
                ) VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(query_key) DO UPDATE SET
                    original_query = excluded.original_query,
                    result_text = excluded.result_text,
                    updated_at = excluded.updated_at,
                    hit_count = websearch_cache.hit_count + 1
                """,
                (normalized_query, original_query, result, now, now),
            )

    def _jina_search(self, query: str) -> str:
        """
        Perform a search via the Jina AI API.

        Args:
            query: Search query string.

        Returns:
            Title and description from search results, limited to the first
            five results.
        """
        jina_api_key = os.getenv("JINA_API_KEY")
        if not jina_api_key:
            raise ValueError("JINA_API_KEY environment variable is not set")

        url = "https://s.jina.ai/"
        headers = {
            "Authorization": f"Bearer {jina_api_key}",
            "X-Engine": "direct",
            "X-Respond-With": "no-content",
        }
        params = {"q": query}

        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return self._extract_title_and_description(response.text, 5)

    def _extract_title_and_description(self, text: str, max_results: int) -> str:
        """
        Extract result titles and descriptions while dropping URLs.

        Args:
            text: Raw search result text.
            max_results: Maximum number of results to keep.

        Returns:
            Joined title and description lines for the top results.
        """
        lines = text.split("\n")
        results = []
        current_result: Dict[str, Any] = {}
        current_result_num = 0

        for line in lines:
            line = line.strip()

            if re.match(r"^\[\d+\]\s*Title:", line):
                if current_result and 0 < current_result_num <= max_results:
                    result_text = f"[{current_result_num}] {current_result.get('title', '')}"
                    if current_result.get("description"):
                        result_text += f" - {current_result['description']}"
                    results.append(result_text)

                match = re.match(r"^\[(\d+)\]", line)
                if match:
                    result_num = int(match.group(1))
                    if result_num > max_results:
                        break
                    current_result_num = result_num
                    current_result = {}

            if "Title:" in line and current_result_num <= max_results:
                title_match = re.search(r"Title:\s*(.+)", line)
                if title_match:
                    current_result["title"] = title_match.group(1)

            if "Description:" in line and current_result_num <= max_results:
                desc_match = re.search(r"Description:\s*(.+)", line)
                if desc_match:
                    current_result["description"] = desc_match.group(1)

        if current_result and 0 < current_result_num <= max_results:
            last_result_text = f"[{current_result_num}] {current_result.get('title', '')}"
            if current_result.get("description"):
                last_result_text += f" - {current_result['description']}"

            if not results or not results[-1].startswith(f"[{current_result_num}]"):
                results.append(last_result_text)

        return "\n".join(results)
