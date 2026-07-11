"""
Base class for GraphR1 tools to reduce code duplication
"""

import json
import os
import shutil
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from urllib.parse import urlsplit
from agent.tool.tool_base import Tool
from agent.tool.tools.edit_result_contract import build_edit_task_response
from agent.tool.tools.hyperedge_state_sync import ensure_hyperedge_state_sidecars

logger = logging.getLogger(__name__)

LEGACY_BAD_LLM_BASE_URL_HOSTS = {
    "code.xiaomocode.site",
}


def _extract_url_host(url: str) -> str:
    try:
        parsed = urlsplit((url or "").strip())
        return (parsed.hostname or "").lower()
    except Exception:
        return ""


def _repair_legacy_llm_env_urls(dotenv_values_map: Dict[str, str], env: Dict[str, str]) -> List[str]:
    repaired = []
    for var_name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        desired_value = (dotenv_values_map.get(var_name) or "").strip()
        current_value = (env.get(var_name) or "").strip()
        if not desired_value or not current_value or desired_value == current_value:
            continue
        if _extract_url_host(current_value) not in LEGACY_BAD_LLM_BASE_URL_HOSTS:
            continue
        env[var_name] = desired_value
        repaired.append(var_name)
    return repaired

# 设置日志记录器
logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import dotenv_values, load_dotenv
    # Try to load .env from project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..", "..")
    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=False)
        repaired_vars = _repair_legacy_llm_env_urls(dotenv_values(env_path), os.environ)
        if repaired_vars:
            logger.warning(
                "[ENV] Repaired legacy LLM base URL from process env using %s for %s",
                env_path,
                ", ".join(repaired_vars),
            )
        logger.info(f"[ENV] Loaded environment variables from: {env_path}")
    else:
        # Try to load from current directory
        env_path_local = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path_local):
            load_dotenv(dotenv_path=env_path_local, override=False)
            repaired_vars = _repair_legacy_llm_env_urls(dotenv_values(env_path_local), os.environ)
            if repaired_vars:
                logger.warning(
                    "[ENV] Repaired legacy LLM base URL from process env using %s for %s",
                    env_path_local,
                    ", ".join(repaired_vars),
                )
            logger.info(f"[ENV] Loaded environment variables from: {env_path_local}")
except ImportError:
    logger.warning("[WARNING] python-dotenv not installed. Please install it: pip install python-dotenv")

# Constants
KB_FILES = ["kv_store_entities.json", "kv_store_hyperedges.json"]
KB_ARTIFACT_FILES = [
    "kv_store_entities.json",
    "kv_store_hyperedges.json",
    "kv_store_text_chunks.json",
    "kv_store_full_docs.json",
    "graph_chunk_entity_relation.graphml",
    "index.bin",
    "corpus.npy",
    "index_entity.bin",
    "corpus_entity.npy",
    "corpus_entity_hashes.json",
    "entity_index_metadata.json",
    "index_hyperedge.bin",
    "corpus_hyperedge.npy",
    "corpus_hyperedge_hashes.json",
    "hyperedge_index_hashes.json",
    "hyperedge_index_metadata.json",
    "hyperedge_content_lookup.json",
    "hyperedge_recent_mutations.json",
]
HYPEREDGE_PREFIX = '<hyperedge>"'
HYPEREDGE_SUFFIX = '"'


# Add GraphR1 path to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..", "..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 使用延迟导入解决循环导入问题
def _get_graphr1_class():
    """懒加载GraphR1类"""
    try:
        from graphr1 import GraphR1
        return GraphR1
    except ImportError as e:
        raise ImportError(f"无法导入GraphR1类: {e}")

def _get_query_param_class():
    """懒加载QueryParam类"""
    try:
        from graphr1 import QueryParam
        return QueryParam
    except ImportError as e:
        raise ImportError(f"无法导入QueryParam类: {e}")

def _create_graphr1_instance(working_dir, **kwargs):
    """创建GraphR1实例"""
    try:
        GraphR1 = _get_graphr1_class()
        return GraphR1(working_dir=working_dir, **kwargs)
    except Exception as e:
        raise RuntimeError(f"无法创建GraphR1实例: {e}")

def _is_graphr1_available():
    """检查GraphR1是否可用"""
    try:
        _get_graphr1_class()
        return True
    except ImportError:
        return False


def _write_json_with_filelock(path: str, data: dict) -> None:
    """Write JSON file with cross-process file locking."""
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


class GraphR1BaseTool(Tool):
    """Base class for GraphR1 tools with common functionality"""

    # 类级别的 hyperedges 内存缓存，所有 tool 实例共享
    # key: working_dir, value: {"data": dict, "dirty": bool}
    _hyperedges_cache: Dict[str, Dict] = {}
    _dirty_graphs: Dict[str, Any] = {}
    _dirty_hyperedge_indexes: Dict[str, bool] = {}
    _dirty_entity_indexes: Dict[str, bool] = {}
    _pending_previous_hyperedges_data: Dict[str, Dict] = {}
    _shared_graphr1_instances: Dict[str, Any] = {}
    _cache_lock = None  # 延迟初始化，避免 fork 问题

    @classmethod
    def _get_cache_lock(cls):
        import threading
        if cls._cache_lock is None:
            cls._cache_lock = threading.Lock()
        return cls._cache_lock

    def __init__(self, name: str, description: str, parameters: Dict):
        super().__init__(name, description, parameters)
        self._working_dir = None  # Store the working directory for this training session

        # 获取统一批量队列单例
        from agent.tool.tools.batch.unified_batch_queue import UnifiedBatchQueue
        self._batch_queue = UnifiedBatchQueue.get_instance()

    def get_batch_queue(self):
        """获取统一批量队列"""
        return self._batch_queue

    def _get_or_create_graphr1(self) -> Optional['GraphR1']:
        """Get or create GraphR1 instance"""

        # 最高优先级：环境变量覆盖
        env_dir = os.getenv("GRAPHR1_WORKING_DIR")
        if env_dir and os.path.exists(env_dir):
            graphr1_working_dir = env_dir
            self._working_dir = graphr1_working_dir
            logger.debug(f"Using working directory from GRAPHR1_WORKING_DIR: {graphr1_working_dir}")

        # If we already have a working directory for this training session, reuse it
        elif self._working_dir is not None:
            graphr1_working_dir = self._working_dir
            logger.debug(f"Reusing existing working directory: {graphr1_working_dir}")
        else:
            # Dynamically detect training dataset and create backup knowledge base
            current_working_dir = os.getcwd()

            # Try to detect dataset from training files or environment
            dataset_name = self._detect_dataset_name()

            if dataset_name:
                # 1) SLURM作业ID的确定性目录
                slurm_job_id = os.getenv("SLURM_JOB_ID")
                if slurm_job_id:
                    fixed_dir_name = f"expr_{dataset_name}_working_{slurm_job_id}"
                    fixed_dir = os.path.join(current_working_dir, fixed_dir_name)
                    graphr1_working_dir = fixed_dir
                    logger.debug(f"Using SLURM-based working directory: {graphr1_working_dir}")
                else:
                    # 2) Fallback to timestamp-based directory
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fallback_dir_name = f"expr_{dataset_name}_working_{timestamp}"
                    fallback_dir = os.path.join(current_working_dir, fallback_dir_name)
                    graphr1_working_dir = fallback_dir
                    logger.debug(f"Using timestamp-based working directory: {graphr1_working_dir}")
            else:
                # Fallback: use current directory
                graphr1_working_dir = current_working_dir
                logger.debug(f"Using current directory as working directory: {graphr1_working_dir}")

            # Store the working directory for reuse in this training session
            self._working_dir = graphr1_working_dir

        # Return None if GraphR1 is not available, but knowledge base setup is done
        if not _is_graphr1_available():
            logger.debug("GraphR1 not available, but knowledge base setup completed")
            return None

        # Use working directory as a shared process-local instance identifier.
        lock = self._get_cache_lock()
        with lock:
            shared_instance = self._shared_graphr1_instances.get(graphr1_working_dir)
        if shared_instance is not None:
            return shared_instance

        if graphr1_working_dir not in self._shared_graphr1_instances:
            try:
                self.flush_dirty_graphs(graphr1_working_dir)

                # Create GraphR1 instance with the working directory
                graphr1_config = {
                    "working_dir": graphr1_working_dir,
                }

                # Get LLM model name
                llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                if os.getenv("SILICONFLOW_API_KEY"):
                    llm_model = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")
                graphr1_config["llm_model_name"] = llm_model

                # Check for API keys
                llm_api_key = os.getenv("OPENAI_API_KEY", "")
                siliconflow_key = os.getenv("SILICONFLOW_API_KEY", "")
                zhipu_key = os.getenv("ZHIPU_API_KEY", "")

                # Add LLM configuration if API key is available
                if llm_api_key or siliconflow_key or zhipu_key:
                    from graphr1.llm import openai_complete_if_cache
                    from functools import wraps

                    llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                    llm_base_url = os.getenv("OPENAI_BASE_URL", "")

                    if siliconflow_key:
                        llm_api_key = siliconflow_key
                        llm_base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
                        llm_model = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")
                        logger.info(f"[ENV] Using SiliconFlow LLM: {llm_model}")
                    elif zhipu_key:
                        logger.info(f"[ENV] Using Zhipu AI LLM")
                    else:
                        logger.info(f"[ENV] Using OpenAI LLM: {llm_model}")

                    def create_llm_wrapper(model_name, api_key, base_url):
                        @wraps(openai_complete_if_cache)
                        async def wrapper(prompt, system_prompt=None, history_messages=[], **kwargs):
                            return await openai_complete_if_cache(
                                model=model_name,
                                prompt=prompt,
                                system_prompt=system_prompt,
                                history_messages=history_messages,
                                api_key=api_key,
                                base_url=base_url if base_url else None,
                                **kwargs
                            )
                        return wrapper

                    graphr1_config["llm_model_func"] = create_llm_wrapper(llm_model, llm_api_key, llm_base_url)

                # Add embedding configuration through the shared BGE manager.
                def create_bge_embedding_func():
                    from agent.tool.tools.bge_model_manager import encode_texts_safe

                    async def bge_embedding(texts):
                        return encode_texts_safe(texts, target_dimension=1024)

                    from graphr1.utils import wrap_embedding_func_with_attrs
                    return wrap_embedding_func_with_attrs(
                        embedding_dim=1024,
                        max_token_size=512
                    )(bge_embedding)

                graphr1_config["embedding_func"] = create_bge_embedding_func()

                # Add any additional configuration if needed
                if os.getenv("GRAPHR1_CONFIG"):
                    try:
                        additional_config = json.loads(os.getenv("GRAPHR1_CONFIG"))
                        graphr1_config.update(additional_config)
                    except Exception:
                        pass

                shared_instance = _get_graphr1_class()(**graphr1_config)
                with lock:
                    self._shared_graphr1_instances[graphr1_working_dir] = shared_instance
                logger.info(f"Created GraphR1 instance in working directory: {graphr1_working_dir}")
            except Exception as e:
                logger.error(f"Failed to create GraphR1 instance: {e}")
                import traceback
                traceback.print_exc()
                return None

        with lock:
            return self._shared_graphr1_instances.get(graphr1_working_dir)

    def _detect_dataset_name(self) -> Optional[str]:
        """Detect dataset name from environment or training files"""
        # 1) Check environment variable
        dataset_name = os.getenv("TRAIN_DATASET")
        if dataset_name:
            return dataset_name

        # 2) Try to detect from training files in current directory
        current_dir = os.getcwd()
        try:
            for file_name in os.listdir(current_dir):
                if file_name.startswith("expr_") and file_name.endswith("_working_"):
                    parts = file_name.split("_")
                    if len(parts) >= 2:
                        return parts[1]
        except OSError:
            pass

        # 3) Check for common dataset patterns in current directory
        dataset_patterns = ["NQ", "2WikiMultiHopQA", "Musique", "HotpotQA"]
        for pattern in dataset_patterns:
            if pattern.lower() in current_dir.lower():
                return pattern

        return None

    def _copy_knowledge_base(self, src_dir: str, dst_dir: str):
        """复制知识库文件到工作目录"""
        os.makedirs(dst_dir, exist_ok=True)
        for filename in KB_ARTIFACT_FILES:
            src = os.path.join(src_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst_dir, filename))

    def validate_args(self, args: Dict) -> tuple:
        """Validate arguments - delegate to Tool base class"""
        return super().validate_args(args)

    def _get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            working_dir = self.get_working_dir()

            # 统计 entities
            entity_count = 0
            entity_path = os.path.join(working_dir, "kv_store_entities.json")
            if os.path.exists(entity_path):
                try:
                    with open(entity_path, "r", encoding="utf-8") as f:
                        entities_data = json.load(f)
                        entity_count = len(entities_data) if entities_data else 0
                except Exception:
                    pass

            # 统计 hyperedges
            hyperedge_count = 0
            hyperedge_path = os.path.join(working_dir, "kv_store_hyperedges.json")
            if os.path.exists(hyperedge_path):
                try:
                    with open(hyperedge_path, "r", encoding="utf-8") as f:
                        hyperedges_data = json.load(f)
                        if hyperedges_data:
                            hyperedge_count = sum(1 for h in hyperedges_data.values()
                                                  if not h.get("deleted", False))
                except Exception:
                    pass

            return {
                "entities": entity_count,
                "hyperedges": hyperedge_count,
                "working_dir": working_dir
            }
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {"entities": 0, "hyperedges": 0, "error": str(e)}

    def _format_response(self, success: bool, message: str, data: Dict = None) -> str:
        """Format response in simplified format"""
        # 延迟执行模式：message为空时，只返回简洁的success状态
        if not message:
            response = {"success": success}
            if data:
                response.update(data)
            return json.dumps(response, ensure_ascii=False)

        # 其他情况保持原格式
        response = {
            "success": success,
            "message": message,
        }
        if os.getenv('TOOL_RESPONSE_TIMESTAMP', 'false').lower() == 'true':
            response["timestamp"] = datetime.now().isoformat()
        if data:
            response.update(data)
        return json.dumps(response, ensure_ascii=False)

    def get_working_dir(self) -> str:
        """获取当前工作目录"""
        # 最高优先级：环境变量覆盖
        env_dir = os.getenv("GRAPHR1_WORKING_DIR")
        if env_dir and os.path.exists(env_dir):
            self._working_dir = env_dir
            return self._working_dir

        # 如果已经有工作目录，直接返回
        if hasattr(self, '_working_dir') and self._working_dir:
            return self._working_dir

        # 如果没有设置工作目录，使用与_get_or_create_graphr1相同的逻辑
        current_working_dir = os.getcwd()
        dataset_name = self._detect_dataset_name()

        if dataset_name:
            slurm_job_id = os.getenv("SLURM_JOB_ID")
            if slurm_job_id:
                fixed_dir_name = f"expr_{dataset_name}_working_{slurm_job_id}"
                fixed_dir = os.path.join(current_working_dir, fixed_dir_name)
                original_kb_dir = os.path.join(current_working_dir, "expr", dataset_name)

                if os.path.exists(fixed_dir):
                    self._working_dir = fixed_dir
                    return self._working_dir
                elif os.path.exists(original_kb_dir):
                    self._copy_knowledge_base(original_kb_dir, fixed_dir)
                    self._working_dir = fixed_dir
                    return self._working_dir
                else:
                    self._working_dir = original_kb_dir
                    return self._working_dir
            else:
                original_kb_dir = os.path.join(current_working_dir, "expr", dataset_name)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_kb_dir = os.path.join(current_working_dir, f"expr_{dataset_name}_working_{timestamp}")

                if os.path.exists(original_kb_dir):
                    self._copy_knowledge_base(original_kb_dir, backup_kb_dir)
                    self._working_dir = backup_kb_dir
                    return self._working_dir
                else:
                    self._working_dir = original_kb_dir
                    return self._working_dir
        else:
            self._working_dir = os.path.join(current_working_dir, "graphr1_kg")
            return self._working_dir

    # ========== 公共辅助方法 - 可被子类复用 ==========

    def _get_hyperedges_data(self, graphr1) -> Dict:
        """
        Get hyperedges data, using in-memory cache when available.
        First call loads from disk; subsequent calls return cached dict.
        """
        try:
            working_dir = graphr1.working_dir if graphr1 else os.getcwd()
            lock = self._get_cache_lock()
            with lock:
                cached = self._hyperedges_cache.get(working_dir)
                if cached is not None:
                    return cached["data"]

            # Cache miss — load from disk (outside lock to avoid blocking)
            kv_store_path = os.path.join(working_dir, "kv_store_hyperedges.json")
            if os.path.exists(kv_store_path):
                with open(kv_store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {}

            with lock:
                # Double-check: another thread may have loaded it
                if working_dir not in self._hyperedges_cache:
                    self._hyperedges_cache[working_dir] = {
                        "data": data, "dirty": False,
                    }
                return self._hyperedges_cache[working_dir]["data"]
        except Exception as e:
            logger.error(f"Error loading hyperedges data: {e}")
            return {}

    @classmethod
    def set_hyperedges_cache(cls, working_dir: str, hyperedges_data: Dict, dirty: bool = False) -> None:
        lock = cls._get_cache_lock()
        with lock:
            cls._hyperedges_cache[working_dir] = {
                "data": hyperedges_data,
                "dirty": dirty,
            }

    @classmethod
    def invalidate_hyperedges_cache(cls, working_dir: str) -> None:
        lock = cls._get_cache_lock()
        with lock:
            cls._hyperedges_cache.pop(working_dir, None)

    @classmethod
    def mark_graph_dirty(cls, graphr1) -> None:
        if graphr1 is None:
            return
        working_dir = getattr(graphr1, "working_dir", None)
        graph_storage = getattr(graphr1, "chunk_entity_relation_graph", None)
        if not working_dir or graph_storage is None:
            return
        lock = cls._get_cache_lock()
        with lock:
            cls._dirty_graphs[working_dir] = graph_storage

    @classmethod
    def mark_hyperedge_index_dirty(cls, target) -> None:
        working_dir = getattr(target, "working_dir", target)
        if not working_dir:
            return
        lock = cls._get_cache_lock()
        with lock:
            cls._dirty_hyperedge_indexes[str(working_dir)] = True

    @classmethod
    def remember_previous_hyperedges_data(cls, target, previous_hyperedges_data: Dict) -> None:
        working_dir = getattr(target, "working_dir", target)
        if not working_dir or previous_hyperedges_data is None:
            return
        lock = cls._get_cache_lock()
        with lock:
            cls._pending_previous_hyperedges_data.setdefault(
                str(working_dir),
                previous_hyperedges_data,
            )

    @classmethod
    def mark_entity_index_dirty(cls, target) -> None:
        working_dir = getattr(target, "working_dir", target)
        if not working_dir:
            return
        lock = cls._get_cache_lock()
        with lock:
            cls._dirty_entity_indexes[str(working_dir)] = True

    @classmethod
    def flush_dirty_graphs(cls, working_dir: str = None) -> int:
        from graphr1.graphr1 import always_get_an_event_loop

        lock = cls._get_cache_lock()
        with lock:
            dirs_to_flush = [working_dir] if working_dir else list(cls._dirty_graphs.keys())
            graph_items = [
                (wd, cls._dirty_graphs.get(wd))
                for wd in dirs_to_flush
                if cls._dirty_graphs.get(wd) is not None
            ]

        flushed = 0
        loop = always_get_an_event_loop()
        for wd, graph_storage in graph_items:
            try:
                loop.run_until_complete(graph_storage.index_done_callback())
                with lock:
                    cls._dirty_graphs.pop(wd, None)
                flushed += 1
                logger.info(f"[GRAPH_FLUSH] Written graph for {wd}")
            except Exception as e:
                logger.error(f"[GRAPH_FLUSH] Error writing graph for {wd}: {e}")
        return flushed

    @classmethod
    def rebuild_hyperedge_index(
        cls,
        working_dir: str,
        hyperedges_data: Dict = None,
        previous_hyperedges_data: Dict = None,
        allow_prefix_reuse: bool = False,
    ) -> bool:
        from agent.tool.tools.hyperedge_index_sync import rebuild_hyperedge_vector_index

        if not working_dir:
            return False

        lock = cls._get_cache_lock()
        if hyperedges_data is None:
            with lock:
                cached = cls._hyperedges_cache.get(working_dir)
                if cached is not None:
                    hyperedges_data = cached.get("data")

        try:
            rebuild_hyperedge_vector_index(
                working_dir,
                hyperedges_data=hyperedges_data,
                previous_hyperedges_data=previous_hyperedges_data,
                allow_prefix_reuse=allow_prefix_reuse,
            )
            with lock:
                cls._dirty_hyperedge_indexes.pop(str(working_dir), None)
            return True
        except Exception as e:
            logger.error(f"[HYPEREDGE_INDEX_FLUSH] Error rebuilding index for {working_dir}: {e}")
            return False

    @classmethod
    def flush_dirty_hyperedge_indexes_detailed(cls, working_dir: str = None) -> Dict[str, Any]:
        lock = cls._get_cache_lock()
        with lock:
            if working_dir:
                dirs_to_flush = [working_dir] if str(working_dir) in cls._dirty_hyperedge_indexes else []
            else:
                dirs_to_flush = list(cls._dirty_hyperedge_indexes.keys())
            previous_states = {
                str(wd): cls._pending_previous_hyperedges_data.get(str(wd))
                for wd in dirs_to_flush
            }

        flushed = 0
        failed_dirs = []
        for wd in dirs_to_flush:
            if cls.rebuild_hyperedge_index(
                wd,
                previous_hyperedges_data=previous_states.get(str(wd)),
            ):
                flushed += 1
                with lock:
                    cls._pending_previous_hyperedges_data.pop(str(wd), None)
            else:
                failed_dirs.append(wd)
        return {
            "flushed": flushed,
            "failed": len(failed_dirs),
            "failed_dirs": failed_dirs,
        }

    @classmethod
    def flush_dirty_hyperedge_indexes(cls, working_dir: str = None) -> int:
        return cls.flush_dirty_hyperedge_indexes_detailed(working_dir).get("flushed", 0)

    @classmethod
    def rebuild_entity_index(
        cls,
        working_dir: str,
        entities_data: Dict = None,
        previous_entities_data: Dict = None,
        allow_prefix_reuse: bool = False,
    ) -> bool:
        from agent.tool.tools.entity_index_sync import rebuild_entity_vector_index

        if not working_dir:
            return False

        try:
            rebuild_entity_vector_index(
                working_dir,
                entities_data=entities_data,
                previous_entities_data=previous_entities_data,
                allow_prefix_reuse=allow_prefix_reuse,
            )
            lock = cls._get_cache_lock()
            with lock:
                cls._dirty_entity_indexes.pop(str(working_dir), None)
            return True
        except Exception as e:
            logger.error(f"[ENTITY_INDEX_FLUSH] Error rebuilding index for {working_dir}: {e}")
            return False

    @classmethod
    def flush_dirty_entity_indexes_detailed(cls, working_dir: str = None) -> Dict[str, Any]:
        lock = cls._get_cache_lock()
        with lock:
            if working_dir:
                dirs_to_flush = [working_dir] if str(working_dir) in cls._dirty_entity_indexes else []
            else:
                dirs_to_flush = list(cls._dirty_entity_indexes.keys())

        flushed = 0
        failed_dirs = []
        for wd in dirs_to_flush:
            if cls.rebuild_entity_index(wd, allow_prefix_reuse=True):
                flushed += 1
            else:
                failed_dirs.append(wd)
        return {
            "flushed": flushed,
            "failed": len(failed_dirs),
            "failed_dirs": failed_dirs,
        }

    @classmethod
    def flush_dirty_entity_indexes(cls, working_dir: str = None) -> int:
        return cls.flush_dirty_entity_indexes_detailed(working_dir).get("flushed", 0)

    def _get_global_config_for_ai_metadata(self, graphr1) -> Dict:
        """获取用于AI元数据的global_config"""
        global_config = {}

        if graphr1 and hasattr(graphr1, 'llm_model_name'):
            global_config = {"llm_model_name": graphr1.llm_model_name}

        if not global_config.get("llm_model_name"):
            global_config["llm_model_name"] = (
                os.getenv("OPENAI_MODEL") or
                os.getenv("SILICONFLOW_MODEL") or
                os.getenv("ZHIPU_MODEL") or
                os.getenv("AI_MODEL_NAME") or
                "unknown"
            )

        return global_config

    def _save_hyperedges_data(self, graphr1, hyperedges_data) -> None:
        """
        保存 hyperedges 数据：更新内存缓存，标记 dirty，调度延迟写盘。
        实际磁盘写入由 _flush_hyperedges_cache 完成（在 batch queue flush 或 shutdown 时触发）。
        """
        working_dir = graphr1.working_dir if graphr1 else os.getcwd()
        self.set_hyperedges_cache(working_dir, hyperedges_data, dirty=True)
        self.mark_hyperedge_index_dirty(working_dir)

    def _save_hyperedges_data_now(self, graphr1, hyperedges_data) -> None:
        """立即写盘（用于需要即时持久化的场景）"""
        working_dir = graphr1.working_dir if graphr1 else os.getcwd()
        kv_store_path = os.path.join(working_dir, "kv_store_hyperedges.json")
        _write_json_with_filelock(kv_store_path, hyperedges_data)
        ensure_hyperedge_state_sidecars(working_dir, hyperedges_data)
        self.set_hyperedges_cache(working_dir, hyperedges_data, dirty=False)
        self.mark_hyperedge_index_dirty(working_dir)

    @classmethod
    def flush_hyperedges_cache(cls, working_dir: str = None) -> int:
        """
        将 dirty 的 hyperedges 缓存写盘。
        Args:
            working_dir: 指定目录，None 则 flush 所有 dirty 缓存
        Returns:
            写盘的目录数
        """
        lock = cls._get_cache_lock()
        flushed = 0
        with lock:
            dirs_to_flush = (
                [working_dir] if working_dir else list(cls._hyperedges_cache.keys())
            )
            for wd in dirs_to_flush:
                cached = cls._hyperedges_cache.get(wd)
                if cached and cached["dirty"]:
                    kv_store_path = os.path.join(wd, "kv_store_hyperedges.json")
                    try:
                        _write_json_with_filelock(kv_store_path, cached["data"])
                        ensure_hyperedge_state_sidecars(wd, cached["data"])
                        cached["dirty"] = False
                        cls._dirty_hyperedge_indexes[wd] = True
                        flushed += 1
                        logger.info(f"[CACHE_FLUSH] Written hyperedges to {kv_store_path}")
                    except Exception as e:
                        logger.error(f"[CACHE_FLUSH] Error writing {kv_store_path}: {e}")
        return flushed

    @classmethod
    def flush_deferred_io(cls) -> None:
        """
        Flush 所有延迟 IO：batch queue + hyperedges cache。
        用于需要即时持久化的场景（如测试验证、训练 checkpoint）。
        """
        from agent.tool.tools.batch.unified_batch_queue import UnifiedBatchQueue
        queue = UnifiedBatchQueue.get_instance()
        queue.flush()
        cls.flush_hyperedges_cache()
        cls.flush_dirty_graphs()
        cls.flush_dirty_hyperedge_indexes()
        cls.flush_dirty_entity_indexes()

    def batch_execute(self, args_list: List[Dict], update_type) -> List[str]:
        """执行批量操作：创建 BatchTask 列表并 submit_many"""
        from agent.tool.tools.batch.unified_batch_queue import BatchTask
        try:
            action_name = update_type.name.lower()
            tasks = []
            for i, args in enumerate(args_list):
                task = BatchTask(
                    task_id=f"{action_name}_{i}_{int(datetime.now().timestamp())}",
                    update_type=update_type,
                    working_dir=self.get_working_dir(),
                    data=args,
                    timestamp=time.time(),
                )
                tasks.append(task)

            before_failed = self._batch_queue.get_stats().get("total_failed", 0)
            self._batch_queue.submit_many(tasks)
            flush_result = self._batch_queue.flush()
            after_stats = self._batch_queue.get_stats()
            failed_count = after_stats.get("total_failed", 0) - before_failed
            task_results = [task.result or {"status": "succeeded"} for task in tasks]
            skipped_count = sum(1 for result in task_results if result.get("status") == "skipped")
            task_failed_count = sum(1 for result in task_results if result.get("status") == "failed")

            if failed_count > 0 or skipped_count > 0 or task_failed_count > 0:
                responses = []
                for task, task_result in zip(tasks, task_results):
                    status = task_result.get("status")
                    is_success = status == "succeeded"
                    response = build_edit_task_response(
                        success=is_success,
                        update_type=update_type.value,
                        task_result=task_result,
                        task_id=task.task_id,
                        task_status=status,
                        message=task_result.get("message") or f"Batch {action_name} {status}",
                        errors=after_stats.get("last_errors", []),
                        partial_success=not is_success,
                        kv_graph_updated=bool(task_result.get("kv_graph_updated", is_success)),
                        hyperedge_index_rebuilt=task_result.get(
                            "hyperedge_index_rebuilt",
                            failed_count == 0 and task_failed_count == 0,
                        ),
                        entity_index_rebuilt=task_result.get(
                            "entity_index_rebuilt",
                            failed_count == 0 and task_failed_count == 0,
                        ),
                        failed_task_count=failed_count or task_failed_count,
                        skipped_task_count=skipped_count,
                        flush_result=flush_result,
                    )
                    responses.append(json.dumps(response, ensure_ascii=False))
                return responses

            return [
                json.dumps(
                    build_edit_task_response(
                        success=True,
                        update_type=update_type.value,
                        task_result=task_results[i],
                        task_id=tasks[i].task_id,
                        task_status=task_results[i].get("status", "succeeded"),
                        hyperedge_index_rebuilt=True,
                        entity_index_rebuilt=True,
                        flush_result=flush_result,
                    ),
                    ensure_ascii=False,
                )
                for i in range(len(args_list))
            ]

        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            return [
                json.dumps(
                    build_edit_task_response(
                        success=False,
                        update_type=update_type.value,
                        task_result={
                            "success": False,
                            "message": f"Batch {action_name} failed: {str(e)}",
                        },
                    ),
                    ensure_ascii=False,
                )
                for _ in args_list
            ]

    def time_batch_submit(self, args: Dict, train_step: int, env_step: int, update_type) -> str:
        """提交任务到统一批量队列"""
        from agent.tool.tools.batch.unified_batch_queue import BatchTask
        try:
            import uuid
            action_name = update_type.name.lower()
            task_id = f"{action_name}_train{train_step}_env{env_step}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
            task = BatchTask(
                task_id=task_id,
                update_type=update_type,
                working_dir=self.get_working_dir(),
                data=args,
                timestamp=time.time(),
                train_step=train_step,
                env_step=env_step,
            )

            self._batch_queue.submit(task)
            logger.info(f"[TIME_BATCH_SUBMIT] Task submitted: {task_id}")
            return self._format_response(True, "")

        except Exception as e:
            logger.error(f"[TIME_BATCH_SUBMIT] Submission failed: {e}")
            return self._format_response(False, f"Time batch submission failed: {str(e)}")

    def process_time_batch(self, train_step: int) -> Dict[str, Any]:
        """手动 flush 统一队列"""
        try:
            before_stats = self._batch_queue.get_stats()
            before_failed = before_stats.get("total_failed", 0)
            before_errors = before_stats.get("last_errors", [])

            flush_result = self._batch_queue.flush()
            after_stats = self._batch_queue.get_stats()
            failed_count = after_stats.get("total_failed", 0) - before_failed
            after_errors = after_stats.get("last_errors", [])
            new_errors = (
                after_errors[len(before_errors):]
                if len(after_errors) >= len(before_errors)
                else after_errors
            )

            if failed_count > 0 or new_errors:
                return {
                    "status": "partial_failed",
                    "success": False,
                    "partial_success": True,
                    "kv_graph_updated": True,
                    "hyperedge_index_rebuilt": False,
                    "entity_index_rebuilt": False,
                    "failed_task_count": failed_count,
                    "flush_result": flush_result,
                    "errors": new_errors or after_errors,
                }

            return {
                **flush_result,
                "success": True,
                "hyperedge_index_rebuilt": True,
                "entity_index_rebuilt": True,
            }
        except Exception as e:
            logger.error(f"[TIME_BATCH_ERROR] Failed to process time batch: {e}")
            return {"status": "error", "message": str(e)}

    def get_time_batch_stats(self) -> Dict[str, Any]:
        """获取统一队列统计信息"""
        return self._batch_queue.get_stats()

    def calculate_reward(self, args: Dict, result: str, reward_on_success: float = 0.1, reward_on_failure: float = -0.1) -> float:
        """计算奖励（子类可以自定义奖励值）"""
        try:
            result_obj = json.loads(result)
            if result_obj.get("success", False):
                return reward_on_success
            else:
                return reward_on_failure
        except Exception:
            return reward_on_failure
