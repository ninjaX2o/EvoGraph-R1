"""
BGE model manager.

This module keeps a process-global BGE encoder and prefers local model paths
before falling back to the Hugging Face model name.
"""

import logging
import importlib.util
import os
import tempfile
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from agent.tool.tools.bge_resolver import resolve_bge_model_reference

logger = logging.getLogger(__name__)

_GLOBAL_BGE_MODEL = None
_GLOBAL_BGE_MODEL_LOADED = False
_GLOBAL_BGE_MODEL_FAILURE_REASON = None
_BGE_MODEL_LOCK = threading.Lock()
_BGE_LOCK_FILE = os.path.join(tempfile.gettempdir(), "bge_model_lock")


class EmbeddingDimensionMismatchError(ValueError):
    """Raised when a real BGE model returns an unexpected embedding size."""


def _acquire_process_lock() -> bool:
    max_retries = 10 if os.getenv("SLURM_JOB_ID") else 30
    retry_delay = 0.5 if os.getenv("SLURM_JOB_ID") else 1.0

    for _ in range(max_retries):
        try:
            if not os.path.exists(_BGE_LOCK_FILE):
                with open(_BGE_LOCK_FILE, "w", encoding="utf-8") as f:
                    f.write(str(os.getpid()))
                time.sleep(0.1)
                return True

            try:
                with open(_BGE_LOCK_FILE, "r", encoding="utf-8") as f:
                    pid = f.read().strip()
            except OSError:
                pid = ""

            if pid and os.path.exists(f"/proc/{pid}"):
                time.sleep(retry_delay)
                continue

            try:
                os.remove(_BGE_LOCK_FILE)
            except OSError:
                pass
        except Exception as e:
            logger.warning(f"Failed to operate BGE lock file: {e}")
            time.sleep(retry_delay)

    logger.warning("Failed to acquire BGE process lock")
    return False


def _release_process_lock() -> None:
    try:
        if os.path.exists(_BGE_LOCK_FILE):
            os.remove(_BGE_LOCK_FILE)
    except OSError:
        pass


def _is_verl_environment() -> bool:
    try:
        import ray

        return ray.is_initialized()
    except ImportError:
        return False


def _build_bge_model() -> object:
    model_reference, is_local_model = resolve_bge_model_reference()
    try:
        FlagModel = _load_flag_model_class()

        if is_local_model:
            logger.info(f"Using local BGE model path: {model_reference}")
        else:
            logger.info(f"Falling back to remote BGE model name: {model_reference}")

        return FlagModel(
            model_reference,
            query_instruction_for_retrieval=(
                "Represent this sentence for searching relevant passages: "
            ),
        )
    except Exception as e:
        logger.warning(
            "FlagEmbedding is unavailable or failed to initialize, "
            f"falling back to transformers/sentence-transformers: {e}"
        )
        if is_local_model:
            import torch
            from sentence_transformers import SentenceTransformer

            requested_device = os.getenv("BGE_DEVICE", "").strip()
            device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Loading SentenceTransformer BGE fallback on device: %s", device)
            return SentenceTransformer(model_reference, device=device)

        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_reference)


def _load_flag_model_class():
    try:
        from FlagEmbedding import FlagModel

        return FlagModel
    except Exception as package_error:
        package_spec = importlib.util.find_spec("FlagEmbedding")
        search_locations = list(package_spec.submodule_search_locations or []) if package_spec else []
        for package_dir in search_locations:
            module_path = os.path.join(package_dir, "flag_models.py")
            if not os.path.exists(module_path):
                continue
            module_spec = importlib.util.spec_from_file_location(
                "_graphr1_flagembedding_flag_models",
                module_path,
            )
            if module_spec is None or module_spec.loader is None:
                continue
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            logger.warning(
                "Loaded FlagModel directly from flag_models.py because "
                f"FlagEmbedding package import failed: {package_error}"
            )
            return module.FlagModel
        raise package_error


def get_bge_model() -> Tuple[Optional[object], bool]:
    global _GLOBAL_BGE_MODEL, _GLOBAL_BGE_MODEL_LOADED, _GLOBAL_BGE_MODEL_FAILURE_REASON

    with _BGE_MODEL_LOCK:
        if _GLOBAL_BGE_MODEL_LOADED:
            return _GLOBAL_BGE_MODEL, _GLOBAL_BGE_MODEL is not None

        if os.getenv("SLURM_JOB_ID") and not _acquire_process_lock():
            _GLOBAL_BGE_MODEL = None
            _GLOBAL_BGE_MODEL_LOADED = True
            _GLOBAL_BGE_MODEL_FAILURE_REASON = "failed to acquire BGE process lock"
            return None, False

        try:
            _GLOBAL_BGE_MODEL = _build_bge_model()
            test_embeddings = _GLOBAL_BGE_MODEL.encode(["test"])
            if test_embeddings is None or len(test_embeddings) == 0:
                raise RuntimeError("BGE model test encode returned empty result")
            _validate_embedding_dimension(
                test_embeddings,
                target_dimension=1024,
                model_source=_describe_model_source(_GLOBAL_BGE_MODEL),
            )
            logger.info("Global BGE model initialized")
            _GLOBAL_BGE_MODEL_LOADED = True
            return _GLOBAL_BGE_MODEL, True
        except EmbeddingDimensionMismatchError:
            _GLOBAL_BGE_MODEL = None
            _GLOBAL_BGE_MODEL_LOADED = False
            _GLOBAL_BGE_MODEL_FAILURE_REASON = None
            raise
        except Exception as e:
            logger.error(f"Failed to initialize global BGE model: {e}")
            _GLOBAL_BGE_MODEL = None
            _GLOBAL_BGE_MODEL_LOADED = True
            _GLOBAL_BGE_MODEL_FAILURE_REASON = str(e)
            return None, False
        finally:
            _release_process_lock()


def encode_text_safe(text: str, target_dimension: int = 1024) -> Optional[np.ndarray]:
    return encode_texts_safe([text], target_dimension=target_dimension)


def encode_texts_safe(texts: List[str], target_dimension: int = 1024) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]

    model, success = get_bge_model()
    if success and model is not None:
        try:
            embeddings = model.encode(texts)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            return _validate_embedding_dimension(
                embeddings,
                target_dimension,
                model_source=_describe_model_source(model),
            )
        except EmbeddingDimensionMismatchError:
            raise
        except Exception as e:
            logger.error(f"BGE batch encode failed: {e}")

    fallback_vectors = [
        _hash_text_fallback(text, target_dimension=target_dimension)[0]
        for text in texts
    ]
    return _validate_embedding_dimension(
        np.asarray(fallback_vectors, dtype=np.float32),
        target_dimension,
        model_source="hash_fallback",
    )


def _describe_model_source(model: object) -> str:
    model_reference, is_local_model = resolve_bge_model_reference()
    model_scope = "local" if is_local_model else "remote"
    return f"{type(model).__name__} via {model_scope}:{model_reference}"


def _validate_embedding_dimension(
    embeddings: np.ndarray,
    target_dimension: int = 1024,
    model_source: str = "unknown",
) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    current_dimension = embeddings.shape[1]
    if current_dimension == target_dimension:
        return embeddings

    raise EmbeddingDimensionMismatchError(
        "BGE embedding dimension mismatch: "
        f"expected {target_dimension}, got {current_dimension}, source={model_source}"
    )


def _hash_text_fallback(text: str, target_dimension: int = 1024) -> np.ndarray:
    import hashlib

    text_normalized = text.lower().strip()
    if not text_normalized:
        return np.zeros((1, target_dimension), dtype=np.float32)

    features = []

    def stable_feature(value: str) -> float:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    for i in range(min(len(text_normalized), target_dimension // 4)):
        features.append(stable_feature(f"char:{text_normalized[i : i + 1]}:{i}"))

    for i, word in enumerate(text_normalized.split()[: target_dimension // 4]):
        features.append(stable_feature(f"word:{word}:{i}"))

    features.append(stable_feature(f"text:{text_normalized}"))

    md5_hash = hashlib.md5(text_normalized.encode()).hexdigest()
    for i in range(0, len(md5_hash), 2):
        if len(features) >= target_dimension:
            break
        hex_val = int(md5_hash[i : i + 2], 16)
        features.append(hex_val / 255.0)

    while len(features) < target_dimension:
        features.append(0.0)

    vector = np.array(features[:target_dimension], dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(vector[0])
    if norm > 0:
        vector[0] = vector[0] / norm
    return vector


def clear_bge_model() -> None:
    global _GLOBAL_BGE_MODEL, _GLOBAL_BGE_MODEL_LOADED, _GLOBAL_BGE_MODEL_FAILURE_REASON

    with _BGE_MODEL_LOCK:
        _GLOBAL_BGE_MODEL = None
        _GLOBAL_BGE_MODEL_LOADED = False
        _GLOBAL_BGE_MODEL_FAILURE_REASON = None


def is_bge_model_loaded() -> bool:
    with _BGE_MODEL_LOCK:
        return _GLOBAL_BGE_MODEL_LOADED and _GLOBAL_BGE_MODEL is not None
