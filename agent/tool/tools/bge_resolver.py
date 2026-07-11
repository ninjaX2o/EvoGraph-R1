import os
from typing import List, Optional, Tuple


DEFAULT_BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
REQUIRED_MODEL_FILES = ("config.json",)
TOKENIZER_MODEL_FILES = ("tokenizer.json", "vocab.txt", "sentencepiece.bpe.model")
HF_CACHE_MODEL_DIR = "models--BAAI--bge-large-en-v1.5"


def get_bge_model_candidates() -> List[str]:
    candidates = [os.getenv("BGE_MODEL_PATH")]
    extra_paths = os.getenv("BGE_MODEL_PATHS")
    if extra_paths:
        candidates.extend(extra_paths.split(os.pathsep))
    return [candidate for candidate in candidates if candidate]


def resolve_local_bge_model_path() -> Optional[str]:
    for candidate in get_bge_model_candidates():
        if is_valid_bge_model_dir(candidate):
            return candidate
    cached_path = resolve_hf_cached_bge_model_path()
    if cached_path:
        return cached_path
    return None


def resolve_bge_model_reference() -> Tuple[str, bool]:
    local_path = resolve_local_bge_model_path()
    if local_path:
        return local_path, True
    return DEFAULT_BGE_MODEL_NAME, False


def is_valid_bge_model_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False

    has_required_files = all(
        os.path.isfile(os.path.join(path, filename))
        for filename in REQUIRED_MODEL_FILES
    )
    has_tokenizer_files = any(
        os.path.isfile(os.path.join(path, filename))
        for filename in TOKENIZER_MODEL_FILES
    )
    return has_required_files and has_tokenizer_files


def resolve_hf_cached_bge_model_path() -> Optional[str]:
    cache_root = (
        os.getenv("HF_HOME")
        or os.getenv("HUGGINGFACE_HUB_CACHE")
        or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    )
    hub_root = cache_root
    if os.path.basename(hub_root) != "hub":
        hub_root = os.path.join(hub_root, "hub")

    model_cache = os.path.join(hub_root, HF_CACHE_MODEL_DIR)
    refs_main = os.path.join(model_cache, "refs", "main")
    snapshot_id = None
    if os.path.isfile(refs_main):
        with open(refs_main, "r", encoding="utf-8") as f:
            snapshot_id = f.read().strip()

    snapshot_candidates = []
    if snapshot_id:
        snapshot_candidates.append(os.path.join(model_cache, "snapshots", snapshot_id))

    snapshots_root = os.path.join(model_cache, "snapshots")
    if os.path.isdir(snapshots_root):
        snapshot_candidates.extend(
            os.path.join(snapshots_root, name)
            for name in os.listdir(snapshots_root)
        )

    for candidate in snapshot_candidates:
        if is_valid_bge_model_dir(candidate):
            return candidate
    return None
