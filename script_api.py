import json
import os
import time
import threading
import gc
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import faiss
import numpy as np
try:
    from FlagEmbedding import FlagModel
except ImportError:
    FlagModel = None
from typing import Dict, List
import argparse
from graphr1 import GraphR1, QueryParam
import asyncio
from tqdm import tqdm
import logging
from agent.tool.tools.hyperedge_index_sync import (
    iter_active_hyperedge_contents,
    iter_searchable_hyperedge_contents,
    load_hyperedge_index_metadata,
    rebuild_hyperedge_vector_index,
)
from agent.tool.tools.entity_index_sync import load_entity_index_metadata
from agent.tool.tools.bge_resolver import resolve_bge_model_reference
from agent.tool.tools.hyperedge_state_sync import (
    HYPEREDGE_LOOKUP_FILE,
    HYPEREDGE_RECENT_MUTATIONS_FILE,
    iter_recent_active_contents,
    load_hyperedge_lookup_payload,
    load_recent_hyperedge_mutations,
    normalize_lookup_content,
)

# 璁剧疆鏃ュ織
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceTransformerBGEModel:
    def __init__(self, model_reference: str, query_instruction: str):
        import torch
        from sentence_transformers import SentenceTransformer

        requested_device = os.getenv("BGE_DEVICE", "").strip()
        device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading SentenceTransformer BGE fallback on device: %s", device)
        self.model = SentenceTransformer(model_reference, device=device)
        self.query_instruction = query_instruction

    def encode_queries(self, queries):
        prefixed_queries = [f"{self.query_instruction}{query}" for query in queries]
        embeddings = self.model.encode(
            prefixed_queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

parser = argparse.ArgumentParser()
parser.add_argument('--data_source', default='2WikiMultiHopQA')
parser.add_argument('--reload_interval', type=int, default=300, help='Auto reload interval in seconds; 0 disables background reload checks.')
parser.add_argument('--check_interval', type=int, default=60, help='Minimum interval in seconds between file-change checks.')
parser.add_argument('--working_dir', type=str, help='Override the GraphR1 working directory.')
parser.add_argument('--host', default='127.0.0.1', help='Host interface for the search API server.')
parser.add_argument('--port', type=int, default=8001, help='Port for the search API server.')
args, _unknown_args = parser.parse_known_args()
data_source = args.data_source

# 妯″瀷灏嗗湪load_embedding_model()鍑芥暟涓姞杞?

# 鑾峰彇宸ヤ綔鐩綍
working_dir = args.working_dir or os.getenv("GRAPHR1_WORKING_DIR", f"expr/{data_source}")
print(f"[DEBUG] Using working directory: {working_dir}")

# 鍏ㄥ眬鍙橀噺瀛樺偍妯″瀷鍜屾暟鎹?
model = None
index_entity = None
corpus_entity = []
index_hyperedge = None
corpus_hyperedge = []
rag = None
hyperedge_lookup_payload = {}
recent_hyperedge_mutations = []
hyperedge_search_metadata = {}
rag_graph_stale = False
last_reload_time = 0
last_check_time = 0
data_changed = False
reload_lock = threading.Lock()


def get_watched_paths(base_dir: str = None) -> List[str]:
    """Files whose mtime should trigger search-index reload."""
    base_dir = base_dir or working_dir
    return [
        f"{base_dir}/kv_store_entities.json",
        f"{base_dir}/kv_store_hyperedges.json",
        f"{base_dir}/index_entity.bin",
        f"{base_dir}/corpus_entity.npy",
        f"{base_dir}/entity_index_metadata.json",
        f"{base_dir}/index_hyperedge.bin",
        f"{base_dir}/corpus_hyperedge.npy",
        f"{base_dir}/hyperedge_index_metadata.json",
        f"{base_dir}/{HYPEREDGE_LOOKUP_FILE}",
        f"{base_dir}/{HYPEREDGE_RECENT_MUTATIONS_FILE}",
    ]

def load_embedding_model():
    """鍔犺浇宓屽叆妯″瀷"""
    global model
    try:
        if FlagModel is None:
            raise ImportError("FlagEmbedding is not installed")
        model_reference, is_local_model = resolve_bge_model_reference()
        if is_local_model:
            logger.info("Using local BGE model path: %s", model_reference)
        else:
            logger.info("Using remote BGE model reference: %s", model_reference)
        query_instruction = "Represent this sentence for searching relevant passages: "
        try:
            model = FlagModel(
                model_reference,
                query_instruction_for_retrieval=query_instruction,
                devices=os.getenv("BGE_DEVICE", "").strip() or None,
            )
        except TypeError as flag_error:
            if "unexpected keyword argument 'dtype'" not in str(flag_error):
                raise
            logger.warning(
                "FlagEmbedding is incompatible with this transformers build (%s); "
                "falling back to SentenceTransformer on CPU",
                flag_error,
            )
            model = SentenceTransformerBGEModel(model_reference, query_instruction)
        logger.info("BGE妯″瀷鍔犺浇鎴愬姛")
        return True
    except Exception as e:
        logger.error(f"BGE妯″瀷鍔犺浇澶辫触: {e}")
        return False

def _load_search_sidecars() -> None:
    global hyperedge_lookup_payload, recent_hyperedge_mutations
    hyperedge_lookup_payload = load_hyperedge_lookup_payload(working_dir)
    recent_hyperedge_mutations = load_recent_hyperedge_mutations(working_dir)


def _get_searchable_hyperedge_lookup() -> Dict[str, str]:
    if not isinstance(hyperedge_lookup_payload, dict):
        return {}
    for section in ("searchable", "all"):
        lookup = hyperedge_lookup_payload.get(section)
        if isinstance(lookup, dict):
            return lookup
    return {}


def _get_active_hyperedge_lookup() -> Dict[str, str]:
    active_lookup = hyperedge_lookup_payload.get("active") if isinstance(hyperedge_lookup_payload, dict) else None
    return active_lookup if isinstance(active_lookup, dict) else {}


def _has_searchable_hyperedge_lookup() -> bool:
    return isinstance(hyperedge_lookup_payload, dict) and (
        "searchable" in hyperedge_lookup_payload or "all" in hyperedge_lookup_payload
    )


def _has_active_hyperedge_lookup() -> bool:
    return isinstance(hyperedge_lookup_payload, dict) and "active" in hyperedge_lookup_payload


def _is_known_searchable_hyperedge(raw_content: str) -> bool:
    searchable_lookup = _get_searchable_hyperedge_lookup()
    if not _has_searchable_hyperedge_lookup():
        return True
    return normalize_lookup_content(raw_content) in searchable_lookup


def _get_hyperedge_search_state(raw_content: str) -> Dict:
    normalized = normalize_lookup_content(raw_content)
    default_state = {
        "deleted": False,
        "active": True,
        "weight": 1.0,
    }
    if not normalized:
        return default_state
    state = hyperedge_search_metadata.get(normalized)
    if not isinstance(state, dict):
        return default_state
    merged = dict(default_state)
    merged.update(state)
    return merged


def _sort_results_by_search_contract(results: List[Dict]) -> List[Dict]:
    scored = []
    for index, item in enumerate(results):
        if not isinstance(item, dict) or "<knowledge>" not in item:
            continue
        state = _get_hyperedge_search_state(item.get("<knowledge>", ""))
        coherence = item.get("<coherence>", 0.0)
        try:
            base_score = float(coherence)
        except (TypeError, ValueError):
            base_score = 0.0
        scored.append(
            (
                (
                    1 if state.get("active", True) else 0,
                    base_score,
                    float(state.get("weight", 1.0)),
                    -index,
                ),
                item,
            )
        )
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored]


def _promote_recent_hyperedges(query_text: str, matched_hyperedges: List[str]) -> List[str]:
    normalized_query = normalize_lookup_content(query_text)
    if not normalized_query:
        return matched_hyperedges

    searchable_lookup = _get_searchable_hyperedge_lookup()
    if not _has_searchable_hyperedge_lookup():
        return matched_hyperedges

    boosted = []
    seen = set()
    if normalized_query in searchable_lookup:
        boosted.extend(
            item
            for item in iter_recent_active_contents(recent_hyperedge_mutations)
            if normalize_lookup_content(item) == normalized_query
            and normalize_lookup_content(item) in searchable_lookup
        )

    for item in [
        mutation.get("content") or mutation.get("plain_content") or ""
        for mutation in recent_hyperedge_mutations
        if isinstance(mutation, dict) and mutation.get("searchable", True)
    ]:
        normalized_item = normalize_lookup_content(item)
        if normalized_item not in searchable_lookup:
            continue
        if normalized_item == normalized_query:
            boosted.append(item)
            continue
        if len(normalized_query) >= 12 and (
            normalized_query in normalized_item or normalized_item in normalized_query
        ):
            boosted.append(item)

    merged = []
    for item in boosted + matched_hyperedges:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged


def _merge_fresh_results(results, matched_hyperedges: List[str]) -> List[Dict]:
    merged = []
    seen = set()
    for item in matched_hyperedges:
        normalized_item = normalize_lookup_content(item)
        if not normalized_item or normalized_item in seen:
            continue
        seen.add(normalized_item)
        merged.append({"<knowledge>": normalized_item, "<coherence>": 1.0})

    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict) or "<knowledge>" not in item:
                continue
            normalized_item = normalize_lookup_content(item.get("<knowledge>", ""))
            if not normalized_item or normalized_item in seen:
                continue
            seen.add(normalized_item)
            merged.append(item)

    return _sort_results_by_search_contract(merged)


def _results_from_matches(entity_matches: List[str], hyperedge_matches: List[str]) -> List[Dict]:
    merged = []
    seen = set()
    filtered_hyperedge_matches = [
        item for item in hyperedge_matches
        if _is_known_searchable_hyperedge(item)
    ]
    for rank, item in enumerate(filtered_hyperedge_matches):
        normalized_item = normalize_lookup_content(item)
        if not normalized_item or normalized_item in seen:
            continue
        seen.add(normalized_item)
        merged.append({"<knowledge>": normalized_item, "<coherence>": 1.0 / (rank + 1)})
    base_rank = len(filtered_hyperedge_matches)
    for offset, item in enumerate(entity_matches):
        normalized_item = normalize_lookup_content(item)
        if not normalized_item or normalized_item in seen:
            continue
        seen.add(normalized_item)
        merged.append({"<knowledge>": normalized_item, "<coherence>": 1.0 / (base_rank + offset + 1)})
    return _sort_results_by_search_contract(merged)


def _should_use_rag_results() -> bool:
    return rag is not None and not rag_graph_stale


def _is_legacy_active_only_hyperedge_index(
    hyperedges: Dict,
    loaded_index_hyperedge,
    loaded_corpus_hyperedge,
) -> bool:
    if loaded_index_hyperedge is None or loaded_corpus_hyperedge is None:
        return False
    searchable_count = sum(1 for _ in iter_searchable_hyperedge_contents(hyperedges))
    active_count = sum(1 for _ in iter_active_hyperedge_contents(hyperedges))
    return (
        searchable_count > active_count
        and loaded_index_hyperedge.ntotal == active_count
        and loaded_corpus_hyperedge.shape[0] == active_count
    )


def _migrate_legacy_hyperedge_index_to_searchable(hyperedges: Dict) -> bool:
    try:
        previous_active_hyperedges_data = {
            hyperedge_id: hyperedge
            for hyperedge_id, hyperedge in hyperedges.items()
            if isinstance(hyperedge, dict) and not hyperedge.get("deleted", False)
        }
        logger.warning(
            "Detected legacy active-only hyperedge index for searchable KV state; rebuilding searchable hyperedge index"
        )
        rebuild_hyperedge_vector_index(
            working_dir,
            hyperedges_data=hyperedges,
            previous_hyperedges_data=previous_active_hyperedges_data,
            allow_prefix_reuse=False,
        )
        return True
    except Exception as e:
        logger.error("Failed to migrate legacy hyperedge index to searchable contract: %s", e)
        return False


def load_data(load_graph: bool = True, allow_hyperedge_searchable_migration: bool = True):
    """鍔犺浇鎵€鏈夋暟鎹拰绱㈠紩"""
    global index_entity, corpus_entity, index_hyperedge, corpus_hyperedge, rag, rag_graph_stale, hyperedge_search_metadata
    
    try:
        # 鍔犺浇瀹炰綋绱㈠紩鍜岃鏂欏簱
        logger.info(f"[DEBUG] LOADING ENTITY EMBEDDINGS from {working_dir}")
        entity_index_path = f"{working_dir}/index_entity.bin"
        entity_corpus_path = f"{working_dir}/corpus_entity.npy"
        entity_kv_path = f"{working_dir}/kv_store_entities.json"
        
        if os.path.exists(entity_index_path) and os.path.exists(entity_corpus_path):
            loaded_index_entity = faiss.read_index(entity_index_path)
            loaded_corpus_entity = np.load(entity_corpus_path, mmap_mode="r")
            corpus_entity = []
            with open(entity_kv_path, 'r', encoding='utf-8') as f:
                entities = json.load(f)
                for item in entities:
                    corpus_entity.append(entities[item]['entity_name'])
            if loaded_index_entity.ntotal != len(corpus_entity) or loaded_corpus_entity.shape[0] != len(corpus_entity):
                logger.error(
                    "Entity index/KV mismatch: index=%s, corpus_rows=%s, kv=%s. "
                    "Run CRUD flush or rebuild index before reload.",
                    loaded_index_entity.ntotal,
                    loaded_corpus_entity.shape[0],
                    len(corpus_entity),
                )
                index_entity = None
                corpus_entity = []
                return False
            elif loaded_corpus_entity.ndim != 2 or loaded_index_entity.d != loaded_corpus_entity.shape[1]:
                logger.error(
                    "Entity index/corpus dimension mismatch: index_dim=%s, corpus_dim=%s",
                    loaded_index_entity.d,
                    loaded_corpus_entity.shape[1] if loaded_corpus_entity.ndim == 2 else None,
                )
                index_entity = None
                corpus_entity = []
                return False
            elif load_entity_index_metadata(working_dir).get("provider") == "openai":
                logger.error(
                    "Entity index provider is openai; this runtime supports BGE indexes only. "
                    "Rebuild the knowledge base with script_build.py."
                )
                index_entity = None
                corpus_entity = []
                return False
            elif loaded_index_entity.d != 1024:
                logger.error(
                    "Entity index dimension is %s, but BGE script_api.py expects 1024; "
                    "use the matching search API or rebuild the index with BGE.",
                    loaded_index_entity.d,
                )
                index_entity = None
                corpus_entity = []
                return False
            else:
                index_entity = loaded_index_entity
            logger.info(f"[DEBUG] ENTITY EMBEDDINGS LOADED: {len(corpus_entity)} entities")
        else:
            logger.warning(f"瀹炰綋绱㈠紩鏂囦欢涓嶅瓨鍦? {entity_index_path}, {entity_corpus_path}")
            index_entity = None
            corpus_entity = []
            return False

        # 鍔犺浇瓒呰竟绱㈠紩鍜岃鏂欏簱
        logger.info(f"[DEBUG] LOADING HYPEREDGE EMBEDDINGS from {working_dir}")
        hyperedge_index_path = f"{working_dir}/index_hyperedge.bin"
        hyperedge_corpus_path = f"{working_dir}/corpus_hyperedge.npy"
        hyperedge_kv_path = f"{working_dir}/kv_store_hyperedges.json"
        
        if os.path.exists(hyperedge_index_path) and os.path.exists(hyperedge_corpus_path):
            loaded_index_hyperedge = faiss.read_index(hyperedge_index_path)
            loaded_corpus_hyperedge = np.load(hyperedge_corpus_path, mmap_mode="r")
            corpus_hyperedge = []
            hyperedge_search_metadata = {}
            with open(hyperedge_kv_path, 'r', encoding='utf-8') as f:
                hyperedges = json.load(f)
                corpus_hyperedge.extend(iter_searchable_hyperedge_contents(hyperedges))
                for hyperedge in hyperedges.values():
                    if not isinstance(hyperedge, dict):
                        continue
                    raw_content = hyperedge.get("content") or hyperedge.get("hyperedge_name") or ""
                    normalized_content = normalize_lookup_content(raw_content)
                    if not normalized_content:
                        continue
                    deleted = bool(hyperedge.get("deleted", False))
                    current_weight = hyperedge.get("weight", 1.0)
                    try:
                        weight = float(current_weight)
                    except (TypeError, ValueError):
                        weight = 1.0
                    existing = hyperedge_search_metadata.get(normalized_content)
                    candidate_state = {
                        "deleted": deleted,
                        "active": not deleted,
                        "weight": weight,
                    }
                    if existing is None:
                        hyperedge_search_metadata[normalized_content] = candidate_state
                    elif existing.get("deleted", False) and not deleted:
                        hyperedge_search_metadata[normalized_content] = candidate_state
            if (
                allow_hyperedge_searchable_migration
                and _is_legacy_active_only_hyperedge_index(
                    hyperedges,
                    loaded_index_hyperedge,
                    loaded_corpus_hyperedge,
                )
            ):
                loaded_corpus_hyperedge = None
                gc.collect()
                if _migrate_legacy_hyperedge_index_to_searchable(hyperedges):
                    return load_data(
                        load_graph=load_graph,
                        allow_hyperedge_searchable_migration=False,
                    )
                return False
            if loaded_index_hyperedge.ntotal != len(corpus_hyperedge) or loaded_corpus_hyperedge.shape[0] != len(corpus_hyperedge):
                logger.error(
                    "Hyperedge index/KV mismatch: index=%s, corpus_rows=%s, searchable_kv=%s. "
                    "Run CRUD flush or rebuild index before reload.",
                    loaded_index_hyperedge.ntotal,
                    loaded_corpus_hyperedge.shape[0],
                    len(corpus_hyperedge),
                )
                index_hyperedge = None
                corpus_hyperedge = []
                return False
            elif loaded_corpus_hyperedge.ndim != 2 or loaded_index_hyperedge.d != loaded_corpus_hyperedge.shape[1]:
                logger.error(
                    "Hyperedge index/corpus dimension mismatch: index_dim=%s, corpus_dim=%s",
                    loaded_index_hyperedge.d,
                    loaded_corpus_hyperedge.shape[1] if loaded_corpus_hyperedge.ndim == 2 else None,
                )
                index_hyperedge = None
                corpus_hyperedge = []
                return False
            elif load_hyperedge_index_metadata(working_dir).get("provider") == "openai":
                logger.error(
                    "Hyperedge index provider is openai; this runtime supports BGE indexes only. "
                    "Rebuild the knowledge base with script_build.py."
                )
                index_hyperedge = None
                corpus_hyperedge = []
                return False
            elif loaded_index_hyperedge.d != 1024:
                logger.error(
                    "Hyperedge index dimension is %s, but BGE script_api.py expects 1024; "
                    "use the matching search API or rebuild the index with BGE.",
                    loaded_index_hyperedge.d,
                )
                index_hyperedge = None
                corpus_hyperedge = []
                return False
            else:
                index_hyperedge = loaded_index_hyperedge
            logger.info(f"[DEBUG] HYPEREDGE EMBEDDINGS LOADED: {len(corpus_hyperedge)} hyperedges")
        else:
            logger.warning(f"瓒呰竟绱㈠紩鏂囦欢涓嶅瓨鍦? {hyperedge_index_path}, {hyperedge_corpus_path}")
            index_hyperedge = None
            corpus_hyperedge = []
            hyperedge_search_metadata = {}

        # 鍔犺浇GraphR1瀹炰緥
        if not load_graph:
            _load_search_sidecars()
            rag_graph_stale = rag is not None
            return True

        try:
            _load_search_sidecars()
            rag = GraphR1(working_dir=working_dir)
            rag_graph_stale = False
            logger.info("GraphR1瀹炰緥鍔犺浇鎴愬姛")
        except Exception as e:
            logger.warning(f"GraphR1瀹炰緥鍔犺浇澶辫触: {e}")
            rag = None
            rag_graph_stale = False

        return True
        
    except Exception as e:
        logger.error(f"鏁版嵁鍔犺浇澶辫触: {e}")
        return False

def check_data_changes():
    """Check whether KV/index files changed since the last successful reload."""
    global last_check_time, data_changed

    current_time = time.time()
    if current_time - last_check_time < args.check_interval:
        return data_changed

    try:
        max_mtime = 0
        for file_path in get_watched_paths(working_dir):
            if os.path.exists(file_path):
                max_mtime = max(max_mtime, os.path.getmtime(file_path))

        data_changed = max_mtime > last_reload_time
        last_check_time = current_time
        return data_changed
    except Exception as e:
        logger.error("Failed to check data changes: %s", e)
        return False


def _get_max_watched_mtime(base_dir: str = None) -> float:
    max_mtime = 0.0
    for file_path in get_watched_paths(base_dir or working_dir):
        if os.path.exists(file_path):
            max_mtime = max(max_mtime, os.path.getmtime(file_path))
    return max_mtime


def reload_data_if_needed():
    """Reload data if watched files changed."""
    global last_reload_time, data_changed

    if not check_data_changes():
        return True

    with reload_lock:
        if not data_changed:
            return True

        logger.info("Detected data changes, reloading search artifacts")
        if load_data(load_graph=False):
            last_reload_time = time.time()
            data_changed = False
            logger.info("Search artifacts reloaded")
            return True

        logger.error("Search artifact reload failed")
        return False


def auto_reload_worker():
    """Background worker that periodically checks watched files."""
    while True:
        try:
            if args.reload_interval > 0:
                time.sleep(args.reload_interval)
                check_data_changes()
            else:
                time.sleep(60)
        except Exception as e:
            logger.error("Auto reload worker error: %s", e)
            time.sleep(10)


async def process_query(query_text, rag_instance, entity_match, hyperedge_match, rag_top_k=10):
    """澶勭悊鏌ヨ"""
    try:
        # 娣诲姞璋冭瘯淇℃伅
        logger.info(f"Processing query with rag_top_k={rag_top_k}")
        logger.info(f"Entity match count: {len(entity_match) if entity_match else 0}")
        logger.info(f"Hyperedge match count: {len(hyperedge_match) if hyperedge_match else 0}")
        
        # 鎵撳嵃鍏蜂綋鐨別ntity鍜宧yperedge鍖归厤缁撴灉
        if entity_match and query_text in entity_match:
            logger.info(f"Entity matches for '{query_text}': {entity_match[query_text][:5]}")  # 鍙樉绀哄墠5涓?
        if hyperedge_match and query_text in hyperedge_match:
            logger.info(f"Hyperedge matches for '{query_text}': {hyperedge_match[query_text][:5]}")  # 鍙樉绀哄墠5涓?
        
        result = await rag_instance.aquery(
            query_text, 
            param=QueryParam(
                only_need_context=True, 
                top_k=rag_top_k,
                max_token_for_text_unit=16000,  # 澶у箙澧炲姞鏂囨湰鍗曞厓token闄愬埗
                max_token_for_global_context=16000,  # 澶у箙澧炲姞鍏ㄥ眬涓婁笅鏂噒oken闄愬埗
                max_token_for_local_context=16000  # 澶у箙澧炲姞鏈湴涓婁笅鏂噒oken闄愬埗
            ), 
            entity_match=entity_match, 
            hyperedge_match=hyperedge_match
        )
        
        # 娣诲姞缁撴灉璁℃暟璋冭瘯淇℃伅
        if isinstance(result, list):
            logger.info(f"GraphR1 returned {len(result)} results")
        else:
            logger.info(f"GraphR1 returned result type: {type(result)}")
            
        return {"query": query_text, "result": result}
    except Exception as e:
        logger.error(f"鏌ヨ澶勭悊澶辫触: {e}")
        return {"query": query_text, "result": {"error": str(e)}}

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def _format_results(results: List, corpus) -> str:
    results_list = []
    
    for i, result in enumerate(results):
        if result < 0 or result >= len(corpus):
            continue
        results_list.append(corpus[result])
    
    return results_list

def queries_to_results(queries: List[str], entity_top_k: int = 5, hyperedge_top_k: int = 5, rag_top_k: int = 10) -> List[str]:
    """灏嗘煡璇㈣浆鎹负缁撴灉"""
    # 妫€鏌ュ苟閲嶆柊鍔犺浇鏁版嵁
    if not reload_data_if_needed():
        return [json.dumps({"error": "鏁版嵁鍔犺浇澶辫触"}) for _ in queries]
    
    if model is None or index_entity is None or index_hyperedge is None:
        return [json.dumps({"error": "妯″瀷鎴栫储寮曟湭鍔犺浇"}) for _ in queries]
    
    try:
        # 缂栫爜鏌ヨ
        embeddings = model.encode_queries(queries)
        if embeddings.ndim != 2 or embeddings.shape[1] != index_hyperedge.d:
            raise RuntimeError(
                "BGE query embedding dimension does not match hyperedge FAISS index: "
                f"query_dim={embeddings.shape[1] if embeddings.ndim == 2 else None}, index_dim={index_hyperedge.d}"
            )
        
        # 鎼滅储瀹炰綋
        _, entity_ids = index_entity.search(embeddings, entity_top_k)
        entity_match = {queries[i]: _format_results(entity_ids[i], corpus_entity) for i in range(len(entity_ids))}
        
        # 鎼滅储瓒呰竟
        _, hyperedge_ids = index_hyperedge.search(embeddings, hyperedge_top_k)
        hyperedge_match = {queries[i]: _format_results(hyperedge_ids[i], corpus_hyperedge) for i in range(len(hyperedge_ids))}
        hyperedge_match = {
            query_text: _promote_recent_hyperedges(query_text, hyperedge_match[query_text])
            for query_text in queries
        }
        
        # 娣诲姞璋冭瘯淇℃伅
        logger.info(f"Search parameters: entity_top_k={entity_top_k}, hyperedge_top_k={hyperedge_top_k}, rag_top_k={rag_top_k}")
        for i, query in enumerate(queries):
            logger.info(f"Query {i}: '{query}' -> {len(entity_ids[i])} entities, {len(hyperedge_ids[i])} hyperedges")
        
        # 澶勭悊鏌ヨ
        results = []
        loop = always_get_an_event_loop()
        
        for query_text in tqdm(queries, desc="Processing queries", unit="query"):
            if _should_use_rag_results():
                result = loop.run_until_complete(
                    process_query(query_text, rag, entity_match[query_text], hyperedge_match[query_text], rag_top_k)
                )
                merged_results = _merge_fresh_results(result["result"], hyperedge_match[query_text])
                results.append(json.dumps({"results": merged_results}))
            else:
                # 濡傛灉GraphR1涓嶅彲鐢紝杩斿洖鎼滅储缁撴灉
                results.append(json.dumps({
                    "results": _results_from_matches(
                        entity_match[query_text],
                        hyperedge_match[query_text],
                    )
                }))
        
        return results
        
    except Exception as e:
        logger.error(f"鏌ヨ澶勭悊澶辫触: {e}")
        return [json.dumps({"error": str(e)}) for _ in queries]
########### PREDEFINE ############

# 鍒涘缓 FastAPI 瀹炰緥
app = FastAPI(title="Search API with Auto-Reload", description="鏀寔瀹炴椂鏁版嵁閲嶈浇鐨勬枃妗ｆ绱PI")

class SearchRequest(BaseModel):
    queries: List[str]
    entity_top_k: int = 5
    hyperedge_top_k: int = 5
    rag_top_k: int = 10

class ReloadRequest(BaseModel):
    force: bool = False
    load_graph: bool = False

@app.post("/search")
def search(request: SearchRequest):
    """鎼滅储鎺ュ彛"""
    results_str = queries_to_results(
        request.queries, 
        entity_top_k=request.entity_top_k,
        hyperedge_top_k=request.hyperedge_top_k,
        rag_top_k=request.rag_top_k
    )
    return results_str

@app.post("/reload")
def reload_data(request: ReloadRequest):
    """éŽµå¬ªå§©é–²å¶ˆæµ‡éç‰ˆåµéŽºãƒ¥å½›"""
    global last_reload_time, data_changed
    with reload_lock:
        current_max_mtime = _get_max_watched_mtime(working_dir)
        if (
            request.load_graph is False
            and current_max_mtime <= last_reload_time
            and index_hyperedge is not None
            and index_entity is not None
        ):
            data_changed = False
            return {"success": True, "message": "Reload skipped; artifacts already fresh", "reloaded": False}
        if load_data(load_graph=request.load_graph):
            last_reload_time = time.time()
            data_changed = False
            return {"success": True, "message": "Reload completed", "reloaded": True}
        else:
            return {"success": False, "message": "Reload failed"}

@app.get("/status")
def get_status():
    """Return runtime status for the search API."""
    return {
        "working_dir": working_dir,
        "model_loaded": model is not None,
        "entity_index_loaded": index_entity is not None,
        "hyperedge_index_loaded": index_hyperedge is not None,
        "entity_count": len(corpus_entity),
        "hyperedge_count": len(corpus_hyperedge),
        "rag_loaded": rag is not None,
        "rag_graph_stale": rag_graph_stale,
        "last_reload_time": last_reload_time,
        "auto_reload_enabled": args.reload_interval > 0,
        "reload_interval": args.reload_interval,
        "check_interval": args.check_interval,
        "data_changed": data_changed,
        "last_check_time": last_check_time
    }

@app.post("/test_params")
def test_params(request: SearchRequest):
    """Echo the received search parameters for diagnostics."""
    return {
        "received_params": {
            "entity_top_k": request.entity_top_k,
            "hyperedge_top_k": request.hyperedge_top_k,
            "rag_top_k": request.rag_top_k
        },
        "queries": request.queries
    }

@app.get("/health")
def health_check():
    """Health probe endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

def main():
    """Start the local search API service."""
    global last_reload_time, data_changed
    # 鍒濆鍖?
    logger.info("Starting search API service...")
    
    # 鍔犺浇妯″瀷
    if not load_embedding_model():
        logger.error("Failed to load embedding model")
        return
    
    # 鍔犺浇鏁版嵁
    if not load_data(load_graph=True):
        logger.error("Failed to load search artifacts")
        return
    last_reload_time = _get_max_watched_mtime(working_dir)
    data_changed = False
    
    # 鍚姩鑷姩閲嶈浇绾跨▼
    if args.reload_interval > 0:
        reload_thread = threading.Thread(target=auto_reload_worker, daemon=True)
        reload_thread.start()
        logger.info(f"Auto reload enabled: reload_interval={args.reload_interval}s, check_interval={args.check_interval}s")
    else:
        logger.info("Auto reload disabled")
    
    # 鍚姩鏈嶅姟
    logger.info(f"Starting uvicorn server on {args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()

