import asyncio
import json
import re
import os
from datetime import datetime
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union
from collections import Counter, defaultdict
import warnings
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


def _ensure_numeric_weight(weight_value, default_weight: float = 1.0) -> float:
    """
    确保权重值是数值类型，避免类型错误
    
    Args:
        weight_value: 权重值，可能是字符串、数字或其他类型
        default_weight: 当转换失败时使用的默认权重
        
    Returns:
        float: 确保为数值类型的权重
    """
    if isinstance(weight_value, str):
        try:
            return float(weight_value)
        except (ValueError, TypeError):
            return default_weight
    elif isinstance(weight_value, (int, float)):
        return float(weight_value)
    else:
        return default_weight


def _add_ai_metadata_to_hyperedge(hyperedge_data: dict, action: str, tool_name: str, global_config: dict) -> dict:
    """
    为单个超边添加AI元数据标记

    Args:
        hyperedge_data: 超边数据
        action: 操作类型 (insert, update, delete)
        tool_name: 工具名称
        global_config: 全局配置

    Returns:
        dict: 添加元数据后的超边数据
    """
    try:
        # 获取当前时间戳（使用本地时间以保持一致）
        timestamp = datetime.now().isoformat()

        # 获取AI模型信息 - 优先使用环境变量，按优先级依次尝试
        # 1. OPENAI_MODEL (OpenAI)
        # 2. SILICONFLOW_MODEL (SiliconFlow)
        # 3. ZHIPU_MODEL (智谱AI)
        # 4. global_config.llm_model_name
        # 5. AI_MODEL_NAME (通用)
        # 6. 默认值
        ai_model = (
            os.getenv("OPENAI_MODEL") or
            os.getenv("SILICONFLOW_MODEL") or
            os.getenv("ZHIPU_MODEL") or
            global_config.get("llm_model_name") if global_config else None or
            os.getenv("AI_MODEL_NAME") or
            "unknown"
        )

        # 初始化元数据
        if "metadata" not in hyperedge_data:
            hyperedge_data["metadata"] = {}

        metadata = hyperedge_data["metadata"]

        # 保留原有的 metadata 字段，只更新 AI 相关的字段
        # 不覆盖 source（可能原来是 "human" 或其他来源）
        if metadata.get("source") not in ["ai_generated", "ai_modified"]:
            metadata["source"] = "ai_generated"

        metadata["ai_modified"] = True
        metadata["last_ai_action"] = action
        metadata["last_ai_timestamp"] = timestamp
        metadata["ai_model"] = ai_model

        # 初始化修改历史（如果不存在）
        if "modification_history" not in metadata:
            metadata["modification_history"] = []

        # 添加当前修改记录
        modification_record = {
            "timestamp": timestamp,
            "action": action,
            "tool": tool_name,
            "ai_model": ai_model
        }
        metadata["modification_history"].append(modification_record)

        # 只在首次添加时设置这些字段，后续保持原值
        if "verification_status" not in metadata:
            metadata["verification_status"] = "pending"
        if "human_reviewed" not in metadata:
            metadata["human_reviewed"] = False

        logger.info(f"Added AI metadata to hyperedge: {action}")
        return hyperedge_data

    except Exception as e:
        logger.error(f"Error adding AI metadata to hyperedge: {str(e)}")
        return hyperedge_data


def _add_ai_metadata_to_entity(entity_data: dict, action: str, tool_name: str, global_config: dict, entity_info: dict = None) -> dict:
    """
    为单个实体添加AI元数据标记

    Args:
        entity_data: 实体数据
        action: 操作类型 (insert, update, delete)
        tool_name: 工具名称
        global_config: 全局配置
        entity_info: 实体信息字典

    Returns:
        dict: 添加元数据后的实体数据
    """
    try:
        # 获取当前时间戳（使用本地时间以保持一致）
        timestamp = datetime.now().isoformat()

        # 获取AI模型信息 - 优先使用环境变量，按优先级依次尝试
        ai_model = (
            os.getenv("OPENAI_MODEL") or
            os.getenv("SILICONFLOW_MODEL") or
            os.getenv("ZHIPU_MODEL") or
            global_config.get("llm_model_name") if global_config else None or
            os.getenv("AI_MODEL_NAME") or
            "unknown"
        )

        # 初始化元数据
        if "metadata" not in entity_data:
            entity_data["metadata"] = {}

        metadata = entity_data["metadata"]

        # 保留原有的 metadata 字段，只更新 AI 相关的字段
        if metadata.get("source") not in ["ai_generated", "ai_modified"]:
            metadata["source"] = "ai_generated"

        metadata["ai_modified"] = True
        metadata["last_ai_action"] = action
        metadata["last_ai_timestamp"] = timestamp
        metadata["ai_model"] = ai_model

        # 初始化修改历史（如果不存在）
        if "modification_history" not in metadata:
            metadata["modification_history"] = []

        # 添加当前修改记录
        modification_record = {
            "timestamp": timestamp,
            "action": action,
            "tool": tool_name,
            "ai_model": ai_model
        }
        metadata["modification_history"].append(modification_record)

        # 只在首次添加时设置这些字段，后续保持原值
        if "verification_status" not in metadata:
            metadata["verification_status"] = "pending"
        if "human_reviewed" not in metadata:
            metadata["human_reviewed"] = False

        # 添加实体特定信息
        if entity_info:
            metadata["entity_type"] = entity_info.get("entity_type", "")
            metadata["source_id"] = entity_info.get("source_id", "")

        logger.info(f"Added AI metadata to entity: {action}")
        return entity_data

    except Exception as e:
        logger.error(f"Error adding AI metadata to entity: {str(e)}")
        return entity_data


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    now_hyper_relation: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"entity"' or now_hyper_relation == "":
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 50.0
    )
    hyper_relation = now_hyper_relation
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        weight=weight,
        hyper_relation=hyper_relation,
        source_id=entity_source_id,
    )


def _handle_single_hyperrelation_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 3 or record_attributes[0] != '"hyper-relation"':
        return None
    # add this record as edge
    knowledge_fragment = clean_str(record_attributes[1])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        hyper_relation="<hyperedge>"+knowledge_fragment,
        weight=weight,
        source_id=edge_source_id,
    )
    

async def _merge_hyperedges_then_upsert(
    hyperedge_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []

    already_hyperedge = await knowledge_graph_inst.get_node(hyperedge_name)
    if already_hyperedge is not None:
        # 确保权重值是数值类型，避免类型错误
        existing_weight = already_hyperedge["weight"]
        if isinstance(existing_weight, str):
            try:
                existing_weight = float(existing_weight)
            except (ValueError, TypeError):
                existing_weight = 1.0  # 默认权重
        elif not isinstance(existing_weight, (int, float)):
            existing_weight = 1.0  # 默认权重
        already_weights.append(existing_weight)
        already_source_ids.extend(
            split_string_by_multi_markers(already_hyperedge["source_id"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in nodes_data] + already_weights)
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    node_data = dict(
        role = "hyperedge",
        weight=weight,
        source_id=source_id,
    )
    
    # 添加 AI metadata
    node_data = _add_ai_metadata_to_hyperedge(node_data, "update", "merge_hyperedges", global_config)
    
    await knowledge_graph_inst.upsert_node(
        hyperedge_name,
        node_data=node_data,
    )
    node_data["hyperedge_name"] = hyperedge_name
    return node_data


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        role="entity",
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    
    # 添加 AI metadata
    entity_info = {
        "entity_type": entity_type,
        "source_id": source_id
    }
    node_data = _add_ai_metadata_to_entity(node_data, "update", "merge_nodes", global_config, entity_info)
    
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    edge_data_list = []
    
    for node in nodes_data:
        source_id = node["source_id"]
        hyper_relation = node["hyper_relation"]
        weight = node["weight"]
        
        already_weights = []
        already_source_ids = []
        
        if await knowledge_graph_inst.has_edge(hyper_relation, entity_name):
            already_edge = await knowledge_graph_inst.get_edge(hyper_relation, entity_name)
            # 确保权重值是数值类型，避免类型错误
            existing_weight = already_edge["weight"]
            if isinstance(existing_weight, str):
                try:
                    existing_weight = float(existing_weight)
                except (ValueError, TypeError):
                    existing_weight = 1.0  # 默认权重
            elif not isinstance(existing_weight, (int, float)):
                existing_weight = 1.0  # 默认权重
            already_weights.append(existing_weight)
            already_source_ids.extend(
                split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
            )
        
        weight = sum([weight] + already_weights)
        source_id = GRAPH_FIELD_SEP.join(
            set([source_id] + already_source_ids)
        )

        # 添加 AI metadata
        edge_data = dict(
            weight=weight,
            source_id=source_id,
        )
        edge_info = {
            "src_id": hyper_relation,
            "tgt_id": entity_name,
            "description": f"hyperedge_relation_to_{entity_name}"
        }
        edge_data = _add_ai_metadata_to_hyperedge(edge_data, "update", "merge_edges", global_config)

        await knowledge_graph_inst.upsert_edge(
            hyper_relation,
            entity_name,
            edge_data=edge_data,
        )

        edge_data_list.append(dict(
            src_id=hyper_relation,
            tgt_id=entity_name,
            weight=weight,
        ))

    return edge_data_list


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    hyperedge_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        # entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)

        final_result = await use_llm_func(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        now_hyper_relation=""
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_relation = _handle_single_hyperrelation_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[if_relation["hyper_relation"]].append(
                    if_relation
                )
                now_hyper_relation = if_relation["hyper_relation"]

            if_entities = _handle_single_entity_extraction(
                record_attributes, chunk_key, now_hyper_relation
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue
            
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        try:
            print(
                f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
                end="",
                flush=True,
            )
        except UnicodeEncodeError:
            pass
        return dict(maybe_nodes), dict(maybe_edges)

    results = []
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    ):
        results.append(await result)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)
            
    logger.info("Inserting hyperedges into storage...")
    all_hyperedges_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_hyperedges_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_edges.items()
            ]
        ),
        total=len(maybe_edges),
        desc="Inserting hyperedges",
        unit="entity",
    ):
        all_hyperedges_data.append(await result)
            
    logger.info("Inserting entities into storage...")
    all_entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Inserting entities",
        unit="entity",
    ):
        all_entities_data.append(await result)

    logger.info("Inserting relationships into storage...")
    all_relationships_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_edges_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Inserting relationships",
        unit="relationship",
    ):
        all_relationships_data.append(await result)

    if not len(all_hyperedges_data) and not len(all_entities_data) and not len(all_relationships_data):
        logger.warning(
            "Didn't extract any hyperedges and entities, maybe your LLM is not working"
        )
        return None

    if not len(all_hyperedges_data):
        logger.warning("Didn't extract any hyperedges")
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    if hyperedge_vdb is not None:
        # 添加 AI metadata 到超边数据
        data_for_vdb = {}
        for dp in all_hyperedges_data:
            hyperedge_id = compute_mdhash_id(dp["hyperedge_name"], prefix="rel-")
            
            # 创建基础数据结构
            hyperedge_data = {
                "content": dp["hyperedge_name"],
                "hyperedge_name": dp["hyperedge_name"],
            }
            
            # 添加 AI metadata
            hyperedge_data = _add_ai_metadata_to_hyperedge(hyperedge_data, "insert", "extract_entities", global_config)
            
            data_for_vdb[hyperedge_id] = hyperedge_data
        
        await hyperedge_vdb.upsert(data_for_vdb)

    if entity_vdb is not None:
        # 添加 AI metadata 到实体数据
        data_for_vdb = {}
        for dp in all_entities_data:
            entity_id = compute_mdhash_id(dp["entity_name"], prefix="ent-")
            
            # 创建基础数据结构
            entity_data = {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            
            # 添加 AI metadata
            entity_data = _add_ai_metadata_to_entity(entity_data, "insert", "extract_entities", global_config, dp)
            
            data_for_vdb[entity_id] = entity_data
        
        await entity_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: list,
    hyperedges_vdb: list,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:
    
    hl_keywords = query
    ll_keywords = query
    keywords = [ll_keywords, hl_keywords]
    context = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        hyperedges_vdb,
        text_chunks_db,
        query_param,
    )
    
    return context



async def _build_query_context(
    query: list,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):

    ll_kewwords, hl_keywrds = query[0], query[1]

    knowledge_list_1 = await _get_node_data(
        ll_kewwords,
        knowledge_graph_inst,
        entities_vdb,
        text_chunks_db,
        query_param,
    )

    knowledge_list_2 = await _get_edge_data(
        hl_keywrds,
        knowledge_graph_inst,
        hyperedges_vdb,
        text_chunks_db,
        query_param,
    )
    
    know_score = dict()
    for i, k in enumerate(knowledge_list_1):
        if k not in know_score:
            know_score[k] = 0
        score = 1/(i+1)
        know_score[k] += score
    for i, k in enumerate(knowledge_list_2):
        if k not in know_score:
            know_score[k] = 0
        score = 1/(i+1)
        know_score[k] += score
    knowledge_list = sorted(know_score.items(), key=lambda x: x[1], reverse=True)[:query_param.top_k]
    knowledge=[]
    for k in knowledge_list:
        knowledge.append({"<knowledge>": k[0], "<coherence>": round(k[1],3)})
    return knowledge


async def _get_node_data(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):  
    results = entities_vdb
    if not len(results):
        return "", "", ""
    # get entity information
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    # get entity degree
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    knowledge_list = [s["description"].replace("<hyperedge>","") for s in use_relations]
    # s_ids = []
    # for r in use_relations:
    #     s_ids.extend(r["source_id"].split(GRAPH_FIELD_SEP))
    # knowledge_list = [(await text_chunks_db.get_by_id(s))["content"] for s in s_ids]
    return knowledge_list


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                    "relation_counts": 0,
                }

            if this_edges:
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(e)
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, "description": k[1], **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    return all_edges_data


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):  
    results = hyperedges_vdb

    if not len(results):
        return "", "", ""

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    # edge_degree = await asyncio.gather(
    #     *[knowledge_graph_inst.node_degree(r["hyperedge_name"]) for r in results]
    # )
    edge_datas = [
        {"hyperedge": k, "rank": v["weight"], **v}
        for k, v in zip(results, edge_datas)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    knowledge_list = [s["hyperedge"].replace("<hyperedge>","") for s in edge_datas]
    # s_ids = []
    # for r in edge_datas:
    #     s_ids.extend(r["source_id"].split(GRAPH_FIELD_SEP))
    # knowledge_list = [(await text_chunks_db.get_by_id(s))["content"] for s in s_ids]
    return knowledge_list


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(edge["hyperedge"]) for edge in edge_datas]
    )
    
    entity_names = []
    seen = set()

    for node_data in node_datas:
        for e in node_data:
            if e[1] not in seen:
                entity_names.append(e[1])
                seen.add(e[1])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                chunk_data = await text_chunks_db.get_by_id(c_id)
                # Only store valid data
                if chunk_data is not None and "content" in chunk_data:
                    all_text_units_lookup[c_id] = {
                        "data": chunk_data,
                        "order": index,
                    }

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units


def combine_contexts(entities, relationships, sources):
    # Function to extract entities, relationships, and sources from context strings
    hl_entities, ll_entities = entities[0], entities[1]
    hl_relationships, ll_relationships = relationships[0], relationships[1]
    # Combine and deduplicate the entities
    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )

    return combined_entities, combined_relationships, ""