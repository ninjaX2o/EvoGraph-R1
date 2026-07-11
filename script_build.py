import os
import json
import time
from graphr1 import GraphR1
import argparse
import numpy as np
from FlagEmbedding import FlagModel
import faiss
from dotenv import load_dotenv
import torch
from agent.tool.tools.hyperedge_index_sync import (
    iter_active_hyperedge_contents,
    write_hyperedge_index_metadata,
)
from agent.tool.tools.entity_index_sync import write_entity_index_metadata

# 加载环境变量
load_dotenv()

# 设置 OpenAI API 配置
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

if base_url:
    os.environ["OPENAI_BASE_URL"] = base_url

def extract_knowledge(rag, unique_contexts):
    print(f"Total insert rounds: {len(unique_contexts)//50 + 1}")
    for i in range(0, len(unique_contexts), 50):
        print(f"This is the {i//50 + 1} round of insertion, remain rounds: {len(unique_contexts)//50 - i//50}")
        retries = 0
        max_retries = 50
        while retries < max_retries:
            try:
                rag.insert(unique_contexts[i:i+50])
                break
            except Exception as e:
                retries += 1
                print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
                time.sleep(10)
        if retries == max_retries:
            print("Insertion failed after exceeding the maximum number of retries")
    
    retries = 0
    max_retries = 50
    while retries < max_retries:
        try:
            rag.insert(unique_contexts)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")
        
def embed_knowledge(data_source):
    print("Loading FlagEmbedding model for vectorization...")
    model = FlagModel(
        'BAAI/bge-large-en-v1.5',
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
        use_fp16=True,
    )
    print("FlagEmbedding model loaded successfully")

    print("Loading text chunks...")
    corpus = []
    with open(f"expr/{data_source}/kv_store_text_chunks.json", encoding='utf-8') as f:
        texts = json.load(f)
        for item in texts:
            corpus.append(texts[item]['content'])

    print("Loading entity descriptions...")
    corpus_entity = []
    corpus_entity_des = []
    with open(f"expr/{data_source}/kv_store_entities.json", encoding='utf-8') as f:
        entities = json.load(f)
        for item in entities:
            corpus_entity.append(entities[item]['entity_name'])
            corpus_entity_des.append(entities[item]['content'])
            
    print("Loading hyperedges...")
    corpus_hyperedge = []
    with open(f"expr/{data_source}/kv_store_hyperedges.json", encoding='utf-8') as f:
        hyperedges = json.load(f)
        corpus_hyperedge.extend(iter_active_hyperedge_contents(hyperedges))

    print(f"Processing {len(corpus)} text chunks with FlagEmbedding...")
    embeddings = model.encode_corpus(corpus, batch_size=512, max_length=256)
    print(f"Text chunks embeddings shape: {embeddings.shape}")
    #save
    np.save(f"expr/{data_source}/corpus.npy", embeddings)

    corpus_numpy = np.load(f"expr/{data_source}/corpus.npy")
    dim = corpus_numpy.shape[-1]
    print(f"Text chunks embedding dimension: {dim}")

    corpus_numpy = corpus_numpy.astype(np.float32)

    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    index.add(corpus_numpy)
    faiss.write_index(index, f"expr/{data_source}/index.bin")
    print("Text chunks FAISS index saved")

    print(f"Processing {len(corpus_entity_des)} entity descriptions...")
    embeddings = model.encode_corpus(corpus_entity_des, batch_size=512, max_length=256)
    print(f"Entity embeddings shape: {embeddings.shape}")
    #save
    np.save(f"expr/{data_source}/corpus_entity.npy", embeddings)

    corpus_numpy = np.load(f"expr/{data_source}/corpus_entity.npy")
    dim = corpus_numpy.shape[-1]
    print(f"Entity embedding dimension: {dim}")

    corpus_numpy = corpus_numpy.astype(np.float32)

    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    index.add(corpus_numpy)
    faiss.write_index(index, f"expr/{data_source}/index_entity.bin")
    write_entity_index_metadata(
        f"expr/{data_source}",
        provider="bge",
        dimension=dim,
        model="BAAI/bge-large-en-v1.5",
        entity_count=len(corpus_entity_des),
    )
    print("Entity FAISS index saved")

    print(f"Processing {len(corpus_hyperedge)} hyperedges...")
    embeddings = model.encode_corpus(corpus_hyperedge, batch_size=512, max_length=256)
    print(f"Hyperedge embeddings shape: {embeddings.shape}")
    #save
    np.save(f"expr/{data_source}/corpus_hyperedge.npy", embeddings)

    corpus_numpy = np.load(f"expr/{data_source}/corpus_hyperedge.npy")
    dim = corpus_numpy.shape[-1]
    print(f"Hyperedge embedding dimension: {dim}")

    corpus_numpy = corpus_numpy.astype(np.float32)

    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    index.add(corpus_numpy)
    faiss.write_index(index, f"expr/{data_source}/index_hyperedge.bin")
    write_hyperedge_index_metadata(
        f"expr/{data_source}",
        provider="bge",
        dimension=dim,
        model="BAAI/bge-large-en-v1.5",
        active_count=len(corpus_hyperedge),
    )
    print("Hyperedge FAISS index saved")

def insert_knowledge(data_source, unique_contexts):
    print(f"Initializing GraphR1 with OpenAI API for LLM and FlagEmbedding for vectors...")
    print(f"Working directory: expr/{data_source}")
    print("OpenAI API key: configured")
    if base_url:
        print(f"OpenAI Base URL: {base_url}")
    
    # 创建 GraphR1 实例
    # GraphR1 会使用环境变量中的 OpenAI 配置进行 LLM 处理
    # 但向量化会使用 FlagEmbedding（在 embed_knowledge 函数中）
    rag = GraphR1(
        working_dir=f"expr/{data_source}"   
    )    
    
    print("Extracting knowledge and building knowledge graph using OpenAI API...")
    extract_knowledge(rag, unique_contexts)
    
    print("Building vector embeddings using FlagEmbedding...")
    embed_knowledge(data_source)
    
    print(f"Knowledge successfully inserted and embedded for {data_source}")
    print("Note: LLM processing used OpenAI API, vectorization used FlagEmbedding")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", type=str, default="2WikiMultiHopQA")
    args = parser.parse_args()
    data_source = args.data_source
    
    # Validate configuration after argument parsing so --help always works.
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set in the environment or .env")
    
    print("=" * 60)
    print("Configuration Summary:")
    print("=" * 60)
    print(f"Data Source: {data_source}")
    print("OpenAI API key: configured")
    if base_url:
        print(f"OpenAI Base URL: {base_url}")
    else:
        print("OpenAI Base URL: Using default endpoint")
    print("Vectorization: FlagEmbedding (BAAI/bge-large-en-v1.5)")
    print("LLM Processing: OpenAI API")
    print("=" * 60)
    
    unique_contexts = []
    with open(f"datasets/{data_source}/corpus.jsonl", encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            unique_contexts.append(data["contents"])
    
    print(f"Loaded {len(unique_contexts)} documents from {data_source}")
    
    insert_knowledge(data_source, unique_contexts)
    
    


