"""Lightweight exports for multimodal KB indexing helpers."""

from .encoders import (
    DEFAULT_EMBEDDING_DIMENSION,
    FUSED_DOCUMENT_INSTRUCTION,
    GME_MODEL_REPO_ID,
    GME_IMAGE_MAX_PIXELS,
    GMEQwen2VLEncoder,
    IMAGE_DOCUMENT_INSTRUCTION,
    QWEN3_VL_MODEL_REPO_ID,
    TEXT_DOCUMENT_INSTRUCTION,
    TEXT_QUERY_INSTRUCTION,
    LocalModelUnavailable,
    MockEmbeddingEncoder,
    Qwen3VLEncoder,
    create_embedding_encoder,
    validate_embedding_dimension,
)
from .faiss_store import write_vector_index


__all__ = [
    "QWEN3_VL_MODEL_REPO_ID",
    "GME_MODEL_REPO_ID",
    "GME_IMAGE_MAX_PIXELS",
    "DEFAULT_EMBEDDING_DIMENSION",
    "TEXT_QUERY_INSTRUCTION",
    "TEXT_DOCUMENT_INSTRUCTION",
    "IMAGE_DOCUMENT_INSTRUCTION",
    "FUSED_DOCUMENT_INSTRUCTION",
    "LocalModelUnavailable",
    "validate_embedding_dimension",
    "MockEmbeddingEncoder",
    "GMEQwen2VLEncoder",
    "Qwen3VLEncoder",
    "create_embedding_encoder",
    "write_vector_index",
]
