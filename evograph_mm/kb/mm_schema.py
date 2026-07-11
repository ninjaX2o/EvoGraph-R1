"""Compatibility imports for the metadata-only multimodal schema."""

from .schema import (
    MM_REQUIRED_EXTRA_INFO_FIELDS,
    MM_REQUIRED_TOP_LEVEL_FIELDS,
    MM_SCHEMA_VERSION,
    MultimodalSchemaError,
    build_multimodal_qa_record,
    parse_e_vqa_answers,
    validate_mm_record,
)

REQUIRED_RL_FIELDS = MM_REQUIRED_TOP_LEVEL_FIELDS
REQUIRED_EXTRA_INFO_FIELDS = MM_REQUIRED_EXTRA_INFO_FIELDS

__all__ = [
    "MM_REQUIRED_EXTRA_INFO_FIELDS",
    "MM_REQUIRED_TOP_LEVEL_FIELDS",
    "MM_SCHEMA_VERSION",
    "REQUIRED_EXTRA_INFO_FIELDS",
    "REQUIRED_RL_FIELDS",
    "MultimodalSchemaError",
    "build_multimodal_qa_record",
    "parse_e_vqa_answers",
    "validate_mm_record",
]
