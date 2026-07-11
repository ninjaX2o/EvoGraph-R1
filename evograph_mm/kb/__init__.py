"""Unified boundary for multimodal KB lifecycle work.

Multimodal KB build, indexing, retrieval, and editing code belongs in this
package.
"""

from importlib import import_module

from .prepare_echosight import (
    DATASET_E_VQA,
    DATASET_INFOSEEK,
    MANIFEST_VERSION,
    NO_DOWNLOAD_POLICY,
    SUPPORTED_DATASETS,
    AssetDefinition,
    AssetStatus,
    MissingAsset,
    MissingRequiredAssetsError,
    ValidationReport,
    asset_catalog,
    checksum_plan_document,
    local_raw_asset_path,
    manifest_document,
    raw_dataset_root,
    validate_required_assets,
    write_planning_files,
)
from .audit_echosight import (
    AUDIT_VERSION,
    FIELD_AUDIT_REPORT_PATH,
    audit_echosight_assets,
)


_LAZY_EXPORTS = {
    "IMPLEMENTATION_RULES": (".boundary", "IMPLEMENTATION_RULES"),
    "KB_BOUNDARY_DOCUMENT": (".boundary", "KB_BOUNDARY_DOCUMENT"),
    "KB_BOUNDARY_PATH": (".boundary", "KB_BOUNDARY_PATH"),
    "KB_OWNED_CAPABILITIES": (".boundary", "KB_OWNED_CAPABILITIES"),
    "VALIDATION_VERSION": (".validate_echosight", "VALIDATION_VERSION"),
    "validate_echosight_full_assets": (
        ".validate_echosight",
        "validate_echosight_full_assets",
    ),
    "MM_SCHEMA_VERSION": (".mm_schema", "MM_SCHEMA_VERSION"),
    "REQUIRED_RL_FIELDS": (".mm_schema", "REQUIRED_RL_FIELDS"),
    "MultimodalSchemaError": (".mm_schema", "MultimodalSchemaError"),
    "build_multimodal_qa_record": (".mm_schema", "build_multimodal_qa_record"),
    "validate_mm_record": (".mm_schema", "validate_mm_record"),
    "SYNTHETIC_DATASET": (".datasets", "SYNTHETIC_DATASET"),
    "evaluate_echosight_readiness": (".datasets", "evaluate_echosight_readiness"),
    "iter_synthetic_records": (".datasets", "iter_synthetic_records"),
    "process_echosight_dataset": (".datasets", "process_echosight_dataset"),
    "process_synthetic_dataset": (".datasets", "process_synthetic_dataset"),
    "build_arg_parser": (".build", "build_arg_parser"),
    "main": (".build", "main"),
    "run_build": (".build", "run_build"),
}


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "DATASET_E_VQA",
    "DATASET_INFOSEEK",
    "MANIFEST_VERSION",
    "NO_DOWNLOAD_POLICY",
    "SUPPORTED_DATASETS",
    "AssetDefinition",
    "AssetStatus",
    "MissingAsset",
    "MissingRequiredAssetsError",
    "ValidationReport",
    "asset_catalog",
    "checksum_plan_document",
    "local_raw_asset_path",
    "manifest_document",
    "raw_dataset_root",
    "validate_required_assets",
    "write_planning_files",
    "AUDIT_VERSION",
    "FIELD_AUDIT_REPORT_PATH",
    "audit_echosight_assets",
]
