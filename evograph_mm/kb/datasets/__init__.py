"""Dataset adapters for metadata-only multimodal preprocessing scaffolding."""

from .echosight import evaluate_echosight_readiness, process_echosight_dataset
from .synthetic import SYNTHETIC_DATASET, iter_synthetic_records, process_synthetic_dataset

__all__ = [
    "SYNTHETIC_DATASET",
    "iter_synthetic_records",
    "process_synthetic_dataset",
    "evaluate_echosight_readiness",
    "process_echosight_dataset",
]
