"""Configure multimodal embedding visibility before model libraries load."""

from __future__ import annotations

import os
import re


RUNTIME_DEVICE_ENV = "EVOGRAPH_MM_EMBED_RUNTIME_DEVICE"


def configure_embedding_device(requested: str | None = None) -> str:
    value = (requested or os.getenv("MM_EMBED_DEVICE", "auto")).strip().lower()
    if value in {"", "auto"}:
        runtime_device = "auto"
    elif value == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        runtime_device = "cpu"
    elif value == "cuda":
        runtime_device = "cuda"
    else:
        match = re.fullmatch(r"cuda:(\d+)", value)
        if match is None:
            raise ValueError(
                "MM_EMBED_DEVICE must be auto, cpu, cuda, or cuda:<index>"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = match.group(1)
        runtime_device = "cuda"

    os.environ[RUNTIME_DEVICE_ENV] = runtime_device
    return runtime_device


def runtime_embedding_device() -> str | None:
    value = os.getenv(RUNTIME_DEVICE_ENV) or os.getenv("MM_EMBED_DEVICE", "auto")
    normalized = value.strip().lower()
    return None if normalized in {"", "auto"} else normalized


__all__ = [
    "RUNTIME_DEVICE_ENV",
    "configure_embedding_device",
    "runtime_embedding_device",
]
