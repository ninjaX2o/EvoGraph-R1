import os

import pytest

from evograph_mm.device_env import (
    RUNTIME_DEVICE_ENV,
    configure_embedding_device,
    runtime_embedding_device,
)


def test_cpu_embedding_device_hides_cuda(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert configure_embedding_device("cpu") == "cpu"
    assert runtime_embedding_device() == "cpu"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_indexed_cuda_device_is_remapped(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert configure_embedding_device("cuda:3") == "cuda"
    assert runtime_embedding_device() == "cuda"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_auto_device_preserves_existing_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")

    assert configure_embedding_device("auto") == "auto"
    assert runtime_embedding_device() is None
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2"


def test_invalid_embedding_device_is_rejected(monkeypatch):
    monkeypatch.delenv(RUNTIME_DEVICE_ENV, raising=False)

    with pytest.raises(ValueError, match="MM_EMBED_DEVICE"):
        configure_embedding_device("gpu")
