"""Embedding encoder abstractions for multimodal KB indexing."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evograph_mm.device_env import runtime_embedding_device


QWEN3_VL_MODEL_REPO_ID = "Qwen/Qwen3-VL-Embedding-2B"
GME_MODEL_REPO_ID = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
DEFAULT_EMBEDDING_DIMENSION = 1536
MAX_EMBEDDING_DIMENSION = 2048
GME_IMAGE_FACTOR = 28
GME_IMAGE_MAX_TOKENS = 1280
GME_IMAGE_MAX_PIXELS = GME_IMAGE_MAX_TOKENS * GME_IMAGE_FACTOR * GME_IMAGE_FACTOR
TEXT_QUERY_INSTRUCTION = "Retrieve multimodal evidence relevant to the question."
TEXT_DOCUMENT_INSTRUCTION = "Represent this evidence passage for multimodal retrieval."
IMAGE_DOCUMENT_INSTRUCTION = "Represent this image for multimodal retrieval."
FUSED_DOCUMENT_INSTRUCTION = "Represent this image and text pair for multimodal retrieval."


class LocalModelUnavailable(RuntimeError):
    """Raised when the requested local embedding model cannot be loaded."""


def validate_embedding_dimension(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("embedding dimension must be an integer")
    if not 1 <= value <= MAX_EMBEDDING_DIMENSION:
        raise ValueError(
            f"embedding dimension must be between 1 and {MAX_EMBEDDING_DIMENSION}"
        )
    return value


class MockEmbeddingEncoder:
    """Deterministic normalized encoder for tests and offline wiring checks."""

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIMENSION):
        self.dimension = validate_embedding_dimension(dimension)

    def encode_texts(
        self,
        texts: Iterable[Any],
        instruction: str | None = None,
    ) -> np.ndarray:
        return self._encode(
            "text",
            texts,
            TEXT_DOCUMENT_INSTRUCTION if instruction is None else instruction,
        )

    def encode_images(
        self,
        images: Iterable[Any],
        instruction: str | None = None,
    ) -> np.ndarray:
        return self._encode(
            "image",
            images,
            IMAGE_DOCUMENT_INSTRUCTION if instruction is None else instruction,
        )

    def encode_fused(
        self,
        items: Iterable[Any],
        instruction: str | None = None,
    ) -> np.ndarray:
        return self._encode(
            "fused",
            items,
            FUSED_DOCUMENT_INSTRUCTION if instruction is None else instruction,
        )

    def _encode(self, mode: str, items: Iterable[Any], instruction: str) -> np.ndarray:
        vectors = [self._vector_for(mode, instruction, item) for item in items]
        if not vectors:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32, copy=False)

    def _vector_for(self, mode: str, instruction: str, item: Any) -> np.ndarray:
        seed_material = f"{mode}\0{instruction}\0{_stable_item_key(item)}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.dimension).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return (vector / norm).astype(np.float32, copy=False)


class Qwen3VLEncoder:
    """Local-only Qwen3-VL embedding encoder wrapper."""

    def __init__(
        self,
        model_path: str | Path,
        embedding_dim: int = DEFAULT_EMBEDDING_DIMENSION,
        batch_size: int = 1,
    ):
        self.model_path = Path(model_path)
        self.embedding_dim = validate_embedding_dimension(embedding_dim)
        self.batch_size = _validate_batch_size(batch_size)
        self.device = runtime_embedding_device()
        if not self.model_path.exists():
            raise LocalModelUnavailable(
                "Local Qwen3-VL embedding model path is missing: "
                f"{self.model_path}. Expected a local checkout of "
                f"{QWEN3_VL_MODEL_REPO_ID}; loading is local_files_only=True. "
                "No download or Hugging Face access is attempted by this loader."
            )

        try:
            embedder_class = _load_local_qwen3_vl_embedder_class(self.model_path)
            self.model = embedder_class(
                model_name_or_path=str(self.model_path),
                local_files_only=True,
                **_qwen_model_load_kwargs(),
            )
        except Exception as exc:
            raise LocalModelUnavailable(
                "Failed to load local Qwen3-VL embedding model from "
                f"{self.model_path}: {exc}. Expected a local checkout of "
                f"{QWEN3_VL_MODEL_REPO_ID}; loading is local_files_only=True. "
                "No download or Hugging Face access is attempted by this loader."
            ) from exc

    def encode_texts(
        self,
        texts: Iterable[Any],
        instruction: str | None = None,
    ) -> np.ndarray:
        payloads = (
            {"text": str(text), "instruction": instruction}
            for text in texts
        )
        return self._encode_payloads(
            payloads,
            TEXT_DOCUMENT_INSTRUCTION if instruction is None else instruction,
        )

    def encode_images(
        self,
        images: Iterable[Any],
        instruction: str | None = None,
    ) -> np.ndarray:
        payloads = (
            {"image": str(image), "instruction": instruction}
            for image in images
        )
        return self._encode_payloads(
            payloads,
            IMAGE_DOCUMENT_INSTRUCTION if instruction is None else instruction,
        )

    def encode_fused(
        self,
        items: Iterable[Any],
        instruction: str | None = None,
    ) -> np.ndarray:
        payloads = (_fused_payload(item, instruction) for item in items)
        return self._encode_payloads(
            payloads,
            FUSED_DOCUMENT_INSTRUCTION if instruction is None else instruction,
        )

    def _encode_payloads(self, payloads: Iterable[dict[str, Any]], instruction: str) -> np.ndarray:
        prepared = []
        for payload in payloads:
            prepared_payload = dict(payload)
            prepared_payload["instruction"] = (
                instruction
                if prepared_payload.get("instruction") is None
                else str(prepared_payload["instruction"])
            )
            prepared.append(prepared_payload)

        if not prepared:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        batches = []
        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            batches.append(
                _truncate_and_normalize(
                    _as_2d_float32(self.model.process(batch)),
                    self.embedding_dim,
                )
            )
        return np.vstack(batches).astype(np.float32, copy=False)


class GMEQwen2VLEncoder:
    """Local-only GME Qwen2-VL embedding encoder wrapper."""

    def __init__(
        self,
        model_path: str | Path,
        embedding_dim: int = DEFAULT_EMBEDDING_DIMENSION,
        batch_size: int = 1,
    ):
        self.model_path = Path(model_path)
        self.embedding_dim = validate_embedding_dimension(embedding_dim)
        self.batch_size = _validate_batch_size(batch_size)
        self.device = runtime_embedding_device()
        if not self.model_path.exists():
            raise LocalModelUnavailable(
                "Local GME Qwen2-VL embedding model path is missing: "
                f"{self.model_path}. Expected a local checkout of "
                f"{GME_MODEL_REPO_ID}; loading is local_files_only=True. "
                "No download or Hugging Face access is attempted by this loader."
            )
        try:
            from sentence_transformers import SentenceTransformer

            model_kwargs = {
                "trust_remote_code": True,
                "local_files_only": True,
            }
            if self.device is not None:
                model_kwargs["device"] = self.device
            self.model = SentenceTransformer(str(self.model_path), **model_kwargs)
            self._backend = "sentence-transformers"
        except Exception as exc:
            try:
                self.model = _load_gme_transformers_model(
                    self.model_path,
                    device=self.device,
                )
                self._backend = "transformers"
            except Exception as fallback_exc:
                raise LocalModelUnavailable(
                    "Failed to load local GME Qwen2-VL embedding model from "
                    f"{self.model_path}: {fallback_exc}. Expected a local checkout of "
                    f"{GME_MODEL_REPO_ID}; loading is local_files_only=True. "
                    "No download or Hugging Face access is attempted by this loader."
                ) from exc

    def encode_texts(
        self,
        texts: Iterable[Any],
        instruction: str | None = None,
    ) -> np.ndarray:
        values = [str(text) for text in texts]
        if self._backend == "sentence-transformers":
            if instruction:
                payloads: list[Any] = [
                    {"text": value, "prompt": instruction} for value in values
                ]
            else:
                payloads = values
            return self._encode_sentence_transformers(payloads)
        return self._encode_transformers_text(values, instruction=instruction)

    def encode_images(
        self,
        images: Iterable[Any],
        instruction: str | None = None,
    ) -> np.ndarray:
        values = [str(image) for image in images]
        prepared_images = _prepare_gme_images(values)
        if self._backend == "sentence-transformers":
            return self._encode_sentence_transformers(
                [{"image": image} for image in prepared_images]
            )
        return self._encode_transformers_images(prepared_images)

    def encode_fused(
        self,
        items: Iterable[Any],
        instruction: str | None = None,
    ) -> np.ndarray:
        payloads = [_fused_payload(item, instruction) for item in items]
        if self._backend == "sentence-transformers":
            return self._encode_sentence_transformers(
                [
                    {"text": item["text"], "image": item["image"]}
                    for item in payloads
                ]
            )
        return self._encode_transformers_fused(
            [item["text"] for item in payloads],
            [item["image"] for item in payloads],
        )

    def _encode_sentence_transformers(self, payloads: list[Any]) -> np.ndarray:
        if not payloads:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        output = self.model.encode(
            payloads,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _truncate_and_normalize(_as_2d_float32(output), self.embedding_dim)

    def _encode_transformers_text(
        self,
        texts: list[str],
        instruction: str | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        output = self.model.get_text_embeddings(texts=texts, instruction=instruction)
        return _truncate_and_normalize(_as_2d_float32(output), self.embedding_dim)

    def _encode_transformers_images(self, images: list[Any]) -> np.ndarray:
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        images = _prepare_gme_images(images)
        _empty_cuda_cache()
        output = self.model.get_image_embeddings(
            images=_gme_image_loader(images, batch_size=getattr(self, "batch_size", 1)),
            is_query=False,
        )
        return _truncate_and_normalize(_as_2d_float32(output), self.embedding_dim)

    def _encode_transformers_fused(
        self,
        texts: list[str],
        images: list[str],
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        output = self.model.get_fused_embeddings(texts=texts, images=images)
        return _truncate_and_normalize(_as_2d_float32(output), self.embedding_dim)


def create_embedding_encoder(
    backend: str,
    model_path: str | Path | None,
    **kwargs: Any,
):
    embedding_dim = kwargs.pop(
        "embedding_dimension",
        kwargs.pop("embedding_dim", DEFAULT_EMBEDDING_DIMENSION),
    )
    batch_size = kwargs.pop("batch_size", 1)
    normalized = (backend or "gme").lower().replace("_", "-")
    if normalized == "mock":
        return MockEmbeddingEncoder(dimension=embedding_dim)
    if normalized in {"gme", "gme-qwen2-vl"}:
        return GMEQwen2VLEncoder(
            model_path=model_path or "",
            embedding_dim=embedding_dim,
            batch_size=batch_size,
        )
    if normalized == "qwen3-vl":
        return Qwen3VLEncoder(
            model_path=model_path or "",
            embedding_dim=embedding_dim,
            batch_size=batch_size,
        )
    raise ValueError(f"Unsupported embedding backend for Task D/E/F: {backend}")


def _load_gme_transformers_model(model_path: Path, device: str | None = None):
    from transformers import AutoModel

    return AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        **_gme_model_load_kwargs(device),
    )


def _load_local_qwen3_vl_embedder_class(model_path: Path):
    script_path = model_path / "scripts" / "qwen3_vl_embedding.py"
    if not script_path.is_file():
        raise LocalModelUnavailable(
            "Local Qwen3-VL embedding helper is missing: "
            f"{script_path}. Expected a local checkout of {QWEN3_VL_MODEL_REPO_ID}."
        )
    spec = importlib.util.spec_from_file_location(
        "evograph_mm_local_qwen3_vl_embedding",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise LocalModelUnavailable(
            f"Failed to import local Qwen3-VL embedding helper from {script_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.Qwen3VLEmbedder
    except AttributeError as exc:
        raise LocalModelUnavailable(
            f"Local Qwen3-VL embedding helper does not define Qwen3VLEmbedder: {script_path}"
        ) from exc


def _validate_batch_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("batch size must be a positive integer")
    return value


def _qwen_model_load_kwargs() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    return {"dtype": torch.float16}


def _gme_model_load_kwargs(device: str | None = None) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {}
    if device == "cpu":
        return {"device_map": "cpu"}
    if device is not None and device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise LocalModelUnavailable(
                f"MM_EMBED_DEVICE={device} requested CUDA, but CUDA is unavailable"
            )
        return {"torch_dtype": torch.float16, "device_map": device}
    if torch.cuda.is_available():
        return {"torch_dtype": torch.float16, "device_map": "cuda"}
    return {}


def _prepare_gme_images(images: Iterable[Any]) -> list[Any]:
    return [_prepare_gme_image(image) for image in images]


def _gme_image_loader(images: list[Any], *, batch_size: int):
    from torch.utils.data import DataLoader

    return DataLoader(
        _GMEImageDataset(images),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_identity_collate,
        num_workers=0,
    )


class _GMEImageDataset:
    def __init__(self, images: list[Any]):
        self.images = images
        self.transform = None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Any:
        image = self.images[index]
        if self.transform is None:
            return image
        return self.transform(image)


def _identity_collate(batch: list[Any]) -> list[Any]:
    return batch


def _prepare_gme_image(image: Any) -> Any:
    try:
        from PIL import Image
    except ImportError:
        return image

    if isinstance(image, Image.Image):
        return _resize_gme_image(image.convert("RGB"))
    path = Path(str(image))
    if not path.is_file():
        return image
    with Image.open(path) as opened:
        return _resize_gme_image(opened.convert("RGB"))


def _resize_gme_image(image: Any) -> Any:
    width, height = image.size
    if width <= 0 or height <= 0 or width * height <= GME_IMAGE_MAX_PIXELS:
        return image
    scale = math.sqrt(GME_IMAGE_MAX_PIXELS / float(width * height))
    resized_width = max(GME_IMAGE_FACTOR, int(width * scale))
    resized_height = max(GME_IMAGE_FACTOR, int(height * scale))
    resized_width = max(
        GME_IMAGE_FACTOR,
        (resized_width // GME_IMAGE_FACTOR) * GME_IMAGE_FACTOR,
    )
    resized_height = max(
        GME_IMAGE_FACTOR,
        (resized_height // GME_IMAGE_FACTOR) * GME_IMAGE_FACTOR,
    )
    from PIL import Image

    return image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)


def _empty_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _truncate_and_normalize(embeddings: np.ndarray, embedding_dim: int) -> np.ndarray:
    if embeddings.shape[1] < embedding_dim:
        raise ValueError(
            "encoder returned embedding dimension "
            f"{embeddings.shape[1]}, expected at least {embedding_dim}"
        )
    if embeddings.shape[1] > embedding_dim:
        embeddings = embeddings[:, :embedding_dim]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        embeddings = embeddings / norms
    if embeddings.shape[1] != embedding_dim:
        raise ValueError(
            "encoder returned embedding dimension "
            f"{embeddings.shape[1]}, expected {embedding_dim}"
        )
    return embeddings.astype(np.float32, copy=False)


def _fused_payload(item: Any, instruction: str | None) -> dict[str, Any]:
    if isinstance(item, dict):
        return {
            "text": str(item.get("text", "")),
            "image": str(item.get("image", "")),
            "instruction": instruction,
        }
    if isinstance(item, tuple) and len(item) == 2:
        return {
            "image": str(item[0]),
            "text": str(item[1]),
            "instruction": instruction,
        }
    raise TypeError("fused embedding items must be dicts or (image, text) tuples")


def _as_2d_float32(output: Any) -> np.ndarray:
    if hasattr(output, "detach") and callable(output.detach):
        output = output.detach()
    if hasattr(output, "cpu") and callable(output.cpu):
        output = output.cpu()
    embeddings = np.asarray(output, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.ndim != 2:
        raise ValueError(f"encoder output must be a 2D array, got shape {embeddings.shape}")
    return embeddings.astype(np.float32, copy=False)


def _stable_item_key(item: Any) -> str:
    if isinstance(item, Path):
        return item.as_posix()
    if isinstance(item, tuple):
        return "(" + "|".join(_stable_item_key(value) for value in item) + ")"
    if isinstance(item, list):
        return "[" + "|".join(_stable_item_key(value) for value in item) + "]"
    if isinstance(item, dict):
        pairs = (
            f"{_stable_item_key(key)}:{_stable_item_key(value)}"
            for key, value in sorted(item.items(), key=lambda pair: str(pair[0]))
        )
        return "{" + "|".join(pairs) + "}"
    return str(item)


__all__ = [
    "QWEN3_VL_MODEL_REPO_ID",
    "GME_MODEL_REPO_ID",
    "DEFAULT_EMBEDDING_DIMENSION",
    "GME_IMAGE_MAX_PIXELS",
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
]
