"""Thin wrapper for the multimodal KB build CLI."""

from __future__ import annotations

import os
from pathlib import Path

from evograph_mm.kb.build import build_arg_parser, main as _build_main
from evograph_mm.device_env import configure_embedding_device


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    configure_embedding_device()
    return _build_main(argv)


__all__ = ["build_arg_parser", "load_env_file", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
