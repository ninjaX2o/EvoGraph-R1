"""CLI entrypoint for the multimodal EvoGraph API server."""

from __future__ import annotations

import argparse
from functools import partial
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from evograph_mm.device_env import configure_embedding_device

configure_embedding_device()

import uvicorn

from evograph_mm.kb.graph_edit import prepare_edit_working_dir
from evograph_mm.kb.api import create_app
from evograph_mm.kb.indexing.encoders import create_embedding_encoder


DEFAULT_DATASET = "E-VQA"
DEFAULT_SUBSET = "paper_tar_5120_128_seed0"
DEFAULT_OUTPUT_ROOT = str(Path(os.getenv("MM_DATA_ROOT", ".")) / "expr_mm")
DEFAULT_MODEL = "runtime_encoder_disabled"
DEFAULT_EMBEDDING_BACKEND = os.getenv("MM_EMBEDDING_BACKEND", "gme")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8003
DEFAULT_RELOAD_INTERVAL = 300
DEFAULT_CHECK_INTERVAL = 60


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the multimodal EvoGraph API.",
        allow_abbrev=False,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--working_dir")
    parser.add_argument("--model")
    parser.add_argument(
        "--embedding-backend",
        choices=("gme", "qwen3-vl"),
        default=DEFAULT_EMBEDDING_BACKEND,
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--reload_interval", type=int, default=DEFAULT_RELOAD_INTERVAL)
    parser.add_argument("--check_interval", type=int, default=DEFAULT_CHECK_INTERVAL)
    parser.add_argument(
        "--prepare_edit_working_dir",
        action="store_true",
        help="Copy the resolved MM KB to a new edit working directory before starting.",
    )
    return parser


def resolve_working_dir(
    working_dir: str | Path | None,
    output_root: str | Path,
    dataset: str,
    subset: str,
) -> Path:
    if working_dir is not None:
        return Path(working_dir)

    env_working_dir = os.environ.get("EVOGRAPH_MM_WORKING_DIR")
    if env_working_dir:
        return Path(env_working_dir)

    return Path(output_root) / dataset


def resolve_model_path(model: str | Path | None) -> Path:
    if model is not None:
        return Path(model)

    env_model_path = os.environ.get("MM_EMBED_MODEL_PATH")
    if env_model_path:
        return Path(env_model_path)

    return Path(DEFAULT_MODEL)


def _has_explicit_arg(argv: list[str], option: str) -> bool:
    return any(item == option or item.startswith(f"{option}=") for item in argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    args = build_arg_parser().parse_args(raw_argv)
    working_dir = resolve_working_dir(
        args.working_dir,
        args.output_root,
        args.dataset,
        args.subset,
    )
    if args.prepare_edit_working_dir:
        working_dir = prepare_edit_working_dir(working_dir)
    has_working_dir_override = (
        args.working_dir is not None
        or bool(os.environ.get("EVOGRAPH_MM_WORKING_DIR"))
        or args.prepare_edit_working_dir
    )
    dataset = args.dataset
    subset = args.subset
    if has_working_dir_override:
        if not _has_explicit_arg(raw_argv, "--dataset"):
            dataset = None
        if not _has_explicit_arg(raw_argv, "--subset"):
            subset = None
    model_path = resolve_model_path(args.model)
    encoder_factory = None
    if model_path != Path(DEFAULT_MODEL):
        encoder_factory = partial(create_embedding_encoder, args.embedding_backend)
    app = create_app(
        working_dir=working_dir,
        model_path=model_path,
        output_root=args.output_root,
        dataset=dataset,
        subset=subset,
        encoder_factory=encoder_factory,
        reload_interval=args.reload_interval,
        check_interval=args.check_interval,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
