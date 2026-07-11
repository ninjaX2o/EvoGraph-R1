"""CLI implementation for guarded multimodal preprocessing."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .datasets.echosight import process_echosight_dataset
from .datasets.synthetic import SYNTHETIC_DATASET, process_synthetic_dataset
from .prepare_echosight import DATASET_E_VQA, DATASET_INFOSEEK


CLI_SYNTHETIC_DATASET = "synthetic"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process guarded multimodal datasets or report readiness blockers.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=(CLI_SYNTHETIC_DATASET, SYNTHETIC_DATASET, DATASET_E_VQA, DATASET_INFOSEEK),
        help=(
            "Dataset to process. Synthetic supports jsonl/parquet. EchoSight datasets support metadata-only "
            "parquet in subset mode (--subset-mode paper with --subset) and full raw mode (--subset-mode none); "
            "without --metadata-only they report readiness."
        ),
    )
    parser.add_argument(
        "--output-root",
        help="Caller-provided output root for processed metadata-only files.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository or staging root containing datasets_mm/ for EchoSight datasets.",
    )
    parser.add_argument(
        "--subset",
        help="EchoSight subset name to process, for example paper_300_20_seed0.",
    )
    parser.add_argument(
        "--subset-mode",
        default="paper",
        choices=("none", "paper"),
        help=(
            "EchoSight processing mode. Use paper for existing subset processing "
            "or none to process the full raw CSV splits in place."
        ),
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write metadata-only outputs for EchoSight subset or full raw processing.",
    )
    parser.add_argument(
        "--format",
        default="jsonl",
        choices=("jsonl", "parquet"),
        help="Output format. Synthetic preprocessing supports jsonl/parquet; EchoSight metadata-only supports parquet.",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    if args.dataset in {CLI_SYNTHETIC_DATASET, SYNTHETIC_DATASET}:
        if not args.output_root:
            raise ValueError("--output-root is required for synthetic preprocessing")
        return process_synthetic_dataset(
            args.output_root,
            output_format=args.format,
            subset=args.subset,
        )

    return process_echosight_dataset(
        args.root,
        dataset=args.dataset,
        subset=args.subset,
        subset_mode=args.subset_mode,
        output_root=args.output_root,
        output_format=args.format,
        metadata_only=args.metadata_only,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        report = run_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
