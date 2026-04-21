#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sci_response.data.registry import load_dataset_spec, load_prepared_dataset
from sci_response.data.splits import (
    SUPPORTED_PROTOCOLS,
    build_split_summary,
    make_protocol_split,
    protocol_support_report,
    save_split,
)
from sci_response.data.io import dump_json


def _resolve_output_paths(
    dataset_name: str,
    protocol: str,
    split_name: str,
    split_root: Path,
    explicit_output: str | None,
) -> tuple[Path, Path]:
    if explicit_output:
        split_path = Path(explicit_output).resolve()
        summary_path = split_path.parent / "split_summary.json"
        return split_path, summary_path
    output_dir = (split_root / dataset_name / protocol).resolve()
    return output_dir / f"{split_name}.json", output_dir / "split_summary.json"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create explicit protocol-aware split files.")
    parser.add_argument("--dataset-config", required=True, help="Path to a dataset YAML config.")
    parser.add_argument(
        "--protocol",
        required=True,
        choices=sorted(SUPPORTED_PROTOCOLS),
        help="Split protocol to generate.",
    )
    parser.add_argument(
        "--group-field",
        action="append",
        default=None,
        help="Optional explicit grouping field. Repeat to build compound grouping keys.",
    )
    parser.add_argument("--split-name", default=None, help="Split file stem. Defaults to seed<seed>.")
    parser.add_argument("--split-root", default="splits", help="Root directory for standard split outputs.")
    parser.add_argument("--output", default=None, help="Optional explicit JSON split path.")
    parser.add_argument("--seed", type=int, default=11, help="Split seed.")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()

    dataset_config = Path(args.dataset_config).resolve()
    spec = load_dataset_spec(dataset_config)
    dataset = load_prepared_dataset(spec)

    support = protocol_support_report(dataset, args.protocol, group_fields=args.group_field)
    if not bool(support["supported"]):
        raise ValueError(
            f"{spec.dataset_name} does not support protocol {args.protocol}: {support['reason']}"
        )

    split = make_protocol_split(
        dataset=dataset,
        protocol=args.protocol,
        group_fields=args.group_field,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    split_name = args.split_name or f"seed{args.seed}"
    split_path, summary_path = _resolve_output_paths(
        dataset_name=spec.dataset_name,
        protocol=args.protocol,
        split_name=split_name,
        split_root=Path(args.split_root),
        explicit_output=args.output,
    )
    save_split(split_path, split)

    summary = build_split_summary(dataset=dataset, split=split, split_name=split_name)
    dump_json(summary_path, summary)

    print(
        f"split_file={split_path} summary_file={summary_path} protocol={split.protocol} "
        f"group_fields={','.join(split.group_fields)} "
        f"train={len(split.train_ids)} val={len(split.val_ids)} test={len(split.test_ids)}"
    )


if __name__ == "__main__":
    main()
