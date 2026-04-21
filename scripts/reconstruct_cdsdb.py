#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


MANIFEST_HEADER = [
    "source_kind",
    "source_url",
    "local_raw_path",
    "license_note",
    "status",
    "notes",
]


def write_bootstrap_manifest(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "cdsdb_reconstruction_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_HEADER)
        writer.writerow(
            [
                "CDS-DB portal export",
                "https://cdsdb.ncpsb.org.cn/",
                _display_path(output_root / "raw" / "cdsdb_export.tsv"),
                "Check CDS-DB terms before export or redistribution.",
                "pending_manual_export",
                "Stage normalized paired signatures and sample metadata here.",
            ]
        )
        writer.writerow(
            [
                "CDS-DB publication",
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC10767794/",
                _display_path(output_root / "reference" / "cdsdb_paper.html"),
                "Paper is public; use it to document schema and provenance.",
                "optional_reference",
                "Use this to annotate fields, not as the data source itself.",
            ]
        )
    return manifest_path


def write_normalized_template(output_root: Path) -> Path:
    normalized = output_root / "normalized_responses.tsv"
    normalized.parent.mkdir(parents=True, exist_ok=True)
    with normalized.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "sample_id",
                "patient_id",
                "cancer_type",
                "intervention_id",
                "intervention_type",
                "dose",
                "time",
                "group_id",
                "x_baseline__GENE1",
                "y_response__GENE1",
            ]
        )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap an explicit CDS-DB reconstruction plan.")
    parser.add_argument(
        "--output-root",
        default="data/drug perturbation/CDSDB",
        help="Root directory for the reconstruction plan and staged exports.",
    )
    args = parser.parse_args()

    output_root = (ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root).resolve()
    manifest = write_bootstrap_manifest(output_root)
    normalized = write_normalized_template(output_root)
    print(f"manifest={_display_path(manifest)}")
    print(f"normalized_template={_display_path(normalized)}")


if __name__ == "__main__":
    main()
