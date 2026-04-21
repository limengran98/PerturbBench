#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sci_response.data.adapters.cdsdb_profile import prepare_cdsdb_profile_dataset
from sci_response.data.adapters.cppa_profile import prepare_cppa_profile_dataset
from sci_response.data.adapters.l1000_profile import prepare_l1000_profile_dataset
from sci_response.data.adapters.scperturb_profile import prepare_scperturb_profile_dataset
from sci_response.data.io import dump_json
from sci_response.data.registry import load_dataset_spec, load_prepared_dataset, save_prepared_dataset
from sci_response.data.schemas import PreparedDataset
from sci_response.pathing import repo_relative_mapping, repo_relative_str


def _safe_float(value: float) -> str:
    numeric = float(value)
    if np.isnan(numeric):
        return "nan"
    return f"{numeric:.8g}"


def _export_prepared_directory(spec, prepared: PreparedDataset) -> Path:
    export_dir = spec.prepared_path.parent
    export_dir.mkdir(parents=True, exist_ok=True)

    samples_path = export_dir / "samples.csv"
    features_path = export_dir / "features.npy"
    feature_names_path = export_dir / "feature_names.json"
    metadata_path = export_dir / "metadata.json"
    log_path = export_dir / "prepare.log"

    metadata_column_names = prepared.metadata_column_names.astype(str).tolist()
    leakage_guard_names = prepared.leakage_guard_names.astype(str).tolist()
    context_feature_names = prepared.context_feature_names.astype(str).tolist()

    sample_fieldnames = [
        "sample_id",
        "split_id",
        "dataset_name",
        "intervention_id",
        "intervention_type",
        "dose",
        "time",
        "group_id",
        "metadata_json",
    ]
    sample_fieldnames.extend(f"meta__{name}" for name in metadata_column_names)
    sample_fieldnames.extend(f"guard__{name}" for name in leakage_guard_names)

    with samples_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fieldnames)
        writer.writeheader()
        for row_idx in range(prepared.sample_count):
            row = {
                "sample_id": str(prepared.sample_ids[row_idx]),
                "split_id": str(prepared.split_ids[row_idx]),
                "dataset_name": prepared.dataset_name,
                "intervention_id": str(prepared.intervention_ids[row_idx]),
                "intervention_type": str(prepared.intervention_types[row_idx]),
                "dose": _safe_float(prepared.doses[row_idx]),
                "time": _safe_float(prepared.times[row_idx]),
                "group_id": str(prepared.group_ids[row_idx]),
                "metadata_json": str(prepared.sample_metadata_json[row_idx]),
            }
            for col_idx, name in enumerate(metadata_column_names):
                row[f"meta__{name}"] = str(prepared.metadata_values[row_idx, col_idx])
            for col_idx, name in enumerate(leakage_guard_names):
                row[f"guard__{name}"] = str(prepared.leakage_guard_values[row_idx, col_idx])
            writer.writerow(row)

    np.save(features_path, prepared.delta_response.astype(np.float32), allow_pickle=False)
    feature_names_path.write_text(
        json.dumps(prepared.feature_names.astype(str).tolist(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    dataset_metadata = json.loads(prepared.dataset_metadata_json)
    blockers = []
    if (
        spec.adapter == "l1000_profile"
        and dataset_metadata.get("source_provenance_status") == "local_derived_bundle_only"
    ):
        blockers.append(
            "Local LINCS bundle is derived profile-level data only; public GEO/CLUE provenance rebuild is still required before main-table use."
        )

    metadata_payload = {
        "dataset_name": prepared.dataset_name,
        "task_family": prepared.task_family,
        "modality": prepared.modality,
        "feature_space": prepared.feature_space,
        "baseline_mode": prepared.baseline_mode,
        "pairing_regime": prepared.pairing_regime,
        "prepared_bundle_file": spec.prepared_path.name,
        "sample_table_file": samples_path.name,
        "feature_matrix_file": features_path.name,
        "feature_matrix_semantics": "delta_response",
        "feature_names_file": feature_names_path.name,
        "sample_count": prepared.sample_count,
        "control_count": prepared.control_count,
        "feature_count": prepared.output_dim,
        "context_feature_count": int(prepared.context_matrix.shape[1]),
        "context_feature_names": context_feature_names,
        "metadata_columns": metadata_column_names,
        "leakage_guard_fields": leakage_guard_names,
        "raw_paths": repo_relative_mapping(spec.raw_paths),
        "license_or_source_note": prepared.license_or_source_note,
        "source_description": prepared.source_description,
        "dataset_metadata": dataset_metadata,
        "blockers": blockers,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(metadata_path, metadata_payload)

    log_lines = [
        f"dataset_name={prepared.dataset_name}",
        f"prepared_bundle={repo_relative_str(spec.prepared_path)}",
        f"sample_table={repo_relative_str(samples_path)}",
        f"feature_matrix={repo_relative_str(features_path)} semantics=delta_response",
        f"sample_count={prepared.sample_count}",
        f"control_count={prepared.control_count}",
        f"feature_count={prepared.output_dim}",
        f"context_feature_count={int(prepared.context_matrix.shape[1])}",
        f"metadata_columns={','.join(metadata_column_names)}",
        f"leakage_guard_fields={','.join(leakage_guard_names)}",
        f"raw_paths={json.dumps(metadata_payload['raw_paths'], sort_keys=True, ensure_ascii=True)}",
        f"dataset_metadata={json.dumps(dataset_metadata, sort_keys=True, ensure_ascii=True)}",
    ]
    if blockers:
        for blocker in blockers:
            log_lines.append(f"blocker={blocker}")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return export_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare a dataset into the canonical reproducible NPZ format."
    )
    parser.add_argument("--dataset-config", required=True, help="Path to a dataset YAML config.")
    args = parser.parse_args()

    dataset_config = Path(args.dataset_config).resolve()
    spec = load_dataset_spec(dataset_config)

    if spec.adapter == "prepared_npz":
        raw_path = spec.raw_paths[0]
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw prepared fixture not found: {raw_path}")
        with np.load(raw_path, allow_pickle=False) as bundle:
            payload = {key: bundle[key] for key in bundle.files}
        save_prepared_dataset(spec.prepared_path, payload)
    elif spec.adapter == "scperturb_profile":
        prepare_scperturb_profile_dataset(spec)
    elif spec.adapter == "l1000_profile":
        prepare_l1000_profile_dataset(spec)
    elif spec.adapter == "cdsdb_profile":
        prepare_cdsdb_profile_dataset(spec)
    elif spec.adapter == "cppa_profile":
        prepare_cppa_profile_dataset(spec)
    else:
        raise ValueError(f"Unsupported dataset adapter: {spec.adapter}")

    prepared = load_prepared_dataset(spec)
    export_dir = _export_prepared_directory(spec, prepared)
    print(
        f"prepared_dataset={spec.prepared_path} "
        f"export_dir={export_dir} samples={prepared.sample_count} "
        f"output_dim={prepared.output_dim} baseline_mode={prepared.baseline_mode}"
    )


if __name__ == "__main__":
    main()
