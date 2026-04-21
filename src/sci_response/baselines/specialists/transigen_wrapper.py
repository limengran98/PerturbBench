from __future__ import annotations

from pathlib import Path
from typing import Optional

from sci_response.baselines.specialists.base import (
    SpecialistBaselineSpec,
    export_lightweight_native_manifest,
    export_transigen_bundle,
    run_specialist_preflight,
    use_lightweight_shared_export,
)
from sci_response.baselines.specialists.shared_runtime import train_shared_runtime_specialist


SPEC = SpecialistBaselineSpec(
    name="TranSiGen",
    slug="transigen",
    archive_name="TranSiGen-main.zip",
    repo_dir_name="TranSiGen-main",
    version_hint="no explicit package version found in archive",
    upstream_hint="https://github.com/myzhengSIMM/TranSiGen",
    source_classification="local_archive_copy_likely_official_snapshot",
    source_note=(
        "Local asset is a GitHub-style source archive. README points to the public TranSiGen repo and release assets, "
        "but branch and commit are unavailable without git metadata."
    ),
    license_name="MIT",
    key_files=(
        "README.md",
        "LICENSE",
        "requirements.txt",
        "src/train_TranSiGen_full_data.py",
        "src/prediction.py",
        "src/model.py",
        "src/dataset.py",
    ),
    entrypoints=(
        "src/train_TranSiGen_full_data.py",
        "src/prediction.py",
    ),
    dependency_files=("requirements.txt",),
    critical_packages=("torch", "cmappy", "rdkit", "numpy", "pandas", "scipy", "sklearn"),
    smoke_import_modules=(),
    help_script="src/train_TranSiGen_full_data.py",
    config_files=(),
    python_requirement="3.6.13",
    torch_requirement="pytorch==1.5.1",
    cuda_requirement="CUDA 10.1 + cuDNN 7.6.5 inferred from exported conda lockfile",
    native_input_format=(
        "HDF5 perturbation bundle plus molecule-path pickles and optional external molecular embedding files "
        "(KPGT or ECFP4)."
    ),
    native_output_format=(
        "Saved model checkpoints, per-split reconstruction CSVs, and optional predicted profile dumps under repo-local results directories."
    ),
    extra_priors=(
        "KPGT molecular embeddings",
        "ECFP4 molecular fingerprints",
        "pretrained shRNA initialization weights",
    ),
    enhanced_information=True,
    enhanced_information_note=(
        "Default TranSiGen workflows depend on external molecular features and pretrained initializations, so they are enhanced-information specialist runs."
    ),
    best_fit_datasets={
        "L1000": "compatible",
        "CDS-DB": "unknown",
    },
    best_fit_notes={
        "L1000": "Local repo ships dedicated LINCS2020 data layout, examples, and training entrypoints.",
        "CDS-DB": "No CDS-DB-specific config or loader is visible in the local archive snapshot.",
    },
    current_blockers=(
        "strict-upstream TranSiGen import still needs cmappy and rdkit",
        "primary execution should use the in-framework shared-runtime representation path rather than the legacy molecular-feature stack",
    ),
)


def run_preflight(baseline_root: Path, artifacts_root: Path, run_id: Optional[str] = None):
    return run_specialist_preflight(SPEC, baseline_root=baseline_root, artifacts_root=artifacts_root, run_id=run_id)


def export_native_inputs(dataset, split, export_root: Path):
    if use_lightweight_shared_export():
        return export_lightweight_native_manifest(
            SPEC,
            dataset,
            split,
            export_root,
            native_export_kind="transigen_shared_runtime_manifest",
            notes=(
                "Primary shared-runtime execution does not require per-split HDF export for TranSiGen",
                "Use fallback upstream mode if strict-upstream HDF assets are needed for audit or comparison",
            ),
        )
    return export_transigen_bundle(SPEC, dataset, split, export_root)


def execute_specialist(
    *,
    dataset,
    split,
    baseline_root: Path,
    config,
    seed: int,
    requested_device: str,
    log_fn,
):
    if str(dataset.dataset_name).lower() not in {"l1000_public", "l1000"}:
        raise ValueError(
            f"TranSiGen in-framework implementation currently supports l1000_public/l1000 only, got {dataset.dataset_name}"
        )
    merged_config = {
        "hidden_dim": int(config.get("hidden_dim", 64)),
        "latent_dim": int(config.get("latent_dim", 32)),
        "dropout": float(config.get("dropout", 0.1)),
        "learning_rate": float(config.get("learning_rate", 1e-3)),
        "weight_decay": float(config.get("weight_decay", 1e-5)),
        "batch_size": int(config.get("batch_size", 2048)),
        "epochs": int(config.get("epochs", 1)),
        "patience": int(config.get("patience", 1)),
        "hash_dim": int(config.get("hash_dim", 128)),
        "latent_l2_weight": float(config.get("latent_l2_weight", 1e-4)),
        "baseline_recon_weight": float(config.get("baseline_recon_weight", 0.2)),
        "max_steps_per_epoch": int(config.get("max_steps_per_epoch", 16)),
        "max_train_samples": int(config.get("max_train_samples", 8192)),
        "max_val_samples": int(config.get("max_val_samples", 2048)),
        "target_kind": "response",
    }
    return train_shared_runtime_specialist(
        model_kind="transigen",
        dataset=dataset,
        split=split,
        config=merged_config,
        seed=int(seed),
        requested_device=str(requested_device),
        log_fn=log_fn,
    )
