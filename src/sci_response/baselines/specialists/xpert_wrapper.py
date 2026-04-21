from __future__ import annotations

from pathlib import Path
from typing import Optional

from sci_response.baselines.specialists.base import (
    SpecialistBaselineSpec,
    export_ann_like_bundle,
    export_lightweight_native_manifest,
    run_specialist_preflight,
    use_lightweight_shared_export,
)
from sci_response.baselines.specialists.shared_runtime import train_shared_runtime_specialist


SPEC = SpecialistBaselineSpec(
    name="XPert",
    slug="xpert",
    archive_name="XPert-main.zip",
    repo_dir_name="XPert-main",
    version_hint="no explicit package version found in archive",
    upstream_hint="https://github.com/GSanShui/XPert",
    source_classification="local_archive_copy_upstream_identity_unverified",
    source_note=(
        "Local asset is a GitHub-style source archive. It looks like the paper code snapshot, "
        "but no git metadata is preserved and officialness cannot be proven from the archive alone."
    ),
    license_name="MIT",
    key_files=(
        "README.md",
        "LICENSE",
        "requirements.txt",
        "train_xpert.py",
        "models/model_XPert.py",
        "configs/config_l1000.yaml",
        "configs/config_cdsdb.yaml",
    ),
    entrypoints=(
        "train_xpert.py",
        "scripts/train.sh",
        "scripts/train_cdsdb.sh",
        "scripts/test.sh",
    ),
    dependency_files=("requirements.txt", "configs/config_l1000.yaml", "configs/config_cdsdb.yaml"),
    critical_packages=(
        "torch",
        "torch_geometric",
        "scanpy",
        "h5py",
        "yaml",
        "pandas",
        "numpy",
    ),
    smoke_import_modules=(),
    help_script="train_xpert.py",
    config_files=("configs/config_l1000.yaml", "configs/config_cdsdb.yaml"),
    python_requirement="3.9 in README environment instructions",
    torch_requirement="torch==2.1.0+cu121",
    cuda_requirement="CUDA 12.1 stack implied by requirements; flash_attn and gradient-scaler path assume GPU-capable environment",
    native_input_format=(
        "Paired pre/post perturbation h5ad where post-treatment expression is in adata.X, baseline in adata.obsm['X_ctl'], "
        "and metadata in adata.obs."
    ),
    native_output_format=(
        "XPert experiment logs, checkpoints, optional predicted profiles, CLS embeddings, and attention outputs; "
        "repo-native outputs are not aligned to this repo's artifact contract."
    ),
    extra_priors=(
        "PPI gene vectors",
        "heterogeneous graph drug embeddings",
        "UniMol / KPGT / Morgan drug features",
        "optional pretrained models",
    ),
    enhanced_information=True,
    enhanced_information_note=(
        "XPert explicitly consumes graph structure and external molecular representations, so it must stay out of the raw-input universal leaderboard."
    ),
    best_fit_datasets={
        "L1000": "compatible",
        "CDS-DB": "compatible",
    },
    best_fit_notes={
        "L1000": "Local repo ships dedicated l1000 configs, scripts, and data-path expectations.",
        "CDS-DB": "Local repo ships a dedicated config_cdsdb.yaml and training script for the independent-dataset setting.",
    },
    current_blockers=(
        "strict-upstream XPert import still needs torch-geometric, scanpy, and UniMol-side assets",
        "Phase 4 needs an explicit policy on which external drug features and graph assets are allowed under the fairness contract",
        "primary execution should use the shared-runtime adaptation path rather than the upstream HG/UniMol stack",
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
            native_export_kind="xpert_shared_runtime_manifest",
            notes=(
                "Primary shared-runtime execution does not require a full h5ad reconstruction bundle for XPert",
                "Use fallback upstream mode if a full AnnData-style export is needed for strict-upstream comparison",
            ),
        )
    return export_ann_like_bundle(
        SPEC,
        dataset,
        split,
        export_root,
        extra_notes=(
            "XPert expects h5ad with post-treatment data in X and matched control in obsm['X_ctl']",
            "The current export preserves those matrices split-wise but does not synthesize a true h5ad without anndata/scanpy",
        ),
    )


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
    if str(dataset.dataset_name).lower() not in {"l1000_public", "l1000", "cdsdb"}:
        raise ValueError(
            f"XPert shared-runtime adaptation currently supports l1000_public/l1000/cdsdb only, got {dataset.dataset_name}"
        )
    merged_config = {
        "hidden_dim": int(config.get("hidden_dim", 48)),
        "num_heads": int(config.get("num_heads", 4)),
        "perturb_rank": int(config.get("perturb_rank", 16)),
        "dropout": float(config.get("dropout", 0.1)),
        "learning_rate": float(config.get("learning_rate", 1e-3)),
        "weight_decay": float(config.get("weight_decay", 1e-5)),
        "batch_size": int(config.get("batch_size", 1024)),
        "epochs": int(config.get("epochs", 2)),
        "patience": int(config.get("patience", 1)),
        "hash_dim": int(config.get("hash_dim", 128)),
        "gene_basis_l2_weight": float(config.get("gene_basis_l2_weight", 1e-4)),
        "max_steps_per_epoch": int(config.get("max_steps_per_epoch", 16)),
        "max_train_samples": int(config.get("max_train_samples", 8192)),
        "max_val_samples": int(config.get("max_val_samples", 2048)),
    }
    runtime_device_override = str(config.get("runtime_device_override", "")).strip()
    return train_shared_runtime_specialist(
        model_kind="xpert",
        dataset=dataset,
        split=split,
        config=merged_config,
        seed=int(seed),
        requested_device=runtime_device_override or str(requested_device),
        log_fn=log_fn,
    )
