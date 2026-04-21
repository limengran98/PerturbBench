from __future__ import annotations

from pathlib import Path
from typing import Optional

from sci_response.baselines.specialists.base import (
    SpecialistBaselineSpec,
    export_ann_like_bundle,
    run_specialist_preflight,
)
from sci_response.baselines.specialists.shared_runtime import train_shared_runtime_specialist


SPEC = SpecialistBaselineSpec(
    name="CellOT",
    slug="cellot",
    archive_name="cellot-main.zip",
    repo_dir_name="cellot-main",
    version_hint="0.1 from setup.py",
    upstream_hint="https://github.com/bunnech/cellot",
    source_classification="local_archive_copy_likely_official_snapshot",
    source_note=(
        "Local asset is a GitHub-style source archive with reproducibility scripts and configs, "
        "but no git metadata survives the archive snapshot."
    ),
    license_name="BSD-3-Clause",
    key_files=(
        "README.md",
        "LICENSE",
        "requirements.txt",
        "setup.py",
        "scripts/train.py",
        "scripts/evaluate.py",
        "cellot/train/experiment.py",
        "configs/tasks/sciplex3.yaml",
    ),
    entrypoints=(
        "scripts/train.py",
        "scripts/evaluate.py",
    ),
    dependency_files=("requirements.txt", "setup.py"),
    critical_packages=("torch", "scanpy", "anndata", "ml_collections", "numpy", "pandas", "scipy", "sklearn"),
    smoke_import_modules=("cellot",),
    help_script="scripts/train.py",
    config_files=("configs/tasks/sciplex3.yaml", "configs/models/cellot.yaml"),
    python_requirement="3.9.5 in README setup instructions",
    torch_requirement="torch==1.11.0",
    cuda_requirement="optional; README documents CPU training as valid but slow",
    native_input_format=(
        "Task-specific h5ad assets declared in YAML configs; sciplex3 config points to a train-only HVG AnnData file "
        "with explicit source/condition fields."
    ),
    native_output_format=(
        "CellOT experiment directory with config.yaml, cache/model.pt, scalar logs, and evaluation outputs; "
        "not aligned to this repo's artifact contract."
    ),
    extra_priors=(),
    enhanced_information=False,
    enhanced_information_note=(
        "CellOT is a specialist transport model, but the local repo does not show mandatory graph or knowledge-prior inputs."
    ),
    best_fit_datasets={
        "sci-Plex3": "compatible",
        "Papalexi RNA": "possibly compatible",
        "Papalexi Protein": "possibly compatible",
    },
    best_fit_notes={
        "sci-Plex3": "Repo contains an explicit sciplex3 task config.",
        "Papalexi RNA": "Method targets unpaired single-cell response distributions, but no local Papalexi-specific task file exists.",
        "Papalexi Protein": "Paper scope includes multiplexed protein imaging, but local repo ships 4i rather than Papalexi-specific adapters.",
    },
    current_blockers=(
        "strict-upstream CellOT import still needs scanpy, anndata, and ml_collections",
        "primary execution should use the in-framework shared-runtime transport path rather than the old h5ad-bound stack",
    ),
)


def run_preflight(baseline_root: Path, artifacts_root: Path, run_id: Optional[str] = None):
    return run_specialist_preflight(SPEC, baseline_root=baseline_root, artifacts_root=artifacts_root, run_id=run_id)


def export_native_inputs(dataset, split, export_root: Path):
    return export_ann_like_bundle(
        SPEC,
        dataset,
        split,
        export_root,
        extra_notes=(
            "CellOT task configs reference h5ad assets with explicit control/treatment annotations",
            "The emitted bundle preserves split-wise source/target matrices but leaves the final h5ad assembly to a dedicated CellOT environment",
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
    if str(dataset.dataset_name).lower() not in {"sciplex3", "papalexi_arrayed_rna", "papalexi_arrayed_protein"}:
        raise ValueError(
            f"CellOT in-framework implementation currently supports sciPlex3/Papalexi RNA/Papalexi Protein only, got {dataset.dataset_name}"
        )
    merged_config = {
        "hidden_dim": int(config.get("hidden_dim", 256)),
        "latent_dim": int(config.get("latent_dim", 128)),
        "dropout": float(config.get("dropout", 0.1)),
        "learning_rate": float(config.get("learning_rate", 1e-3)),
        "weight_decay": float(config.get("weight_decay", 1e-5)),
        "batch_size": int(config.get("batch_size", 256)),
        "epochs": int(config.get("epochs", 20)),
        "patience": int(config.get("patience", 4)),
        "hash_dim": int(config.get("hash_dim", 64)),
        "transport_weight": float(config.get("transport_weight", 1e-3)),
        "mmd_weight": float(config.get("mmd_weight", 1e-3)),
        "target_kind": "response",
    }
    return train_shared_runtime_specialist(
        model_kind="cellot",
        dataset=dataset,
        split=split,
        config=merged_config,
        seed=int(seed),
        requested_device=str(requested_device),
        log_fn=log_fn,
    )
