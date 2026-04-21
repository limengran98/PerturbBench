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
    name="CPA",
    slug="cpa",
    archive_name="cpa-main.zip",
    repo_dir_name="cpa-main",
    version_hint="0.8.8 from pyproject.toml",
    upstream_hint="https://github.com/theislab/cpa",
    source_classification="local_archive_copy_likely_official_snapshot",
    source_note=(
        "Local asset is a GitHub-style source archive. Repo version is visible in pyproject.toml, "
        "but remote, branch, and commit are unavailable without preserved git metadata."
    ),
    license_name="BSD-3-Clause",
    key_files=(
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "setup.py",
        "cpa/__init__.py",
        "cpa/_model.py",
        "cpa/_api.py",
    ),
    entrypoints=(
        "cpa/__init__.py",
        "examples/tune_script.py",
        "docs/tutorials/Norman.ipynb",
        "docs/tutorials/combosciplex.ipynb",
    ),
    dependency_files=("pyproject.toml", "setup.py"),
    critical_packages=("torch", "anndata", "scanpy", "scvi", "lightning", "jax", "jaxlib", "rdkit", "ray"),
    smoke_import_modules=("cpa",),
    help_script=None,
    config_files=(),
    python_requirement=">=3.9, <3.11",
    torch_requirement=">1.8.0, <=2.0.1",
    cuda_requirement="optional; inherited from torch/scvi stack",
    native_input_format=(
        "Preprocessed AnnData with perturbation, dosage, and optional cell type / batch covariates in adata.obs; "
        "counts preserved in adata.layers['counts']."
    ),
    native_output_format=(
        "CPA model checkpoints, latent embeddings, and predicted perturbation responses through the Python API; "
        "native repo does not use this repo's artifact schema."
    ),
    extra_priors=("optional external drug embeddings such as RDKit",),
    enhanced_information=False,
    enhanced_information_note=(
        "Base CPA can run without extra priors, but the repo also includes external-embedding tutorials. "
        "Any embedding-augmented CPA run must be reported separately as enhanced-information."
    ),
    best_fit_datasets={
        "sci-Plex3": "possibly compatible",
        "Norman": "compatible",
        "Adamson": "possibly compatible",
    },
    best_fit_notes={
        "sci-Plex3": "Repo contains combo-sciPlex tutorials, but local mapping to our prepared sci-Plex3 export is not implemented.",
        "Norman": "Repo ships a Norman tutorial and gene-perturbation support.",
        "Adamson": "Gene perturbations should be conceptually compatible, but no explicit local Adamson example is bundled here.",
    },
    current_blockers=(
        "strict-upstream CPA import still needs scvi-tools, scanpy, lightning, jax, and rdkit",
        "primary execution should use the shared-runtime adaptation path rather than the upstream scvi stack",
        "Phase 4 needs a frozen decision on whether external embeddings are allowed",
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
            "CPA expects AnnData with perturbation metadata in obs and counts preserved in layers['counts']",
            "This export preserves split-wise matrices and metadata so a dedicated CPA env can reconstruct the required h5ad",
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
    if str(dataset.dataset_name).lower() not in {"norman", "adamson", "sciplex3"}:
        raise ValueError(f"CPA shared-runtime adaptation currently supports Norman/Adamson/sci-Plex3 only, got {dataset.dataset_name}")
    merged_config = {
        "hidden_dim": int(config.get("hidden_dim", 256)),
        "latent_dim": int(config.get("latent_dim", 128)),
        "dropout": float(config.get("dropout", 0.1)),
        "learning_rate": float(config.get("learning_rate", 1e-3)),
        "weight_decay": float(config.get("weight_decay", 1e-5)),
        "batch_size": int(config.get("batch_size", 256)),
        "epochs": int(config.get("epochs", 25)),
        "patience": int(config.get("patience", 5)),
        "hash_dim": int(config.get("hash_dim", 64)),
        "latent_l2_weight": float(config.get("latent_l2_weight", 1e-4)),
    }
    return train_shared_runtime_specialist(
        model_kind="cpa",
        dataset=dataset,
        split=split,
        config=merged_config,
        seed=int(seed),
        requested_device=str(requested_device),
        log_fn=log_fn,
    )
