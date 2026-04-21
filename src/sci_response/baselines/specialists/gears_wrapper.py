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
    name="GEARS",
    slug="gears",
    archive_name="GEARS-master.zip",
    repo_dir_name="GEARS-master",
    version_hint="0.1.2 from gears/version.py",
    upstream_hint="https://github.com/snap-stanford/GEARS",
    source_classification="local_archive_copy_likely_official_snapshot",
    source_note=(
        "Local asset is a GitHub-style source archive. README states it is the official implementation, "
        "but no git metadata is preserved in the zip."
    ),
    license_name="MIT",
    key_files=(
        "README.md",
        "LICENSE",
        "requirements.txt",
        "setup.py",
        "gears/__init__.py",
        "gears/model.py",
        "gears/pertdata.py",
    ),
    entrypoints=(
        "gears/__init__.py",
        "demo/tutorial_inference_Norman.ipynb",
    ),
    dependency_files=("requirements.txt", "setup.py"),
    critical_packages=("torch", "torch_geometric", "scanpy", "networkx", "numpy", "pandas", "scipy", "sklearn"),
    smoke_import_modules=("gears",),
    help_script=None,
    config_files=(),
    python_requirement="not pinned in repo; practical modern Python required by current dependency stack",
    torch_requirement="torch (unpinned) + torch_geometric",
    cuda_requirement="optional CUDA for faster training; API examples use explicit cuda device strings",
    native_input_format=(
        "scanpy AnnData or processed PertData bundle; custom data path requires adata.var['gene_name'] "
        "and adata.obs['condition'], adata.obs['cell_type']."
    ),
    native_output_format=(
        "Saved GEARS model plus predicted transcriptional response profiles via Python API; "
        "no native artifact contract matching this repo."
    ),
    extra_priors=("gene ontology graph", "gene co-expression graph inferred from training data"),
    enhanced_information=True,
    enhanced_information_note=(
        "GEARS uses graph structure beyond the shared universal feature budget, so it belongs in the "
        "enhanced-information specialist bucket."
    ),
    best_fit_datasets={"Norman": "compatible", "Adamson": "compatible"},
    best_fit_notes={
        "Norman": "README includes a Norman inference tutorial and built-in dataset loader example.",
        "Adamson": "README explicitly lists Adamson among paper datasets handled by PertData.",
    },
    current_blockers=(
        "strict-upstream GEARS import still needs scanpy and torch_geometric",
        "primary execution should use the shared-runtime adaptation path rather than the upstream PertData stack",
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
            "GEARS ultimately expects AnnData/PertData with obs['condition'] and obs['cell_type'] plus var['gene_name']",
            "This bundle keeps the benchmark split explicit but stops short of writing h5ad in the current environment",
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
    if str(dataset.dataset_name).lower() not in {"norman", "adamson"}:
        raise ValueError(f"GEARS shared-runtime adaptation currently supports Norman/Adamson only, got {dataset.dataset_name}")
    merged_config = {
        "hidden_dim": int(config.get("hidden_dim", 256)),
        "perturb_rank": int(config.get("perturb_rank", 32)),
        "dropout": float(config.get("dropout", 0.1)),
        "learning_rate": float(config.get("learning_rate", 1e-3)),
        "weight_decay": float(config.get("weight_decay", 1e-5)),
        "batch_size": int(config.get("batch_size", 256)),
        "epochs": int(config.get("epochs", 25)),
        "patience": int(config.get("patience", 5)),
        "hash_dim": int(config.get("hash_dim", 64)),
        "gene_specific_l1_weight": float(config.get("gene_specific_l1_weight", 1e-4)),
    }
    return train_shared_runtime_specialist(
        model_kind="gears",
        dataset=dataset,
        split=split,
        config=merged_config,
        seed=int(seed),
        requested_device=str(requested_device),
        log_fn=log_fn,
    )
