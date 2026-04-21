from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from sci_response.baselines.specialists.base import (
    SpecialistBaselineSpec,
    export_gperturb_bundle,
    run_specialist_preflight,
)
from sci_response.data.splits import indices_from_split


SPEC = SpecialistBaselineSpec(
    name="GPerturb",
    slug="gperturb",
    archive_name="GPerturb-main.zip",
    repo_dir_name="GPerturb-main",
    version_hint="0.0.1 from setup.py",
    upstream_hint="https://github.com/hwxing3259/GPerturb",
    source_classification="local_archive_copy_likely_official_snapshot",
    source_note=(
        "Local asset is a GitHub-style source archive. README and setup.py match the public GPerturb package, "
        "but remote and commit identity cannot be verified from the zip alone."
    ),
    license_name="MIT",
    key_files=(
        "README.md",
        "LICENSE",
        "setup.py",
        "GPerturb/__init__.py",
        "GPerturb/GPerturb_model.py",
    ),
    entrypoints=(
        "GPerturb/__init__.py",
        "numerical_examples/Norman_GPerturb.ipynb",
        "numerical_examples/SciPlex2_GPerturb.ipynb",
    ),
    dependency_files=("setup.py",),
    critical_packages=("torch", "numpy", "pandas", "matplotlib"),
    smoke_import_modules=("GPerturb",),
    help_script=None,
    config_files=(),
    python_requirement="Python >= 3.8.0",
    torch_requirement="torch==2.2.2",
    cuda_requirement="optional; README runtime example uses an NVIDIA RTX2060 GPU but method can run on CPU",
    native_input_format=(
        "Three explicit matrices: gene expression matrix X, cell-level covariate matrix C, and perturbation matrix P."
    ),
    native_output_format=(
        "Predicted expression matrix plus sparse gene-level perturbation effect matrix; no built-in artifact contract."
    ),
    extra_priors=(),
    enhanced_information=False,
    enhanced_information_note=(
        "Local code path is a pure modeling baseline without mandatory external graph or knowledge priors."
    ),
    best_fit_datasets={
        "Norman": "compatible",
        "Adamson": "possibly compatible",
        "sci-Plex3": "possibly compatible",
    },
    best_fit_notes={
        "Norman": "Repo ships a Norman numerical example.",
        "Adamson": "Matrix-based input contract makes Adamson plausible, but no local Adamson example is bundled.",
        "sci-Plex3": "Repo demonstrates SciPlex2 rather than sci-Plex3, so compatibility is plausible but not yet frozen.",
    },
    current_blockers=(),
)


def run_preflight(baseline_root: Path, artifacts_root: Path, run_id: Optional[str] = None):
    return run_specialist_preflight(SPEC, baseline_root=baseline_root, artifacts_root=artifacts_root, run_id=run_id)


def export_native_inputs(dataset, split, export_root: Path):
    return export_gperturb_bundle(SPEC, dataset, split, export_root)


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
    import torch

    repo_dir = baseline_root / SPEC.repo_dir_name
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from GPerturb.GPerturb_model import GPerturb_Gaussian, Gaussian_estimates  # type: ignore

    split_indices = indices_from_split(dataset.sample_ids, split)
    vocabulary = sorted(np.unique(dataset.intervention_ids.astype(str)).tolist())
    vocab_lookup = {label: idx for idx, label in enumerate(vocabulary)}

    def build_p(indices: np.ndarray) -> np.ndarray:
        matrix = np.zeros((indices.shape[0], len(vocabulary)), dtype=np.float32)
        for row_idx, label in enumerate(dataset.intervention_ids[indices].astype(str).tolist()):
            matrix[row_idx, vocab_lookup[label]] = 1.0
        return matrix

    contexts = dataset.context_matrix.astype(np.float32)
    responses = dataset.y_response.astype(np.float32)
    perturbations = build_p(np.arange(dataset.sample_ids.shape[0], dtype=np.int64))

    hidden_node = int(config.get("hidden_node", 128))
    hidden_layer_1 = int(config.get("hidden_layer_1", 2))
    hidden_layer_2 = int(config.get("hidden_layer_2", 2))
    learning_rate = float(config.get("learning_rate", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))
    batch_size = int(config.get("batch_size", 256))
    epochs = int(config.get("epochs", 20))
    patience = int(config.get("patience", 5))
    tau_value = float(config.get("tau", 1.0))
    nu_1 = float(config.get("nu_1", 1.0))
    nu_2 = float(config.get("nu_2", 0.1))
    nu_3 = float(config.get("nu_3", 1.0))
    nu_4 = float(config.get("nu_4", 0.1))
    nu_5 = float(config.get("nu_5", 1.0))
    nu_6 = float(config.get("nu_6", 0.1))

    if requested_device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(requested_device)
    else:
        device = torch.device("cpu")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    model = GPerturb_Gaussian(
        conditioner_dim=int(perturbations.shape[1]),
        output_dim=int(responses.shape[1]),
        base_dim=int(contexts.shape[1]),
        data_size=int(split_indices["train"].shape[0]),
        hidden_node=hidden_node,
        hidden_layer_1=hidden_layer_1,
        hidden_layer_2=hidden_layer_2,
        tau=torch.tensor(tau_value, device=device),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_indices = split_indices["train"].astype(np.int64)
    val_indices = split_indices["val"].astype(np.int64)
    rng = np.random.default_rng(int(seed))

    def to_tensor(array: np.ndarray):
        return torch.from_numpy(array).to(device=device, dtype=torch.float32)

    def context_tensor(indices: np.ndarray):
        if contexts.shape[1] == 0:
            return None
        return to_tensor(contexts[indices])

    def loss_on_indices(indices: np.ndarray):
        obs = to_tensor(responses[indices])
        cond = to_tensor(perturbations[indices])
        cell_info = context_tensor(indices)
        return model.normal_loss(
            observation=obs,
            conditioner=cond,
            cell_info=cell_info,
            nu_1=nu_1,
            nu_2=nu_2,
            nu_3=nu_3,
            nu_4=nu_4,
            nu_5=nu_5,
            nu_6=nu_6,
        )

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    for epoch in range(epochs):
        order = rng.permutation(train_indices.shape[0])
        batch_losses = []
        model.train()
        for start in range(0, train_indices.shape[0], batch_size):
            batch_pick = train_indices[order[start : start + batch_size]]
            loss = loss_on_indices(batch_pick)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_on_indices(val_indices).detach().cpu().item())
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        log_fn(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"device={device.type} vocab={len(vocabulary)}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                log_fn(f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict_indices(indices: np.ndarray):
        obs = to_tensor(responses[indices])
        cond = to_tensor(perturbations[indices])
        cell_info = context_tensor(indices)
        with torch.no_grad():
            estimates = Gaussian_estimates(model, obs=obs, cond=cond, cell_info=cell_info)
        y_pred = np.asarray(estimates["pert_mean"], dtype=np.float32)
        delta_pred = y_pred - dataset.x_baseline[indices].astype(np.float32)
        return {"y_pred": y_pred, "delta_pred": delta_pred}

    return {
        "execution_mode": "gperturb_native_torch",
        "training_summary": {
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "epochs_requested": int(epochs),
            "hidden_node": hidden_node,
            "hidden_layer_1": hidden_layer_1,
            "hidden_layer_2": hidden_layer_2,
            "batch_size": batch_size,
            "device": str(device),
        },
        "predictions": {
            split_name: predict_indices(indices)
            for split_name, indices in split_indices.items()
        },
    }
