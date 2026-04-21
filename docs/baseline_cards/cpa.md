# CPA

- Method name: CPA
- Category: specialist baseline
- Method summary: compositional perturbation autoencoder for perturbation, dose, and context-aware response prediction
- Current local path: `baseline/cpa-main`
- Official/local version info: local archive copy; likely official upstream snapshot from `https://github.com/theislab/cpa`; local version evidence `0.8.8` from `pyproject.toml`; no git metadata preserved
- Dependencies: Python `>=3.9,<3.11`, `torch<=2.0.1`, `scvi-tools`, `scanpy`, `jax`, `lightning`, `ray`, `rdkit`
- Suitable datasets: Norman `compatible`; sci-Plex3 `possibly compatible`; Adamson `possibly compatible`
- Input requirements: preprocessed AnnData with perturbation, dosage, optional cell type / batch covariates in `adata.obs`, and counts saved in `adata.layers['counts']`
- Output requirements: CPA checkpoints, perturbation embeddings, and API-level predictions
- Enhanced-information: base CPA no; if external drug embeddings such as RDKit are enabled, that run must be relabeled as enhanced-information
- Minimal smoke test entry: `python3 scripts/preflight_specialist.py --baseline CPA --baseline-root baseline`
- Current blocker: repo is extracted and wrapped, but the current machine still lacks the CPA `scvi` stack; current benchmark artifacts emit a CPA-targeted AnnData-like bundle, and embedding-augmented CPA must remain outside the raw-input leaderboard
