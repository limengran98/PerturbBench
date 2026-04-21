# CellOT

- Method name: CellOT
- Category: specialist baseline
- Method summary: neural optimal transport model for learning single-cell perturbation responses from unpaired control and treated distributions
- Current local path: `baseline/cellot-main`
- Official/local version info: local archive copy; likely official upstream snapshot from `https://github.com/bunnech/cellot`; local version evidence `0.1` from `setup.py`; no git metadata preserved
- Dependencies: Python `3.9.5`, `torch==1.11.0`, `scanpy==1.8.1`, `anndata==0.7.6`, `ml-collections==0.1.0`
- Suitable datasets: sci-Plex3 `compatible`; Papalexi RNA `possibly compatible`; Papalexi Protein `possibly compatible`
- Input requirements: task-specific h5ad assets referenced by YAML config files; local repo includes `configs/tasks/sciplex3.yaml`
- Output requirements: experiment directory with `config.yaml`, cached model state, and evaluation outputs
- Enhanced-information: no mandatory external graph or knowledge prior is visible in the local repo
- Minimal smoke test entry: `python3 scripts/preflight_specialist.py --baseline CellOT --baseline-root baseline`
- Shared-runtime status: runnable in-framework implementation is now available for `sci-Plex3`, `Papalexi RNA`, and `Papalexi Protein`
- Shared-runtime run:
  - `python3 scripts/run_benchmark.py --config configs/experiments/specialist_cellot_sciplex3.yaml --device cuda:0 --cuda-visible-devices 0`
  - `python3 scripts/run_benchmark.py --config configs/experiments/specialist_cellot_papalexi_rna.yaml --device cuda:0 --cuda-visible-devices 0`
  - `python3 scripts/run_benchmark.py --config configs/experiments/specialist_cellot_papalexi_protein.yaml --device cuda:0 --cuda-visible-devices 0`
- Current blocker: strict-upstream CellOT still depends on the old `scanpy/anndata/ml_collections` stack; the benchmark-default path is the in-framework implementation
