# GEARS

- Method name: GEARS
- Category: specialist enhanced-information baseline
- Method summary: graph-based single-cell perturbation predictor for single-gene and combinatorial gene perturbations
- Current local path: `baseline/GEARS-master`
- Official/local version info: local archive copy; likely official upstream snapshot from `https://github.com/snap-stanford/GEARS`; local version evidence `0.1.2` from `gears/version.py`; no git metadata preserved
- Dependencies: `torch`, `torch_geometric`, `scanpy`, `networkx`, `numpy`, `pandas`, `scipy`, `scikit-learn`
- Suitable datasets: Norman `compatible`; Adamson `compatible`
- Input requirements: AnnData / PertData with `gene_name` in `adata.var` and `condition`, `cell_type` in `adata.obs`
- Output requirements: GEARS checkpoint plus predicted transcriptional profiles through the GEARS API
- Enhanced-information: yes; code constructs gene ontology and co-expression graphs
- Minimal smoke test entry: `python3 scripts/preflight_specialist.py --baseline GEARS --baseline-root baseline`
- Current blocker: repo is extracted and wrapped, but the shared environment still lacks `scanpy` and `torch_geometric`; current benchmark artifacts emit an AnnData-like export bundle rather than a true GEARS-ready PertData object
