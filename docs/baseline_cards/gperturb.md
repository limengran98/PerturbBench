# GPerturb

- Method name: GPerturb
- Category: specialist baseline
- Method summary: sparse Bayesian distributional regression model that estimates perturbation effects at gene level
- Current local path: `baseline/GPerturb-main`
- Official/local version info: local archive copy; likely official upstream snapshot from `https://github.com/hwxing3259/GPerturb`; local version evidence `0.0.1` from `setup.py`; no git metadata preserved
- Dependencies: Python `>=3.8`, `torch==2.2.2`, `numpy==1.26.4`, `pandas==2.2.1`, `matplotlib==3.8.0`
- Suitable datasets: Norman `compatible`; Adamson `possibly compatible`; sci-Plex3 `possibly compatible`
- Input requirements: three explicit matrices: expression `X`, cell covariates `C`, perturbation design `P`
- Output requirements: fitted expression predictions plus sparse perturbation-effect matrix
- Enhanced-information: no mandatory external prior is visible in the local code
- Minimal smoke test entry: `python3 scripts/run_benchmark.py --dataset-config configs/datasets/norman_scperturb.yaml --split-path splits/norman/cold_intervention/seed11.json --specialist gperturb --run-id specialist_gperturb_exec_smoke --device cpu`
- Current blocker: Norman is now runnable end-to-end, but Adamson and sci-Plex3 have not yet been validated through the same GPerturb path and cold-intervention performance is currently weak
