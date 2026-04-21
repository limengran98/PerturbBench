# TranSiGen

- Method name: TranSiGen
- Category: specialist enhanced-information baseline
- Method summary: deep representation learning model for chemical-induced transcriptional profiles and downstream phenotype-based drug discovery
- Current local path: `baseline/TranSiGen-main`
- Official/local version info: local archive copy; likely official upstream snapshot from `https://github.com/myzhengSIMM/TranSiGen`; no git metadata preserved
- Dependencies: Python `3.6.13`, `pytorch==1.5.1`, `cmappy==4.0.1`, `rdkit==2020.09.1`
- Suitable datasets: L1000 `compatible`; CDS-DB `unknown`
- Input requirements: HDF5 perturbation bundle plus molecule mapping pickles and optional KPGT / ECFP4 molecular embeddings
- Output requirements: saved checkpoints, reconstruction CSVs, predicted profile dumps, and downstream repurposing outputs under repo-local `results/`
- Enhanced-information: yes; default workflows use external molecular embeddings and pretrained shRNA initialization
- Minimal smoke test entry: `python3 scripts/preflight_specialist.py --baseline TranSiGen --baseline-root baseline`
- Shared-runtime status: runnable in-framework implementation is now available for `l1000_public` and `l1000` (internal)
- Shared-runtime run:
  - `python3 scripts/run_benchmark.py --config configs/experiments/specialist_transigen_l1000_public.yaml --device cuda:0 --cuda-visible-devices 0`
  - `python3 scripts/run_benchmark.py --config configs/experiments/specialist_transigen_l1000_internal.yaml --device cuda:0 --cuda-visible-devices 0`
- Current blocker: strict-upstream TranSiGen still depends on legacy `cmappy/rdkit/Python 3.6`; the benchmark-default path is the in-framework implementation
