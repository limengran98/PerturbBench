# XPert

- Method name: XPert
- Category: specialist enhanced-information baseline
- Method summary: biologically informed dual-branch Transformer for drug-response profile prediction across dose-time conditions
- Current local path: `baseline/XPert-main`
- Official/local version info: local archive copy with upstream hint `https://github.com/GSanShui/XPert`; official upstream identity is plausible but not fully verifiable because no git metadata is preserved
- Dependencies: Python `3.9`, `torch==2.1.0+cu121`, `torch-geometric==2.6.1`, `flash_attn==2.6.0.post1`, `scanpy==1.9.8`, `unimol-tools`
- Suitable datasets: L1000 `compatible`; CDS-DB `compatible`
- Input requirements: paired h5ad with treated profile in `adata.X`, baseline profile in `adata.obsm['X_ctl']`, and metadata in `adata.obs`
- Output requirements: experiment logs, checkpoints, optional predicted profiles, CLS embeddings, and attention dumps
- Enhanced-information: yes; local configs require PPI vectors, heterogeneous-graph drug embeddings, and external molecular features such as UniMol or KPGT
- Minimal smoke test entry: `python3 scripts/preflight_specialist.py --baseline XPert --baseline-root baseline`
- Shared-runtime smoke run: `python3 scripts/run_benchmark.py --config configs/experiments/specialist_xpert_cdsdb.yaml --device cuda:0 --cuda-visible-devices 0`
- Shared-runtime status: runnable shared-runtime adaptation is now implemented for `l1000_public`, `l1000` (internal), and `cdsdb`; strict-upstream repo execution remains a fallback path only
- Current blocker: strict-upstream execution still lacks the GPU graph stack and `scanpy`; shared-runtime L1000 runs are GPU-preferred because the public dataset is much larger than CDS-DB
