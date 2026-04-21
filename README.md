# PerturbBench

This repository mirrors the public benchmark execution layer only.

Included:
- benchmark configs, splits, env definitions, scripts, source code, and specialist baseline integrations
- direct-code, random-edit, and HPO control-line runtimes
- public placeholder LLM config templates for the direct-code control line

Excluded:
- the main structured-agent v1-v8 line
- legacy CodeEvo / agent-research entrypoints
- paper-writing assets
- run outputs and result tables
- prepared/raw data
- private LLM configs and API keys
- unrelated packaging utilities

Primary entrypoints:
- `bash scripts/benchmark_packs/run_norman_pack.sh`
- `bash scripts/benchmark_packs/run_adamson_pack.sh`
- `bash scripts/benchmark_packs/run_sciplex3_pack.sh`
- `bash scripts/benchmark_packs/run_papalexi_rna_pack.sh`
- `bash scripts/benchmark_packs/run_papalexi_protein_pack.sh`
- `bash scripts/benchmark_packs/run_l1000_public_pack.sh`
- `bash scripts/benchmark_packs/run_cdsdb_pack.sh`

Supporting workflows:
- data preparation via `scripts/prepare_dataset.py`
- split generation via `scripts/make_splits.py`
- baseline/specialist execution via `scripts/run_benchmark.py` and `scripts/run_dataset_pack.py`
- control-line execution via `scripts/run_direct_code_pack.py`, `scripts/run_random_edit_pack.py`, and `scripts/run_hpo_pack.py`

LLM setup note:
- `configs/agent/llm.yaml` is a placeholder template with no secret
- copy or edit it locally before direct-code runs, or pass `--llm-config path/to/your_local_llm.yaml`
