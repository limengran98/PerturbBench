#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_KEY="${1:?dataset key is required}"
shift || true

DEVICE="${DEVICE:-cuda:0}"
CUDA_MASK="${CUDA_VISIBLE_DEVICES_OVERRIDE:-0}"
SEEDS="${SEEDS:-11,12,13}"
INCLUDE_SUPP="${INCLUDE_SUPPLEMENTARY:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUNTIME_MODE="${RUNTIME_MODE:-primary}"
RUNTIME_ENV_GROUP_OVERRIDE="${RUNTIME_ENV_GROUP_OVERRIDE:-}"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-}"

ARGS=(
  --dataset-key "${DATASET_KEY}"
  --device "${DEVICE}"
  --cuda-visible-devices "${CUDA_MASK}"
  --seeds "${SEEDS}"
  --runtime-mode "${RUNTIME_MODE}"
)

if [[ -n "${RUNTIME_ENV_GROUP_OVERRIDE}" ]]; then
  ARGS+=(--runtime-env-group "${RUNTIME_ENV_GROUP_OVERRIDE}")
fi

if [[ "${INCLUDE_SUPP}" == "1" ]]; then
  ARGS+=(--include-supplementary)
fi

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  ARGS+=(--skip-existing)
fi

if [[ -n "${ARTIFACTS_ROOT}" ]]; then
  ARGS+=(--artifacts-root "${ARTIFACTS_ROOT}")
fi

python3 "${ROOT}/scripts/run_dataset_pack.py" "${ARGS[@]}" "$@"
