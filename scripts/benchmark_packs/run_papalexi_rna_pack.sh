#!/usr/bin/env bash
set -euo pipefail
bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_run_dataset_pack.sh" papalexi_arrayed_rna "$@"
