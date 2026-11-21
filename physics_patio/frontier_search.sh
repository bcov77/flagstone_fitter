#!/usr/bin/env bash
set -euo pipefail
frontier_root="$(cd "$(dirname "$0")/.." && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
exec "$frontier_root/.venv/bin/python" -u "$frontier_root/analysis/frontier_search/run.py" "$@"
