#!/usr/bin/env bash
set -euo pipefail
trajectory_root="$(cd "$(dirname "$0")/.." && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
exec "$trajectory_root/.venv/bin/python" "$trajectory_root/analysis/physics_trajectories/run.py" "$@"
