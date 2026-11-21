#!/bin/bash
# Fresh adaptive-only search; repeat the same command to resume.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/run_allocation_campaigns.sh \
  --out examples/adaptive_production_20261121_v2 \
  --policies adaptive --repeats 10 --hours 60 --workers 10 \
  --runtime-hours 72 --seed 20261121 "$@"
