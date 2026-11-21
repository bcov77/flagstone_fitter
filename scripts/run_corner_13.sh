#!/usr/bin/env bash
# Foreground controller run with readable live logs and a durable tee transcript.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec .venv/bin/python scripts/watch_corner_run.py --help
fi
if [[ $# -gt 0 && "$1" != --* ]]; then
  task_output="$1"
  shift
else
  task_output="examples/watched_controller_13_$(date +%Y%m%d_%H%M%S)_$$"
fi
for task_arg in "$@"; do
  if [[ "$task_arg" == --output || "$task_arg" == --output=* ]]; then
    echo 'Pass the output directory as the first positional argument.' >&2
    exit 2
  fi
done
mkdir "$task_output"
.venv/bin/python -u scripts/watch_corner_run.py --output "$task_output" "$@" 2>&1 | tee "$task_output/console.log"
