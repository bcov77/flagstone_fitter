#!/usr/bin/env bash
set -euo pipefail
physics_root="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${physics_root}${PYTHONPATH:+:${PYTHONPATH}}"
exec "$physics_root/.venv/bin/python" -m patio_physics.server "$@"
