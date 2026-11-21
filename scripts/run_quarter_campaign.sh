#!/usr/bin/env bash
# User-submitted foreground campaign; no detaching or automatic submission.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec .venv/bin/python scripts/quarter_campaign.py --help
fi
task_output=""
task_resume=false
task_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output|--output=*)
      echo 'Pass the directory positionally, or use --resume DIRECTORY.' >&2; exit 2 ;;
    --resume|--report-only)
      task_resume=true
      task_args+=("$1")
      shift
      if [[ $# -gt 0 && "$1" != --* && -z "$task_output" ]]; then
        task_output="$1"; shift
      fi ;;
    --cores|--hours|--heartbeat|--trial|--native-python)
      if [[ $# -lt 2 ]]; then echo "Missing value for $1" >&2; exit 2; fi
      task_args+=("$1" "$2"); shift 2 ;;
    --*) task_args+=("$1"); shift ;;
    *)
      if [[ -n "$task_output" ]]; then echo "Unexpected argument: $1" >&2; exit 2; fi
      task_output="$1"; shift ;;
  esac
done
if [[ -z "$task_output" ]]; then
  if [[ "$task_resume" == true ]]; then
    echo 'Specify the campaign directory: --resume examples/quarter_campaign_...' >&2
    exit 2
  fi
  task_output="examples/quarter_campaign_$(date +%Y%m%d_%H%M%S)_$$"
fi
# Naming an existing campaign means continue it, not overwrite or start over.
if [[ "$task_resume" == false && -f "$task_output/state.json" && -f "$task_output/plan.json" ]]; then
  task_resume=true
  task_args+=(--resume)
fi
if [[ "$task_resume" == false ]]; then
  mkdir "$task_output"
fi
# Keep macOS awake while this foreground command exists; terminal must stay open.
task_awake=()
if [[ -x /usr/bin/caffeinate ]]; then
  task_awake=(/usr/bin/caffeinate -i)
fi
"${task_awake[@]}" .venv/bin/python -u scripts/quarter_campaign.py --output "$task_output" "${task_args[@]}" 2>&1 | tee -a "$task_output/console.log"
