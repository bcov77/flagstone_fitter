# Working on the patio solver

This repository explores packing irregular, full-size stones into a patio, with
clearance, coverage and maximum-gap checks. It contains both reusable solver
code and experimental campaign drivers. The working folder also holds large
private inputs and historical outputs; those are not needed to review the code.

The local Git repository tracks explicitly selected source and documentation.
No `.gitignore` is required: data and results remain untracked. Add individual
files deliberately rather than using `git add .`. Use `git status -uno` to see
changes to tracked files without listing the untracked workspaces. A fresh clone
contains the code described below, but does not include private inputs or runs.

## Where to start

| Location | Responsibility |
| --- | --- |
| `src/flagstone/` | Core geometry, layouts, inventory, placement, search, relaxation, checkpoints, quarter-patio search, scoring and gap audits. |
| `src/flagstone/phase1.py` | Layout representation, JSON loading/saving, synthetic examples and geometric validation. |
| `src/flagstone/quarter.py`, `quarter_search2.py`, `quarter_gaps.py` | Quarter-patio construction/refinement and gap measurements. |
| `physics_patio/patio_physics/` | Pymunk rigid-body backend, recording and browser editor assets. |
| `analysis/physics_trajectories/` | Geometry proposals, physical transactions, state restoration, measurements and verification. This is runtime code despite its directory name. |
| `analysis/aggressive_trajectories/` | Current aggressive selector, eligibility rules, physical evaluation and policy configuration. |
| `analysis/physics_placement_campaign/` | Shared boundary/contact/near-fit geometry imported by trajectory search, plus older experiment tools. |
| `scripts/full_trajectories.py` | Isolated numerical operations, bounded timeouts, immutable receipts, code/input pins, checkpointing and integrity verification. |
| `scripts/allocation_campaigns.py` | Worker-budget allocation, independent campaign streams, adaptive probe promotion and reporting. |
| `scripts/quarter_native.py` | Subprocess bridge to the external PackingSolver constructor. |
| `tests/` | Core tests. Some integration tests require omitted real inputs or historical outputs. |
| `analysis/allocation_campaigns/`, `analysis/full_trajectories/` | Focused scheduler/transaction tests and protocol documentation. |

Current production flow:

```text
run_adaptive_production.sh
  -> allocation_campaigns.py (campaign scheduling and budgets)
    -> full_trajectories.py (bounded operations and durable receipts)
      -> quarter_native.py + external PackingSolver (fresh construction)
      -> aggressive_trajectories + physics_trajectories (propose and evaluate)
        -> flagstone geometry + patio_physics engine
    -> final integrity verification and reports
```

Good first contributions include proposal search, geometry performance, gap
measurement and scheduler diagnostics. Preserve original outlines, unique stone
IDs, locks, minimum clearance, physical acceptance/rollback and receipt integrity
when changing numerical behavior. Start a new output directory after changing
pinned production code; existing runs intentionally reject changed code.

## Three-day production command on Brian's existing machine

From the original project root:

```bash
bash scripts/run_adaptive_production.sh
```

This selects adaptive only: ten independently seeded streams, 60 worker-hours
each, ten concurrent workers, and a 72-hour elapsed window per invocation.
Each batch uses two fresh construction-only parents, three 8-minute probes per
parent, then one exact-checkpoint continuation to 90 cumulative minutes.
Promotion remains 80% highest absolute coverage and 20% uniform among the other
five. Total sampling allocation is 600 worker-hours, with transaction-boundary
overshoot retained; final verification is additional overhead within the window.
This scales repetition of the tested policy without changing the physics policy.

Outputs: `examples/adaptive_production_20261121_v1/`. The shell wrapper prevents
idle sleep on macOS. Keep the machine powered and the terminal session running;
the wrapper is a foreground command. It does not prevent shutdown or lid sleep.
Use another terminal for inspection:

```bash
bash scripts/run_adaptive_production.sh --status
bash scripts/run_adaptive_production.sh --stop
bash scripts/run_adaptive_production.sh --verify
```

Ctrl-C also requests a safe checkpointed stop. Rerun the original command to
resume. Each invocation supplies another 72-hour elapsed window while retaining
the same total sampling budgets. `EXECUTION_COMPLETE.json` means simulation
budgets ended; `COMPLETE.json` means all final integrity verifiers passed. A
deadline or low disk space can leave work checkpointed and require resumption.
Do not edit pinned runtime code, change its environment, or run another search
against this output while it is active. Use an independent checkout for your
friend's development. Historical experiments and their archived drivers remain
separate; a changed driver is not an exact-command resume of an old experiment.

## Source-only handoff

Build a ZIP from the original project root:

```bash
python3 scripts/export_source.py
```

The archive is `exports/elliot_patio_source.zip`, containing a single
`elliot_patio_source/` root folder. Its README is this guide. The export uses an
explicit source allowlist: core package, top-level scripts, physics backend and
web assets, direct Python files in the analysis subdirectories, selected protocol
docs, the aggressive policy configuration, tests, dependency metadata and lockfile.
Keeping analysis paths avoids breaking existing imports while exposing the
experimental code to collaborators.

It excludes stone outlines/inventories, patio measurements, photographs, results,
recordings, code snapshots inside results, virtual environments, caches, vendored
segmentation models and image-extraction experiments. `generate_flagstones/` and
`generate_patio_outline/` are upstream image-processing workspaces, rather than
dependencies of the packing/physics pipeline, and are omitted. The exported
`.gitignore` keeps generated data and environments out of a new repository.

Unzip and use that folder as the GitHub repository root. No GitHub repository is
created or published by the export command. No license is selected automatically.

## Running without private data

Python 3.12+ is required. The existing package metadata/lockfile describe the
original environment and include MLX, used for acceleration on supported hardware.
For the original environment, use `uv sync`. Physics campaigns additionally need
`pymunk==7.3.1`; the browser sandbox also needs the dependencies in
`physics_patio/requirements.txt` (including aiohttp). A source-only checkout can
exercise synthetic examples without any real flagstone inputs:

```bash
uv sync
uv run python -m flagstone --case grid --output examples/synthetic_grid
uv run python -m unittest discover -s analysis/allocation_campaigns -p 'test_*.py'
```

For CPU development on a machine without MLX support, create a separate environment
and install the core dependencies explicitly, avoiding installation of the
original package's mandatory MLX dependency:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy scipy shapely matplotlib ortools pyvoronoi pytest pymunk==7.3.1
PYTHONPATH=src .venv/bin/python -m flagstone --case grid --output examples/synthetic_grid
.venv/bin/python -m unittest discover -s analysis/allocation_campaigns -p 'test_*.py'
```

The CPU setup is a development starting point, not a reproduction of the pinned
production environment. MLX-dependent tools still need supported hardware and MLX.

The production command requires inputs intentionally omitted from the archive.
Currently `scripts/full_trajectories.py` defines `DEFAULT_SOURCE` as the local
task-576 JSON (used for inventory and boundary, not a reused starting layout),
and `DEFAULT_NATIVE` as Brian's separate conda Python containing PackingSolver.
For another machine, supply a compatible inventory/boundary JSON in feet and
configure those paths before preparing a fresh campaign. The native environment
here has `packingsolver==0.1.1334`; its availability/build requirements vary by
platform. Production also loads `analysis/aggressive_trajectories/config.json`,
which is included. Synthetic examples and scheduler unit tests do not require
the external constructor or private inputs. A code-only handoff therefore supports
development immediately, but cannot reproduce the real patio search by itself.
