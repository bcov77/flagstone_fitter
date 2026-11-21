# Full-trajectory sampling calibration

The single entry point is `scripts/full_trajectories.py`. From the repository root:

```bash
caffeinate -i .venv/bin/python scripts/full_trajectories.py
```

It runs in the foreground, stops when the finite design completes, verifies the
saved results, and writes `COMPLETE.json`. It has not been launched for the full
production experiment by the assistant. Small validation outputs are separate.

Default output: `examples/full_trajectories_20261119/`.

## What runs

- **12 independent classical lineages**, using fresh native constructions from
  the original 137-stone inventory on the same synthetic quarter-area square.
  Historical task 576 supplies original geometry, not a warm-start arrangement.
  Each finite native call has a 30-second ceiling and may finish early. Its
  highest-area original-clearance-valid pattern is the initial board.
- **Four classical handoffs:** construction only and approximately 1, 4, and 16
  minutes of classical refinement. Refinement uses the existing shifted-grid
  energy-based continuous repair and periodic three-stone pocket rebuilding.
  The refinement time does not include native construction or boundary audits;
  total worker costs include them. Longer refinement can improve gaps without
  adding coverage, which is one reason to test these handoffs downstream.
- **Three independent physics continuations per handoff:** 144 total. Physics
  uses the existing aggressive proposal/relaxation policy with normal sampling
  throughout. It preserves original full-size outlines, inventory, half-inch
  clearance, >=45-degree deliberate reorientation and no net downsizing.
  There is no Golden/Silver policy comparison in this calibration.
- **Physics observations near 2, 8, and 30 worker-minutes.** A quarter of the
  continuations, chosen randomly before seeing results, continue to 90 minutes.
  No parent or continuation is pruned because of early apparent performance.

The first experiment deliberately holds the physics policy fixed. It measures
time allocation and the possible opening/finishing transition. It does **not**
yet compare a new frontier-finishing operator policy or branch multiple children
from each late intermediate board. The previous late-board frontier results
remain evidence for a subsequent matched finishing experiment.

Nominal physics allocation is **108 worker-hours**: 72 for the first 30 minutes
of all 144 continuations and 36 for the random extensions. Classical refinement
adds about 3.2 hours, plus construction, audits, process startup, records, and
transaction overshoot. Eight workers suggest roughly 14+ elapsed hours; this
is an allocation estimate, not a measured completion promise.

Use fewer workers if desired; this is the only execution setting that can change
on resume without changing the experimental design:

```bash
caffeinate -i .venv/bin/python scripts/full_trajectories.py --workers 4
```

All source/configuration/library pins are checked before resuming. Original
compute and inputs are copied under `code/` and `source.json`. Do not modify
pinned code during the run. Changes require a new output directory/version.

## Stop, resume, and inspect

Ctrl-C asks workers to finish the current bounded numerical operation, save
their state and exit. Rerun the **same command** to resume; workers do not reset
their budgets or discard selected proposals. A selected physical transaction
may remain pending at a stop and is executed with the same seed on resume.

From a second terminal:

```bash
.venv/bin/python scripts/full_trajectories.py --status
.venv/bin/python scripts/full_trajectories.py --stop
```

The status report refreshes about every 30 seconds while active; the running
operation's time is committed at its boundary. Selection and physical operations
each have a five-minute hard cap; timeout handling discards any uncommitted
partial result. Ctrl-C can therefore take several minutes to finish cleanly.
Empty/failed proposal searches consume compute time and do not count as physical
attempts. Physical timeouts count as evaluated attempts and leave the board intact.
Unexpected software errors stop scheduling and preserve diagnostics in `errors/`.

A 10-GiB free-disk reserve requests a safe stop. This is a reserve, not an
estimated total output size. Full candidate, rejected-endpoint, and geometry
records can occupy substantial space. The script never deletes historical runs.

Force-killing processes can leave a pending operation. A surviving child PID
causes recovery to refuse overlapping work; inspect it before retrying. A dead
interrupted operation can be replayed with its request and seed preserved.
Its observed heartbeat cost is retained and explicitly marked as a lower bound.
Normal Ctrl-C/resume is the preferred path.

## Budget semantics

Times are **active worker wall seconds**, not simulated physics seconds or
attempt counts. Native work, proposal generation, empty searches, subprocess
startup, physics, audits, timeouts, and recorded orchestration overhead are
accounted for. Operation CPU seconds are also saved, but killed-operation CPU
is unknown and is explicitly counted as missing. Parallel CPU time is not
substituted for worker wall time.

Sampling horizons are soft boundaries: a selected proposal is physically tested
before the continuation records its next sample. A 2-minute sample might
therefore be committed after 4 minutes. Several boundaries may refer to the same
state if one transaction crosses them. Use **actual recorded times**, not nominal
labels, for allocation comparisons. No timed-out operation is silently accepted.

Each operation saves immutable input, output and cost receipts before its state
transition commits. Resume can reuse completed operations without double-counting
their cost. All attempted endpoints and current/best boards remain reviewable.

## Results to interpret afterward

- `report.txt` / `report.json`: completion counts, aggregate costs and coverage
  champion. `COMPLETE.json` exists only after all jobs complete and verification
  succeeds. Partial runs have reports too; absence of COMPLETE matters.
- `parents.csv`: handoff coverage/gaps and actual construction/refinement costs.
- `samples.csv`: each continuation's best coverage, gap diagnostics, actual time,
  acceptance, no-proposal and timeout counts at each observation horizon.
- `parents/pNNN/handoff_HH.json`: full original-outline handoff boards.
- `physics/pNNN_hHH_rRR/`: attempts, samples, checkpoint, source handoff, final and
  best-coverage layouts, plus a separate `best_gaps.json` champion ranked by the
  normalized interior/edge gap maximum. Full intermediate poses remain in the
  operation/attempt records; timed samples contain small metrics and references.
- `operations/op_NNNNNN/`: exact requests, candidate pools or physical outcomes,
  seeds, logs, hashes, and cost receipts. `sessions/` records orchestration time.
- `verification.json`: integrity and physical acceptance checks. Stored gap
  audits are retained, not independently recomputed by this verifier.

Regenerate or check results after stopping:

```bash
.venv/bin/python scripts/full_trajectories.py --report
.venv/bin/python scripts/full_trajectories.py --verify
```

The three continuation replicate seeds are matched across handoff times within
a classical lineage. Different lineages and replicates have distinct seeds;
related handoffs remain correlated. Compare parent versus continuation variance,
downstream outcome versus early features, fixed versus inactivity stopping, and
best coverage at a complete fixed campaign budget. Charge shared classical
prefixes once when replaying allocation policies. Aggregate worker cost of the
calibration includes every sampled child and must not be confused with the cost
of producing its champion in isolation.

All intermediate boards remain in operation/attempt records, including useful
gap improvements with unchanged area. Coverage alone does not establish a
finished patio; retain the 2-inch interior / 5-inch edge diagnostics separately.
The calibration has only 12 independent roots; rare-tail policy claims require
fresh prospective campaigns after fitting allocation rules.

## Validation

```bash
.venv/bin/python -m unittest discover -s analysis/full_trajectories -p 'test_*.py' -v
```

Tests cover balanced identities and seeds, invalid horizons, empty-search cost,
selection and physical timeouts, rejection rollback, pending-proposal stop/resume,
random extensions, receipt reuse and tampering, completed-child crash recovery,
real subprocess termination, compute-pin changes, compact pose reconstruction,
report caching/accounting, gap champions, and zero-attempt verification.

**17 tests pass.** Real-board validation in
`examples/full_trajectories_validation_20261119_v2/` exercised two classical
handoffs and two accepted physical transactions, including an actual SIGINT
during native construction followed by exact-command resume. The committed native
receipt was byte-identical after resume. Integrity checks passed for all eight
operation receipts, original outlines/coverage, clearance, eligibility, choice
probabilities, trajectories, rollback, consecutive pre-stop settling, exported
champions and actual time boundaries. See
[`real_validation_v2.json`](real_validation_v2.json) and its companion transcript.
The numerical validation revision is archived with that output; subsequent
changes only optimize derived report caching and status formatting. This small
validation establishes operation and recovery, not statistical performance.
