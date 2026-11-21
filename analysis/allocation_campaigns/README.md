# Prospective allocation comparison — November 20, 2026

## Adaptive-only production — November 21

The completed comparison favors adaptive as the next production allocation;
see the results in `first_real_patio_quarter_full_trajectories.md`. Run the new
prepared search from the repository root:

```bash
bash scripts/run_adaptive_production.sh
```

The wrapper selects `--policies adaptive --repeats 10 --hours 60 --workers 10
--runtime-hours 72 --seed 20261121`, with a new output directory at
`examples/adaptive_production_20261121_v1/`. This means 600 allocated worker-hours
across ten streams, plus separately timed final verification. It uses the
unchanged adaptive batch/promotion and numerical policies described below.
It is prepared, with no simulation jobs started. The wrapper runs in the
foreground under macOS idle-sleep prevention. Append `--status`, `--stop`, or
`--verify` as needed; rerun the original command to resume. Each invocation has
a fresh 72-hour window but retains the same total sampling budget. Deadline or
disk-reserve stopping can require resumption. Do not change pinned runtime code
or libraries during production.

The generic runner now accepts `--policies broad multiple adaptive` (default)
or a selected subset. Fourteen allocation tests and the 17 transaction tests
pass. Previous completed experiments retain their archived drivers; this new
driver revision belongs in a new output directory.

## Original comparison command

Run or resume from the repository root:

```bash
bash scripts/run_allocation_campaigns.sh
```

Production output: `examples/allocation_campaigns_20261120_v1/`.
The default is **10 workers, 12 independent campaigns, 14 worker-hours each**:
168 allocated worker-hours, with a 20-hour elapsed-time window per invocation.
Do not run a second copy while the output lock is held.

## Predeclared comparison

Four independently seeded campaigns per policy, with fresh construction-only
parents from the original inventory. No calibration boards or physics endpoints
are reused. All policies retain the existing aggressive normal physics selector
and original-outline physical constraints.

| Policy | Allocation cycle |
| --- | --- |
| `broad` | Construct a fresh parent, run one physics continuation to 30 minutes, repeat. |
| `multiple` | Construct a fresh parent, run three independent continuations to 30 minutes each, repeat. |
| `adaptive` | Construct two fresh parents; run three 8-minute probes per parent; select one of the six exact checkpoints and continue it to a cumulative 90 minutes; repeat. |

Adaptive promotion uses highest **absolute best coverage** with 80% probability;
with 20% probability it selects uniformly from the other five probes. Ties break
by key. Selection seeds and the full candidate ranking are recorded before
dispatch. The longer continuation uses the original checkpoint, RNG derivation,
and operation sequence; it does not restart from the parent. The opening summary
is archived under `stages/` before extension. Short probes also save 2-minute
observations; fixed continuations save 2- and 8-minute observations.

Campaigns have separate seeds and inventories copied from the same original
geometry. Parents are not shared across campaigns. Scheduling rotates ready
campaigns through a deterministic shuffled order, with at most one active action
per campaign and ten workers overall. This avoids serializing entire policies
under different machine loads. Four campaigns per policy provide a prospective
comparison, with substantial uncertainty about rare successes.

## Budget and stopping

Charge native construction, boundary audits, every proposal search and physical
operation (including failures/timeouts), worker session overhead, preparation,
and controller/report work. Shared parents and continuation prefixes are charged
once. Physical simulated time and attempt counts do not substitute for compute.
The last continuation horizon is shortened to the remaining campaign budget.
Like the calibration, a selected physical transaction finishes before sampling:
**budgets are soft at transaction boundaries**. Actual costs and overshoot are
retained in every campaign report and must accompany policy comparisons.

The scheduler stops dispatching 11 minutes before the 20-hour deadline, allowing
two bounded five-minute operations and shutdown. A stopped campaign can be
resumed with the same command; committed targets and decisions do not change.
Each invocation opens another 20-hour execution window, while preserving all
worker budgets. A 20-GiB disk reserve requests a safe stop. Unexpected worker
errors stop scheduling and save diagnostics. The runtime can finish with partial
campaigns if physical overhead exceeds the allocation estimate; those campaigns
must not be treated as completed fixed-budget replicates.

Full post-run integrity verification is separately timed overhead, outside the
168-hour sampling budget but within the remaining 20-hour elapsed window.
It may be expensive. `EXECUTION_COMPLETE.json` means all campaign budgets ended;
`COMPLETE.json` exists only after all final verifiers pass. Verification completed
for each campaign is retained; an audit interrupted by the elapsed deadline is
repeated for that campaign when resumed. The verifier checks geometry, receipts,
chains, physical gates, exports and time boundaries; saved gap audits are not
independently recomputed. Coverage alone never establishes a finished patio.

## Inspect, stop, resume

```bash
.venv/bin/python scripts/allocation_campaigns.py --status
.venv/bin/python scripts/allocation_campaigns.py --stop
bash scripts/run_allocation_campaigns.sh
```

Ctrl-C and SIGTERM request the same checkpointed stop during execution. Do not
force-kill numerical workers. `--workers` may change on resume; sampling settings
and driver/code/library hashes are pinned. To finish only the final verification:

```bash
bash scripts/run_allocation_campaigns.sh --verify
```

The root `report.txt`/`report.json` summarize each campaign's actual costs, best
coverage, parent/continuation counts and status. Each campaign contains its own
`allocation_state.json` (including committed promotion decisions and pending
action), `allocation_report.json`, action completion markers, immutable operation
requests/results/receipts, timed samples, and original-outline layouts. Inspect
`parents/pNNN/handoff_00.json`, `physics/pNNN_h00_rRR/best_coverage.json` and the
separate `best_gaps.json`. The process transcript is `run.log` when launched by
the assistant. `launch.json` records the current process and deadline.

Compare best coverage and target success across the four complete campaigns per
policy, alongside actual cost/overshoot and gap diagnostics. Preserve individual
root/campaign identities; siblings are correlated. Do not select a policy solely
from its average trajectory gain or count incomplete campaigns as full budgets.

## Validation

```bash
.venv/bin/python -m unittest discover -s analysis/allocation_campaigns -p 'test_*.py' -v
.venv/bin/python -m unittest discover -s analysis/full_trajectories -p 'test_*.py' -v
```

The allocation suite covers remaining-budget horizons, prefix charging, failures,
pending dispatch recovery, complete-batch promotion, reproducible exploration,
exact-directory continuation, opening-summary preservation, and completion gates.
The existing 17 transaction tests cover physical rollback, timeout and immutable
receipt recovery. Real numerical validation is separate from production under
`examples/allocation_validation_20261120_v1/`.

**11 allocation checks and all 17 existing transaction checks pass.** The real
three-policy validation completed and passed all three full verifiers, with seven
accepted physical transactions. A safe stop during proposal selection followed
by exact-command resume retained pending selections and byte-identical native
receipts. Evidence: [`real_validation.json`](real_validation.json).

A separate copied validation trajectory was continued from its saved physics
checkpoint, adding a fourth accepted transaction while retaining every earlier
receipt byte hash. Its full verifier passed. Evidence:
[`extension_validation.json`](extension_validation.json); output:
`examples/allocation_extension_validation_20261120_v1/`. Production boards are
fresh and do not reuse either validation. The first validation archives its
driver revision; later changes add safe audit stopping and final status refresh,
covered by the focused tests and the final warm-continuation validation.
