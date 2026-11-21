# Adaptive production performance audit — November 21, 2026

The next production search benefits from exact CPU geometry improvements. The
current adaptive path does not invoke the earlier Metal/MLX batch scorer. Its
parents are construction-only, so adding GPU classical refinement would change
the allocation policy that won the prospective comparison.

## Evidence and accounting

Read all saved operation receipts from the four completed adaptive campaigns:
**56.291 worker-hours**, 48 fresh parents, 1,756 proposal searches and 1,496
physical evaluations. These are complete campaign costs, rather than simulated
physics seconds. Final whole-run verification is additional overhead.

| Operation | Share of sampling worker time |
| --- | ---: |
| Proposal selection | 63.62% |
| Physical evaluation, including endpoint gap measurement | 30.19% |
| Native construction, including its endpoint measurement | 0.67% |
| Separate construction handoff audits | 0.57% |
| Worker/controller overhead | 4.95% |

Fifteen proposal operations hit their five-minute timeout. None of the physical
operations timed out. The timeout count does not identify the expensive function
inside those killed processes. Detailed profiles below use successful requests.

Evidence lives locally in `examples/adaptive_profile_20261121_v1/`; it is not
included in the source repository. `historical_costs.json` retains category
counts, costs and representative request paths. The baseline implementation is
available in Git commit `f7d9bda`. Saved production requests/results remain intact.

## Profiles and implemented changes

Replay saved requests in separate processes with the original seeds, boards,
configuration and full-size outlines. Profiles are diagnostic; compare speed
using separate unprofiled executions, since cProfile adds overhead.

The median-duration successful search spent 25.83 of 46.89 profiled seconds in
polygon buffers. A later champion-lineage search spent 11.56 of 18.85 seconds
there. Repeated original-outline copies and boundary/joint diagnostics also
contribute. Its 1,508 FFT convolutions together took only 0.176 seconds.

The median-duration physical operation spent 22.34 of 35.74 profiled seconds in
endpoint measurement, including 14.92 seconds finding nearest original stone
polygons. Actual simulation advancement took 1.03 profiled seconds; document
copying/export also costs time. These nested times overlap and cannot be added.

Two changes are implemented:

1. In `src/flagstone/quarter_gaps.py`, use a spatial index of original boundary
   segments for outside-point distances. Prepared polygon coverage preserves
   distance zero inside/on a stone, including correct handling of holes. This
   changes the distance lookup, not either segment-Voronoi diagram, local-maxima
   classification, tolerances or acceptance thresholds. An isolated complete
   dual-scale audit fell from 18.95 to 5.40 seconds (3.51x), with its entire
   returned dictionary exactly equal. Shapely's [spatial-index documentation](https://shapely.readthedocs.io/en/stable/strtree.html)
   describes the nearest-distance queries used here.
2. Region-ownership diagnostics reuse each exact stone-buffer boundary across
   regions of the same assessed board. The lazy cache ends with that assessment;
   it is never reused across changed boards or clearances.

| Matched full operation, without profiler | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Representative proposal search | 43.272 s | 41.901 s | 1.033x |
| Late proposal search | 17.119 s | 17.045 s | 1.004x |
| Representative physical evaluation, including measurements | 26.198 s | 12.032 s | 2.177x |

All three replays exactly match their historical saved result values, including
the complete proposal pool/selection or physical result and gap measurements.
Two additional physical replays (held-release and a late rejected free-pulse
attempt) also exactly match, covering both protocols and rollback behavior.
These are individual requests, not a statistical estimate over the entire
production workload. The small proposal-search difference may include timing
noise; most of its buffer cost remains. Applying the representative factors to
the historical category shares suggests roughly 18% less worker time per
equivalent work, or 23% more work in a fixed budget. That is an extrapolation,
not a measured full-campaign speedup or a coverage-success forecast.

## Worker count and GPU decision

The local machine reports 14 CPU cores (10 performance, 4 efficiency), with
48 GiB memory. A short fixed-work throughput test runs the same search followed
by the same physical operation in each independent worker. Every replay checks
its complete saved result. Process startup and IO are included.

| Concurrent workers | Completed request pairs per hour, extrapolated from short batch |
| --- | ---: |
| 1 | 126 |
| 5 | 578 |
| 10, initial sample | 923 |
| 10, repeat after the bounded smoke campaign ended | 1,025 |
| 14 | 1,112 |

The initial scaling sweep overlapped the bounded numerical smoke campaign;
the repeated ten-worker measurement and fourteen-worker measurement remove
that extra numerical load. Other desktop activity and short-batch variation
remain. The cleaner 10/14 comparison suggests about 8.5% more aggregate throughput
at fourteen, but a request pair takes roughly 45 rather than 35 seconds per
worker. Thus fourteen uses about 29% more summed worker wall time per matched
pair. It also requires at least fourteen independent campaign streams: this
scheduler allows only one active action per campaign.

Keep the agreed **ten streams / ten workers / 600 worker-hours** for the prepared
production command. Fourteen is an option when maximizing throughput per elapsed
hour is the priority, but it is not a free improvement at the current worker-time
budget and changes timed-probe behavior. The short tests do not establish a
universal optimal concurrency or long-duration thermal performance.

GPU work is not currently the best-supported next optimization:

- The existing GPU scorer in `src/flagstone/quarter_batch.py` accelerates dense
  batches in older classical searches. This production allocation skips them.
- Polygon buffers, Boolean overlays and irregular spatial-index queries dominate
  the measured search. The existing MLX scorer does not implement those operations.
- Raster/FFT proposal scoring could be batched on a GPU, but its measured share
  in the late sample was below 1% of search time. A GPU rewrite of that slice
  alone has little end-to-end headroom, even before transfers and scheduling.
- Accelerating the physical integrator alone would address about one profiled
  second in a 36-second operation. Replacing the backend would also require
  revalidating contacts, quiet integration and acceptance; it is a separate
  numerical-policy experiment. Pymunk's [API documentation](https://www.pymunk.org/en/latest/pymunk.html)
  also notes that its default single solver thread preserves determinism; the
  current independent-process arrangement is retained.

A future batched GPU geometry/planning implementation may be worthwhile, but
the present evidence favors CPU geometry and reduced copying first. No GPU
speedup is claimed from these changes, and no audit or physical gate is skipped.

## Validation and production readiness

Regression checks cover exact distances for concavities, holes, overlapping
stones, boundaries, empty inputs and 2,000 seeded random points, plus exact
complete-audit parity and cache reuse without changed ownership. Existing gap,
GPU-score parity, physics-engine, planning, aggressive-selector, transaction and
allocation checks also pass.
There are **127 passing focused checks** across these groups.

A separate fresh adaptive smoke campaign used a 90-worker-second nominal budget
and completed two accepted physical transactions. Actual cost was approximately
119 worker-seconds after transaction-boundary overshoot, and its full verifier
passed. It is a bounded integration check, not a complete adaptive promotion
batch or a performance estimate. Earlier promotion/checkpoint tests cover those
transitions. Its output is `examples/adaptive_profile_20261121_v1/real_smoke/`.

Because numerical code pins changed, preserve the original, unstarted prepared
production directory `examples/adaptive_production_20261121_v1/`. The same user
command now prepares/resumes a fresh **v2** directory:

```bash
bash scripts/run_adaptive_production.sh
```

The budget, horizon, selection probabilities and physical policy stay fixed.
Faster operations can perform more attempts before a timed checkpoint, so full
campaign trajectories may differ even when individual request results agree.
The 600-worker-hour cap remains; the expected benefit is more search in that
budget. The three-day production search has not been started.

## Reproducing the audit

```bash
.venv/bin/python scripts/profile_adaptive.py \
  --inventory examples/allocation_campaigns_20261120_v1 \
  --out examples/my_profile/historical_costs.json

.venv/bin/python scripts/profile_adaptive.py \
  --request PATH_TO_SAVED_OPERATION/request.json.gz \
  --out examples/my_profile/operation.json

# Use --no-profile for ordinary operation timing and complete-result parity.
.venv/bin/python scripts/benchmark_adaptive_workers.py \
  --request PATH_TO_SELECT/request.json.gz PATH_TO_PHYSICS/request.json.gz \
  --workers 1 5 10 --out examples/my_profile/worker_scaling
```

These tools require the local saved inputs. Native-constructor replay is refused
because it writes artifacts. Sampling profiles and throughput tests are outside
production budgets. Never write profiling output into an existing campaign.
