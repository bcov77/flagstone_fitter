# Quarter patio: sampling complete construction-to-physics trajectories

## Performance audit before production — November 21, 2026

The four completed adaptive campaigns spent 63.62% of sampling worker time on
proposal search, 30.19% on physical evaluations including endpoint measurements,
0.67% on native construction, 0.57% on separate handoff audits and 4.95% on
worker/controller overhead. This construction-only production path does not
invoke the GPU batch scorer from earlier classical searches.

Profiles identify polygon buffering in selection and original-polygon nearest
queries in the complete gap audit as major costs. An exact boundary-segment
distance index, retaining zero distances inside/on polygons, reduced one full
dual-scale gap audit from 18.95 to 5.40 seconds, with exactly equal returned
candidates and measurements. Reusing exact buffer boundaries within one fixed
board assessment gives a smaller selection improvement. No original outlines,
diagrams, tolerances, physical protocols or acceptance gates are changed.

Unprofiled matched full-operation replays: representative selection **43.272 ->
41.901 seconds**; late selection **17.119 -> 17.045 seconds**; representative
physical evaluation **26.198 -> 12.032 seconds**. All complete saved result values
match exactly. Additional held-release and rejected late-board physical replays
also match. These are request-level checks, not a measured whole-campaign
speedup. Historical cost shares suggest approximately 18% less worker time per
equivalent work, with substantial uncertainty; fixed-budget production uses the
savings for more attempts rather than reducing its 600-worker-hour cap.

Ten-worker fixed-work throughput was about 1,025 request pairs/hour in a clean
repeat; fourteen reached about 1,112, but consumed roughly 29% more worker wall
time per matched pair. Keep the agreed ten-stream/ten-worker allocation for the
current worker-time budget. These short measurements do not establish optimal
concurrency under every workload or long-duration thermal conditions.

The GPU's existing dense classical scoring path is inactive here. The late
selection sample spent under 1% of its time in FFT convolution; simulation
advancement was about one of 36 profiled physical-evaluation seconds. Moving
those small slices to the GPU has limited end-to-end headroom. Faster GPU
geometry/planning remains possible future work, requiring a separate backend
and validation rather than enabling the old scorer in this policy.

**127 focused checks pass**, including new exact-distance/cache regressions and
the existing geometry, physics, planning, aggressive policy, allocation and
transaction checks. A separate fresh bounded adaptive campaign completed two
accepted physical transactions and passed its full verifier. The production
wrapper now uses `examples/adaptive_production_20261121_v2/`; the unstarted v1
preparation is retained because code pins changed. The 72-hour run is still
**not launched**. Full methods, timing limits and reproduction commands:
[`analysis/adaptive_performance/README.md`](analysis/adaptive_performance/README.md).

## Completed allocation comparison — November 21, 2026

All 12 prospective campaigns completed, and all 12 final integrity verifiers
passed. Production sampling consumed **168.712 worker-hours** against 168
allocated, including retained transaction-boundary overshoot. Final verification
added **1.356 elapsed hours** of separately timed overhead. Evidence:
[`report.txt`](examples/allocation_campaigns_20261120_v1/report.txt),
[`report.json`](examples/allocation_campaigns_20261120_v1/report.json), and
[`COMPLETE.json`](examples/allocation_campaigns_20261120_v1/COMPLETE.json).

Each policy had four independent campaigns with a nominal 14 worker-hours each.
The primary outcome below is each campaign's best saved coverage, rather than
average endpoint coverage across its continuations.

| Policy | Mean campaign-best coverage | Best coverage | Campaigns reaching 80% | Actual worker-hours |
| --- | ---: | ---: | ---: | ---: |
| Adaptive: six 8-minute probes across two parents, extend one checkpoint toward 90 minutes | 80.749% | 81.518% | 4/4 | 56.291 |
| Broad: one 30-minute continuation per fresh parent | 80.327% | 81.205% | 3/4 | 56.126 |
| Multiple: three 30-minute continuations per fresh parent | 79.478% | 80.098% | 1/4 | 56.295 |

Adaptive is the leading policy to carry forward: mean campaign-best coverage
exceeded broad by 0.422 percentage points and multiple by 1.271 points. Broad's
advantage over multiple suggests fresh-parent breadth was more useful than
spending the same budget on three short continuations per parent. Adaptive's
result supports screening followed by deeper continuation as a complete policy;
this comparison does not isolate the effects of screening, depth, or exploration.
Four campaigns per policy leave substantial uncertainty about rare successes
and do not establish an optimal schedule or exploration fraction.

The champion is `adaptive_01/physics/p011_h00_r01`, with **81.517802% coverage**
and saved gap upper estimates of **5.374 inches interior** and **4.794 inches
edge**. It remains below the previous **82.625%** coverage record and fails the
2-inch interior target. None of the 12 campaign coverage champions meets the
interior target. The final verifiers check original-outline geometry, coverage,
clearance, physical gates, receipts, chains, exports and time boundaries; saved
gap audits are not independently recomputed.

### Proposed multi-day production run

Use the tested adaptive policy unchanged for the next production search: fresh
construction-only parents, three independent 8-minute probes per parent, two
parents per selection batch, then one exact-checkpoint continuation to a
cumulative 90 minutes. Preserve the tested 80% highest-absolute-coverage / 20%
uniform-other-candidate promotion rule and the current aggressive normal physics
policy. Repeat batches throughout the run, retaining every best-so-far board.
This is the best-supported next allocation, not a claim of global optimality.

A concrete starting proposal is **10 independent adaptive campaign streams**, up
to **10 concurrent workers**, and **60 worker-hours per stream**: 600 nominal
worker-hours, with a **72-hour elapsed window** for scheduling, overhead, safe
checkpointing and final verification. Actual throughput determines completion;
600 worker-hours at full occupancy is already 60 elapsed hours. Report actual
sampling cost, overshoot and verification overhead separately. Use fresh seeds
and a new versioned output directory; preserve all earlier experiments.

Keep the selection and numerical policy fixed during this production search.
Save timed checkpoints, promotion rankings and exploration decisions, immutable
receipts, rejected transactions, and separate coverage/gap champions. Retain
safe stop/resume, pinned code/configuration/inputs, idle-sleep prevention and
disk-reserve stopping. End with full integrity verification. Crossing a coverage
target alone does not establish gap compliance or a finished patio.

The production command is now implemented and prepared, **not launched**:

```bash
bash scripts/run_adaptive_production.sh
```

It selects adaptive only, ten streams at 60 worker-hours each, ten workers, a
72-hour elapsed window and seed 20261121. Outputs are separate under
`examples/adaptive_production_20261121_v2/`. Add `--status`, `--stop`, or `--verify`
for inspection, safe stopping or final verification; rerun the original command
to resume. Preparation verified local input, code, library and native-constructor
pins, with zero construction or physics jobs executed. All 14 allocation checks
and 17 existing transaction checks pass. Adaptive-only preparation, independent
seeds, pinned resume and preservation of the default three-policy comparison
are covered by the allocation checks.

Scaling total campaign budgets repeats the existing 90-minute trajectory policy;
extending individual trajectories for many hours or branching historical
champions would be separate, untested policy changes. See
[`PROJECT_LAYOUT.md`](PROJECT_LAYOUT.md) for operations and a source-only
collaborator handoff. The production environment should stay fixed while running;
development belongs in a separate checkout.

## Prospective allocation comparison — November 20, 2026

The overnight calibration produced all 12 classical lineages, 144 physics
continuations, and 468 timed observations, accounting for 114.512 worker-hours.
Its saved champion is 81.391736% coverage (`p000_h00_r00`, construction only,
90-minute continuation), with a 5.745-inch interior and 4.484-inch edge gap.
It does not meet the interior gap target or exceed the previous 82.625% record.
The whole-run integrity verifier was started separately; completion of simulation
jobs must not be confused with a final `COMPLETE.json` marker.

At the 30-minute physics horizon, mean downstream coverage was 77.685%, 77.451%,
76.596%, and 75.438% for construction-only, 1-, 4-, and 16-minute classical
handoffs. The current gap-energy refinement traded area for gaps: mean classical
handoff coverage fell from 69.085% to 65.483% by 16 minutes. Longer refinement
underperformed construction-only in 11 of 12 lineages at 30 minutes, averaging
over seeds. These are related lineage comparisons, not 144 independent roots.

All 36 randomly extended continuations improved between 30 and 90 minutes;
mean gain was 2.146 percentage points, median 1.606, and 12 reached 80% coverage.
No continuation reached 80% at the 30-minute observation. Across the three seeds
per handoff, the mean coverage range at 30 minutes was 2.124 points. Early weak
runs could recover substantially. These observations motivate a prospective
allocation comparison; they do not establish the optimal schedule.

The new separate entry point is:

```bash
bash scripts/run_allocation_campaigns.sh
```

It uses **10 workers** for **four independent campaigns per policy**, with
**14 worker-hours per campaign** (168 allocated worker-hours total), and a
**20-hour elapsed window per invocation**. Every policy uses fresh
construction-only parents and the existing aggressive normal physics policy.

1. **Broad:** one 30-minute continuation per fresh parent, then a new parent.
2. **Multiple:** three independent 30-minute continuations per parent.
3. **Adaptive:** two fresh parents with three 8-minute probes each; extend one
   exact checkpoint to a cumulative 90 minutes. Select highest absolute coverage
   with 80% probability, or uniformly among the other five with 20% probability.

Charge parents and shared continuation prefixes once, include failed searches
and all recorded worker/controller costs, and shorten the final continuation to
the remaining allocation. Physical transactions can overshoot a boundary;
compare actual costs and retain overshoot explicitly. Final full verification
is separately timed overhead within the available elapsed window. A conservative
dispatch cutoff allows bounded operations to checkpoint before the deadline.

Production outputs are separate under
`examples/allocation_campaigns_20261120_v1/`. Driver, numerical code, library,
source and configuration pins prevent silent changes on resume. Seeds,
promotion rankings, exploration decisions, pending actions, original-outline
boards, rejected transactions and receipt hashes are retained. Full details and
status/stop/resume commands are in
[`analysis/allocation_campaigns/README.md`](analysis/allocation_campaigns/README.md).
The old experiment and its pinned numerical code remain intact.

**Launched November 20 at 08:36 Mountain time**, detached with idle-sleep
prevention. The elapsed deadline is approximately November 21 at 04:37 Mountain
time. `run.log`, `launch.json`, and `report.txt` in the production directory
record the live process and progress. Validation: 11 allocation checks and the
17 existing transaction checks pass; three bounded real campaigns passed their
full verifiers after stop/resume, and a separate saved-checkpoint continuation
passed with a new accepted transaction and unchanged earlier receipt hashes.

## Calibration runner — November 19, 2026

The single finite foreground entry point is
[`scripts/full_trajectories.py`](scripts/full_trajectories.py):

```bash
caffeinate -i .venv/bin/python scripts/full_trajectories.py
```

Default output: `examples/full_trajectories_20261119/`. The full experiment is
prepared for the user to launch; the assistant has run only separate bounded
validation experiments. Ctrl-C requests a checkpointed stop; the same command
resumes the same jobs and budgets. `--status`, `--stop`, `--report`, and
`--verify` provide inspection, stopping, report regeneration, and integrity checks.

The implemented calibration uses **12 fresh classical lineages**, saves
**construction-only plus 1/4/16-minute refinement handoffs**, and gives each
handoff **three physics continuations**. Adding the construction-only control
to the three proposed refined handoffs makes **144 continuations**, rather than
108. Physics observations are near **2/8/30 worker-minutes**; a preselected random
quarter continue to **90 minutes**. This allocates **108 nominal physics
worker-hours**, plus about 3.2 classical refinement hours, construction, audits,
orchestration and bounded transaction overshoot. Eight workers are the default.

The classical phase uses the existing finite native constructor followed by
shifted-grid energy refinement and periodic pocket rebuilding. The physics
phase holds the existing aggressive normal-sampling policy fixed throughout.
This first experiment measures duration, parent quality and continuation
variability; a changed frontier-finishing policy and multiple children per late
physics checkpoint remain subsequent controlled comparisons.

Budgets count active worker wall time. Proposal selection and physical evaluation
have separate hard time caps; empty searches and timeouts consume time and remain
visible. Observation boundaries are soft: a selected physical transaction finishes
before recording a sample, and actual times are exported. Some short horizons
can share an endpoint after one expensive transaction crosses both boundaries.
Operation requests/results preserve compact rigid poses against frozen original
geometry, selected proposals, candidate pools, RNG seeds, rejected endpoints,
cost receipts and byte hashes. Unexpected software errors stop scheduling.

See the [runner instructions](analysis/full_trajectories/README.md) for full
accounting, resume semantics, validation and output interpretation. The initial
research design below remains the rationale, not a claim that its provisional
parameters are optimal.

Validation: **17 focused checks pass**. A separate real-board experiment
completed two accepted physical transactions and passed its integrity verifier.
It also exercised actual Ctrl-C during construction, exact-command resume, and
byte-preservation of the committed construction receipt. Evidence:
[`analysis/full_trajectories/real_validation_v2.json`](analysis/full_trajectories/real_validation_v2.json).
These are integration checks, not calibration results. The full production
output is prepared with zero parent or physics jobs executed.

## Initial research design — November 19, 2026

This is the new working research log following
[the physics research log](first_real_patio_quarter_first_physics.md).
This historical entry proposes the experiments implemented in the runner above.
Keep the previous log and all historical experiments intact.

The question is how to allocate a fixed compute budget across independent
classical starts, classical refinement, physics continuations, and late-board
branches to maximize the best verified coverage found. Longer trajectories and
more samples compete for the same resources. The best allocation will depend
on the total budget and the currently available boards; there need not be one
universal best duration or retention fraction.

### What the existing evidence establishes

- Classical construction can be fast. The corrected early native pilot finished
  its finite construction in about 0.62 solver seconds despite a 50-second
  ceiling, and saved 16 patterns. The complete pilot took 68.40 seconds including
  refinement and certification. A larger timeout alone does not make that finite
  constructor explore longer. We must distinguish new starts/patterns from
  continuous refinement. See the [early pilot](first_real_patio_quarter_then_repair.md#first-numerical-evidence).
- Long classical refinement showed diminishing returns. In the second campaign,
  the best checked interior upper was 4.817 inches by midday; roughly sixteen
  further elapsed hours brought it to 4.688 inches. This was gap-oriented
  optimization, not a controlled coverage-versus-runtime comparison. Its final
  task-576 board had 73.524% coverage. See the
  [completed campaign](first_real_patio_quarter_then_repair.md#completed-second-campaign--november-15-2026).
- The aggressive physics campaign repeatedly started from that same task-576
  board. Of 103 completed aggressive runs, 47 reached 80%, five reached 81%, and
  one reached 82%; the best was 82.035%. This demonstrates substantial downstream
  variability from one parent, but cannot tell us which classical parents are
  best or how much classical refinement they deserve. See the
  [campaign review](analysis/aggressive_campaign_review_20261118/findings.md).
- A late regime is plausible, but acceptance depends strongly on the operator.
  At coverage >=80%, probe replacements passed 16/213 trials, ordinary
  replacements 64/69, and probe insertions 5/8. An aggregate rejection rate
  mixes these different opportunities. High acceptance can also reflect
  area-neutral reorientation rather than useful progress.
- Late refinement still has value. The four-root cached frontier continuation
  improved the previously available coverage champions by 0.458–1.199 percentage
  points over approximately 23 elapsed hours. Two-stone repacks were productive;
  three-stone repacks used 38.5% of combined recorded worker time for seven
  balanced area gains. These are adaptive observations, not randomized operator
  comparisons. See the [final physics synthesis](first_real_patio_quarter_first_physics.md#final-synthesis--november-19-2026).

### Objective and resource accounting

The primary sampling objective is **the expected best verified coverage found by
budget B**, not average final coverage or average acceptance rate. Also report
the probability of reaching predeclared coverage targets and the distribution
of best coverage across repeated campaigns. Keep best-so-far states, since a
last endpoint need not be the best result.

Report interior and edge gap maxima alongside coverage, retaining separate gap
and Pareto champions. The historical interior <=2-inch and edge <=5-inch targets
remain quality diagnostics; none of the four latest coverage champions meets
the interior target. Do not silently make gap compliance an eligibility gate
for this coverage experiment, or claim a coverage champion is a finished patio.

Use allocated worker time as the primary budget, including construction,
proposal generation, empty searches, physics, auditing, checkpointing, and
discarded branches. Report actual CPU time and elapsed time at fixed concurrency
as well; shared GPU work needs explicit accounting if used. Include startup and
validation overhead. Physical simulated seconds and number of attempts are
diagnostics, not equivalent compute budgets.

In particular, the current retry-until-proposal policy can spend arbitrarily
long without consuming a physical attempt. The new protocol needs bounded
proposal searches and safe time limits independently of its attempt limits.

### Treat the pipeline as a tree

Use three allocation levels initially:

1. **Classical parent generation:** independent finite constructions plus optional
   classical refinement, saving several timed checkpoints per lineage.
2. **Physics opening:** independent continuations from each selected checkpoint,
   saving intermediate boards and all costs.
3. **Physics finishing:** fresh continuations or branches from promising opening
   checkpoints, with productive insertions/repacking and bounded setup work.

A second physics seed from a classical board, a restart from an intermediate
physics board, and a repeat of the same proposal with a new physical seed are
different experiments. Record separate proposal-selection and physics RNG seeds
where supported. Initially measure variability between complete continuations;
only add a dedicated same-proposal seed experiment if execution variability
looks consequential.

The opening/finishing split should first be an allocation boundary with the
same proposal policy on either side. Test a changed finishing policy separately
so we can distinguish gains from better allocation and gains from new moves.

### First calibration experiment

Start with a balanced experiment before learning aggressive pruning rules:

1. Generate **12 independent classical lineages** with varied seeds/input orders,
   using the same geometry and physical constraints. Save the finite constructor
   outputs and refinement checkpoints near **1, 4, and 16 worker-minutes** after
   construction. These are provisional geometric spacing, not measured optima;
   adjust after a small timing pilot if overhead makes them unsuitable.
2. From each lineage/time checkpoint, run **three independent physics
   continuations**, provisionally up to **30 worker-minutes** each, with saved
   checkpoints near 2, 8, and 30 minutes. This is 108 continuations and up to
   **54 worker-hours for physics alone**; reduce the design openly if that budget
   is too large. Do not describe it as a cheap smoke test.
3. Randomize dispatch across classical durations and parents. Avoid running all
   short-parent trials first and all long-parent trials later under different
   load. Record actual completed transaction times when a safe boundary crosses
   a nominal deadline.
4. Continue a randomly selected **quarter of opening continuations** to a longer
   horizon, independently of their apparent quality. Budget this extension
   separately. This measures delayed recovery and protects against concluding
   that early losers never improve merely because we stopped observing them.
5. Reserve fresh classical lineages for a later, prospective comparison of the
   chosen allocation policies at equal total budgets. The initial 12 are a
   calibration sample, not enough to establish rare-tail optima.

Using timed prefixes lets us compare many fixed horizons without rerunning all
prefix work. Checkpoints from the same lineage remain correlated. Offline
replay is valid for prefix-based allocation only when the saved continuations
match the proposed actions and horizons; it cannot invent unsampled branches or
new operator policies. Charge every selected prefix and new branch, including
its parent-generation cost, once in a simulated campaign.

### How to answer the four sampling questions

| Question | Initial comparison | Evidence to collect |
| --- | --- | --- |
| How long to run classical work? | Constructor only versus progressively longer refinement | Best downstream coverage per complete campaign budget, not just handoff quality |
| Which classical outputs to retain? | All versus top fractions versus score-plus-diversity | Downstream coverage, tail success, and missed winners on held-out lineages |
| How long to run physics? | Fixed horizons versus calibrated inactivity stopping | Chance and size of the next useful gain versus additional worker time |
| How many continuations per parent? | One broadly versus several per parent versus adaptive allocation | Between-parent and within-parent outcome variability; best-of-k curves |

Give every parent an initial downstream probe during calibration. For later
allocation, a cheap physics opening may predict downstream potential better than
static classical metrics. Candidate predictors include initial coverage, recent
gain rate, interior/edge gaps, usable pocket sizes and fragmentation, unused
inventory size/shape compatibility, wall fit, and opening-stage useful gains per
worker-minute. Feature computation itself must be costed.

Predict **absolute downstream coverage at a specified added budget**, with
incremental gain as a secondary outcome. Ranking only by gain can favor weak
parents that improve substantially while remaining inferior. Compare predictors
against initial coverage alone. Split validation by classical root lineage;
siblings and time checkpoints must not leak across training and validation.

If parent quality dominates downstream variance, screen more parents and deepen
the promising ones. If seed variation within a parent dominates, give selected
parents more independent continuations. Three seeds provide an initial estimate,
not a reliable characterization of rare success probabilities.

### Inactivity stopping and the physics transition

Use a minimum initial budget, a calibrated patience interval, and an absolute
time cap. Reset coverage patience only after a meaningful best-so-far coverage
gain, with tolerance above floating-point ties. Track a separate bounded setup
allowance for gap improvements or purposeful rearrangements; accepting an
area-neutral pulse must not reset the coverage clock indefinitely.

Estimate from longer observed traces how often a trajectory improves after a
given inactive period, by how much, and at what cost. Compare continuing that
trajectory with starting a new classical parent, probing another existing
parent, or branching a saved physics state. Stop when continuation looks less
valuable than these alternatives, while preserving a random exploration share.

Do not switch phases solely at 80% coverage or after a few rejections. Candidate
transition signals are a sustained fall in useful coverage gain per worker-minute,
low proposal yield, fragmented remaining space, and operator-specific failure
rates. Separate no-proposal searches, clearance rollbacks, settling failures,
valid area-neutral moves, and positive gains. Use minimum observation windows
and retain uncertainty when counts are small.

Evaluate setup moves over short descendant sequences, not just immediate area
gain. Existing paired work demonstrates one specific useful setup, but also a
case where the supposed agitation benefit was actually a missed direct insertion.
See the [late-board findings](analysis/late_board_branching_20261118/findings.md).
Keep a bounded exploration allowance rather than requiring every step to improve
coverage or treating every later gain as proof its setup was necessary.

### Candidate production policy, pending calibration

Maintain queues for new parents, short physics probes, continued trajectories,
and late branches. Give all eligible new parents a small probe; promote a mixture
of high-performing and geometrically diverse states; add continuations in small
batches where they have the best estimated chance of exceeding the current
champion. Reserve a provisional 10–20% exploration budget, to be tested rather
than treated as optimal. Revisit parked states when evidence changes.

Compare this policy prospectively with (a) one deep classical parent plus deep
physics, (b) many shallow complete trajectories, and (c) several parents with
several physics continuations each. Repeat whole campaigns at fixed budgets and
concurrency. Report uncertainty by independent root/campaign, not by treating
every move or sibling as independent. A higher mean improvement rate alone is
not enough to select the policy that most often finds the best board.

All phases preserve original full-size outlines, unique inventory, locks,
no net downsizing for physics substitutions, >=45-degree deliberate same-piece
edits, half-inch whole-board clearance, observed settling, and exact rollback.
Save code/config/input pins, lineage, RNG state, rejected endpoints, costs, and
stop reasons in new versioned outputs. Numerical failures remain distinct from
unsuccessful geometric/physical searches.

The next concrete work is a read-only timing/trace audit to choose feasible
calibration horizons and determine which historical checkpoints can be reused.
No allocation parameters are established as optimal by the existing evidence.
