# Online physics trajectories

**Current revision: r3 resumes preserved checkpoints after fixing a missing math import in filler planning.** See [recovery details](RECOVERY.md). The current default output is `examples/physics_trajectories_20261117_r3/`; the original batch remains unchanged.

From the repository root, run **in the foreground**:

```bash
caffeinate -i ./physics_patio/overnight.sh
```

`caffeinate -i` keeps this Mac awake during the job. The launcher sets numerical
library threads to one and uses **ten worker processes**, each running an
independent trajectory. Every trajectory starts from the exact task-576 board
used by `./physics_patio/run.sh`, not the editor autosave or the previous run's
final board. Each trajectory makes **30 serial attempts**, then the worker
starts another trajectory indefinitely. Global run numbers 1–8 are normal,
9 Golden, 10 Silver; this repeats every ten. Completion order can differ.

**Ctrl-C** stops scheduling and asks each worker to finish its current attempt,
atomically checkpoint, and exit. This may take several minutes if a selector or
gap audit is in progress. The same command resumes unfinished trajectories
before allocating new numbers. A second process cannot use the same output
folder. `touch examples/physics_trajectories_20261117_r3/STOP` also requests a safe
stop. Starting again clears that stop request. No background process is launched
by the assistant, and the editor server/session is untouched.

## What is chosen, and why

All proposals are generated against the **current retained board**, not replayed
from the frozen v4/v5 source poses. Each attempt samples natural openings and
replacement targets. Poorly connected stones are sampled more often for
replacement, while provisionally stable stones can still be revisited. Explicit
locks are respected. A piece already chosen in a trajectory is not chosen again
in that trajectory; failed attempts also consume that piece's selection.

Original outlines and a half-inch joint define free space. Morphological throat
cuts suggest pockets without turning their artificial mouths into physical
walls. FFT position/rotation proposals and bounded geometric refinement search
unused full-size shapes. Two-piece proposals search a second distinct stone in
the first one's residual opening and check their mutual clearance. They undergo
**one shared relaxation and count as one attempt**. They may replace one original
stone, or insert into an existing opening. Reorienting an existing stone is also
an allowed attempt. This first runner does not remove two existing stones at once.

Initial shape gates require at least 55% coverage of the identified pocket,
limited overhang, at most 0.4 inches of clearance intrusion, at most 5% intrusion
area, no wall-clearance intrusion and at most four initially affected neighbors.
This allows limited near-fits for physics to resolve; it does not rescale stones.
Only a bounded subset of targets, inventory pieces, angles and poses is searched.
If no proposal passes, the failed bounded search is recorded as an attempt.

The online planning layer reuses the v5 geometry functions with smaller repair
budgets. It compares with the current original/empty control and any searched
same-stone reorientations. The risk tier comes first: new local wall/corner,
rigid-pocket and indent risks; lost future flat-side options; corner reach;
and provisional stability tradeoffs. Within tiers it favors contiguous joints
against walls/locks, then stable neighbors, then other completed runs. Area is
a later tie breaker. Relevant neighbors' whole outlines are in the measurement
ROI, including their far sides. Unused-piece repair probes expose actual bounded
witnesses or explicit unsuccessful/unsearched regions. They are not promises
of a repair sequence. The saved reasons identify owners and conditional options.

These are **experimental, explicit heuristics**, not fitted human approval
predictions. Thresholds are not validated and this version retains the v5
heuristic's known false cautions and missed corner concerns. The selector cannot
claim to find the best available placement globally. Stability is a continuous
hint, not a new permanent lock.

## Probabilities

Selection samples distinct target/piece actions, so many poses of one piece do
not multiply its chance. Among the best available risk tier, the top 20% of the
pool (at least three when that tier has enough choices) form a uniform best set.
A normal decision mixes **70% best-set sampling, 25% decreasing rank-weighted
sampling, and 5% uniform exploration**. Thus every screened option has a route
to selection. Golden decisions always use the best set, with diversity within it.
Silver decisions use **80% best-set sampling plus 20% the normal distribution**.
The total chance of a best-set choice in Silver is consequently above 80%.

Every probability vector, sampled index, piece/pose, alternative assessment,
selection tier, target and reason is saved. Probabilities are conditional on
the sampled candidate pool, not on all possible pieces/poses on the patio.
Target/inventory sampling rules and the generated target set are also recorded.

## Relaxation and rollback

The chosen piece(s) are held during two simulated seconds of jiggling while
neighbors respond. Then temporary holds are released and drive/tether forces
are removed for up to six simulated seconds of quiet settling. Original locks
remain. Positions anchor to the current attempt's start.

Original-outline checks cover the **whole board**, not only proposed pieces.
Any half-inch clearance failure triggers one quick **0.50 → 1.10 → 0.50 inch**
gap cycle (0.75 simulated seconds inflated and 0.75 deflated), followed by the
same quiet settling test. Acceptance requires half-inch clearance plus two
consecutive low-motion observations (below 0.02 in/s and 0.1 degrees/s).
Measurements precede the editor engine's velocity-zeroing stop function.
Unresolved clearance or settling timeouts roll back the complete attempt.

A valid settled move is retained even when area or gaps worsen. This preserves
exploratory trajectories instead of silently pruning their bad outcomes. The
reason records the pre-physics hypothesis; measured endpoints show what actually
happened. There are no physics mid-frame images or pose traces. Phase diagnostics
store motion/clearance measurements only.

## Saved artifacts

Default output: `examples/physics_trajectories_20261117_r3/`.

- `source.json`, `manifest.json`: original inventory once, source and compute
  hashes, library versions, config, baseline measurements and policy.
- `runs/run_000001/attempt_001.json.gz` etc.: compact start/proposed/attempted-end/
  retained-end frames; full candidate pool/reasons/probabilities; repair and
  settling outcomes; endpoint coverage and gap sizes. Trial records are gzip-compressed JSON without dropping data. An atomic attempt
  file is the transaction boundary. `pending_attempt.json.gz` preserves a selected trial
  while physics is running and is removed after commit.
- Each completed run: `final.json` opens directly in the physics editor,
  `summary.json`, `start_end.png`, `trajectory.png` (all retained endpoints),
  and `metrics.png`. Render failures are explicit and can be retried.
- `progress.json` within each run: phase and completed attempts. Root
  `summary.json` tracks completed runs. Raw rejected endpoints remain saved.

Coverage is union area inside the patio, as a percentage. Gap measurements use
**the full original-outline segment-Voronoi audit at two integer scales** and
save padded largest interior and edge diameters, locations and stability.
Half-inch clearance and maximum-gap compliance remain separate. A failed audit
is explicitly marked unavailable; it is never presented as a passing layout.
Endpoint auditing is a significant part of compute cost.

After stopping, build a pooled analysis table and refresh images:

```bash
.venv/bin/python analysis/physics_trajectories/report.py \
  --out examples/physics_trajectories_20261117_r3
```

This writes `attempts.csv` and `outcomes.json`. Associations across adaptive,
selected trajectories are not causal effects. Selection probabilities help
interpret the experiment but do not by themselves remove confounding.

Inspect status without starting work:

```bash
./physics_patio/overnight.sh --status
```

Open a completed endpoint in the editor:

```bash
./physics_patio/run.sh --layout examples/physics_trajectories_20261117_r3/runs/run_000001/final.json
```

Export a rejected endpoint (or `--frame start` / `end`) for inspection:

```bash
.venv/bin/python analysis/physics_trajectories/report.py \
  --out examples/physics_trajectories_20261117_r3 --run 1 \
  --export-attempt 3 --frame attempted_end --output /tmp/attempt_003_rejected.json
```

Optional limits: `--workers 2 --runs 2` executes/resumes two trajectories, then
exits. `--config path.json --out examples/a_fresh_experiment` changes budgets in
a separate experiment. Resume rejects changed compute code, libraries, config
or the saved source. Do not alter compute files during a live run. Dependency
versions are in `requirements.txt`; the launcher uses the root `.venv`.

Independent continuity/clearance verification after stopping:

```bash
.venv/bin/python analysis/physics_trajectories/verify.py \
  --out examples/physics_trajectories_20261117_r3
```

Validation details and example outputs are in [VALIDATION.md](VALIDATION.md).
