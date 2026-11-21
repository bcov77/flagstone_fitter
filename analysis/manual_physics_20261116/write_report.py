"""Write a concise research report from completed summaries."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
summary=json.loads((OUT/'summary.json').read_text());trials=json.loads((OUT/'trials.json').read_text())['trials']
valid=sum(r['after']['passes_clearance'] for r in trials)
text=f'''# Initial fit study: manual physics packing

Completed **{len(trials)} full-board trials**, covering three final-board slots,
two seeds, direct settling and a 0.5 → 1.1 → 0.5-inch gap cycle. Compared human
pieces, unused alternatives selected by three screening hypotheses, and
empty-slot controls. Each trial ran 16 simulated seconds, including a final
four seconds with jitter magnitudes zero and the user's saved tethers retained.

**{valid}/{len(trials)} trials passed exact original-outline half-inch clearance.**
These are learning experiments, not new accepted layouts. No solver, source
layout, settings defaults or running server was changed.

## What the measurements mean

The broad-void proxy measures area of grid points more than one inch from every
stone and wall. It tracks wide spaces within gaps; it is not total unoccupied
area or the existing certified local-maximum gap audit. Far-region measurements
exclude a two-foot radius around the original slot's centroid. The table gives
means over two seeds. Negative deltas mean less broad void than the matched
human-piece reference. Pair deficit is the sum, over stone pairs, of each
shortfall below half an inch; it omits wall shortfalls, which are recorded
separately in the CSV.

## Direct settling results

| Slot | Candidate | Selection | Pair deficit (in, summed) | Broad void (ft²) | Local void Δ vs human (ft²) | Far void Δ vs human (ft²) |
|---|---|---|---:|---:|---:|---:|
'''
for r in summary:
 if r['protocol']!='direct':continue
 text+=f"| {r['target']} | {r['candidate'] or 'empty'} | {r['selected_by']} | {r['pair_deficit_sum_inches']:.2f} | {r['broad_void_sqft']:.3f} | {r['local_void_delta_vs_human_sqft']:+.3f} | {r['far_void_delta_vs_human_sqft']:+.3f} |\n"
text+='''
## Effect of the inflated-gap maneuver

These are paired changes from direct settling to the gap cycle for the same
candidate and seeds. Negative values mean improvement in that measure.

| Slot | Candidate | Pair deficit change (in, summed) | Broad void change (ft²) |
|---|---|---:|---:|
'''
for r in summary:
 if r['protocol']!='gap_cycle':continue
 direct=next(x for x in summary if x['target']==r['target'] and x['candidate']==r['candidate'] and x['protocol']=='direct')
 text+=f"| {r['target']} | {r['candidate'] or 'empty'} | {r['pair_deficit_sum_inches']-direct['pair_deficit_sum_inches']:+.2f} | {r['broad_void_sqft']-direct['broad_void_sqft']:+.3f} |\n"
text+='''
## Interpretation

- Boundary profiles capture a failure mode that overlap area misses. In the
  synthetic example, a tiny 0.075-in² intrusion into the required-clearance
  region forces a 0.75-inch increase in the main edge's separation. Both the
  flat and bumped examples still have the same 0.5-inch minimum clearance.
- Keep local void change and remote void change separate. A local improvement
  need not be a whole-board improvement. `outcomes.csv` provides matched
  per-seed comparisons against both human and empty-slot controls.
- The current edge and scaled scores are uncalibrated hypotheses. A scaled
  match may select a piece that is too large at full size. A low intrusion score
  may select a piece too small to fill the slot. Both require full-size trials.
- A gap cycle is an optional search move, not an unconditional improvement.
  It needs outcome-based acceptance and a way to restore the previous board.
- Finite compliant contact stiffness and persistent tethers can sustain
  clearance violations even after jitter stops. These short experiments do
  not establish that the boards are infeasible. A separate feasibility phase
  and exact acceptance checks remain necessary before declaring packing gains.

## Next experimental changes

1. Use additional human-labeled good/bad placements and early/late board
   snapshots. This experiment has only one crowded final board.
2. Compare contiguous matching boundary runs and directional cavity widths
   with today's average boundary-distance score. Normalize cavity benefit so
   small candidates cannot win simply by avoiding all neighbors.
3. Test final feasibility independently of candidate selection: retain the
   user's preferred exploration settings, then compare explicit final-phase
   tether/drive and hardness policies, always checking original geometry.
4. After that, expand candidate pose search and calibrate local/remote damage
   acceptance using held-out neighborhoods. Preserve the best verified state.

See [README.md](README.md) for method, limitations and reproduction commands,
[settled_comparison.png](settled_comparison.png) for labeled local views, and
[bump_example.png](bump_example.png) for the synthetic edge example.
'''
probe=OUT/'convergence_probe.json'
if probe.exists():
 last=json.loads(probe.read_text())['samples'][-1]
 text+=f'''\n## Short-time response probe\n\nAfter two seconds with jitter magnitudes zero, the separate probe retained\nmean translational speed {last['mean_speed_inches_per_second']:.2f} inches/second\n(maximum {last['max_speed_inches_per_second']:.2f}) and\n{last['pair_clearance_violations']} pair-clearance violations. Turning jitter off\nalone did not establish equilibrium in this probe. The main batch's quiet\nphases are finite observation windows, not measured convergence certificates.\nRaw samples are in `convergence_probe.json`.\n'''
text+='''\n## Measurement-grid check\n\nAt half-inch grid spacing, 35 → 150 decreased local broad void in both seeds\n(0.137/0.111 ft²) and increased remote broad void (0.014/0.095 ft²). Total broad\nvoid still decreased. These are competing objectives; this does not establish\nthat remote damage outweighs the local benefit. See `grid_sensitivity.json`.\n'''
(OUT/'report.md').write_text(text)
print('Wrote report.md')
