"""Summarize the captured live run and isolated paired replay evidence."""
import json,gzip
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def read(path):
 return json.load(gzip.open(path,'rt') if str(path).endswith('.gz') else open(path))
snap=read(HERE/'run_snapshot.json');bench=read(HERE/'benchmark_results.json');cached=read(HERE/'cache_results.json')
cache={r['job']['number']:r['cache_only'] for r in cached['cases']}
lines=['# Frontier geometry review — November 18, 2026','',
'The live run is working, but geometry is doing substantial avoidable work and the fixed operator schedule does not learn which searches are productive. The most defensible first change is an exact cache fix; the more exploratory changes should stay separate until their recall is better established.','',
'## Captured run','',
'The snapshot contains '+str(sum(r['jobs'] for r in snap['operators']))+' completed jobs. Best coverage by root at capture: '+', '.join(str(r['root'])+': '+format(r['best'],'.3f')+'%' for r in snap['summary']['roots'])+'. The original roots were 79.231%, 80.259%, 81.152%, 82.035%; pilot warm starts were already present, so not all gain over the originals is attributable to this run.','',
'| Operator | Jobs | Physics | Accepted | Worker hours |','|---|---:|---:|---:|---:|']
for r in snap['operators']:lines.append(f"| {r['operator']} | {r['jobs']} | {r['physics']} | {r['accepted']} | {r['worker_hours']:.2f} |")
lines+=['',
'Replacement planning averages about 690 seconds before selection, versus 63 seconds from selection through physics/audit. These stage estimates use file timestamps and include small scheduling/write overheads. Three-piece repacking has spent about 10.7 worker-hours to produce three trials. Empty pair fills are extremely cheap; an empty count alone is not evidence of an important performance problem. Reorientation and pulses can prepare later fills despite having zero immediate inventory-area gain.','',
'## What the code is doing','',
'- It performs guided FFT shape fitting, followed by original-outline fit and eligibility checks. It is not enumerating every placement or every inventory combination.','- For every scaled replacement pose, `planner.proposal` reconstructs all active polygons, unions/buffers the entire board and intersects the same target with the same original free space. Those values are invariant for that target. `planner_cached.py` computes this once per target without changing search order, random draws, scoring, gates or candidate poses.','- The current per-piece shortlist can be occupied by nearly identical poses. The global coarse shortlist and the four-state compound beam can likewise concentrate on very few piece choices. The prototype suppresses near duplicates, reserves space across piece IDs and distinct piece sets, and filters inventory/angles that provably cannot meet existing eligibility rules.','- The richer old residual-pocket, boundary/alignment and actual-filler assessment in `online_geometry.py` remains absent from frontier ranking. This review does not claim to have restored it. Restore that assessment on a diverse bounded shortlist, and use it as guidance rather than vetoing temporary setup gaps.','- There is no fitted outcome model or operator adaptation. The scheduler cycles fixed operators, visits promising saved boards and caps repeated selected actions by context. Empty searches and failed families do not change the operator schedule. A fresh board context also resets the exact-action cache.','',
'At inspection, the 82% root had 20 replacement physics trials: all were scaled proposals and all rolled back for clearance failure. This is evidence for family/context feedback and better full-size feasibility ranking, not evidence that no larger replacement could ever work.','',
'## Paired geometry replay','',
'Each case supplies one saved opening and its parent board; the search receives available inventory but no winning piece IDs or poses. Target discovery runs normally and is then restricted to that opening. These are bounded single-opening tests, not full-job throughput or autonomous board-improvement experiments. Baseline and prototype use the saved job seed, phase, grids and inventory budgets. The openings were chosen from prior accepted trials where available; the 82% replacement and triple cases have no accepted trial in the captured data.','',
'| Job / operator | Baseline seconds | Cache only seconds | Diverse prototype seconds | Baseline → prototype candidates / distinct piece sets |','|---|---:|---:|---:|---|']
recall=[]
for row in bench['cases']:
 j=row['job'];a=row['baseline'];b=row['prototype'];c=cache.get(j['number']);cs=f"{c['seconds']:.1f}" if c else '—'
 lines.append(f"| {j['number']} / {j['operator']} | {a['seconds']:.1f} | {cs} | {b['seconds']:.1f} | {a['candidates']} / {a['piece_sets']} → {b['candidates']} / {b['piece_sets']} |")
 result=read(ROOT/'examples/frontier_search_20261118/jobs'/f"job_{j['number']:08d}"/'result.json.gz')
 known=tuple(sorted(x['stone'] for x in result.get('chosen',{}).get('placements',[])))
 if known:
  record=dict(job=j['number'],previous_physics_accepted=result.get('accepted'),previous_piece_set=known)
  for label in ['baseline','prototype']:
   pool=read(HERE/f"job_{j['number']}_{label}_pool.json.gz")['candidates'];record[label+'_retrieves_piece_set']=any(tuple(sorted(x['stone'] for x in p['placements']))==known for p in pool)
  recall.append(record)
(HERE/'piece_set_retrieval.json').write_text(json.dumps(recall,indent=2)+'\n')
lines+=['',
'Cache-only cases return exactly equal ordered candidate objects to baseline. The separate 12-call scaled-proposal microbenchmark took 15.37 seconds before caching and 1.38 seconds afterward, with equal outputs. It isolates redundant proposal work and is not a whole-run speedup estimate.','',
'The diverse prototype can trade away candidates: on job 10 it kept one distinct piece instead of two. Piece-set retrieval is recorded separately; retrieving a piece set at another pose is not rediscovering its accepted physical endpoint. These results do not justify claiming that more diversity will improve boards.','',
'## Recommended revision','',
'1. Carry forward the exact cache optimization in a separately pinned continuation, seeded from the current retained boards. Keep the original run/checkpoints intact.','2. Add stage timings, gate-rejection reasons and per-root/operator/family success records. Allocate more work to productive contexts while retaining exploration; reduce repeated costly triple searches with no demonstrated followup benefit. Do not eliminate all zero-gain setup moves.','3. Validate a mixed shortlist that preserves strong original candidates while reserving additional slots for distinct poses/piece combinations. The current prototype is experimental, not a ready production replacement.','4. Restore bounded old surrounding-geometry assessment and test actual followup fills and physical outcomes before calling the revised ranking better. Broader known-pose rediscovery and physical end-to-end comparisons remain outstanding.','',
'## Validation and live-job protection','',
'Five focused regression checks pass for diverse shortlists, early angle filtering, cached scaled-proposal equality, locks and no downsizing. All 14 existing frontier recovery/constraint checks also pass with the prototype planner substituted in an isolated Python process. No physical trials were launched by this review.','',
'The live runner, compute files, manifest, source inputs and checkpoints were not modified or stopped. Its full compute/input/library preflight passed after the prototype work. New code lives only in this review directory, which the live campaign does not import.','',
'Artifacts: `run_snapshot.json` pins the captured result files; `profile.txt` profiles a real triple-repack opening; `benchmark_results.json` and candidate pools preserve paired results; `cache_results.json` preserves cache-only equality checks; `validation.json` records test scope. Runtime comparisons are single sequential measurements on a machine also running eight workers.']
(HERE/'findings.md').write_text('\n'.join(lines)+'\n')
print('Wrote findings for',len(bench['cases']),'paired cases and',len(cache),'cache-only cases.')
