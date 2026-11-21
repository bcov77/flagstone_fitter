"""Summarize pilot checkpoints, independently recheck validity, and render review."""
import argparse,copy,csv,json,math,statistics
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch,Circle
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
# Run in the repository plotting environment. Engine audit is separate.
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path(__file__).with_name('pilot_v1'));args=parser.parse_args();out=args.out
 m=json.loads((out/'manifest.json').read_text());rows=[json.loads(p.read_text()) for p in sorted((out/'results').glob('*.json'))]
 assert len(rows)==len(m['tasks']),(len(rows),len(m['tasks']))
 lookup={(r['task']['case']['name'],r['task']['seed'],r['task']['protocol'],r['task']['finish']):r for r in rows}
 flat=[]
 for r in rows:
  t=r['task'];c=t['case'];control=lookup[(c['control'],t['seed'],t['protocol'],t['finish'])]
  aq=r['after_quality'];cq=control['after_quality']
  flat.append(dict(task_id=t['id'],case=c['name'],family=c['family'],candidate=None if not c['candidate'] else c['candidate']['id'],seed=t['seed'],protocol=t['protocol'],finish=t['finish'],valid=r['after']['valid'],converged=r['converged'],timeout=r['timeout'],pair_violations=r['after']['pair_violations'],wall_violations=r['after']['wall_violations'],pair_deficit_inches=r['after']['pair_deficit_inches'],minimum_pair_inches=r['after']['minimum_pair_inches'],quiet_seconds=r['quiet_seconds'],wall_seconds=r['wall_seconds'],covered_area_sqft=aq['covered_area_sqft'],coverage_gain_sqft=aq['covered_area_sqft']-cq['covered_area_sqft'],broad_void_sqft=aq['broad_void_1'],local_void_change_sqft=(aq['broad_void_1']-aq['far_void_1'])-(cq['broad_void_1']-cq['far_void_1']),remote_void_change_sqft=aq['far_void_1']-cq['far_void_1'],max_translation_inches=r['max_translation_inches'],final_max_speed_ips=r['final_motion']['max_speed_ips']))
 with (out/'outcomes.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
 # Original-piece restoration is a deliberately duplicated control, used to
 # check reset/order/anchor consistency across independent worker processes.
 reset_diffs=[]
 for seed in [17,41]:
  for protocol in ['direct','gap_cycle']:
   for finish in ['retain','release','margin_release']:
    a=lookup[('full_unchanged',seed,protocol,finish)]['final_poses'];b=lookup[('replace_original',seed,protocol,finish)]['final_poses']
    reset_diffs.append(max(abs(x-y) for sid in a for x,y in zip(a[sid],b[sid])))
 assert max(reset_diffs)<1e-10,max(reset_diffs)
 lookup={(r['task']['case']['name'],r['task']['seed'],r['task']['protocol'],r['task']['finish']):r for r in rows}
 table=[]
 for r in rows:
  t=r['task'];control=lookup[(t['case']['control'],t['seed'],t['protocol'],t['finish'])]
  aq=r['after_quality'];cq=control['after_quality']
  table.append(dict(id=t['id'],case=t['case']['name'],family=t['case']['family'],seed=t['seed'],protocol=t['protocol'],finish=t['finish'],valid=r['after']['valid'],converged=r['converged'],quiet_seconds=r['quiet_seconds'],pair_deficit_inches=r['after']['pair_deficit_inches'],wall_violations=r['after']['wall_violations'],area_gain_sqft=aq['covered_area_sqft']-cq['covered_area_sqft'],broad_void_delta_sqft=aq['broad_void_1']-cq['broad_void_1'],far_void_delta_sqft=aq['far_void_1']-cq['far_void_1'],local_void_delta_sqft=(aq['broad_void_1']-aq['far_void_1'])-(cq['broad_void_1']-cq['far_void_1']),max_translation_inches=r['max_translation_inches'],wall_seconds=r['wall_seconds']))
 with (out/'outcomes.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
 summary=[]
 for finish in ['retain','release','margin_release']:
  for protocol in ['direct','gap_cycle']:
   subset=[r for r in flat if r['finish']==finish and r['protocol']==protocol]
   summary.append(dict(finish=finish,protocol=protocol,n=len(subset),valid=sum(r['valid'] for r in subset),converged=sum(r['converged'] for r in subset),mean_quiet_seconds=statistics.mean(r['quiet_seconds'] for r in subset),mean_deficit_inches=statistics.mean(r['pair_deficit_inches'] for r in subset)))
 gate={}
 for finish in ['retain','release','margin_release']:
  controls=[r for r in flat if r['finish']==finish and r['case'] in ['full_unchanged','loose_unchanged','replace_original']]
  easy=[r for r in flat if r['finish']==finish and r['case']=='insert_known_good']
  gate[finish]={'control_valid':sum(r['valid'] for r in controls),'control_total':len(controls),'easy_valid':sum(r['valid'] for r in easy),'easy_total':len(easy),'control_converged':sum(r['converged'] for r in controls),'easy_converged':sum(r['converged'] for r in easy),'passes':all(r['valid'] and r['converged'] for r in controls+easy)}
 result=dict(completed=len(rows),valid=sum(r['valid'] for r in flat),converged=sum(r['converged'] for r in flat),groups=summary,expansion_gate=gate,reset_max_pose_difference=max(reset_diffs),median_trial_wall_seconds=statistics.median(r['wall_seconds'] for r in flat),sum_trial_wall_seconds=sum(r['wall_seconds'] for r in flat))
 (out/'summary.json').write_text(json.dumps(result,indent=2))
 text=f'''# Physics placement pilot results\n\nCompleted {len(rows)}/{len(m['tasks'])} trials. Exact half-inch clearance passed in
{result['valid']} endpoints; {result['converged']} also met the provisional motion/
stability criterion. These are feasibility measurements, not certification of
the final-layout local-maximum gap rule. No campaign layout is promoted here.

## Experiment

Eight cases: unchanged full and synthetic loose boards; known-good, perturbed,
and reserve-piece insertions; original-piece, reoriented-piece, and reserve-piece
replacements. Two seeds, paired direct/gap-cycle protocols, three finish policies.
Only two target regions are used: this pilot tests finishing and measurement,
not learned rules across many gaps. Controls and seed repeats are correlated.

Exploration uses the user's jitter amplitudes/frequencies and constraints at
30, with loose piece/wall hardness 18/40. Explore for 0.5, 0.5, 1.0 seconds;
only the middle phase inflates the gap in the cycle protocol. All final phases
use hardness 48/89 and zero jitter magnitude. Retain keeps restoring gains;
release zeros all four restoring stiffness/force gains. Margin-release also
uses a 0.6-inch collision gap during finishing. All trials end with a 0.5-inch
gap and another 0.5 seconds of observation.

Quiet phases sample motion and exact clearance every 0.5 seconds, up to four
seconds. Provisional convergence needs two consecutive low-motion, valid,
stable-deficit samples: max translation speed <0.02 in/s, max angular speed
<0.1 deg/s, summed pair-deficit change <0.002 in. After returning to the final
0.5-inch gap, validity and low motion must hold again. These provisional
thresholds were declared before the run; this pilot evaluates their usefulness.

## Finishing-policy results

| Finish | Protocol | Valid / trials | Converged / trials | Mean quiet seconds | Mean pair deficit (in, summed) |
|---|---|---:|---:|---:|---:|
'''
 for r in summary:text+=f"| {r['finish']} | {r['protocol']} | {r['valid']}/{r['n']} | {r['converged']}/{r['n']} | {r['mean_quiet_seconds']:.2f} | {r['mean_deficit_inches']:.3f} |\n"
 text+='\n## Expansion gate\n\nA policy passes if all unchanged/original-piece controls and all known-good\ninsertions reach both valid clearance and provisional convergence.\n\n'
 for finish,g in gate.items():text+=f"- {finish}: {'PASS' if g['passes'] else 'NOT PASSED'}; valid controls {g['control_valid']}/{g['control_total']}, valid easy insertions {g['easy_valid']}/{g['easy_total']}; converged {g['control_converged']} controls and {g['easy_converged']} easy insertions.\n"
 text+='\n## Results by action (all settings, four repeats per finish)\n\n| Case | Retain valid / converged | Release valid / converged | Margin-release valid / converged |\n|---|---:|---:|---:|\n'
 for case in m['cases']:
  cells=[]
  for finish in ['retain','release','margin_release']:
   rs=[r for r in flat if r['case']==case['name'] and r['finish']==finish]
   cells.append(f"{sum(r['valid'] for r in rs)}/4; {sum(r['converged'] for r in rs)}/4")
  text+='| '+case['name']+' | '+' | '.join(cells)+' |\n'
 text+=f'''\n## Checks and limits\n\nAll twelve independently run original-piece controls reproduce the untouched
full-board control to a maximum pose-coordinate difference of
{max(reset_diffs):.3g}. Geometric tests cover valid sources/easy positive setups,
injected pair/wall damage, and balanced matched controls. An additional engine
checker audit is recorded separately in `audit.json` when complete.

Median per-trial wall time was {result['median_trial_wall_seconds']:.2f} seconds
under the worker concurrency used. Throughput is measured for this small pilot,
not a promise for denser boards or longer searches. `wall_seconds` excludes
proposal generation and parent scheduling overhead.

The synthetic loose board removes 50, 60 and 61 from the real 27-stone winner.
Known-good insertion restores 60 at its original pose; it tests a positive
geometry case, not discovery of a new inventory choice. Reserve candidates
are screened using a declared area/intrusion rule, not selected randomly across
many targets. Motion checks do not yet enforce a time window of stable broad-
void measurements. A stopped endpoint is not proof of global optimality or
long-term stability. The sampled void metrics are research proxies.

Raw results include features, before/after validity and coverage, compact phase
poses, quiet traces, motion, displacement and local/remote changes against
matched controls. No model was fitted to these few related neighborhoods.
The next dataset must broaden target diversity and use grouped holdouts.

See `gallery.png`, `finishing_traces.png`, `outcomes.csv`, and `manifest.json`.
Source inventory is stored once in `source.json`; source and runner/engine hashes
are recorded in the manifest. The live server and user saves are untouched.
'''
 text+='''
## Pilot conclusion and next configuration

All three policies passed the feasibility gate in these two target regions.
Release reached the provisional quiet criterion fastest (1.5 simulated seconds
on average; retain about 2.45, margin-release 2.0). Use release as the default
finishing policy for the next dataset, retaining the user's maximum constraints
during exploration. Keep margin-release as a fallback to test on harder cases.
The explicit motion/clearance gate stays mandatory; do not replace it with
a universal 1.5-second timeout.

All tested candidates ended feasible, so this pilot cannot estimate rejection
rules or performance on dense/tightly constrained cavities. It validates the
measurement and finishing path. The main batch needs random and borderline
proposals across many targets, with failures deliberately retained. Local/remote
quality still varies even when every candidate passes clearance.

The full 93-task batch after the three smoke trials took 111.6 wall seconds
with four workers; the smoke batch took 5.0 seconds with three workers. These
are observed timings for these cases. A linear estimate for approximately
4,000 equally easy trials is roughly 1.3--1.5 hours including overhead, but
crowded cases, longer budgets and candidate generation can cost much more.
A time-bounded resumable campaign is preferable to promising an overnight count.

No larger campaign was started as part of this pilot.
'''
 (out/'report.md').write_text(text)
 doc=json.loads((out/'source.json').read_text());stones={str(s['id']):s for s in doc['stones']}
 def poly(sid,pose):
  x,y,t=pose;c,s=math.cos(t),math.sin(t)
  return affine_transform(Polygon(stones[sid]['outline']),(c,-s,s,c,x,y))
 fig,axes=plt.subplots(2,4,figsize=(16,11))
 for ax,case in zip(axes.ravel(),m['cases']):
  r=lookup[(case['name'],17,'direct','margin_release')];candidate=case['candidate'];cx,cy=case['center']
  for sid,pose in r['final_poses'].items():
   g=poly(sid,pose);ax.add_patch(Patch(list(g.exterior.coords),facecolor='#c98b5c' if candidate and sid==candidate['id'] else '#ddd3c0',edgecolor='#554a39',linewidth=.5))
   q=g.representative_point();ax.text(q.x,q.y,sid,fontsize=6,ha='center',va='center',clip_on=True)
  if candidate:
   g=poly(candidate['id'],candidate['pose']);x,y=g.exterior.xy;ax.plot(x,y,'--',color='#387aaf',linewidth=.8)
  patio=Polygon(doc['patio']);x,y=patio.exterior.xy;ax.plot(x,y,'k-',linewidth=.7)
  ax.set_xlim(cx-1.7,cx+1.7);ax.set_ylim(cy-1.7,cy+1.7);ax.set_aspect('equal')
  ax.set_title(f"{case['name']}\nvalid={r['after']['valid']}; converged={r['converged']}",fontsize=9)
 fig.suptitle('Pilot review: direct protocol, seed 17, margin-release finish\nOrange: proposed piece; blue dashed: initial candidate pose; original outlines')
 fig.subplots_adjust(left=.04,right=.99,bottom=.05,top=.87,wspace=.16,hspace=.35);fig.savefig(out/'gallery.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for finish in ['retain','release','margin_release']:
  r=lookup[('replace_reserve',17,'direct',finish)]
  axes[0].plot([x['seconds'] for x in r['trace']],[max(x['max_speed_ips'],1e-6) for x in r['trace']],marker='.',label=finish)
  axes[1].plot([x['seconds'] for x in r['trace']],[x['pair_deficit_inches'] for x in r['trace']],marker='.',label=finish)
 axes[0].set_yscale('log');axes[0].axhline(.02,linestyle='--',color='gray');axes[0].set_ylabel('Maximum speed (in/s)')
 axes[1].set_ylabel('Summed pair-clearance deficit (in)')
 for ax in axes:ax.set_xlabel('Quiet-phase seconds');ax.legend(fontsize=8)
 fig.suptitle('Reserve replacement: same starting state and random drives, different finishing')
 fig.tight_layout();fig.savefig(out/'finishing_traces.png',dpi=160);plt.close(fig)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
