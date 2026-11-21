"""Paired geometry-only replay on saved openings, never supplied winning IDs/poses."""
import sys, importlib.util, collections
from pathlib import Path
REVIEW=Path(__file__).resolve().parent
ROOT=REVIEW.parents[1]
sys.path.insert(0,str(ROOT/'analysis/frontier_search'))
from core import *
import planner as baseline
spec=importlib.util.spec_from_file_location('geometry_prototype',REVIEW/'planner_v2.py')
prototype=importlib.util.module_from_spec(spec);spec.loader.exec_module(prototype)

def snapshot():
 out=DEFAULT_OUT;rows=[];stats=collections.defaultdict(list);pins={}
 for f in sorted((out/'jobs').glob('job_*/result.json.gz')):
  j=read(f.parent/'job.json');r=read(f);stats[j['operator']].append((j,r,f));rows.append((j,r));pins[str(f.relative_to(ROOT))]=sha(f)
 table=[]
 for op,rs in stats.items():
  gains=[];planning=[];endpoint=[]
  for j,r,f in rs:
   selected=f.parent/'selected.json.gz'
   if selected.exists():
    planning.append(max(0,selected.stat().st_mtime-(f.parent/'job.json').stat().st_mtime));endpoint.append(max(0,f.stat().st_mtime-selected.stat().st_mtime))
   if r.get('accepted'):
    parent=read(out/'nodes'/f'{j["parent"]}.json.gz');gains.append(r['end_metrics']['area_percent']-parent['metrics']['area_percent'])
  table.append(dict(operator=op,jobs=len(rs),physics=sum(bool(r.get('physics_evaluated')) for j,r,f in rs),accepted=sum(bool(r.get('accepted')) for j,r,f in rs),worker_hours=sum(r['seconds'] for j,r,f in rs)/3600,positive_gains=sum(g>1e-6 for g in gains),summed_branch_gain_points=sum(gains),mean_planning_seconds=float(np.mean(planning)) if planning else None,mean_physics_audit_seconds=float(np.mean(endpoint)) if endpoint else None))
 write(REVIEW/'run_snapshot.json',dict(captured_unix=time.time(),summary=read(out/'summary.json'),operators=table,result_hashes=pins,compute_hashes=dependencies(),timing_caveat='Filesystem selection/result timestamps approximate stage durations; include scheduling and writes. Branch gains are not cumulative best-board gains.'))
 return rows

def run():
 rows=snapshot();cases=[]
 for op,root in [('replace',62),('replace',76),('pair_repack',62),('pair_repack',70),('triple_repack',76),('insert',62),('reorient',105)]:
  found=[(j,r) for j,r in rows if j['operator']==op and j['root']==root and r.get('accepted')]
  if not found:found=[(j,r) for j,r in rows if j['operator']==op and j['root']==root]
  j,r=found[0];target=r.get('chosen',{}).get('target')
  if not target:target=r['diagnostics']['targets'][0]['target']
  cases.append((j,target))
 results=[]
 for j,target in cases:
  out=DEFAULT_OUT;node=read(out/'nodes'/f'{j["parent"]}.json.gz');doc=restore(read(out/'source.json'),node['frame']);cfg=read(out/'manifest.json')['config'];row=dict(job=j,target=target,scope='Single saved opening; full original inventory/budgets/grids; no winning piece IDs or poses passed to search')
  for label,module in [('baseline',baseline),('prototype',prototype)]:
   old_replace=module.replacement_targets;old_insert=module.insertion_targets
   def replace(*args):return [t for t in old_replace(*args) if t['id']==target]
   def insert(*args):return [t for t in old_insert(*args) if t['id']==target]
   module.replacement_targets=replace;module.insertion_targets=insert
   start=time.monotonic()
   try:pool,diag=module.generate(doc,node['metrics'],cfg,j['operator'],j['phase'],np.random.default_rng(j['seed']),lambda:False)
   finally:module.replacement_targets=old_replace;module.insertion_targets=old_insert
   write(REVIEW/f'job_{j["number"]}_{label}_pool.json.gz',dict(candidates=pool,diagnostics=diag))
   row[label]=dict(seconds=time.monotonic()-start,candidates=len(pool),piece_sets=len({tuple(sorted(x['stone'] for x in p['placements'])) for p in pool}),best_gain_sqft=max((p['area_gain_sqft'] for p in pool),default=None),all_eligible=all(eligible(doc,p,cfg['physics']) for p in pool))
   print(j['number'],target,label,row[label],flush=True)
  results.append(row);write(REVIEW/'benchmark_results.json',dict(cases=results,prototype_sha=sha(REVIEW/'planner_v2.py'),baseline_sha=sha(ROOT/'analysis/frontier_search/planner.py'),scope='Geometry candidate recall and runtime only; does not establish physical validity or search gains.'))

if __name__=='__main__':run()
