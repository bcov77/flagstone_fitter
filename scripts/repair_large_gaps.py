"""Try validated direct insertions near the largest sampled empty circle."""
import argparse,json,math,os,time
from pathlib import Path
from flagstone.diagnostics import sampled_voids
from flagstone.phase1 import load_json, save_json, validate
from flagstone.search import greedy_insert, coverage

ap=argparse.ArgumentParser()
ap.add_argument('--input',default='examples/mixed_parallel_search/run/best.json')
ap.add_argument('--output',default='examples/gap_repair')
ap.add_argument('--seconds',type=float,default=30)
ap.add_argument('--seed',type=int,default=42)
args=ap.parse_args()
if not math.isfinite(args.seconds) or args.seconds<=0: ap.error('seconds must be positive')
x=load_json(args.input,seed=args.seed); gap=edge=1/24
if x.units!='ft': raise ValueError('This repair experiment requires input units in feet')
errors=validate(x,gap=gap,edge=edge)
if errors: raise ValueError(f'Invalid input: {len(errors)} violations')
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
if (out/'best.json').exists() or (out/'report.json').exists():
    raise FileExistsError('Output already has a repair result; choose another output directory')
start=time.monotonic(); before=sampled_voids(x,spacing=1/24,diameter=1/3)
initial_coverage=coverage(x); initial_count=int(x.active.sum()); attempts=0; passes=0; history=[]
while time.monotonic()-start<args.seconds:
    sampled=sampled_voids(x,spacing=1/24,diameter=1/3)
    if not sampled['largest_center'] or sampled['maximum_diameter']<1/3: break
    cx,cy=sampled['largest_center']; radius=1.5
    unused=sorted((sid for i,sid in enumerate(x.ids) if not x.active[i]),key=lambda sid:x.shapes[sid].area)[:120]
    if not unused: break
    remaining=args.seconds-(time.monotonic()-start)
    if remaining<=0: break
    result=greedy_insert(x,gap=gap,edge=edge,proposals=60,strategy='random',stone_ids=unused,
        proposal_bounds=(cx-radius,cy-radius,cx+radius,cy+radius),time_limit=min(4,remaining))
    passes+=1;attempts+=result.attempts
    if result.accepted:
        if validate(x,gap=gap,edge=edge): raise RuntimeError('Accepted invalid layout')
        tmp=out/'best.tmp.json';save_json(x,tmp);os.replace(tmp,out/'best.json')
        history.append({'elapsed_seconds':time.monotonic()-start,'accepted':result.accepted,'coverage':coverage(x),'center':[cx,cy]})
        print(json.dumps(history[-1]),flush=True)
tmp=out/'best.tmp.json';save_json(x,tmp);os.replace(tmp,out/'best.json')
after=sampled_voids(x,spacing=1/24,diameter=1/3)
report={'input':args.input,'seed':args.seed,'time_budget_seconds':args.seconds,'elapsed_seconds':time.monotonic()-start,
        'attempts':attempts,'passes':passes,'before':before,'after':after,'coverage_before':initial_coverage,
        'coverage_after':coverage(x),'placed_before':initial_count,'placed_after':int(x.active.sum()),'history':history,
        'note':'Sampled circle diameter and core area are estimates, not exact hole measurements; no stones were moved.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
