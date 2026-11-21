"""Bounded two-depth late-board experiment. Read-only campaign inputs."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import argparse,copy,hashlib,json,math,sys,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'analysis/aggressive_trajectories'),str(ROOT/'analysis/physics_trajectories'),str(ROOT/'physics_patio'),str(ROOT/'analysis/physics_placement_campaign'),str(ROOT/'src')]
import numpy as np
import shapely
from shapely.ops import unary_union
from scipy.optimize import minimize
from common import read,write,sha,state,restore,shapes,world,measure,clearance,GAP
from selector import apply,coarse,pockets,parts,fit
from aggressive_policy import generate
from aggressive_geometry import eligible,canonical,board_key
from aggressive_physics import relax
from aggressive_archive import enrich
from physics import session_for,quiet,motion
BATCH=ROOT/'examples/aggressive_trajectories_20261118_r2/aggressive'
CASES=[62,105,70,76]
ARMS=['ordinary_insert','ordinary_replace','probe_insert','probe_replace','compound_fill','compound_replace','neighbor_refill','reorientation']
SEEDS=[2026111801,2026111802]

def config():
 cfg=read(ROOT/'analysis/aggressive_trajectories/config.json')
 cfg.update(target_count=8,target_budget=8,ordinary_piece_budget=48,scaled_piece_budget=96)
 return cfg

def arm(p):
 ids={x['stone'] for x in p['placements']};removed=set(p['remove'])
 if len(ids)>1:
  return 'neighbor_refill' if ids&removed else 'compound_replace' if removed else 'compound_fill'
 if len(removed)>1:return 'compound_replace'
 if removed==ids:return 'reorientation'
 return ('probe_' if p['physics_protocol']=='free_pulses' else 'ordinary_')+('replace' if removed else 'insert')

def paired(doc,cfg,rng):
 """Full-size two-piece search using the existing ordinary fit prerequisites."""
 board=shapes(doc);patio=shapely.Polygon(doc['patio']);local={str(s['id']):shapely.Polygon(s['outline']) for s in doc['stones']}
 free,holes=pockets(patio,board)
 targets=[dict(id=f'pair_insert_{i}',old=None,pocket=p,free=free) for i,p in enumerate(holes[:6])]
 locked={sid for sid,v in doc.get('physics',{}).get('pieces',{}).items() if v.get('locked')}
 # Revisit stones bordering the largest residual holes, without supplying piece IDs/poses.
 focus=unary_union(holes[:4]) if holes else patio
 changeable=sorted((s for s in board if s not in locked),key=lambda s:(board[s].distance(focus),board[s].area))[:6]
 for sid in changeable:
  f,ps=pockets(patio,{s:g for s,g in board.items() if s!=sid})
  relevant=[p for p in ps if p.intersection(board[sid]).area>.35*board[sid].area]
  if relevant:targets.append(dict(id=f'neighbor_{sid}',old=sid,pocket=max(relevant,key=lambda p:p.intersection(board[sid]).area),free=f))
 result=[];diagnostics=[]
 search_cfg={**cfg,'piece_budget':96,'coarse_per_target':200}
 for t in targets:
  available=[s for s in local if s not in board] if t['old'] is None else [t['old']]
  candidates=coarse(local,available,t['pocket'],t['free'],search_cfg,rng,min_fraction=.20)
  count=0
  for _,sid,pose in candidates[:16]:
   if t['old']:
    single=dict(remove=[sid],placements=[dict(stone=sid,pose=pose)])
    if not eligible(doc,single,cfg):continue
   pl=[dict(stone=sid,pose=pose)]
   first=fit(pl,t,local,board,patio,cfg,min_coverage=.20)
   if first is None:continue
   fg=world(local[sid],pose);residual=t['pocket'].difference(fg.buffer(GAP))
   chunks=sorted(parts(residual.buffer(-.04).buffer(.04).intersection(residual)),key=lambda p:-p.area)
   for chunk in chunks[:2]:
    if chunk.area<.06:continue
    free2=t['free'].difference(fg.buffer(GAP))
    others=[s for s in local if s not in board and s!=sid]
    for _,second,q in coarse(local,others,chunk,free2,search_cfg,rng,min_fraction=.20)[:4]:
     # Improve only the new fill pose, with the held first piece as a physical boundary.
     def objective(v):
      g=world(local[second].simplify(.004,preserve_topology=True),v)
      return -g.intersection(chunk).area+g.difference(chunk).area+30*g.difference(free2).area
     bounds=[(q[0]-.09,q[0]+.09),(q[1]-.09,q[1]+.09),(q[2]-.10,q[2]+.10)]
     q=minimize(objective,q,method='Powell',bounds=bounds,options={'maxiter':10,'xtol':.001,'ftol':.001}).x.tolist()
     placements=pl+[dict(stone=second,pose=q)];f=fit(placements,t,local,board,patio,cfg)
     if f is None:continue
     gain=sum(local[x['stone']].area for x in placements)-(local[t['old']].area if t['old'] else 0)
     p=dict(family='compound',physics_protocol='held_release',target=t['id'],remove=[t['old']] if t['old'] else [],placements=placements,fit=f,area_gain_sqft=gain,rank_key=[0,f['clearance_intrusion_fraction'],-f['coverage'],-gain],research_operator='joint full-size pair / neighbor reorientation plus new fill')
     if eligible(doc,p,cfg):result.append(p);count+=1
  diagnostics.append(dict(target=t['id'],first_poses=len(candidates),eligible_pairs=count))
 return result,diagnostics

def portfolio(doc,cfg,seed,dest):
 if dest.exists():return read(dest)
 start=time.monotonic();rng=np.random.default_rng(seed)
 pool,diag=generate(doc,cfg,rng)
 extra,pair_diag=paired(doc,cfg,rng);pool.extend(extra)
 unique={canonical(dict(remove=p['remove'],placements=p['placements'])):p for p in pool if eligible(doc,p,cfg)}
 groups={a:[] for a in ARMS}
 for p in unique.values():groups[arm(p)].append(p)
 for a in groups:groups[a].sort(key=lambda p:p['rank_key'])
 result=dict(groups=groups,diagnostics=diag,pair_diagnostics=pair_diag,seconds=time.monotonic()-start)
 write(dest,result)
 print(f'{dest.parent.name}: searched portfolio in {result["seconds"]:.1f}s '+str({a:len(v) for a,v in groups.items()}),flush=True)
 return result

def evaluate(doc):return enrich(doc,measure(doc))

def roi_for(doc,p):
 pre=shapes(doc);post=shapes(apply(doc,p));ids=set(p['remove'])|{x['stone'] for x in p['placements']}
 core=unary_union([g for s in ids for g in [pre.get(s),post.get(s)] if g is not None])
 neighbors=[g for g in pre.values() if g.distance(core)<.25]
 return unary_union([core.buffer(.3),*[g.buffer(.1) for g in neighbors]]).intersection(shapely.Polygon(doc['patio']))

def local_metrics(doc,roi):
 board=shapes(doc);patio=shapely.Polygon(doc['patio']);union=unary_union(list(board.values()))
 x0,y0,x1,y1=roi.bounds;step=.5/12
 xx,yy=np.meshgrid(np.arange(x0+step/2,x1,step),np.arange(y0+step/2,y1,step));points=shapely.points(np.column_stack([xx.ravel(),yy.ravel()]));points=points[shapely.contains(roi,points)]
 tree=shapely.STRtree([*board.values(),patio.boundary]);_,dist=tree.query_nearest(points,all_matches=False,return_distance=True)
 return dict(stone_area_sqft=union.intersection(roi).area,broad_void_sqft=float(np.sum(dist>1/12)*step**2),largest_sampled_gap_inches=float(24*max(dist,default=0)),scope='Fixed action neighborhood; approximate half-inch grid void/max gap; stone area from full outlines')

def control(doc,cfg,seed):
 session=session_for(doc,seed);settled,trace=quiet(session,cfg);velocity=motion(session);end=session.export();c=clearance(end);session.stop()
 return end,dict(accepted=bool(settled and c['valid']),settled=settled,clearance=c,final_motion=velocity,phases=[dict(name='quiet_control',trace=trace)],reason='quiet_control')

def trial(folder,ident,doc,p,cfg,seed,parent=None,roi=None):
 path=folder/f'{ident}.json.gz'
 if path.exists():return read(path)
 start=time.monotonic()
 if p is None:end,result=control(doc,cfg,seed)
 else:
  if not eligible(doc,p,cfg):raise ValueError('Ineligible action')
  result=relax(doc,p,cfg,seed);end=result.pop('attempted_document')
 attempted=evaluate(end)
 accepted=result['accepted'];retained=end if accepted else doc
 r=dict(id=ident,parent=parent,seed=seed,action=p,start=state(doc),attempted_end=state(end),end=state(retained),attempted_metrics=attempted,end_metrics=attempted if accepted else evaluate(doc),seconds=time.monotonic()-start,**result)
 if roi is not None:r.update(roi_wkt=roi.wkt,local_before=local_metrics(doc,roi),local_attempted=local_metrics(end,roi),local_end=local_metrics(retained,roi))
 write(path,r)
 print(f'{folder.name} {ident}: {"accepted" if accepted else "rollback"} {r["end_metrics"]["area_percent"]:.3f}% ({r["seconds"]:.1f}s)',flush=True)
 return r

def quality(r,root):
 m=r['end_metrics'];q=m['quality'];base=root['quality']
 return (round(max(0,base['wall_joint_inches']-q['wall_joint_inches'])/3),q['broad_void_sqft'],m['gaps']['largest_interior_inches'],-m['area_percent'])

def run_case(number):
 cfg=config();folder=HERE/f'run_{number:06d}';folder.mkdir(exist_ok=True)
 source_path=BATCH/f'runs/run_{number:06d}/final.json';doc=read(source_path)
 if not (79<=evaluate(doc)['area_percent']<=82.1):raise ValueError('Unexpected root coverage')
 write(folder/'source.json',doc);root=evaluate(doc);write(folder/'root_metrics.json',root)
 pool=portfolio(doc,cfg,20261118+number,folder/'portfolio.json.gz')
 controls=[];trials=[];roots=[]
 for si,seed in enumerate(SEEDS):
  c=trial(folder,f'control_s{si}',doc,None,cfg,seed);controls.append(c)
 for a in ARMS:
  options=pool['groups'][a]
  if not options:continue
  # One deterministic top-ranked hypothesis per arm, replicated under two seeds.
  p=options[0];roi=roi_for(doc,p)
  for si,seed in enumerate(SEEDS):
   r=trial(folder,f'root_{a}_s{si}',doc,p,cfg,seed,roi=roi);r['arm']=a;trials.append(r)
   if r['accepted']:roots.append(r)
 # Independently per physics seed, extend area- and quality-favored endpoints.
 follow=[]
 for si,seed in enumerate(SEEDS):
  candidates=[r for r in roots if r['seed']==seed]
  if not candidates:continue
  best_area=max(candidates,key=lambda r:r['end_metrics']['area_percent'])
  best_quality=min(candidates,key=lambda r:quality(r,root))
  parents={r['id']:r for r in [best_area,best_quality]}
  for parent in parents.values():
   current=restore(doc,parent['end']);roi=shapely.from_wkt(parent['roi_wkt'])
   candidate_pool=portfolio(current,cfg,seed+100,folder/f'follow_pool_{parent["id"]}.json.gz')
   choices=[]
   for group in ['ordinary_insert','probe_insert','compound_fill','neighbor_refill','ordinary_replace','probe_replace']:
    for p in candidate_pool['groups'][group]:
     proposed=shapes(apply(current,p));g=unary_union([proposed[x['stone']] for x in p['placements']])
     if g.intersection(roi).area>.05:
      choices.append((group,p));break
   # Give the next step an insertion opportunity, plus a distinct other operator.
   for ci,(group,p) in enumerate(choices[:2]):
    ident=f'follow_{parent["id"]}_{ci}_{group}'
    r=trial(folder,ident,current,p,cfg,seed+1000+ci,parent=parent['id'],roi=roi);r['arm']=group;follow.append(r)
 all_results=trials+follow
 valid=[r for r in all_results if r['accepted']]
 best=max(valid,key=lambda r:r['end_metrics']['area_percent']) if valid else None
 if best:write(folder/'best_coverage.json',restore(doc,best['end']))
 write(folder/'summary.json',dict(run=number,root=root,available={a:len(v) for a,v in pool['groups'].items()},controls=[c['id'] for c in controls],trials=[{k:r[k] for k in ['id','parent','seed','arm','accepted','reason','end_metrics','seconds']} for r in all_results],best_coverage=best['id'] if best else None,complete=True))
 return number

def manifest():
 paths=[Path(__file__),ROOT/'analysis/aggressive_trajectories/config.json',*sorted((ROOT/'analysis/aggressive_trajectories').glob('*.py')),*sorted((ROOT/'analysis/physics_trajectories').glob('*.py')),ROOT/'physics_patio/patio_physics/engine.py',*sorted((ROOT/'analysis/physics_placement_campaign').glob('*geometry.py')),*[ROOT/f'src/flagstone/{p}.py' for p in ['quarter_gaps','quarter','phase1']]]
 return dict(cases=CASES,seeds=SEEDS,config=config(),sources={str(p.relative_to(ROOT)):sha(p) for n in CASES for p in [BATCH/f'runs/run_{n:06d}/final.json']},dependencies={str(p.relative_to(ROOT)):sha(p) for p in paths},scope='Separate bounded research pilot; same full-size acceptance and eligibility. Broader proposal budgets, one top hypothesis/arm, two seeds, two-depth selective branching. No causal generalization from selected branches.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',type=int);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
 m=manifest();path=HERE/'manifest.json'
 if path.exists():assert read(path)==m,'Code/config/input changed; use a versioned experiment'
 else:write(path,m)
 if args.case:run_case(args.case)
 else:
  with ProcessPoolExecutor(max_workers=args.workers) as ex:
   jobs={ex.submit(run_case,n):n for n in CASES}
   for f in as_completed(jobs):print(f'COMPLETED run {f.result()}',flush=True)
 print('Experiment complete',flush=True)
