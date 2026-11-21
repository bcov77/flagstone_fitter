"""Reproducible finishing pilot. Run with physics_patio/.venv and PYTHONPATH.
No live-server access. Checkpoints contain compact poses, not copied outlines.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,random,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
from shapely.ops import unary_union
from patio_physics.engine import PhysicsSession
ROOT=Path(__file__).resolve().parents[2]
DEFAULT=Path(__file__).resolve().parent/'pilot_v1'
SOURCE=ROOT/'examples/quarter2_20h_ready_20261114/tasks/task_576_escape/best.json'
CACHE={};DOC=None

def write(path,data):
 path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def world(s):
 x,y,t=s['pose'];c,sn=math.cos(t),math.sin(t)
 return affine_transform(Polygon(s['outline']),(c,-sn,sn,c,x,y))
def poses(session):return {sid:p.pose[:] for sid,p in session.pieces.items() if p.active}
def geometry(session):
 return {sid:world({'pose':p.pose,'outline':p.outline}) for sid,p in session.pieces.items() if p.active}
def clearance(session):
 gs=geometry(session);items=list(gs.items());minimum=1e6;deficit=0.;pairs=0;overlap=0.;walls=0;wall_deficit=0.
 for i,(sid,g) in enumerate(items):
  wd=g.distance(session.patio.boundary)*12
  if not session.patio.covers(g) or wd<.5-1.2e-7:walls+=1
  wall_deficit+=max(0,.5-wd)
  for tid,h in items[i+1:]:
   d=g.distance(h)*12;minimum=min(minimum,d);deficit+=max(0,.5-d)
   if d<.5-1.2e-7:pairs+=1
   if d==0:overlap+=g.intersection(h).area
 outside=sum(g.difference(session.patio).area for g in gs.values())
 return dict(valid=not pairs and not walls,pair_violations=pairs,wall_violations=walls,minimum_pair_inches=minimum,
  pair_deficit_inches=deficit,wall_deficit_inches=wall_deficit,overlap_sqft=overlap,outside_sqft=outside)
def motion(session):
 bs=list(session.bodies.values())
 return dict(max_speed_ips=max((b.velocity.length*12 for b in bs),default=0),
  mean_speed_ips=float(np.mean([b.velocity.length*12 for b in bs])),
  max_angular_speed_dps=max((abs(math.degrees(b.angular_velocity)) for b in bs),default=0))
def quality(session,center):
 gs=geometry(session);union=unary_union(list(gs.values()));patio=session.patio
 xx,yy=np.meshgrid(np.arange(1/24,patio.bounds[2],1/12),np.arange(1/24,patio.bounds[3],1/12))
 coords=np.c_[xx.ravel(),yy.ravel()];points=shapely.points(coords)
 idx,ds=shapely.STRtree(list(gs.values())).query_nearest(points,return_distance=True,all_matches=False)
 distance=np.zeros(len(points));distance[idx[0]]=ds;distance=np.minimum(distance,shapely.distance(points,patio.boundary))
 far=np.linalg.norm(coords-np.array(center),axis=1)>2
 result={'covered_area_sqft':union.intersection(patio).area,'active_count':len(gs),'nominal_coverage':sum(g.area for g in gs.values())/patio.area}
 for radius in [.5,1,1.5]:
  mask=distance>radius/12;key=str(radius)
  result['broad_void_'+key]=float(mask.sum()/144);result['far_void_'+key]=float((mask&far).sum()/144)
 return result

def proposal_features(document,target):
 stones={str(s['id']):s for s in document['stones']};candidate=stones[target];g=world(candidate)
 neighbors=[world(s) for sid,s in stones.items() if s['active'] and sid!=target]
 patio=Polygon(document['patio']);obs=unary_union(neighbors);exclusion=unary_union([obs.buffer(.5/12),patio.buffer(3).difference(patio.buffer(-.5/12))])
 samples=shapely.line_interpolate_point(g.exterior,np.linspace(0,g.length,192,endpoint=False))
 distances=np.minimum(shapely.distance(samples,obs),shapely.distance(samples,patio.boundary))*12
 return dict(area_sqft=g.area,clearance_intrusion_sqft=g.intersection(exclusion).area,
  mean_excess_gap_inches=float(np.mean(np.clip(distances-.5,0,4))),gap_quantiles_inches=np.quantile(distances,[.1,.5,.9]).tolist(),
  near_fraction=float(np.mean((distances>=.4)&(distances<=1))),feature_only_fullsize=True)

def prepare(out):
 out.mkdir(parents=True,exist_ok=True);manifest=out/'manifest.json'
 if manifest.exists():
  m=json.loads(manifest.read_text())
  if m['runner_sha256']!=digest(__file__) or m['engine_sha256']!=digest(ROOT/'physics_patio/patio_physics/engine.py'):raise ValueError('Code changed: use a new output directory')
  if digest(out/'source.json')!=m['source_sha256']:raise ValueError('Source copy hash mismatch')
  return m
 raw=SOURCE.read_bytes();d=json.loads(raw);(out/'source.json').write_bytes(raw)
 stones={str(s['id']):s for s in d['stones']};active={sid for sid,s in stones.items() if s['active']}
 center=world(stones['60']).centroid.coords[0]
 # Reserve proposals chosen geometrically; recorded explicitly, not ground truth.
 reserve=[sid for sid in stones if sid not in active]
 def choose(target,removed):
  old=world(stones[target]);ct=old.centroid;obstacles=unary_union([world(stones[sid]).simplify(.003) for sid in active-set(removed)])
  patio=Polygon(d['patio']);forbidden=unary_union([obstacles.buffer(.5/12),patio.buffer(3).difference(patio.buffer(-.5/12))])
  options=sorted(reserve,key=lambda sid:abs(math.log(Polygon(stones[sid]['outline']).area/old.area)))[:12]
  best=None
  for sid in options:
   shape=Polygon(stones[sid]['outline']).simplify(.003);lc=shape.centroid
   for deg in range(0,360,30):
    t=math.radians(deg);c,sn=math.cos(t),math.sin(t)
    pose=[ct.x-c*lc.x+sn*lc.y,ct.y-sn*lc.x-c*lc.y,t]
    g=world({'outline':list(shape.exterior.coords),'pose':pose});score=8*g.intersection(forbidden).area-g.area
    if best is None or score<best[0]:best=(score,sid,pose)
  return {'id':best[1],'pose':best[2],'pool':options,'screen':'12 closest-area reserve pieces, centered, 30-degree rotations, minimum 8*intrusion-area'}
 insert=choose('60',['50','60','61']);replace=choose('61',['61'])
 target61center=world(stones['61']).centroid.coords[0]
 cases=[
 dict(name='full_unchanged',family='control',remove=[],candidate=None,center=list(target61center),control='full_unchanged'),
 dict(name='loose_unchanged',family='control',remove=['50','60','61'],candidate=None,center=list(center),control='loose_unchanged'),
 dict(name='insert_known_good',family='insertion',remove=['50','60','61'],candidate={'id':'60','pose':stones['60']['pose']},center=list(center),control='loose_unchanged'),
 dict(name='insert_perturbed',family='insertion',remove=['50','60','61'],candidate={'id':'60','pose':[stones['60']['pose'][0]+.15,stones['60']['pose'][1]+.15,stones['60']['pose'][2]+math.radians(10)]},center=list(center),control='loose_unchanged'),
 dict(name='insert_reserve',family='insertion',remove=['50','60','61'],candidate=insert,center=list(center),control='loose_unchanged'),
 dict(name='replace_original',family='replacement_control',remove=['61'],candidate={'id':'61','pose':stones['61']['pose']},center=list(target61center),control='full_unchanged'),
 dict(name='replace_reoriented',family='replacement',remove=['61'],candidate={'id':'61','pose':[stones['61']['pose'][0],stones['61']['pose'][1],stones['61']['pose'][2]+math.radians(15)]},center=list(target61center),control='replace_original'),
 dict(name='replace_reserve',family='replacement',remove=['61'],candidate=replace,center=list(target61center),control='replace_original')]
 tasks=[]
 for case in cases:
  for seed in [17,41]:
   for protocol in ['direct','gap_cycle']:
    for finish in ['retain','release','margin_release']:
     task=dict(case=case,seed=seed,protocol=protocol,finish=finish);task['id']=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()[:16];tasks.append(task)
 m=dict(version=1,source=str(SOURCE),source_sha256=hashlib.sha256(raw).hexdigest(),runner_sha256=digest(__file__),engine_sha256=digest(ROOT/'physics_patio/patio_physics/engine.py'),cases=cases,tasks=tasks,
  settings={'piece_hardness':18,'wall_hardness':40,'angular_stiffness':30,'translation_stiffness':30,'rotation_magnitude':28,'rotation_force':12,'rotation_frequency':5,'translation_magnitude':.25,'translation_force':12,'translation_frequency':1,'gap_inches':.5},
  timing={'explore_phase_seconds':[.5,.5,1.0],'quiet_max_seconds':4,'quiet_sample_seconds':.5,'post_margin_seconds':.5},
  convergence={'max_speed_ips':.02,'max_angular_speed_dps':.1,'max_pair_deficit_change_inches':.002,'consecutive_samples':2},
  notes=['Synthetic loose context removes 50/60/61; known-good insertion restores 60 at its original valid pose.','3 finishing policies: retain restoring gains; release all restoring gains; release with 0.6-inch collision gap then 0.5-inch gap.','No full factorial hardness ablation in this first pilot. Exploration uses loose 18/40; all finishing uses 48/89.','Convergence thresholds are provisional and evaluated, not a final recommended policy.'])
 write(manifest,m);return m

def initialize(out):
 global DOC,CACHE
 DOC=json.loads((Path(out)/'source.json').read_text());CACHE={}
def run_one(args):
 out,task,manifest=args;begin=time.monotonic();case=task['case'];d=copy.deepcopy(DOC)
 d['physics']={'settings':manifest['settings'],'pieces':{str(s['id']):{'anchor':s['pose'][:],'locked':False} for s in d['stones']}}
 for s in d['stones']:
  sid=str(s['id'])
  if sid in case['remove']:s['active']=False
  if case['candidate'] and sid==case['candidate']['id']:
   s['active']=True;s['pose']=case['candidate']['pose'][:];d['physics']['pieces'][sid]['anchor']=s['pose'][:]
 features=proposal_features(d,case['candidate']['id']) if case['candidate'] else None
 active={str(s['id']):s['active'] for s in d['stones']}
 for s in d['stones']:s['active']=False
 session=PhysicsSession(d,seed=task['seed']);session.proxy_cache=CACHE
 for sid,p in session.pieces.items():p.active=active[sid]
 session._build()
 initial=poses(session);before=clearance(session);before_quality=quality(session,case['center']);stages=[]
 for phase,seconds in enumerate(manifest['timing']['explore_phase_seconds']):
  gap=1.1 if task['protocol']=='gap_cycle' and phase==1 else .5
  session.edit('settings',values={'gap_inches':gap});session.start()
  for sid in session.drives:
   rng=random.Random(f'{task["seed"]}:{sid}:{phase}')
   session.drives[sid]=[(rng.uniform(.35,1.35),rng.uniform(0,2*math.pi)) for _ in range(3)]
  session.advance(seconds);stages.append(dict(phase='explore_'+str(phase),seconds=seconds,gap_inches=gap,poses=poses(session),motion=motion(session)))
  session.stop()
 changes={'rotation_magnitude':0,'translation_magnitude':0,'piece_hardness':48,'wall_hardness':89,'gap_inches':.6 if task['finish']=='margin_release' else .5}
 if task['finish']!='retain':changes.update(angular_stiffness=0,translation_stiffness=0,rotation_force=0,translation_force=0)
 session.edit('settings',values=changes);session.start();stable=0;last=None;converged=False;trace=[]
 for step in range(8):
  session.advance(.5);c=clearance(session);v=motion(session)
  if c['valid'] and v['max_speed_ips']<.02 and v['max_angular_speed_dps']<.1 and last is not None and abs(c['pair_deficit_inches']-last['pair_deficit_inches'])<.002:stable+=1
  else:stable=0
  trace.append(dict(seconds=(step+1)*.5,**c,**v));last=c
  if stable>=2:converged=True;break
 stages.append(dict(phase='finish',gap_inches=session.settings['gap_inches'],poses=poses(session)))
 session.stop()
 # Return inflated collision geometry to the physical contract and observe again.
 session.edit('settings',values={'gap_inches':.5});session.start();session.advance(.5)
 final_motion=motion(session);after=clearance(session);after_quality=quality(session,case['center']);finalposes=poses(session)
 converged=converged and after['valid'] and final_motion['max_speed_ips']<.02 and final_motion['max_angular_speed_dps']<.1
 session.stop()
 displacements=[math.hypot(finalposes[sid][0]-pose[0],finalposes[sid][1]-pose[1])*12 for sid,pose in initial.items()]
 result=dict(task=task,features=features,before=before,before_quality=before_quality,after=after,after_quality=after_quality,initial_poses=initial,stages=stages,final_poses=finalposes,trace=trace,final_motion=final_motion,converged=converged,timeout=not converged,quiet_seconds=len(trace)*.5,mean_translation_inches=float(np.mean(displacements)),max_translation_inches=max(displacements),wall_seconds=time.monotonic()-begin)
 write(Path(out)/'results'/f'{task["id"]}.json',result);return task['id'],case['name'],task['finish'],after['valid'],converged,result['wall_seconds']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=DEFAULT);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--max-wall-seconds',type=float,default=1200);ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--limit',type=int);a=ap.parse_args()
 if a.workers<1 or a.max_wall_seconds<=0:ap.error('workers and wall budget must be positive')
 m=prepare(a.out);(a.out/'results').mkdir(exist_ok=True)
 if a.prepare_only:print('Prepared',len(m['tasks']),'tasks');return
 remaining=[t for t in m['tasks'] if not (a.out/'results'/f'{t["id"]}.json').exists()]
 if a.limit:remaining=remaining[:a.limit]
 started=time.monotonic();completed=0
 with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(str(a.out),)) as pool:
  pending={};index=0
  while pending or index<len(remaining):
   while len(pending)<a.workers and index<len(remaining) and time.monotonic()-started<a.max_wall_seconds:
    task=remaining[index];pending[pool.submit(run_one,(str(a.out),task,m))]=task['id'];index+=1
   if not pending:break
   done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
   for future in done:
    tid=pending.pop(future)
    try:
     result=future.result();completed+=1;print(completed,'/',len(remaining),result,flush=True)
    except Exception as exc:
     write(a.out/f'error_{tid}.json',{'task':tid,'error':repr(exc)});raise
 print('Finished',completed,'new trials; elapsed',round(time.monotonic()-started,1),'seconds; checkpoints',len(list((a.out/'results').glob('*.json'))),flush=True)
if __name__=='__main__':main()
