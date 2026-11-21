"""Full-board counterfactual trials with original geometry; no server access.
Uses physics_patio/.venv; result metrics are descriptive, not certified gap audit.
"""
import copy,json,math,random,time,sys
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
from patio_physics.engine import PhysicsSession
OUT=Path(__file__).resolve().parent
TARGET=sys.argv[1] if len(sys.argv)>1 else None
TARGET_SEED=int(sys.argv[2]) if len(sys.argv)>2 else None
RESULT=OUT/(f'trials_{TARGET}_{TARGET_SEED}.json' if TARGET_SEED else f'trials_{TARGET}.json' if TARGET else 'trials.json')
SCREEN=json.loads((OUT/'screening.json').read_text())
DOC=json.loads(Path(SCREEN['source']).read_text())
P=Polygon(DOC['patio']); GAP=.5/12
# Fixed one-inch grid; sensitivity checked in separate analysis if conclusions depend on it.
x=np.arange(1/24,P.bounds[2],1/12);y=np.arange(1/24,P.bounds[3],1/12)
xx,yy=np.meshgrid(x,y);XY=np.column_stack([xx.ravel(),yy.ravel()]);POINTS=shapely.points(XY)

def shapes(session):
 out={}
 for sid,p in session.pieces.items():
  if not p.active:continue
  x,y,t=p.pose;c,s=math.cos(t),math.sin(t)
  out[sid]=affine_transform(Polygon(p.outline),(c,-s,s,c,x,y))
 return out

def metrics(session,initial,center):
 gs=shapes(session);items=list(gs.items());pairs=[];overlap=0.;deficit=0.
 for i,(sid,g) in enumerate(items):
  for tid,h in items[i+1:]:
   d=g.distance(h)*12
   if d<.5-1.2e-7:pairs.append({'ids':[sid,tid],'inches':d})
   deficit+=max(0,.5-d)
   if d==0:overlap+=g.intersection(h).area
 wall=[sid for sid,g in items if not P.covers(g) or g.distance(P.boundary)<GAP-1e-8]
 # query_nearest output indices are explicitly mapped back; ordering is not assumed.
 idx,ds=shapely.STRtree(list(gs.values())).query_nearest(POINTS,return_distance=True,all_matches=False)
 distances=np.zeros(len(POINTS));distances[idx[0]]=ds
 distances=np.minimum(distances,shapely.distance(POINTS,P.boundary))
 broad=distances>1/12
 far=np.linalg.norm(XY-np.asarray(center),axis=1)>2
 moved=[]
 for sid,g in items:
  if sid in initial:
   old=initial[sid];now=session.pieces[sid].pose
   moved.append(math.hypot(now[0]-old[0],now[1]-old[1])*12)
 return dict(active_count=len(gs),nominal_coverage=sum(g.area for g in gs.values())/P.area,
  clearance_pair_count=len(pairs),clearance_wall_count=len(wall),pair_deficit_sum_inches=deficit,
  minimum_pair_inches=min((g.distance(h)*12 for i,(_,g) in enumerate(items) for _,h in items[i+1:]),default=None),
  pair_overlap_sqft=overlap,outside_sqft=sum(g.difference(P).area for g in gs.values()),
  broad_void_sqft=float(broad.sum()/144),far_broad_void_sqft=float((broad&far).sum()/144),
  mean_translation_inches=float(np.mean(moved)),max_translation_inches=max(moved,default=0),
  passes_clearance=not pairs and not wall)

results={'method':{'seeds':[17,41],'protocols':['direct','gap_cycle'],'phase_seconds':[4,4,4,4],
 'gap_cycle_inches':[.5,1.1,.5],'quiet_phase':'last 4 seconds: jitter magnitudes zero; saved tethers retained',
 'drives':'Stable seed per piece ID and phase, matched across candidates and protocols',
 'global_void_proxy':'one-inch grid area with empty-circle radius >1 inch; not a certified local-maximum gap audit',
 'far_region':'points more than two feet from original target centroid',
 'anchors':'Preserve saved neighbor anchors; replacement anchor at proposed pose',
 'limitations':'Final-state counterfactuals, narrow local pose scan, two seeds, 16 simulated seconds, arbitrary screening coefficients. All full-board trials use original geometry.'},'trials':[]}
if RESULT.exists():
 results=json.loads(RESULT.read_text())
completed={(r['target'],r['candidate'],r['seed'],r['protocol']) for r in results['trials']}
proxy_cache={}
start=time.time()
for slot in SCREEN['slots']:
 if TARGET and slot['target']!=TARGET:continue
 target=slot['target']; original=next(s for s in DOC['stones'] if str(s['id'])==target)
 c=Polygon(original['outline']).centroid;x,y,t=original['pose'];center=[x+math.cos(t)*c.x-math.sin(t)*c.y,y+math.sin(t)*c.x+math.cos(t)*c.y]
 # Removing the target is a control for whether inserting anything repairs or damages the board.
 cases=slot['selected']+[{'id':None,'selected_by':['leave_empty']}]
 for case in cases:
  for seed in [17,41]:
   if TARGET_SEED and seed!=TARGET_SEED:continue
   for protocol in ['direct','gap_cycle']:
    if (target,case['id'],seed,protocol) in completed:continue
    d=copy.deepcopy(DOC)
    for s in d['stones']:
     if str(s['id'])==target:s['active']=False
     if case['id'] is not None and str(s['id'])==case['id']:
      s['active']=True;s['pose']=case['pose']
      d['physics']['pieces'][case['id']]={'anchor':case['pose'],'locked':False}
    active={str(s['id']):s['active'] for s in d['stones']}
    for s in d['stones']:s['active']=False
    session=PhysicsSession(d,seed=seed)
    for sid,p in session.pieces.items():p.active=active[sid]
    session.proxy_cache=proxy_cache
    session._build()
    initial={sid:p.pose[:] for sid,p in session.pieces.items() if p.active}
    before=metrics(session,initial,center)
    for phase in range(4):
     changes={'gap_inches':1.1 if protocol=='gap_cycle' and phase==1 else .5}
     if phase==3:changes.update(rotation_magnitude=0,translation_magnitude=0)
     session.edit('settings',values=changes);session.start()
     for sid in session.drives:
      rng=random.Random(f'{seed}:{sid}:{phase}')
      session.drives[sid]=[(rng.uniform(.35,1.35),rng.uniform(0,2*math.pi)) for _ in range(3)]
     session.advance(4);session.stop()
    after=metrics(session,initial,center)
    record=dict(target=target,candidate=case['id'],selected_by=case['selected_by'],seed=seed,protocol=protocol,before=before,after=after,
                poses={sid:p.pose for sid,p in session.pieces.items() if p.active})
    results['trials'].append(record)
    RESULT.write_text(json.dumps(results,indent=2))
    print(f"{len(results['trials'])}: slot {target} candidate {case['id']} seed {seed} {protocol}: pair violations {after['clearance_pair_count']}, broad void {after['broad_void_sqft']:.2f}; elapsed {time.time()-start:.0f}s",flush=True)
