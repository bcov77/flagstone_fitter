"""Exploratory, read-only candidate screening. Feet/radians throughout.
Run in repository .venv, then run settle_trials.py in physics_patio/.venv.
Scores are hypotheses, not acceptance criteria. Original geometry used in trials.
"""
from pathlib import Path
import json, math
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.affinity import affine_transform, scale
from shapely.ops import unary_union

OUT=Path(__file__).resolve().parent
SOURCE=Path('/Users/brian/Downloads/patio-physics-2026-11-16T15-45-13.444Z.json')
D=json.loads(SOURCE.read_text()); P=Polygon(D['patio']); GAP=.5/12
STONES={str(s['id']):s for s in D['stones']}
def world(outline,pose):
 x,y,t=pose;c,s=math.cos(t),math.sin(t)
 return affine_transform(outline,[c,-s,s,c,x,y])
ACTIVE={sid:world(Polygon(s['outline']),s['pose']) for sid,s in STONES.items() if s['active']}

def profile(g,obstacles):
 # Equal arc-length samples avoid overweighting densely traced outline vertices.
 points=shapely.line_interpolate_point(g.exterior,np.linspace(0,g.length,96,endpoint=False))
 ds=np.minimum(shapely.distance(points,obstacles),shapely.distance(points,P.boundary))
 return dict(close_fraction=float(np.mean((ds>=.4/12)&(ds<=1/12))),
             excess_gap_inches=float(np.mean(np.clip(ds*12-.5,0,4))),
             deep_fraction=float(np.mean(ds<.25/12)))

output={'source':str(SOURCE),'settings':D['physics']['settings'],'method':
 {'targets':['29','119','35'],'angles_degrees':15,'centroid_offsets_feet':[-.2,0,.2],
 'area_ratio_range':[.65,1.35],'boundary_samples':96,'screen_simplification_inches':.036,
 'scale_probe':.9,'note':'Known final slots, not reconstructed historical insertions. Screening uses simplified polygons; simulations use originals.'},'slots':[]}
for target in output['method']['targets']:
 ref=ACTIVE[target];center=ref.centroid
 obs=unary_union([g.simplify(.003) for sid,g in ACTIVE.items() if sid!=target])
 exclusion=unary_union([obs.buffer(GAP),P.buffer(2).difference(P.buffer(-GAP))])
 candidates=[]
 for sid,s in STONES.items():
  if s['active'] and sid!=target:continue
  shape=Polygon(s['outline']).simplify(.003)
  if not .65<=shape.area/ref.area<=1.35:continue
  bests={}
  for deg in range(0,360,15):
   t=math.radians(deg);c,sn=math.cos(t),math.sin(t);ct=shape.centroid
   for dx in [-.2,0,.2]:
    for dy in [-.2,0,.2]:
     pose=[center.x+dx-c*ct.x+sn*ct.y,center.y+dy-sn*ct.x-c*ct.y,t]
     g=world(shape,pose)
     intrusion=g.intersection(exclusion).area
     # Area-only control rewards occupying more space, penalizes clearance intrusion.
     base=8*intrusion-g.area
     metrics=profile(g,obs)
     edge=base+.2*metrics['excess_gap_inches']*g.length
     small=scale(g,xfact=.9,yfact=.9,origin='centroid')
     sm=profile(small,obs)
     scaled=8*small.intersection(exclusion).area-g.area+.2*sm['excess_gap_inches']*small.length
     row=dict(id=sid,pose=pose,area_sqft=g.area,intrusion_sqft=intrusion,**metrics,
              area_score=base,profile_score=edge,scaled_score=scaled)
     for rule,score in [('area_score',base),('profile_score',edge),('scaled_score',scaled)]:
      if rule not in bests or score<bests[rule][rule]:bests[rule]=row
  candidates.append({'id':sid,'best':bests})
 # Hold human placement fixed as a reference; choose one top unused piece per rule.
 selected=[dict(id=target,pose=STONES[target]['pose'],selected_by=['human'])]
 for rule in ['area_score','profile_score','scaled_score']:
  options=[c['best'][rule] for c in candidates if c['id']!=target]
  winner=min(options,key=lambda r:r[rule])
  prior=next((r for r in selected if r['id']==winner['id'] and r['pose']==winner['pose']),None)
  if prior:prior['selected_by'].append(rule)
  else:selected.append(dict(winner,selected_by=[rule]))
 output['slots'].append(dict(target=target,candidates_screened=len(candidates),selected=selected,screened=candidates))
 print(target,'screened',len(candidates),'selected',[(x['id'],x['selected_by']) for x in selected],flush=True)
 (OUT/'screening.json').write_text(json.dumps(output,indent=2))
