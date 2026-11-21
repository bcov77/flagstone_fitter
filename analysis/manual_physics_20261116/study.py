"""Read-only measurements of manual placement and original starting board."""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
from shapely.geometry import Polygon
from shapely.affinity import affine_transform

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCES={
 'original':ROOT/'examples/quarter2_20h_ready_20261114/tasks/task_576_escape/best.json',
 'manual':Path('/Users/brian/Downloads/patio-physics-2026-11-16T15-45-13.444Z.json'),
}
fig,axes=plt.subplots(1,2,figsize=(15,8))
results={}
for ax,(name,path) in zip(axes,SOURCES.items()):
 d=json.loads(path.read_text()); patio=Polygon(d['patio']); shapes={}
 for stone in d['stones']:
  if not stone.get('active',False):continue
  x,y,t=stone['pose'];c,s=math.cos(t),math.sin(t)
  shapes[str(stone['id'])]=affine_transform(Polygon(stone['outline']),(c,-s,s,c,x,y))
 pairs=[]; violations=[]; wall_violations=[]
 for i,(sid,a) in enumerate(shapes.items()):
  if not patio.covers(a) or a.distance(patio.boundary)*12<.5-1.2e-7:
   wall_violations.append({'id':sid,'distance_inches':a.distance(patio.boundary)*12,'outside_area_sqft':a.difference(patio).area})
  for tid,b in list(shapes.items())[i+1:]:
   dist=a.distance(b)*12
   pairs.append(dist)
   if dist<.5-1.2e-7:violations.append({'ids':[sid,tid],'distance_inches':dist,'overlap_sqft':a.intersection(b).area})
 coverage=sum(g.area for g in shapes.values())/patio.area
 results[name]={'source':str(path),'active_count':len(shapes),'active_ids':list(shapes),'patio_area_sqft':patio.area,'stone_area_sqft':sum(g.area for g in shapes.values()),'nominal_coverage':coverage,'minimum_pair_inches':min(pairs) if pairs else None,'pair_clearance_violations':violations,'boundary_clearance_violations':wall_violations,'locked_active_count':sum(bool(d.get('physics',{}).get('pieces',{}).get(sid,{}).get('locked',False)) for sid in shapes),'local_maximum_gap_rule_checked':False}
 for sid,g in shapes.items():
  ax.add_patch(Patch(list(g.exterior.coords),facecolor='#d6c2a0',edgecolor='#554a39',linewidth=.6))
  centroid=g.representative_point();ax.text(centroid.x,centroid.y,sid,ha='center',va='center',fontsize=7)
 x,y=patio.exterior.xy;ax.plot(x,y,color='black');ax.set_aspect('equal');ax.set_xlim(-.1,patio.bounds[2]+.1);ax.set_ylim(-.1,patio.bounds[3]+.1)
 ax.set_title(f'{name.capitalize()}: {len(shapes)} stones, {coverage:.1%} nominal coverage')
 ax.set_xlabel('Feet');ax.set_ylabel('Feet')
fig.suptitle('Original outlines at saved poses — IDs shown; no simulation run')
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
results['added_ids']=sorted(set(results['manual']['active_ids'])-set(results['original']['active_ids']),key=int)
results['removed_ids']=sorted(set(results['original']['active_ids'])-set(results['manual']['active_ids']),key=int)
(OUT/'measurements.json').write_text(json.dumps(results,indent=2)+'\n')
for k,v in results.items():
 if isinstance(v,dict):print(k,{key:(len(val) if isinstance(val,list) else val) for key,val in v.items() if key!='active_ids'})
 else:print(k,v)
