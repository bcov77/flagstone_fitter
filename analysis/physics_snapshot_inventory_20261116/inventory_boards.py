"""Inventory real saved boards; preserve source bytes and never contact live server."""
import json,math,hashlib
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sources=[('Original start',ROOT/'examples/quarter2_20h_ready_20261114/tasks/task_576_escape/best.json'),('Earlier manual save',ROOT/'physics_patio/sessions/recovery_before_gap_upgrade_20261115_102954.json'),('Final manual export',Path('/Users/brian/Downloads/patio-physics-2026-11-16T15-45-13.444Z.json'))]
fig,axes=plt.subplots(1,3,figsize=(18,6.4));results=[]
for ax,(label,path) in zip(axes,sources):
 raw=path.read_bytes();d=json.loads(raw);patio=Polygon(d['patio']);gs={}
 for s in d['stones']:
  if not s.get('active'):continue
  x,y,t=s['pose'];c,sn=math.cos(t),math.sin(t)
  gs[str(s['id'])]=affine_transform(Polygon(s['outline']),(c,-sn,sn,c,x,y))
 pair=[];walls=[];deficit=0
 for i,(sid,g) in enumerate(gs.items()):
  wd=g.distance(patio.boundary)*12
  if not patio.covers(g) or wd<.5-1.2e-7:walls.append(sid)
  for tid,h in list(gs.items())[i+1:]:
   distance=g.distance(h)*12;deficit+=max(0,.5-distance)
   if distance<.5-1.2e-7:pair.append({'ids':[sid,tid],'distance_inches':distance})
 area=sum(g.area for g in gs.values());stats=dict(label=label,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),file_modified=datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),active_count=len(gs),active_ids=list(gs),nominal_coverage=area/patio.area,pair_clearance_violations=pair,wall_clearance_violations=walls,pair_deficit_sum_inches=deficit,passes_clearance=not pair and not walls,settings=d.get('physics',{}).get('settings'),gap_rule_checked=False)
 results.append(stats)
 for sid,g in gs.items():
  ax.add_patch(Patch(list(g.exterior.coords),facecolor='#d6c2a0',edgecolor='#554a39',linewidth=.5))
  p=g.representative_point();ax.text(p.x,p.y,sid,ha='center',va='center',fontsize=6)
 x,y=patio.exterior.xy;ax.plot(x,y,'k-',linewidth=1);ax.set_xlim(-.1,patio.bounds[2]+.1);ax.set_ylim(-.1,patio.bounds[3]+.1);ax.set_aspect('equal')
 ax.set_title(f'{label}\n{len(gs)} stones; {area/patio.area:.1%} nominal coverage\nClearance violations: {len(pair)} pairs, {len(walls)} walls',fontsize=10)
 ax.set_xlabel('Feet')
fig.tight_layout();fig.savefig(OUT/'saved_boards.png',dpi=170,bbox_inches='tight');plt.close(fig)
(OUT/'inventory.json').write_text(json.dumps(results,indent=2))
for r in results:print(r['label'],r['active_count'],round(r['nominal_coverage']*100,2),len(r['pair_clearance_violations']),len(r['wall_clearance_violations']),round(r['pair_deficit_sum_inches'],2))
