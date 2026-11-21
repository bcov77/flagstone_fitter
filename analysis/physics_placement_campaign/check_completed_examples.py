"""Repeat selected candidate/control void comparisons on a finer 0.5-inch grid."""
import json,math
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
OUT=Path(__file__).with_name('campaign_v1');DEST=OUT/'completed_analysis'
m=json.loads((OUT/'manifest.json').read_text());doc=json.loads((OUT/'source.json').read_text());stones={str(s['id']):s for s in doc['stones']};cases={c['id']:c for c in m['cases']};patio=Polygon(doc['patio'])
lookup={(t['case'],t['seed'],t['protocol']):t['id'] for t in m['tasks']}
selected=json.loads((DEST/'selected_examples.json').read_text());result=[]
def quality(tid,center):
 r=json.loads((OUT/'results'/f'{tid}.json').read_text());gs=[]
 for sid,(x,y,t) in r['final_poses'].items():
  c,s=math.cos(t),math.sin(t);gs.append(affine_transform(Polygon(stones[sid]['outline']),(c,-s,s,c,x,y)))
 a=np.arange(.5/24,patio.bounds[2],.5/12);xx,yy=np.meshgrid(a,a);coords=np.c_[xx.ravel(),yy.ravel()];pts=shapely.points(coords)
 idx,ds=shapely.STRtree(gs).query_nearest(pts,return_distance=True,all_matches=False);distances=np.zeros(len(pts));distances[idx[0]]=ds;distances=np.minimum(distances,shapely.distance(pts,patio.boundary))
 broad=distances>1/12;far=np.linalg.norm(coords-np.asarray(center),axis=1)>2
 return float(broad.sum()/576),float((broad&far).sum()/576)
for example in selected:
 case=cases[example['case']];control=lookup[(case['control'],example['seed'],example['protocol'])]
 a,af=quality(example['task_id'],case['center']);b,bf=quality(control,case['center'])
 row={'label':example['label'],'task':example['task_id'],'grid_inches':.5,'void_delta_sqft':a-b,'far_delta_sqft':af-bf,'local_delta_sqft':(a-af)-(b-bf)};result.append(row);print(row,flush=True)
(DEST/'example_grid_sensitivity.json').write_text(json.dumps(result,indent=2))
