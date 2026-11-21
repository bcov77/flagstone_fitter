"""Check selected broad-void comparisons at 1-inch and 0.5-inch grid spacing."""
import json,math
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
OUT=Path(__file__).resolve().parent
screen=json.loads((OUT/'screening.json').read_text());doc=json.loads(Path(screen['source']).read_text())
split=list(OUT.glob('trials_*_*.json'))
trials=[r for p in split for r in json.loads(p.read_text())['trials']] if split else json.loads((OUT/'trials.json').read_text())['trials']
stones={str(s['id']):s for s in doc['stones']};patio=Polygon(doc['patio'])
def poly(sid,pose):
 x,y,t=pose;c,s=math.cos(t),math.sin(t)
 return affine_transform(Polygon(stones[sid]['outline']),(c,-s,s,c,x,y))
output=[]
for target,candidate,protocol in [('119','119','direct'),('119','117','direct'),('29','47','direct'),('29','47','gap_cycle'),('35','35','direct'),('35','150','direct')]:
 for seed in [17,41]:
  r=next(r for r in trials if (r['target'],r['candidate'],r['protocol'],r['seed'])==(target,candidate,protocol,seed))
  tree=shapely.STRtree([poly(sid,pose) for sid,pose in r['poses'].items()]);center=np.array(poly(target,stones[target]['pose']).centroid.coords[0])
  for step_inches in [1,.5]:
   a=np.arange(step_inches/24,patio.bounds[2],step_inches/12);xx,yy=np.meshgrid(a,a);coords=np.c_[xx.ravel(),yy.ravel()];pts=shapely.points(coords)
   idx,ds=tree.query_nearest(pts,return_distance=True,all_matches=False);distances=np.zeros(len(pts));distances[idx[0]]=ds
   distances=np.minimum(distances,shapely.distance(pts,patio.boundary));broad=distances>1/12;far=np.linalg.norm(coords-center,axis=1)>2
   output.append(dict(target=target,candidate=candidate,protocol=protocol,seed=seed,step_inches=step_inches,
    broad_void_sqft=float(broad.sum()*(step_inches/12)**2),far_broad_void_sqft=float((broad&far).sum()*(step_inches/12)**2)))
(OUT/'grid_sensitivity.json').write_text(json.dumps(output,indent=2))
print('Grid sensitivity checked:',len(output),'measurements')
