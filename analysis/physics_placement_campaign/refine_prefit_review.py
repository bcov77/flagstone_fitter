"""Find fixed-neighbor feasible poses; isolated protrusions cannot hide in area means."""
import json,math,time
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
import build_prefit_review as b

DEST=b.DEST/'refined_results'

def refine(path):
 r=json.loads(path.read_text());t=r['target'];p=shapely.from_wkt(t['pocket']);free=shapely.from_wkt(t['free'])
 # Simplified, slightly inset region for pose finding only. All final metrics
 # use the exact unscaled stone against the original full-outline free space.
 inset=p.simplify(.002,preserve_topology=True).buffer(-.004)
 boundary=inset.boundary
 active=set(b.ACTIVE)-set(t['removed'])-({t['old']} if t['old'] else set());obstacles=unary_union([b.ORIGINAL[s] for s in active])
 seen=set();out=[]
 candidates=[]
 for c in r['candidates']:
  if c['stone'] in seen:continue
  seen.add(c['stone'])
  if c['metrics']['clearance_intrusion_fraction']>.07:continue
  candidates.append(c)
 for c in candidates[:14]:
  sid=c['stone'];q0=np.array(c['pose']);local=shapely.get_coordinates(shapely.segmentize(b.SIMPLE[sid].exterior,.04))
  def objective(q):
   angle=q[2];co,si=math.cos(angle),math.sin(angle);xy=local@np.array([[co,si],[-si,co]])+q[:2]
   bad=~shapely.contains_xy(inset,xy[:,0],xy[:,1])
   if not np.any(bad):return 1e-8*float(np.sum((q-q0)**2))
   ds=np.asarray(shapely.distance(shapely.points(xy[bad]),boundary))
   return float(ds.max()+5*np.mean(ds**2))
  # Several small angular starts help avoid flat tangencies at narrow bumps.
  best=None
  for angular in [0,-.08,.08]:
   bounds=[(q0[0]-.3,q0[0]+.3),(q0[1]-.3,q0[1]+.3),(q0[2]-.26,q0[2]+.26)]
   start=q0.copy();start[2]+=angular
   opt=minimize(objective,start,method='Powell',bounds=bounds,options={'maxiter':60,'xtol':.00008,'ftol':.0001})
   if best is None or opt.fun<best.fun:best=opt
   if opt.fun<1e-7:break
  q=best.x.tolist();g=b.world(sid,q);m=b.measures(g,p,free,obstacles)
  result=dict(stone=sid,pose=q,metrics=m,score=m['coverage']-4*m['clearance_intrusion_fraction']-.8*m['outside_pocket_fraction'],feasibility_objective=float(best.fun))
  if r['baseline']:result['area_gain_sqft']=g.area-r['baseline']['area_sqft']
  out.append(result)
 out.sort(key=lambda c:-c['score']);r['candidates']=out
 (DEST/path.name).write_text(json.dumps(r,indent=2))
 eligible=[c for c in out if c['metrics']['coverage']>=.70 and c['metrics']['max_clearance_intrusion_inches']<=.10 and c['metrics']['clearance_intrusion_fraction']<=.001 and c['metrics']['physical_overlap_sqft']<1e-6]
 return t['id'],[(c['stone'],round(c['metrics']['coverage'],2),round(c['metrics']['max_clearance_intrusion_inches'],3)) for c in eligible[:3]]

if __name__=='__main__':
 DEST.mkdir(exist_ok=True)
 paths=sorted((b.DEST/'search_results').glob('*.json'));todo=[p for p in paths if not (DEST/p.name).exists()]
 with ProcessPoolExecutor(max_workers=10) as pool:
  for x in pool.map(refine,todo):print(x,flush=True)
