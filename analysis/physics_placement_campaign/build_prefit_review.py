"""Geometry-only proposal review: explicit pockets, fixed neighbors, no physics.
Writes only prefit_review_v1; frozen campaigns and numbered human reviews untouched.
"""
from pathlib import Path
import json, math, hashlib, time, argparse
import numpy as np
import shapely
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from shapely.affinity import affine_transform
from scipy import signal, ndimage, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
from concurrent.futures import ProcessPoolExecutor

HERE=Path(__file__).resolve().parent; SRC=HERE/'campaign_v1'; DEST=HERE/'prefit_review_v1'
GAP=.5/12; STEP=.035; THROAT=1.25/12
SOURCE=json.loads((SRC/'source.json').read_text()); MANIFEST=json.loads((SRC/'manifest.json').read_text())
STONES={str(s['id']):s for s in SOURCE['stones']}; LOCAL={s:Polygon(v['outline']) for s,v in STONES.items()}
SIMPLE={s:g.simplify(.004,preserve_topology=True) for s,g in LOCAL.items()}
PATIO=Polygon(SOURCE['patio']); ACTIVE=[s for s,v in STONES.items() if v['active']]

def world(sid,pose,exact=True):
 x,y,t=pose;c,s=math.cos(t),math.sin(t)
 return affine_transform(LOCAL[sid] if exact else SIMPLE[sid],[c,-s,s,c,x,y])
ORIGINAL={s:world(s,STONES[s]['pose']) for s in ACTIVE}

def parts(g):
 if g.is_empty:return []
 return [p for p in (g.geoms if hasattr(g,'geoms') else [g]) if p.geom_type=='Polygon']

def pockets(shapes):
 # Split free space at narrow throats. Dilated cores are clipped to real free
 # space. Their artificial cap edges are shown explicitly in the review.
 free=PATIO.buffer(-GAP).difference(unary_union(list(shapes.values())).buffer(GAP))
 cores=parts(free.buffer(-THROAT))
 out=[]
 for core in cores:
  if core.area<.025:continue
  patch=core.buffer(THROAT).intersection(free)
  patch=max(parts(patch),key=lambda p:p.intersection(core).area)
  if patch.area>.25:out.append(patch)
 return free, sorted(out,key=lambda p:-p.area)

def make_targets():
 targets=[]
 # Original layout and genuine earlier synthetic-removal contexts, labeled.
 for context in MANIFEST['contexts']:
  shapes={s:g for s,g in ORIGINAL.items() if s not in context['removed']}
  free,ps=pockets(shapes)
  for i,p in enumerate(ps):
   # Avoid duplicate unchanged holes across contexts. Search changed pockets.
   if context['removed'] and not any(p.intersection(ORIGINAL[s].buffer(.15)).area>.1 for s in context['removed']):continue
   targets.append(dict(id=f"{context['id']}_insert_pocket_{i:02}",family='insertion',context=context['id'],removed=context['removed'],old=None,pocket=shapely.to_wkt(p,rounding_precision=10),free=shapely.to_wkt(free,rounding_precision=10)))
 # Replacements on the actual 27-stone source board. Preserve original baseline.
 for old in ACTIVE:
  shapes={s:g for s,g in ORIGINAL.items() if s!=old};free,ps=pockets(shapes)
  relevant=[p for p in ps if p.intersection(ORIGINAL[old]).area>ORIGINAL[old].area*.35]
  if not relevant:continue
  p=max(relevant,key=lambda p:p.intersection(ORIGINAL[old]).area)
  targets.append(dict(id=f'board_00_replace_{old}',family='replacement',context='board_00',removed=[],old=old,pocket=shapely.to_wkt(p,rounding_precision=10),free=shapely.to_wkt(free,rounding_precision=10)))
 (DEST/'targets.json').write_text(json.dumps(targets,indent=2))
 fig,axs=plt.subplots(2,4,figsize=(18,9))
 for ax,c in zip(axs.flat,MANIFEST['contexts']):
  shapes={s:g for s,g in ORIGINAL.items() if s not in c['removed']};_,ps=pockets(shapes)
  for g in shapes.values():paint(ax,g,facecolor='#d8d4cb',edgecolor='#666',linewidth=.5)
  for i,p in enumerate(ps):
   paint(ax,p,facecolor='#b4deea',edgecolor='#047a9e',linewidth=.7,alpha=.65)
   pt=p.representative_point();ax.text(pt.x,pt.y,f'{i}: {p.area:.1f}',fontsize=7,ha='center')
  ax.set_title(c['id']+(' · original' if not c['synthetic'] else ' · synthetic removal'));ax.set_aspect('equal');ax.set_xlim(0,7.81);ax.set_ylim(0,7.81)
 fig.tight_layout();fig.savefig(DEST/'pocket_extraction.png',dpi=150);plt.close(fig)
 return targets

def paint(ax,g,**kw):
 for p in parts(g):ax.add_patch(Patch(list(p.exterior.coords),**kw))

def mask_at(g,xs,ys):
 return shapely.contains_xy(g,xs[None,:],ys[:,None]).astype(float)

def measures(g,p,free,obstacles):
 inside=g.intersection(p).area; outside=g.difference(p).area
 intr=g.difference(free).area
 phys=g.intersection(obstacles).area+g.difference(PATIO).area
 # Maximum depth outside admissible space at densified full-size boundary.
 dense=shapely.segmentize(g.exterior,.015);pts=shapely.get_coordinates(dense)
 bad=~shapely.contains_xy(free,pts[:,0],pts[:,1])
 depth=float(np.max(shapely.distance(shapely.points(pts[bad]),free))) if np.any(bad) else 0.
 return dict(area_sqft=g.area,pocket_area_sqft=p.area,coverage=inside/p.area,
             outside_pocket_fraction=outside/g.area,clearance_intrusion_sqft=intr,
             clearance_intrusion_fraction=intr/g.area,physical_overlap_sqft=phys,
             max_clearance_intrusion_inches=12*depth,
             unfilled_pocket_sqft=p.area-inside)

def search(target):
 start=time.time();p=shapely.from_wkt(target['pocket']);free=shapely.from_wkt(target['free'])
 active=set(ACTIVE)-set(target['removed'])-({target['old']} if target['old'] else set())
 obstacles=unary_union([ORIGINAL[s] for s in active]);available=[s for s in STONES if s not in active]
 # Exclude exact restoration of deleted stones from insertion suggestions; it
 # would make the panel trivially resemble the original by construction.
 if target['family']=='insertion':available=[s for s in available if s not in target['removed']]
 x0,y0,x1,y1=p.bounds;pad=.25
 xs=np.arange(x0-pad,x1+pad+STEP,STEP);ys=np.arange(y0-pad,y1+pad+STEP,STEP)
 pm=mask_at(p,xs,ys);fm=mask_at(free,xs,ys)
 # Signed space cost rewards filling the explicit pocket; penalizes crossing
 # its artificial caps; heavily penalizes invading actual clearance exclusion.
 costs=np.where(pm>.5,-1.,.8)+np.where(fm<.5,24.,0.)
 ranked=[]
 for sid in available:
  area=LOCAL[sid].area
  if area<p.area*.55 or area>p.area*1.08:continue
  coords=np.asarray(SIMPLE[sid].exterior.coords)
  candidates=[]
  for deg in range(0,360,5):
   a=math.radians(deg);c,s=math.cos(a),math.sin(a)
   xy=coords@np.array([[c,s],[-s,c]])
   rg=Polygon(xy);rx0,ry0,rx1,ry1=rg.bounds
   kxs=np.arange(rx0+STEP/2,rx1,STEP);kys=np.arange(ry0+STEP/2,ry1,STEP)
   if len(kxs)>len(xs) or len(kys)>len(ys) or not len(kxs) or not len(kys):continue
   km=mask_at(rg,kxs,kys)
   conv=signal.fftconvolve(costs,km[::-1,::-1],mode='valid')*STEP**2
   iy,ix=np.unravel_index(np.argmin(conv),conv.shape)
   pose=[float(xs[ix]-kxs[0]),float(ys[iy]-kys[0]),a]
   candidates.append((float(conv[iy,ix]),pose))
  if candidates:
   candidates.sort(key=lambda z:z[0]);ranked.extend((score,sid,pose) for score,pose in candidates[:2])
 ranked.sort(key=lambda z:z[0]);refined=[]
 # Refine the strongest coarse poses geometrically, with all neighbors fixed.
 for _,sid,pose in ranked[:28]:
  def objective(q):
   g=world(sid,q,False)
   return (-g.intersection(p).area+1.2*g.difference(p).area+35*g.difference(free).area)
  bounds=[(pose[0]-.10,pose[0]+.10),(pose[1]-.10,pose[1]+.10),(pose[2]-.10,pose[2]+.10)]
  opt=optimize.minimize(objective,pose,method='Powell',bounds=bounds,options={'maxiter':50,'xtol':.00015,'ftol':.00005})
  q=opt.x.tolist();g=world(sid,q)
  # Exact, full-size outlines decide review eligibility, not raster masks.
  met=measures(g,p,free,obstacles)
  score=met['coverage']-4*met['clearance_intrusion_fraction']-.8*met['outside_pocket_fraction']
  refined.append(dict(stone=sid,pose=q,metrics=met,score=score))
 # Baseline and same-stone poses may be diagnostic but are marked explicitly.
 baseline=None
 if target['old']:
  baseline=measures(ORIGINAL[target['old']],p,free,obstacles)
  for r in refined:r['area_gain_sqft']=r['metrics']['area_sqft']-baseline['area_sqft']
 refined.sort(key=lambda r:-r['score'])
 result=dict(target=target,baseline=baseline,candidates=refined,seconds=time.time()-start)
 (DEST/'search_results'/f"{target['id']}.json").write_text(json.dumps(result,indent=2))
 return target['id'],len(refined),round(time.time()-start,1),[(r['stone'],round(r['metrics']['coverage'],2),round(r['metrics']['clearance_intrusion_fraction'],4)) for r in refined[:2]]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--search',action='store_true');ap.add_argument('--workers',type=int,default=10);args=ap.parse_args()
 DEST.mkdir(exist_ok=True);(DEST/'search_results').mkdir(exist_ok=True)
 if args.prepare:targets=make_targets();print('Prepared',len(targets),'explicit pockets',flush=True)
 if args.search:
  targets=json.loads((DEST/'targets.json').read_text());todo=[t for t in targets if not (DEST/'search_results'/f"{t['id']}.json").exists()]
  print('Searching',len(todo),'pockets with',args.workers,'workers; zero physics steps',flush=True)
  with ProcessPoolExecutor(max_workers=args.workers) as pool:
   for r in pool.map(search,todo):print(r,flush=True)
if __name__=='__main__':main()
