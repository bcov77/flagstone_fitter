"""De-parallelize prominent edges while preserving every stone's area and ID."""
from pathlib import Path
import json, math
import numpy as np
from shapely.affinity import scale
from flagstone.phase1 import Layout, load_json, local_shape, polygon, save_json

def parallel_long_edges(shape):
    """Triangles exempt; compare up to four sides at least half the longest.

    Edge directions are unoriented: parallel and antiparallel are equivalent.
    """
    v = np.asarray(shape.exterior.coords[:-1]); edges = np.roll(v,-1,axis=0)-v
    if len(edges) <= 3: return False
    lengths = np.linalg.norm(edges,axis=1)
    indices = np.argsort(lengths)[::-1]
    indices = [i for i in indices if lengths[i] >= .5 * max(lengths)][:4]
    angles = np.arctan2(edges[:,1],edges[:,0])
    return any(abs((angles[i]-angles[j]+math.pi/2)%math.pi-math.pi/2) <= math.radians(15)
               for n,i in enumerate(indices) for j in indices[n+1:])

original = load_json('examples/flagstone_reference/revised_25ft_inventory.json')
rng=np.random.default_rng(126); shapes={}; attempts=[]
for sid in original.ids:
    stone=original.shapes[sid]; points=np.array(stone.exterior.coords[:-1]); span=np.ptp(points,axis=0)
    candidate=stone
    if parallel_long_edges(candidate):
        for attempt in range(1,4001):
            # Unequal per-vertex shifts break parallelism, notches and smooth
            # curves are not injected; reject invalid or very distorted shapes.
            try: candidate=polygon(points+rng.normal(0, .18, points.shape)*span)
            except ValueError: continue
            if candidate.area < stone.area*.48 or candidate.area > stone.area*1.8: continue
            if not parallel_long_edges(candidate): break
        else: raise RuntimeError(f'Could not generate valid shape for {sid}')
        attempts.append(attempt)
    factor=math.sqrt(stone.area/candidate.area)
    candidate=scale(candidate,xfact=factor,yfact=factor,origin='centroid')
    shapes[sid]=local_shape(candidate.exterior.coords)
layout=Layout(original.patio,shapes,original.poses,original.active,original.ids,rng,units='ft')
out=Path('examples/nonparallel_search');out.mkdir(parents=True,exist_ok=True)
save_json(layout,out/'inventory.json')
violations=[sid for sid in layout.ids if parallel_long_edges(layout.shapes[sid])]
report={'count':len(layout.ids),'triangles_exempt':sum(len(s.exterior.coords)-1==3 for s in shapes.values()),
        'nontriangle_parallel_violations':len(violations),'allowed_maximum':math.floor(.1*len(layout.ids)),
        'adjusted':len(attempts),'max_attempts':max(attempts,default=0),
        'total_inventory_to_patio_area':sum(s.area for s in shapes.values())/layout.patio.area,
        'interpretation':'Triangles exempt; among other stones, no pair of up to four longest sides at least half the longest is parallel within 15 degrees.'}
(out/'inventory_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
