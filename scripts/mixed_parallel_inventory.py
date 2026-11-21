"""Construct the requested 50/40/10 distribution of near-parallel sides."""
from collections import Counter
from pathlib import Path
import json, math
import numpy as np
from shapely.affinity import scale
from flagstone.edge_analysis import near_parallel_long_edge_pairs as pair_count
from flagstone.phase1 import Layout, load_json, local_shape, polygon, save_json

original=load_json('examples/flagstone_reference/revised_25ft_inventory.json')
rng=np.random.default_rng(222); ids=list(original.ids); rng.shuffle(ids)
multi=[s for s in ids if pair_count(original.shapes[s])>=2][:47]
one=[s for s in ids if s not in multi and pair_count(original.shapes[s])==1][:188]
targets={sid:(2 if sid in multi else 1 if sid in one else 0) for sid in original.ids}
shapes={}; attempts=[]
for sid in original.ids:
    stone=original.shapes[sid]; target=targets[sid]; candidate=stone
    current=pair_count(stone)
    if (current if current<2 else 2)!=target:
        points=np.array(stone.exterior.coords[:-1]); span=np.ptp(points,axis=0)
        for attempt in range(1,4001):
            try: candidate=polygon(points+rng.normal(0,.18,points.shape)*span)
            except ValueError: continue
            if candidate.area<stone.area*.48 or candidate.area>stone.area*1.8: continue
            count=pair_count(candidate)
            if (count if count<2 else 2)==target: break
        else: raise RuntimeError(f'Unable to create target category for {sid}')
        attempts.append(attempt)
    factor=math.sqrt(stone.area/candidate.area)
    shapes[sid]=local_shape(scale(candidate,xfact=factor,yfact=factor,origin='centroid').exterior.coords)
x=Layout(original.patio,shapes,original.poses,original.active,original.ids,rng,units='ft')
out=Path('examples/mixed_parallel_search');out.mkdir(parents=True,exist_ok=True)
save_json(x,out/'inventory.json')
counts=Counter(min(2,pair_count(s)) for s in shapes.values())
report={'count':len(x.ids),'no_near_parallel_pairs':counts[0], 'one_pair':counts[1],
        'multiple_pairs':counts[2], 'triangles_exempt':sum(len(s.exterior.coords)-1==3 for s in shapes.values()),
        'adjusted':len(attempts),'max_attempts':max(attempts,default=0),
        'inventory_to_patio_area':sum(s.area for s in shapes.values())/x.patio.area,
        'interpretation':'Triangles count in the no-pair group and are unrestricted. For other shapes compare up to four longest sides at least half the longest; a pair is near-parallel within 15 degrees.'}
(out/'inventory_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
