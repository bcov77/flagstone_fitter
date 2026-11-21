"""Compare batched NumPy and Shapely-tree pair filtering with exact contacts."""
import json
from pathlib import Path
import platform
import time
import numpy as np
import shapely
from shapely.geometry import box
from shapely.strtree import STRtree
from flagstone.broadphase import PaddedGeometry, candidate_pairs
from flagstone.generate import inventory
from flagstone.phase1 import transformed

rng = np.random.default_rng(31)
rows = []
for n in (50, 100, 200):
    x = inventory(seed=7, counts=(n,), areas=(.001,), vertex_profile='detailed')
    poses = np.empty((8,n,3)); poses[:,:,:2] = rng.uniform(0,1,(8,n,2)); poses[:,:,2] = rng.uniform(-np.pi,np.pi,(8,n))
    active = rng.random((8,n)) > .1
    gap = .01
    t = time.perf_counter()
    packed = PaddedGeometry.from_layout(x)
    candidates = candidate_pairs(packed, poses, active, gap=gap)
    numpy_seconds = time.perf_counter()-t
    t = time.perf_counter()
    geoms = [[transformed(x.shapes[sid], poses[b,i]) for i,sid in enumerate(x.ids)] for b in range(len(poses))]
    transform_seconds = time.perf_counter()-t
    t = time.perf_counter()
    tree_pairs = []
    for b in range(len(poses)):
        indices = np.flatnonzero(active[b])
        tree = STRtree([geoms[b][i] for i in indices])
        pairs = []
        for local_i, i in enumerate(indices):
            lo_x, lo_y, hi_x, hi_y = geoms[b][i].bounds
            for local_j in tree.query(box(lo_x-gap,lo_y-gap,hi_x+gap,hi_y+gap)):
                j = int(indices[local_j])
                if j > i:
                    pairs.append((int(i), j))
        tree_pairs.append(set(pairs))
    tree_seconds = time.perf_counter()-t
    assert all(tree_pairs[b].issubset(set(map(tuple,candidates[b]))) for b in range(len(poses)))
    def exact(groups):
        return [{(int(i),int(j)) for i,j in groups[b] if geoms[b][i].distance(geoms[b][j]) < gap or geoms[b][i].intersection(geoms[b][j]).area > 0} for b in range(len(poses))]
    t=time.perf_counter(); exact_numpy=exact(candidates); numpy_contact=time.perf_counter()-t
    t=time.perf_counter(); exact_tree=exact(tree_pairs); tree_contact=time.perf_counter()-t
    assert exact_numpy == exact_tree
    rows.append({'stones':n,'batch':len(poses),'numpy_pack_and_filter_seconds':numpy_seconds,
                 'geos_transform_seconds':transform_seconds,'tree_filter_seconds':tree_seconds,
                 'numpy_exact_contact_seconds':numpy_contact,'tree_exact_contact_seconds':tree_contact,
                 'numpy_total_seconds':numpy_seconds+transform_seconds+numpy_contact,
                 'tree_total_seconds':transform_seconds+tree_seconds+tree_contact,
                 'numpy_candidates':sum(map(len,candidates)), 'tree_candidates':sum(map(len,tree_pairs)),
                 'exact_violations':sum(map(len,exact_numpy)),
                 'padded_vertex_bytes':packed.vertices.nbytes+packed.vertex_mask.nbytes,
                 'estimated_pair_mask_bytes':len(poses)*n*n})
out = {'python':platform.python_version(),'numpy':np.__version__,'shapely':shapely.__version__,
       'note':'Single local run per size; transform and exact GEOS contacts included; no patio checks, optimizer, GPU, or transfer costs.', 'results':rows}
Path('examples/broadphase_benchmark.json').write_text(json.dumps(out,indent=2))
print(json.dumps(rows,indent=2))
