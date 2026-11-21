"""Compare complete batched validation with independent scalar validation."""
import argparse
import json
import platform
from pathlib import Path
import time
import tracemalloc
import numpy as np
import shapely
from flagstone.batch import validate_batch
from flagstone.generate import inventory
from flagstone.phase1 import Layout, validate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('examples/batch_validation_benchmark.json'))
args = parser.parse_args()
x = inventory(seed=21, counts=(18,36,72), vertex_profile='detailed')
rng = np.random.default_rng(77)
rows = []
for batch_size in (1, 8, 32):
    poses = np.empty((batch_size,len(x.ids),3)); poses[:,:,:2] = rng.uniform(0,1,poses[:,:,:2].shape); poses[:,:,2] = rng.uniform(-np.pi,np.pi,poses[:,:,2].shape)
    active = rng.random((batch_size,len(x.ids))) > .2
    margins = {sid:.001 for sid in x.ids}
    opts = dict(gap=.01, edge=.01, stone_margins=margins, patio_margin=.001)
    start=time.perf_counter(); batched=validate_batch(x, poses, active, **opts); batched_seconds=time.perf_counter()-start
    start=time.perf_counter(); scalar=[]
    for b in range(batch_size):
        candidate = Layout(x.patio, x.shapes, poses[b], active[b], x.ids, np.random.default_rng(0))
        scalar.append(validate(candidate, **opts))
    scalar_seconds=time.perf_counter()-start
    assert batched == scalar
    tracemalloc.start(); validate_batch(x, poses, active, **opts); _, peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    rows.append({'batch':batch_size, 'stones':len(x.ids), 'batched_seconds':batched_seconds,
                 'scalar_seconds':scalar_seconds, 'python_peak_bytes':peak,
                 'violation_count':sum(map(len,batched))})
out={'python':platform.python_version(), 'numpy':np.__version__, 'shapely':shapely.__version__,
     'note':'Single local run per batch size; entire validation included. Python peak excludes some native GEOS allocations. No GPU, optimization or transfer cost.', 'results':rows}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out,indent=2))
print(json.dumps(rows,indent=2))
