"""Compare complete scalar and batched random insertion under equal budgets."""
import json
from pathlib import Path
import platform
import time
import numpy as np
import shapely
from flagstone.generate import inventory
from flagstone.search import greedy_insert, batched_random_insert

runs=[]
for seed in range(3):
    for mode in ('scalar', 'batch8'):
        x=inventory(seed=seed+10, counts=(18,36,72), vertex_profile='detailed')
        start=time.perf_counter(); attempts=accepted=passes=0
        while time.perf_counter()-start < 2 and not x.active.all():
            remaining=max(0, 2-(time.perf_counter()-start))
            if mode == 'scalar':
                result=greedy_insert(x, strategy='random', proposals=20, time_limit=remaining)
            else:
                result=batched_random_insert(x, proposals=20, batch_size=8, time_limit=remaining)
            attempts+=result.attempts; accepted+=result.accepted; passes+=1
        runs.append({'seed':seed+10,'mode':mode,'coverage':result.coverage,'attempts':attempts,
                     'accepted':accepted,'passes':passes,'elapsed_seconds':time.perf_counter()-start,
                     'violations':result.violations})
summary={mode:{'mean_coverage':float(np.mean([r['coverage'] for r in runs if r['mode']==mode])),
               'mean_attempts':float(np.mean([r['attempts'] for r in runs if r['mode']==mode]))}
         for mode in ('scalar','batch8')}
out={'config':{'seconds_per_run':2,'seed_count':3,'stones':126,'vertex_profile':'detailed',
               'note':'Whole-search wall time; checked between batches, no relaxation. Batched trajectories consume extra RNG draws. Single machine.',
               'python':platform.python_version(),'numpy':np.__version__,'shapely':shapely.__version__},'summary':summary,'runs':runs}
Path('examples/batched_search_benchmark.json').write_text(json.dumps(out,indent=2))
print(json.dumps(summary,indent=2))
