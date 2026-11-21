"""Compare local and global checkpointed rebuilds from identical layouts."""
import copy
import json
from pathlib import Path
import platform
import tempfile
import time
import numpy as np
import shapely
from flagstone.generate import inventory
from flagstone.phase1 import Layout, validate
from flagstone.runner import start_rebuild, continue_rebuild
from flagstone.search import greedy_insert, coverage

runs=[]
with tempfile.TemporaryDirectory() as directory:
    for seed in (51,52,53):
        initial = inventory(seed=seed, counts=(6,50,100), areas=(.025,.008,.003), vertex_profile='detailed')
        start = time.perf_counter()
        while time.perf_counter()-start < 2:
            greedy_insert(initial, strategy='edge', proposals=20, time_limit=max(0, 2-(time.perf_counter()-start)))
        for mode in ('global', 'local'):
            x = Layout(initial.patio, initial.shapes, initial.poses, initial.active, initial.ids, np.random.default_rng(0))
            x.rng.bit_generator.state = copy.deepcopy(initial.rng.bit_generator.state)
            path=Path(directory)/f'{seed}-{mode}.json'
            opts={'patch_size':3, 'proposals':8, 'strategy':'edge'}
            if mode=='local': opts.update(patch_padding=.06, max_new_stones=24)
            start_rebuild(x, path, **opts)
            started=time.perf_counter(); rounds=0
            while time.perf_counter()-started < 3:
                rounds+=1
                best, state = continue_rebuild(path, rounds)
                if state['rounds'] < rounds: break
            runs.append({'seed':seed,'mode':mode,'initial_coverage':coverage(initial),
                         'initial_count':int(initial.active.sum()),'final_coverage':coverage(best),
                         'final_count':int(best.active.sum()),'rounds':state['rounds'],
                         'attempts':state['attempts'],'improvements':state['improvements'],
                         'elapsed_seconds':time.perf_counter()-started,'violations':validate(best)})
summary={mode:{'mean_final_coverage':float(np.mean([r['final_coverage'] for r in runs if r['mode']==mode])),
               'mean_gain':float(np.mean([r['final_coverage']-r['initial_coverage'] for r in runs if r['mode']==mode])),
               'mean_rounds':float(np.mean([r['rounds'] for r in runs if r['mode']==mode]))} for mode in ('global','local')}
out={'config':{'initial_seconds':2,'rebuild_seconds':3,'seeds':[51,52,53], 'inventory_counts':[6,50,100],
               'areas':[.025,.008,.003],'note':'Nominal equal time after identical starts; round-boundary timing includes checkpoint I/O and can overrun.',
               'python':platform.python_version(),'numpy':np.__version__,'shapely':shapely.__version__}, 'summary':summary,'runs':runs}
Path('examples/benchmark_patch.json').write_text(json.dumps(out,indent=2))
print(json.dumps(summary,indent=2))
