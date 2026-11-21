"""Compare greedy rebuilds with and without equal-coverage pocket ranking."""
import copy
import json
import platform
from pathlib import Path
import tempfile
import time
import numpy as np
import shapely
from flagstone.generate import inventory
from flagstone.phase1 import Layout, validate
from flagstone.runner import start_rebuild, continue_rebuild
from flagstone.scoring import score_layout
from flagstone.search import greedy_insert

runs=[]
with tempfile.TemporaryDirectory() as directory:
    for seed in (81,82,83):
        initial=inventory(seed=seed, counts=(6,12,24))
        greedy_insert(initial, strategy='edge', proposals=3)
        for mode in ('coverage_only','secondary'):
            x=Layout(initial.patio,initial.shapes,initial.poses,initial.active,initial.ids,np.random.default_rng(0))
            x.rng.bit_generator.state=copy.deepcopy(initial.rng.bit_generator.state)
            path=Path(directory)/f'{seed}-{mode}.json'
            opts={'strategy':'edge','patch_size':2,'proposals':4,'patch_padding':.08,'max_new_stones':12}
            if mode=='secondary':
                opts.update(secondary_resolution=16,secondary_broad_distance=.12,secondary_edge_strip_width=.03)
            start_rebuild(x,path,**opts)
            start=time.perf_counter(); rounds=0
            while time.perf_counter()-start < 2:
                rounds+=1
                best,state=continue_rebuild(path,rounds)
                if state['rounds'] < rounds: break
            elapsed=time.perf_counter()-start
            scored=score_layout(best,resolution=16,broad_distance=.12,edge_strip_width=.03)
            runs.append({'seed':seed,'mode':mode,'coverage':scored['coverage'],'secondary':scored['secondary'],
                         'rounds':state['rounds'],'area_improvements':state['improvements'],
                         'secondary_improvements':state['secondary_improvements'],'elapsed_seconds':elapsed,
                         'violations':validate(best)})
summary={mode:{'mean_coverage':float(np.mean([r['coverage'] for r in runs if r['mode']==mode])),
               'mean_interior_broad_fraction':float(np.mean([r['secondary']['interior_broad_area_fraction'] for r in runs if r['mode']==mode])),
               'mean_rounds':float(np.mean([r['rounds'] for r in runs if r['mode']==mode]))}
         for mode in ('coverage_only','secondary')}
out={'config':{'seeds':[81,82,83],'seconds_per_rebuild':2,'resolution':16,'broad_distance':.12,'edge_strip_width':.03,
               'note':'Nominal equal time after identical starts, checks between checkpointed rounds; results depend on grid resolution.',
               'python':platform.python_version(),'numpy':np.__version__,'shapely':shapely.__version__},'summary':summary,'runs':runs}
Path('examples/benchmark_secondary.json').write_text(json.dumps(out,indent=2))
print(json.dumps(summary,indent=2))
