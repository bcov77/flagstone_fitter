"""Compare checkpointed greedy rebuild and bounded exploration from identical starts."""
import argparse
import json
from pathlib import Path
import platform
import tempfile
import time
import numpy as np
import shapely
from flagstone.generate import inventory
from flagstone.search import greedy_insert, coverage
from flagstone.runner import start_rebuild, continue_rebuild
from flagstone.exploration import start_exploration, continue_exploration
from flagstone.phase1 import validate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--seconds', type=float, default=1)
p.add_argument('--seeds', type=int, default=3)
p.add_argument('--output', type=Path, default=Path('examples/benchmark_exploration.json'))
a = p.parse_args()
if not np.isfinite(a.seconds) or a.seconds <= 0 or a.seeds < 1:
    p.error('seconds and seeds must be positive')
runs = []
with tempfile.TemporaryDirectory() as directory:
    for seed in range(a.seeds):
        # Reconstruct identical starting layouts and RNG states for each mode.
        for mode in ('greedy', 'exploration'):
            x = inventory(seed=seed, counts=(6,12,24))
            greedy_insert(x, proposals=3, strategy='opening')
            initial = coverage(x)
            path = Path(directory) / f'{seed}-{mode}.json'
            if mode == 'greedy':
                start_rebuild(x, path, patch_size=2, proposals=3, strategy='opening')
            else:
                start_exploration(x, path, patch_size=2, proposals=3, strategy='opening', max_coverage_drop=.05)
            t = time.perf_counter(); rounds = 0
            while time.perf_counter() - t < a.seconds:
                rounds += 1
                if mode == 'greedy':
                    best, state = continue_rebuild(path, rounds)
                    if state['rounds'] < rounds:
                        break
                else:
                    _, best, state = continue_exploration(path, rounds)
                    if state['rounds'] < rounds:
                        break
            runs.append({'seed': seed, 'mode': mode, 'initial_coverage': initial, 'best_coverage': coverage(best),
                         'rounds': state['rounds'], 'elapsed_seconds': time.perf_counter()-t,
                         'violations': validate(best)})
summary = {mode: {'mean_best_coverage': float(np.mean([r['best_coverage'] for r in runs if r['mode'] == mode])),
                  'mean_improvement': float(np.mean([r['best_coverage']-r['initial_coverage'] for r in runs if r['mode'] == mode]))}
           for mode in ('greedy','exploration')}
output = {'config': {'seconds_per_run': a.seconds, 'seeds': a.seeds, 'budget_scope': 'post-insertion search, including checkpoint I/O; checked between rounds',
                     'python': platform.python_version(), 'numpy': np.__version__, 'shapely': shapely.__version__},
          'summary': summary, 'runs': runs}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(output, indent=2))
print(json.dumps(summary, indent=2))
