"""Measure CPU reference operations on deterministic dense synthetic layouts."""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
import shapely
from flagstone.generate import inventory
from flagstone.phase1 import transformed, validate
from flagstone.relaxation import energy, relax, relax_neighborhood

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sizes', nargs='+', type=int, default=[25, 50, 100, 200])
parser.add_argument('--output', type=Path, default=Path('examples/reference_profile.json'))
parser.add_argument('--all-mobile', action='store_true')
parser.add_argument('--detailed', action='store_true', help='use higher-vertex-count outlines')
parser.add_argument('--local', action='store_true', help='use neighborhood relaxation')
args = parser.parse_args()
results = []
for n in args.sizes:
    if n < 2:
        parser.error('sizes must be at least 2')
    layout = inventory(seed=4, counts=(n,), areas=(.0005,), mixed=True, vertex_profile='detailed' if args.detailed else 'basic')
    side = int(np.ceil(np.sqrt(n)))
    for i in range(n):
        layout.poses[i] = [(.5 + i % side) / side, (.5 + i // side) / side, 0]
    layout.active[:] = True
    # A deliberate contact gives relaxation something to do.
    layout.poses[1] = layout.poses[0]
    def timed(fn):
        start = time.perf_counter(); result = fn(); return time.perf_counter()-start, result
    transform_seconds, _ = timed(lambda: [transformed(layout.shapes[s], layout.poses[i]) for i, s in enumerate(layout.ids)])
    energy_seconds, value = timed(lambda: energy(layout))
    validation_seconds, violations = timed(lambda: validate(layout))
    mobility = np.ones(n) if args.all_mobile else np.zeros(n)
    mobility[1] = 1
    solver = relax_neighborhood if args.local else relax
    def solve_and_check():
        result = solver(layout, max_steps=1, mobility=mobility)
        return result, validate(layout)
    relaxation_seconds, (result, final_violations) = timed(solve_and_check)
    results.append({'stones': n, 'mean_vertices': float(np.mean([len(g.exterior.coords)-1 for g in layout.shapes.values()])), 'transform_seconds': transform_seconds, 'energy_seconds': energy_seconds,
                    'validation_seconds': validation_seconds, 'one_relaxation_step_plus_full_validation_seconds': relaxation_seconds,
                    'final_violation_count': len(final_violations),
                    'initial_energy': value, 'initial_violation_count': len(violations),
                    'relaxation_reason': result.reason})
out = {'python': platform.python_version(), 'numpy': np.__version__, 'shapely': shapely.__version__,
       'mobile_stones': 'all' if args.all_mobile else 1, 'solver': 'neighborhood' if args.local else 'full', 'vertex_profile': 'detailed' if args.detailed else 'basic',
       'note': 'Single timing per size; relaxation timing includes full-layout validation; no GPU comparison.', 'results': results}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out, indent=2))
print(json.dumps(results, indent=2))
