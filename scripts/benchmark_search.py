"""Compare insertion strategies under equal per-run wall-clock budgets."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time
import numpy as np
import shapely
from flagstone.generate import inventory, inventory_summary
from flagstone.search import greedy_insert, STRATEGIES

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seconds', type=float, default=1.)
parser.add_argument('--seeds', type=int, default=3)
profile = parser.add_mutually_exclusive_group()
profile.add_argument('--detailed', action='store_true')
profile.add_argument('--vertex-profile', choices=('basic', 'detailed', 'varied'))
parser.add_argument('--no-prefilter', action='store_true')
parser.add_argument('--counts', type=int, nargs=3, default=[18,36,72])
parser.add_argument('--areas', type=float, nargs=3, default=[.03,.012,.003])
parser.add_argument('--seed-start', type=int, default=0)
parser.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=list(STRATEGIES))
parser.add_argument('--output', type=Path, default=Path('examples/benchmark.json'))
args = parser.parse_args()
if args.seconds <= 0 or args.seeds < 1:
    parser.error('seconds and seeds must be positive')
records = []
vertex_profile = args.vertex_profile or ('detailed' if args.detailed else 'basic')
for seed in range(args.seed_start, args.seed_start + args.seeds):
    for strategy in args.strategies:
        x = inventory(seed=seed, counts=args.counts, areas=args.areas, vertex_profile=vertex_profile)
        summary = inventory_summary(x, [0, .006, .02, .04])
        vertex_counts = [len(shape.exterior.coords)-1 for shape in x.shapes.values()]
        summary['vertex_count'] = {'min': min(vertex_counts), 'mean': float(np.mean(vertex_counts)), 'max': max(vertex_counts)}
        started = time.perf_counter()
        attempts = accepted = passes = 0
        while time.perf_counter() - started < args.seconds and not x.active.all():
            remaining = args.seconds - (time.perf_counter() - started)
            result = greedy_insert(x, strategy=strategy, proposals=20, time_limit=max(0, remaining), use_prefilter=not args.no_prefilter)
            attempts += result.attempts
            accepted += result.accepted
            passes += 1
        records.append({'seed': seed, 'strategy': strategy, 'inventory': summary,
                        'attempts': attempts, 'accepted': accepted, 'passes': passes,
                        'coverage': result.coverage, 'violations': result.violations,
                        'elapsed_seconds': time.perf_counter() - started})
summary = {}
for strategy in args.strategies:
    values = [r['coverage'] for r in records if r['strategy'] == strategy]
    summary[strategy] = {'mean_coverage': statistics.mean(values), 'min_coverage': min(values), 'max_coverage': max(values)}
out = {'config': {'seconds_per_run': args.seconds, 'seed_count': args.seeds, 'use_prefilter': not args.no_prefilter, 'seed_start': args.seed_start, 'strategies': args.strategies, 'vertex_profile': vertex_profile, 'counts': args.counts, 'areas': args.areas, 'proposals_per_stone_per_pass': 20,
                  'note': 'Time checked between proposals; grid construction and final validation may exceed budget.',
                  'python': platform.python_version(), 'numpy': np.__version__, 'shapely': shapely.__version__},
       'summary': summary, 'runs': records}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out, indent=2))
print(json.dumps(summary, indent=2))
