"""Compare checkpointed patch sizes from the same saved incumbent."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import shapely
from flagstone.checkpoint import load_checkpoint
from flagstone.phase1 import save_json, validate
from flagstone.runner import continue_rebuild, start_rebuild
from flagstone.search import coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('examples/detailed_156_optimized/rebuild/checkpoint.json'))
    parser.add_argument('--output', type=Path, default=Path('examples/benchmark_patch_size'))
    parser.add_argument('--seconds', type=float, default=7)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--seed-start', type=int, default=101)
    parser.add_argument('--sizes', type=int, nargs='+', default=[2, 3, 5])
    parser.add_argument('--restore-removed', action='store_true')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 0 or args.seeds < 1 or any(n < 1 for n in args.sizes) or len(set(args.sizes)) != len(args.sizes) or args.seed_start < 0:
        parser.error('seconds must be finite and nonnegative, seeds and unique sizes positive, and seed-start nonnegative')
    initial, previous = load_checkpoint(args.input)
    if previous.get('kind') != 'rebuild-run-v1':
        parser.error('input must be a rebuild checkpoint')
    initial_coverage, initial_placed = coverage(initial), int(initial.active.sum())
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for size in args.sizes:
            path = args.output / f'seed-{seed}-size-{size}.json'
            if path.resolve() == args.input.resolve():
                parser.error('output checkpoint must differ from input')
            layout, previous = load_checkpoint(args.input)
            layout.rng = np.random.default_rng(seed)
            config = dict(previous['config'])
            config['patch_size'] = size
            config['restore_removed'] = args.restore_removed
            state = start_rebuild(layout, path, uncertainty_model=previous.get('uncertainty_model'), **config)
            started = time.perf_counter()
            while time.perf_counter() - started < args.seconds:
                old = state['rounds']
                layout, state = continue_rebuild(path, old + 1)
                if state['rounds'] == old:
                    break
            elapsed = time.perf_counter() - started
            errors = validate(layout, config['gap'], config['edge'], config.get('tolerance', 1e-9),
                              stone_margins=config.get('stone_margins'), patio_margin=config.get('patio_margin', 0))
            if errors:
                raise ValueError(f'invalid result: {errors}')
            save_json(layout, args.output / f'seed-{seed}-size-{size}-layout.json')
            runs.append({'seed': seed, 'patch_size': size, 'placed': int(layout.active.sum()),
                         'coverage': coverage(layout), 'rounds': state['rounds'], 'improvements': state['improvements'],
                         'seconds': elapsed, 'violations': len(errors)})
    summary = {}
    for size in args.sizes:
        subset = [r for r in runs if r['patch_size'] == size]
        summary[str(size)] = {'mean_coverage': float(np.mean([r['coverage'] for r in subset])),
                              'min_coverage': min(r['coverage'] for r in subset),
                              'max_coverage': max(r['coverage'] for r in subset),
                              'mean_placed': float(np.mean([r['placed'] for r in subset])),
                              'mean_rounds': float(np.mean([r['rounds'] for r in subset]))}
    report = {'input': str(args.input), 'initial': {'coverage': initial_coverage, 'placed': initial_placed},
              'configuration': {'seconds_per_run': args.seconds, 'seeds': args.seeds, 'seed_start': args.seed_start,
                                'sizes': args.sizes, 'restore_removed': args.restore_removed,
                                'other_search_config': previous['config'],
                                'note': 'Equal nominal wall-clock time from one common synthetic incumbent; checks occur between checkpointed rounds.'},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'shapely': shapely.__version__},
              'summary': summary, 'runs': runs}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
