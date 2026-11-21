"""Compare greedy and bounded exploratory local patches from one incumbent."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import shapely
from flagstone.checkpoint import load_checkpoint
from flagstone.exploration import start_exploration, continue_exploration
from flagstone.phase1 import save_json, validate
from flagstone.runner import start_rebuild, continue_rebuild
from flagstone.search import coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('examples/detailed_156_optimized/rebuild/checkpoint.json'))
    parser.add_argument('--output', type=Path, default=Path('examples/benchmark_local_exploration'))
    parser.add_argument('--seconds', type=float, default=8)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--modes', nargs='+', choices=('greedy', 'exploration', 'nudge'), default=['greedy', 'exploration'])
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 0 or args.seeds < 1:
        parser.error('seconds must be finite and nonnegative, and seeds positive')
    initial, input_state = load_checkpoint(args.input)
    if input_state.get('kind') != 'rebuild-run-v1':
        parser.error('input must be a rebuild checkpoint')
    saved = input_state['config']
    if saved.get('stone_margins') or saved.get('patio_margin') or saved.get('frozen_ids') or input_state.get('uncertainty_model'):
        parser.error('this benchmark expects the unmodified synthetic example')
    initial_coverage = coverage(initial)
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in range(301, 301 + args.seeds):
        for mode in args.modes:
            layout, _ = load_checkpoint(args.input)
            layout.rng = np.random.default_rng(seed)
            path = args.output / f'{seed}-{mode}-checkpoint.json'
            if path.resolve() == args.input.resolve():
                parser.error('output checkpoint must differ from input')
            options = {'gap': saved['gap'], 'edge': saved['edge'], 'tolerance': saved.get('tolerance', 1e-9),
                       'patch_size': 3, 'proposals': 8, 'strategy': 'edge',
                       'patch_padding': .06, 'max_new_stones': 24}
            if mode == 'greedy':
                state = start_rebuild(layout, path, **options)
            else:
                nudge = {'nudge_probability': .5, 'nudge_translation': .01, 'nudge_angle': .1} if mode == 'nudge' else {}
                state = start_exploration(layout, path, max_coverage_drop=.01, **nudge, **options)
            current = best = layout
            start = time.perf_counter()
            while time.perf_counter() - start < args.seconds:
                old = state['rounds']
                if mode == 'greedy':
                    current, state = continue_rebuild(path, old + 1)
                    best = current
                else:
                    current, best, state = continue_exploration(path, old + 1)
                if state['rounds'] == old:
                    break
            elapsed = time.perf_counter() - start
            for item in (current, best):
                if validate(item, gap=options['gap'], edge=options['edge'], tolerance=options['tolerance']):
                    raise ValueError('invalid candidate')
            save_json(best, args.output / f'{seed}-{mode}-best.json')
            save_json(current, args.output / f'{seed}-{mode}-current.json')
            runs.append({'seed': seed, 'mode': mode, 'best_coverage': coverage(best),
                         'best_placed': int(best.active.sum()), 'current_coverage': coverage(current),
                         'rounds': state['rounds'], 'improvements': state['improvements'],
                         'seconds': elapsed, 'violations': 0,
                         'nudge_attempts': state.get('nudge_attempts', 0),
                         'nudge_valid': state.get('nudge_valid', 0)})
    summary = {mode: {'mean_best_coverage': float(np.mean([r['best_coverage'] for r in runs if r['mode'] == mode])),
                      'mean_best_placed': float(np.mean([r['best_placed'] for r in runs if r['mode'] == mode])),
                      'mean_rounds': float(np.mean([r['rounds'] for r in runs if r['mode'] == mode]))}
               for mode in args.modes}
    report = {'input': str(args.input), 'initial_coverage': initial_coverage,
              'initial_placed': int(initial.active.sum()),
              'configuration': {'seconds_per_run': args.seconds, 'seeds': args.seeds,
                                'seed_start': 301, 'search': options, 'max_coverage_drop': .01,
                                'modes': args.modes, 'nudge': {'probability': .5, 'translation': .01, 'angle': .1},
                                'note': 'Same incumbent and initial RNG per pair; nominal time includes checkpoint I/O and is checked between rounds.'},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'shapely': shapely.__version__},
              'summary': summary, 'runs': runs}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
