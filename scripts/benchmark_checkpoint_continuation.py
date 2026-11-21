"""Time fixed-work continuation from copies of an iterative checkpoint."""
import argparse
import json
import platform
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import shapely
from flagstone.iterative import continue_iterative
from flagstone.phase1 import validate
from flagstone.search import coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('examples/known_fit_large_60s/checkpoint.json'))
    parser.add_argument('--stones', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('examples/benchmark_checkpoint_continuation.json'))
    args = parser.parse_args()
    if args.stones < 0 or args.repeats < 1:
        parser.error('stones must be nonnegative and repeats must be positive')
    runs = []
    with tempfile.TemporaryDirectory() as directory:
        for i in range(args.repeats):
            path = Path(directory) / f'checkpoint-{i}.json'
            shutil.copyfile(args.input, path)
            started = time.perf_counter()
            layout, state = continue_iterative(path, total_passes=1000, max_stones=args.stones)
            elapsed = time.perf_counter() - started
            config = state['config']
            errors = validate(layout, config['gap'], config['edge'], config.get('tolerance', 1e-9),
                              stone_margins=config.get('stone_margins'), patio_margin=config.get('patio_margin', 0))
            if errors:
                raise ValueError(f'invalid result: {errors}')
            runs.append({'seconds': elapsed, 'stones_tried_total': state['stones_tried'],
                         'accepted_total': state['accepted'], 'placed': int(layout.active.sum()),
                         'coverage': coverage(layout), 'violations': len(errors)})
    report = {'input': str(args.input), 'additional_stones_limit': args.stones, 'repeats': args.repeats,
              'note': 'Fixed work from identical checkpoint copies, including checkpoint writes; timing depends on hardware and load.',
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'shapely': shapely.__version__},
              'runs': runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
