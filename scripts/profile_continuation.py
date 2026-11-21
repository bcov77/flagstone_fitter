"""Profile a fixed checkpointed insertion continuation on the CPU."""
import argparse
import cProfile
import json
import platform
import pstats
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
    parser.add_argument('--input', type=Path, default=Path('examples/varied_156/run/iterative_checkpoint.json'))
    parser.add_argument('--stones', type=int, default=100)
    parser.add_argument('--output', type=Path, default=Path('examples/profile_varied_continuation.json'))
    args = parser.parse_args()
    if args.stones < 0:
        parser.error('stones must be nonnegative')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'checkpoint.json'
        shutil.copyfile(args.input, path)
        profiler = cProfile.Profile()
        started = time.perf_counter()
        profiler.enable()
        layout, state = continue_iterative(path, 1000, max_stones=args.stones)
        profiler.disable()
        elapsed = time.perf_counter() - started
    c = state['config']
    violations = validate(layout, c['gap'], c['edge'], c.get('tolerance', 1e-9),
                          stone_margins=c.get('stone_margins'), patio_margin=c.get('patio_margin', 0))
    if violations:
        raise ValueError(f'invalid profiled result: {violations}')
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        rows.append({'function': name, 'file': Path(filename).name, 'line': line,
                     'calls': calls, 'primitive_calls': primitive,
                     'self_seconds': own, 'cumulative_seconds': cumulative})
    rows.sort(key=lambda item: item['cumulative_seconds'], reverse=True)
    report = {'input': str(args.input), 'additional_stone_limit': args.stones,
              'active_stones': int(layout.active.sum()), 'inventory': len(layout.ids),
              'coverage': coverage(layout), 'violations': len(violations), 'profiled_seconds': elapsed,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'shapely': shapely.__version__},
              'note': 'CPU cProfile of checkpointed work; profiling adds overhead. Cumulative times overlap and must not be summed. No GPU, transfer, or native-memory measurement.',
              'top_cumulative_functions': rows[:35]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
