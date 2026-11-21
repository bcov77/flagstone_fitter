"""Measure checkpointed insertion against a generated, fully feasible inventory."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import shapely

from flagstone.generate import cut_witness
from flagstone.iterative import continue_iterative, start_iterative
from flagstone.phase1 import load_json, plot, save_json, validate
from flagstone.search import coverage, STRATEGIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=12, help='grid rows and columns')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--proposals', type=int, default=10)
    parser.add_argument('--strategy', default='edge', choices=STRATEGIES)
    parser.add_argument('--output', type=Path, default=Path('examples/known_fit_large'))
    args = parser.parse_args()
    if args.size < 1 or not np.isfinite(args.seconds) or args.seconds < 0 or args.proposals < 1:
        parser.error('size and proposals must be positive and seconds must be finite and nonnegative')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    gap, edge, inset = .01, .005, .006
    witness = cut_witness(seed=args.seed, rows=args.size, cols=args.size, clearance=inset)
    violations = validate(witness, gap=gap, edge=edge)
    if violations:
        raise ValueError(f'Witness does not satisfy benchmark constraints: {violations}')
    witness_coverage = coverage(witness)
    save_json(witness, out / 'witness.json')
    plot(witness, out / 'witness.png')
    # Loading separately provides a deterministic search RNG independent of generation.
    initial = load_json(out / 'witness.json', seed=args.seed + 1)
    initial.active[:] = False
    initial.poses[:] = 0
    save_json(initial, out / 'input.json')
    checkpoint = out / 'checkpoint.json'
    start_iterative(initial, checkpoint, gap=gap, edge=edge, proposals=args.proposals, strategy=args.strategy)
    start = time.perf_counter()
    result, state = continue_iterative(checkpoint, 1000, max_seconds=args.seconds)
    elapsed = time.perf_counter() - start
    violations = validate(result, gap=gap, edge=edge)
    if violations:
        raise ValueError(f'Search result failed validation: {violations}')
    save_json(result, out / 'result.json')
    plot(result, out / 'result.png')
    result_coverage = coverage(result)
    report = {
        'configuration': {'grid_rows': args.size, 'grid_cols': args.size, 'seed': args.seed,
                          'search_seed': args.seed + 1, 'gap': gap, 'edge': edge, 'inset': inset,
                          'proposals_per_stone': args.proposals, 'strategy': args.strategy,
                          'time_limit_seconds': args.seconds, 'max_passes': 1000,
                          'note': 'Synthetic near-equal quadrilaterals, not representative of a measured mixed inventory. Time is checked between checkpointed stones.'},
        'witness': {'placed': int(witness.active.sum()), 'coverage': witness_coverage, 'violations': len(validate(witness, gap=gap, edge=edge))},
        'result': {'placed': int(result.active.sum()), 'coverage': result_coverage,
                   'fraction_of_witness_area': result_coverage / witness_coverage,
                   'violations': len(violations), 'elapsed_seconds': elapsed,
                   'stones_tried': state['stones_tried'], 'attempts': state['attempts'],
                   'completed_passes': state['completed_passes']},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'shapely': shapely.__version__},
    }
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
