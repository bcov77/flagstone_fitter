"""Try checkpointed local rebuilds from a known-fit insertion checkpoint."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from flagstone.checkpoint import load_checkpoint
from flagstone.phase1 import plot, save_json, validate
from flagstone.runner import continue_rebuild, start_rebuild
from flagstone.search import coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('examples/known_fit_large_60s/checkpoint.json'))
    parser.add_argument('--output', type=Path, default=Path('examples/known_fit_large_rebuild'))
    parser.add_argument('--seconds', type=float, default=30)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 0:
        parser.error('seconds must be finite and nonnegative')
    layout, insertion = load_checkpoint(args.input)
    if insertion.get('kind') != 'iterative-insertion-v1':
        parser.error('input must be an iterative insertion checkpoint')
    config = insertion['config']
    if config.get('stone_margins') or config.get('patio_margin') or config.get('frozen_ids') or insertion.get('uncertainty_model'):
        parser.error('this benchmark supports only the unmodified synthetic witness')
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / 'checkpoint.json'
    if checkpoint.resolve() == args.input.resolve():
        parser.error('output checkpoint must differ from input checkpoint')
    gap, edge, tolerance = config['gap'], config['edge'], config.get('tolerance', 1e-9)
    initial = {'placed': int(layout.active.sum()), 'coverage': coverage(layout)}
    state = start_rebuild(layout, checkpoint, gap=gap, edge=edge, tolerance=tolerance,
                          strategy='edge', patch_size=3, proposals=8, patch_padding=.06, max_new_stones=24)
    started = time.perf_counter()
    while time.perf_counter() - started < args.seconds:
        old_rounds = state['rounds']
        layout, state = continue_rebuild(checkpoint, old_rounds + 1)
        if state['rounds'] == old_rounds:
            break
    elapsed = time.perf_counter() - started
    violations = validate(layout, gap=gap, edge=edge, tolerance=tolerance)
    if violations:
        raise ValueError(f'rebuild produced invalid layout: {violations}')
    save_json(layout, args.output / 'result.json')
    plot(layout, args.output / 'result.png')
    report = {'input_checkpoint': str(args.input), 'initial': initial,
              'configuration': {'seconds': args.seconds, 'patch_size': 3, 'proposals': 8,
                                'patch_padding': .06, 'max_new_stones': 24, 'strategy': 'edge',
                                'note': 'Budget checked between checkpointed rounds; this is one synthetic run.'},
              'result': {'placed': int(layout.active.sum()), 'coverage': coverage(layout),
                         'rounds': state['rounds'], 'improvements': state['improvements'],
                         'elapsed_seconds': elapsed, 'violations': len(violations)}}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
