"""Measure Python-traced memory for chunked CPU validation of saved geometry."""
import argparse
import json
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np
import shapely
from flagstone.batch import validate_batch
from flagstone.phase1 import load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('examples/varied_156/run/layout.json'))
    parser.add_argument('--output', type=Path, default=Path('examples/benchmark_chunked_validation.json'))
    parser.add_argument('--candidates', type=int, default=96)
    args = parser.parse_args()
    if args.candidates < 1:
        parser.error('candidates must be positive')
    if args.input.resolve() == args.output.resolve():
        parser.error('output must differ from input')
    template = load_json(args.input)
    rng = np.random.default_rng(752)
    poses = np.repeat(template.poses[None], args.candidates, axis=0)
    poses[:,:,:2] += rng.uniform(-.002, .002, poses[:,:,:2].shape)
    poses[:,:,2] += rng.uniform(-.02, .02, poses[:,:,2].shape)
    masks = np.repeat(template.active[None], args.candidates, axis=0)
    rows = []
    expected = None
    for size in (None, 32, 8):
        tracemalloc.start()
        started = time.perf_counter()
        results = validate_batch(template, poses, masks, gap=.01, edge=.01, max_batch_size=size)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if expected is None:
            expected = results
        elif results != expected:
            raise ValueError('chunked validation differs from unchunked validation')
        rows.append({'chunk_size': size, 'traced_peak_bytes': peak, 'instrumented_seconds': elapsed,
                     'total_violations': sum(map(len, results))})
    report = {'input': str(args.input), 'candidates': args.candidates, 'inventory': len(template.ids),
              'active_per_candidate': int(template.active.sum()), 'seed': 752,
              'configuration': {'gap': .01, 'edge': .01, 'translation_jitter_per_axis': .002,
                                'angle_jitter': .02},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'shapely': shapely.__version__},
              'note': 'Python-traced peak includes returned violations and may omit native GEOS memory; timings include tracing overhead. Poses are perturbed and many layouts are invalid. CPU only.',
              'results': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
