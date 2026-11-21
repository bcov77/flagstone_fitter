#!/usr/bin/env python3
"""Measure independent-process throughput on the same saved numerical requests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

import full_trajectories as f


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, nargs='+', required=True)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 5, 10])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if min(args.workers) < 1:
        parser.error('Positive worker counts required')
    args.out.mkdir(parents=True, exist_ok=True)

    def run(index, workers):
        rows = []
        for j, request in enumerate(args.request):
            out = args.out/f'w{workers:02d}_job{index:02d}_request{j:02d}.json'
            result = subprocess.run([sys.executable, str(f.ROOT/'scripts/profile_adaptive.py'),
                '--request', str(request), '--no-profile', '--out', str(out)],
                capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError(result.stderr)
            row = f.read(out)
            if row['exact_saved_result_match'] is not True:
                raise RuntimeError(f'Replay differs: {out}')
            rows.append(dict(kind=row['kind'], seconds=row['wall_seconds']))
        return rows

    results = []
    for workers in args.workers:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = list(pool.map(lambda i:run(i, workers), range(workers)))
        seconds = time.perf_counter()-started
        row = dict(workers=workers, matched_request_sequences=workers, wall_seconds=seconds,
                   sequences_per_hour=workers*3600/seconds, jobs=jobs)
        results.append(row)
        f.write(args.out/'summary.json', dict(requests=list(map(str,args.request)), results=results,
                 note='Fixed requests, one sequence per worker; process startup/IO included. '
                      'Short throughput sample, not an entire adaptive campaign.'))
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
