#!/usr/bin/env python3
"""Read saved cost receipts or replay one immutable numerical request for profiling.

Replays never write into a campaign and do not consume its budget. Native
construction is excluded because it writes constructor artifacts. Use separate
processes to retain the production operation's process-isolation behavior.
"""
import argparse
import collections
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import statistics
import time

import full_trajectories as f


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def inventory(root):
    grouped = collections.defaultdict(list)
    campaigns = []
    for campaign in sorted(root.glob('adaptive_*')):
        if not (campaign/'allocation_report.json').exists():
            continue
        report = f.read(campaign/'allocation_report.json')
        campaigns.append(dict(name=campaign.name, worker_seconds=report['costs']['worker_seconds']))
        for path in sorted(campaign.glob('*/p*/operations/op_*/receipt.json')):
            rec = f.read(path)
            grouped[rec['kind']].append(dict(path=str(path.parent), **rec))
    worker = sum(c['worker_seconds'] for c in campaigns)
    kinds = {}
    for kind, rows in grouped.items():
        times = [r['worker_seconds'] for r in rows]
        ordered = sorted(rows, key=lambda r:r['worker_seconds'])
        kinds[kind] = dict(count=len(rows), worker_seconds=sum(times),
                           fraction_of_campaign_cost=sum(times)/worker,
                           median_seconds=statistics.median(times),
                           p90_seconds=ordered[int(.9*(len(rows)-1))]['worker_seconds'],
                           cpu_seconds=sum(r['cpu_seconds'] or 0 for r in rows),
                           timeouts=sum(r['status']=='timeout' for r in rows),
                           representative_request=ordered[len(rows)//2]['path']+'/request.json.gz')
    operation_total = sum(k['worker_seconds'] for k in kinds.values())
    return dict(campaigns=campaigns, worker_seconds=worker, operations=kinds,
                worker_and_controller_overhead_seconds=worker-operation_total,
                note='Historical wall costs at production concurrency; CPU unknown for timed-out operations.')


def replay(path, profile=True):
    request = f.read(path)
    if request['kind'] == 'native':
        raise ValueError('Native requests write artifacts; use a separate construction experiment.')
    inputs = [path]
    if 'source' in request:
        inputs.append(Path(request['source']))
    before = {str(p):f.digest(p) for p in inputs}
    profiler = cProfile.Profile()
    started, cpu = time.perf_counter(), time.process_time()
    if profile:
        profiler.enable()
    value = f.operation(request)
    if profile:
        profiler.disable()
    elapsed, cpu = time.perf_counter()-started, time.process_time()-cpu
    if 'document' in value and 'source' in request:
        from common import state
        value['end_frame'] = state(value.pop('document'))
    after = {str(p):f.digest(p) for p in inputs}
    if before != after:
        raise RuntimeError('Replay modified an input')
    saved = path.parent/'result.json.gz'
    expected = f.read(saved)['value'] if saved.exists() else None
    rows = []
    if profile:
        for (file, line, name), (primitive, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
            rows.append(dict(file=file, line=line, function=name, primitive_calls=primitive,
                             calls=calls, self_seconds=own, cumulative_seconds=cumulative))
    return dict(request=str(path), kind=request['kind'], profiled=profile,
                wall_seconds=elapsed, cpu_seconds=cpu, inputs_unchanged=before==after,
                result_sha256=fingerprint(value),
                exact_saved_result_match=None if expected is None else fingerprint(value)==fingerprint(expected),
                top_cumulative=sorted(rows,key=lambda r:-r['cumulative_seconds'])[:70],
                top_self=sorted(rows,key=lambda r:-r['self_seconds'])[:40],
                note='cProfile adds overhead; cumulative times overlap. Replay is outside campaign budgets.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inventory', type=Path)
    mode.add_argument('--request', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--no-profile', action='store_true')
    args = parser.parse_args()
    output = args.out.resolve()
    if args.inventory and output.is_relative_to(args.inventory.resolve()):
        parser.error('Write profiling output outside the saved campaign root')
    if args.request:
        for parent in args.request.resolve().parents:
            if (parent/'allocation_state.json').exists() and (parent/'manifest.json').exists():
                if output.is_relative_to(parent):
                    parser.error('Write profiling output outside the saved campaign')
                break
    result = inventory(args.inventory) if args.inventory else replay(args.request, not args.no_profile)
    f.write(args.out, result)
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('top_')}, indent=2))


if __name__ == '__main__':
    main()
