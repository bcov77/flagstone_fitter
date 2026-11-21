#!/usr/bin/env python3
"""Balanced, finite construction/refinement/physics sampling; rerun to resume.

Run with the repository .venv Python. No historical compute or output is changed.
Budgets are active worker wall time, with bounded transaction overshoot. See
analysis/full_trajectories/README.md for the design and interpretation limits.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
DEFAULT_SOURCE = ROOT / 'examples/quarter2_20h_ready_20261114/tasks/task_576_escape/best.json'
DEFAULT_OUT = ROOT / 'examples/full_trajectories_20261119'
DEFAULT_NATIVE = Path.home() / 'miniconda3/envs/patio-packingsolver/bin/python'
_IDLE_STARTED = None
_OVERHEAD_SECONDS = 0.
_SOURCE_PATH = None
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
sys.path[:0] = [str(ROOT / 'analysis/aggressive_trajectories'),
               str(ROOT / 'analysis/physics_trajectories'), str(ROOT / 'src')]


def read(path):
    import gzip
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def write(path, value):
    import gzip
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n').encode()
    if str(path).endswith('.gz'):
        raw = gzip.compress(raw, compresslevel=3, mtime=0)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for(*parts):
    return int(hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:15], 16)


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def code_paths():
    paths = {SCRIPT, ROOT / 'scripts/quarter_native.py'}
    for directory in ('src/flagstone', 'physics_patio/patio_physics',
                      'analysis/aggressive_trajectories', 'analysis/physics_trajectories',
                      'analysis/physics_placement_campaign'):
        paths.update((ROOT / directory).glob('*.py'))
    return sorted(paths)


def versions():
    return {name: importlib.metadata.version(name) for name in
            ('numpy', 'scipy', 'shapely', 'pymunk', 'pyvoronoi', 'matplotlib')}


def pins():
    return {str(p.relative_to(ROOT)): digest(p) for p in code_paths()}


def verify_pins(out):
    m = read(out / 'manifest.json')
    check(digest(out / 'source.json') == m['source_sha256'], 'Frozen inventory changed')
    check(pins() == m['code'], 'Compute changed; preserve this output and use a new --out')
    check(versions() == m['versions'], 'Library versions changed')
    check(sys.version == m['python'], 'Python version changed')
    check(digest(m['design']['native_python']) == m['native_python_sha256'], 'Native Python changed')
    return m


def native_version(python):
    return subprocess.check_output([str(python), '-c',
        'import importlib.metadata; print(importlib.metadata.version("packingsolver"))'],
        text=True, timeout=30).strip()


def prepare(out, design, source):
    if (out / 'manifest.json').exists():
        m = verify_pins(out)
        check(m['design'] == design, 'Design differs; resume with original flags or choose a new --out')
        check(digest(source) == m['source_sha256'], 'Requested inventory differs')
        check(native_version(design['native_python']) == m['native_version'], 'Native package changed')
        return m
    from common import clearance
    d = read(source)
    check(d.get('units') == 'ft' and len(d['stones']) > 0, 'Requires a full inventory in feet')
    ids = [str(s['id']) for s in d['stones']]
    check(len(set(ids)) == len(ids), 'Duplicate inventory IDs')
    check(not any(v.get('locked') for v in d.get('physics', {}).get('pieces', {}).values()),
          'Fresh classical construction requires an unlocked inventory')
    check(clearance(d)['valid'], 'Reference layout has invalid clearance')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'source.json').write_bytes(Path(source).read_bytes())
    frozen = out / 'code'
    for p in code_paths():
        target = frozen / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    cfg = read(ROOT / 'analysis/aggressive_trajectories/config.json')
    from aggressive_policy import validate
    validate(cfg)
    # The extension sample is selected before any outcomes exist.
    keys = [f'p{p:03d}_h{h:02d}_r{r:02d}' for p in range(design['lineages'])
            for h in range(len(design['classical_minutes'])) for r in range(design['replicates'])]
    selected = random.Random(design['seed']).sample(keys, round(len(keys) * design['extension_fraction']))
    m = dict(schema='full-trajectories-v1', created_unix=time.time(), design=design,
             source=str(Path(source).resolve()), source_sha256=digest(source), code=pins(),
             versions=versions(), python=sys.version, physics_config=cfg,
             native_python_sha256=digest(design['native_python']),
             native_version=native_version(design['native_python']),
             extension_keys=sorted(selected),
             objective='Expected best verified original-outline coverage at a fixed total worker-time budget',
             budget='Active worker wall seconds, including empty searches, timeouts, audit, and IO; soft horizon boundaries',
             policy='Normal aggressive selector throughout; no outcome-based pruning in calibration')
    write(out / 'manifest.json', m)
    return m


def operation(request):
    """One isolated numerical operation. All physical gates use existing code."""
    import numpy as np
    from common import clearance, measure, restore
    kind = request['kind']
    doc = request.get('document')
    if 'frame' in request:
        doc = restore(read(request['source']), request['frame'])
    if kind == 'native':
        from flagstone.phase1 import load_json, _layout_data, validate
        from flagstone.quarter import native_input, apply_records, GAP
        folder = Path(request['folder'])
        layout = load_json(request['source'], seed=request['seed'])
        layout.active[:] = False
        write(folder / 'native_input.json', native_input(layout, request['seed'], request['angle']))
        # A replay uses a new subdirectory; never overwrite partial native bytes.
        index = 0
        while (folder / f'native_{index:03d}').exists():
            index += 1
        native = folder / f'native_{index:03d}'
        with (folder / f'native_{index:03d}.log').open('wb') as log:
            subprocess.run([request['native_python'], str(ROOT / 'scripts/quarter_native.py'),
                            '--input', str(folder / 'native_input.json'), '--output', str(native),
                            '--seconds', str(request['seconds'])], stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        patterns = read(native / 'patterns.json')['patterns']
        best = None
        valid = 0
        for pattern in patterns:
            apply_records(layout, pattern['records'])
            if validate(layout, gap=GAP, edge=GAP):
                continue
            valid += 1
            candidate = _layout_data(layout)
            area = sum(layout.shapes[layout.ids[i]].area for i in np.flatnonzero(layout.active))
            if best is None or area > best[0]:
                best = area, candidate
        check(best is not None, 'No original-outline-clearance-valid native construction')
        return dict(document=best[1], metrics=measure(best[1]), patterns=len(patterns), valid_patterns=valid)
    if kind == 'classical':
        from flagstone.phase1 import load_json, _layout_data, validate
        from flagstone import quarter_search2 as search
        from flagstone.quarter import GAP
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as snapshot:
            json.dump(doc, snapshot)
            snapshot.flush()
            layout = load_json(snapshot.name, seed=request['seed'])
        layout.rng.bit_generator.state = request['rng']
        points = search.points_for(layout, shifted=True)
        before = search.measure(layout, points)
        center = before['largest_center']
        action = search.continuous(layout, search.neighborhood(layout, center, count=6), points,
                                   seconds=request['seconds'], maxiter=240, radius_ft=.35)
        # Keep the old adaptive-grid family, with bounded periodic pocket rebuilding.
        structural = None
        if request['round'] % 5 == 4:
            structural = search.repack(layout, center, points, seconds=request['structural_seconds'], count=3)
        check(not validate(layout, gap=GAP, edge=GAP), 'Classical action returned invalid geometry')
        result = _layout_data(layout)
        return dict(document=result, rng=layout.rng.bit_generator.state,
                    action=action, structural=structural, sampled=search.measure(layout, points))
    if kind == 'audit':
        return dict(metrics=measure(doc))
    if kind == 'select':
        from aggressive_policy import generate, choose
        from proposal_search import search_config
        from shapely.errors import GEOSException
        rng = np.random.default_rng(request['seed'])
        cfg = search_config(request['config'], doc, request['search_round'])
        try:
            pool, diagnostic = generate(doc, cfg, rng, excluded=set(request['excluded']))
            chosen, selection = choose(pool, 'normal', request['config'], rng)
            return dict(chosen=chosen, selection=selection, candidates=pool,
                        diagnostics=diagnostic, search_config=cfg,
                        outcome='proposal' if chosen else 'no_candidate')
        except GEOSException:
            return dict(chosen=None, outcome='selection_geometry_failure', traceback=traceback.format_exc())
    if kind == 'physics':
        from aggressive_physics import relax
        from common import state
        result = relax(doc, request['chosen'], request['config'], request['seed'])
        attempted = result.pop('attempted_document')
        result['attempted_end'] = state(attempted)
        result['attempted_metrics'] = measure(attempted)
        result['document'] = attempted if result['accepted'] else doc
        return result
    raise ValueError(kind)


def operation_main(input_path, result_path):
    import resource
    started = time.time()
    cpu0 = time.process_time()
    try:
        request = read(input_path)
        value = operation(request)
        if 'document' in value and 'source' in request:
            from common import state
            value['end_frame'] = state(value.pop('document'))
        result = dict(status='ok', value=value)
    except Exception:
        result = dict(status='error', traceback=traceback.format_exc())
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(started_unix=started, finished_unix=time.time(),
                  cpu_seconds=time.process_time() - cpu0 + children.ru_utime + children.ru_stime)
    write(result_path, result)


def invoke(folder, index, request, timeout):
    """Cache immutable operation receipts so crash recovery never double-commits."""
    global _IDLE_STARTED, _OVERHEAD_SECONDS
    if _IDLE_STARTED is not None:
        _OVERHEAD_SECONDS += time.monotonic() - _IDLE_STARTED
    _IDLE_STARTED = None
    try:
        if _SOURCE_PATH is not None and 'document' in request:
            from common import state
            request = {k:v for k,v in request.items() if k!='document'} | dict(
                frame=state(request['document']), source=str(_SOURCE_PATH))
        return _invoke(folder, index, request, timeout)
    finally:
        _IDLE_STARTED = time.monotonic()


def _invoke(folder, index, request, timeout):
    function_start = time.monotonic()
    directory = folder / 'operations' / f'op_{index:06d}'
    directory.mkdir(parents=True, exist_ok=True)
    inp, result, receipt = (directory / name for name in ('request.json.gz', 'result.json.gz', 'receipt.json'))
    if inp.exists():
        check(read(inp) == request, f'Replay request mismatch: {directory}')
    else:
        write(inp, request)
    if receipt.exists():
        rec = read(receipt)
        check(rec['request_sha256'] == digest(inp), 'Saved operation request changed')
        if rec['result_sha256']:
            check(result.exists() and rec['result_sha256'] == digest(result), 'Saved operation result changed')
        recovered = read(result) if result.exists() else None
        global _OVERHEAD_SECONDS
        _OVERHEAD_SECONDS += time.monotonic()-function_start
        return rec, recovered
    running = directory / 'running.json'
    lost = read(directory / 'interrupted.json') if (directory / 'interrupted.json').exists() else []
    recovered_elapsed = 0.
    if running.exists():
        previous = read(running)
        if result.exists():
            recovered_elapsed = max(0., read(result)['finished_unix']-previous['started_unix'])
        else:
            try:
                os.kill(previous['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f'Previous operation PID {previous["pid"]} may still be active: {directory}')
            lost.append(previous)
            write(directory / 'interrupted.json', lost)
        running.unlink()
    start = time.monotonic()
    start_unix = time.time()
    timeout_hit = False
    if not result.exists():
        with (directory / 'console.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, str(SCRIPT), '--_operation', str(inp), str(result)],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                elapsed = time.monotonic() - start
                write(running, dict(pid=process.pid, started_unix=start_unix,
                                    observed_worker_seconds=elapsed))
                if elapsed >= timeout:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    timeout_hit = True
                    break
                time.sleep(.2)
    r = read(result) if result.exists() else None
    # A completed child result can survive a killed worker before receipt commit.
    elapsed = max(time.monotonic() - function_start, recovered_elapsed)
    if r and running.exists():
        original_start = read(running)['started_unix']
        elapsed = max(elapsed, r['finished_unix'] - original_start)
    elapsed += sum(v['observed_worker_seconds'] for v in lost)
    rec = dict(kind=request['kind'], status='timeout' if timeout_hit else r['status'] if r else 'process_failure',
               worker_seconds=elapsed, cpu_seconds=r.get('cpu_seconds') if r else None,
               interrupted_replays=len(lost), interrupted_accounting_lower_bound=bool(lost),
               request_sha256=digest(inp), result_sha256=digest(result) if r else None)
    write(receipt, rec)
    running.unlink(missing_ok=True)
    return rec, r


def required_value(rec, result, source=None):
    check(rec['status'] == 'ok', f'Numerical operation failed: {rec}; {result}')
    value = result['value']
    if 'end_frame' in value:
        from common import restore
        value = dict(value, document=restore(read(source or _SOURCE_PATH), value['end_frame']))
    return value


def stopped(out):
    return (out / 'STOP').exists()


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def save_state(folder, s):
    global _IDLE_STARTED, _OVERHEAD_SECONDS
    if _IDLE_STARTED is not None:
        _OVERHEAD_SECONDS += time.monotonic() - _IDLE_STARTED
        _IDLE_STARTED = time.monotonic()
    prior = s.get('_session_overhead_charged', 0.)
    s['worker_seconds'] += _OVERHEAD_SECONDS - prior
    s['_session_overhead_charged'] = _OVERHEAD_SECONDS
    write(folder / 'checkpoint.json', s)
    if 'attempts' in s:
        write(folder/'progress.json', {k:s[k] for k in (
            'attempts','accepted','worker_seconds','best_metrics','best_attempt','next_horizon',
            'empty_searches','search_timeouts','physics_timeouts')})


def process_job(out_string, task):
    global _IDLE_STARTED, _OVERHEAD_SECONDS, _SOURCE_PATH
    _IDLE_STARTED, _OVERHEAD_SECONDS = time.monotonic(), 0.
    out = Path(out_string)
    m = read(out / 'manifest.json')
    _SOURCE_PATH = out/'source.json'
    d = m['design']
    folder = out / ('parents' if task['kind'] == 'classical' else 'physics') / task['key']
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(folder / 'task.json', task)
        if (folder / 'summary.json').exists():
            return read(folder / 'summary.json')
        try:
            if task['kind'] == 'classical':
                return classical_job(out, folder, task, m)
            return physics_job(out, folder, task, m)
        finally:
            if _IDLE_STARTED is not None:
                _OVERHEAD_SECONDS += time.monotonic() - _IDLE_STARTED
            write(folder/'sessions'/f'{time.time_ns()}_{os.getpid()}.json',
                  dict(orchestration_worker_seconds=_OVERHEAD_SECONDS))


def classical_job(out, folder, task, m):
    import numpy as np
    d = m['design']
    horizons = d['classical_minutes']
    if (folder / 'checkpoint.json').exists():
        s = read(folder / 'checkpoint.json')
        s['_session_overhead_charged'] = 0.
    else:
        rec, result = invoke(folder, 0, dict(kind='native', source=str(out / 'source.json'),
            folder=str(folder), seed=task['seed'], angle=(task['parent'] % 4) * 90,
            native_python=d['native_python'], seconds=d['native_seconds']), d['native_seconds'] + 180)
        v = required_value(rec, result)
        s = dict(document=v['document'], metrics=v['metrics'], rng=np.random.default_rng(task['seed']).bit_generator.state,
                 next_op=1, round=0, worker_seconds=rec['worker_seconds'], refinement_seconds=0.,
                 native_seconds=rec['worker_seconds'], next_horizon=0)
        write(folder / 'native_summary.json', dict(patterns=v['patterns'], valid_patterns=v['valid_patterns']))
        save_state(folder, s)
    while s['next_horizon'] < len(horizons):
        if stopped(out):
            return dict(key=task['key'], status='checkpointed')
        h = s['next_horizon']
        target = horizons[h] * 60
        if s['refinement_seconds'] >= target:
            # Boundary audits are deliberately charged but do not buy refinement time.
            rec, result = invoke(folder, s['next_op'], dict(kind='audit', document=s['document']), d['audit_timeout'])
            v = required_value(rec, result)
            s['worker_seconds'] += rec['worker_seconds']
            s['next_op'] += 1
            s['metrics'] = v['metrics']
            endpoint = dict(document=s['document'], metrics=s['metrics'], parent=task['parent'],
                            horizon_index=h, nominal_minutes=horizons[h],
                            actual_refinement_seconds=s['refinement_seconds'],
                            cumulative_worker_seconds=s['worker_seconds'], seed=task['seed'])
            write(folder / f'handoff_{h:02d}.json', endpoint)
            s['next_horizon'] += 1
            save_state(folder, s)
            continue
        request = dict(kind='classical', document=s['document'], seed=task['seed'],
                       rng=s['rng'], round=s['round'], seconds=min(d['patch_seconds'], target-s['refinement_seconds']),
                       structural_seconds=min(8., max(.1, target-s['refinement_seconds'])))
        rec, result = invoke(folder, s['next_op'], request, d['classical_timeout'])
        v = required_value(rec, result)
        s.update(document=v['document'], rng=v['rng'], round=s['round']+1, next_op=s['next_op']+1,
                 worker_seconds=s['worker_seconds']+rec['worker_seconds'],
                 refinement_seconds=s['refinement_seconds']+rec['worker_seconds'])
        save_state(folder, s)
    summary = dict(key=task['key'], status='complete', worker_seconds=s['worker_seconds'],
                   handoffs=len(horizons), metrics=s['metrics'])
    write(folder / 'summary.json', summary)
    return summary


def physics_job(out, folder, task, m):
    from common import state
    from aggressive_policy import action_key
    d, cfg = m['design'], m['physics_config']
    parent = read(out / 'parents' / f'p{task["parent"]:03d}' / f'handoff_{task["horizon"]:02d}.json')
    horizons = list(d['physics_minutes'])
    if task['key'] in m['extension_keys']:
        horizons.append(d['extension_minutes'])
    if (folder / 'checkpoint.json').exists():
        s = read(folder / 'checkpoint.json')
        s['_session_overhead_charged'] = 0.
    else:
        s = dict(document=parent['document'], metrics=parent['metrics'], worker_seconds=0., next_op=0,
                 attempts=0, accepted=0, next_horizon=0, search_round=1, excluded=[], pending=None,
                 best_document=parent['document'], best_metrics=parent['metrics'], best_seconds=0.,
                 best_attempt=0, empty_searches=0, search_timeouts=0, physics_timeouts=0)
        s['gap_champion_metrics'], s['gap_champion_document'] = parent['metrics'], parent['document']
        write(folder / 'initial.json', parent)
        write(folder / 'best_coverage.json', s['best_document'])
        write(folder / 'best_gaps.json', s['gap_champion_document'])
        save_state(folder, s)
    while s['next_horizon'] < len(horizons):
        if stopped(out):
            return dict(key=task['key'], status='checkpointed')
        h = s['next_horizon']
        # Finish selected physical transactions before crossing a sampling boundary.
        if s['pending'] is None and s['worker_seconds'] >= horizons[h] * 60:
            write(folder / f'sample_{h:02d}.json', dict(
                nominal_minutes=horizons[h], actual_worker_seconds=s['worker_seconds'],
                best_metrics=s['best_metrics'], end_metrics=s['metrics'], attempts=s['attempts'],
                accepted=s['accepted'], best_seconds=s['best_seconds'], best_attempt=s['best_attempt'],
                gap_champion_metrics=s['gap_champion_metrics'], empty_searches=s['empty_searches'],
                search_timeouts=s['search_timeouts'], physics_timeouts=s['physics_timeouts']))
            s['next_horizon'] += 1
            save_state(folder, s)
            continue
        index = s['next_op']
        if s['pending'] is None:
            request = dict(kind='select', document=s['document'], config=cfg,
                           seed=seed_for(task['seed'], 'select', index),
                           search_round=s['search_round'], excluded=s['excluded'])
            rec, result = invoke(folder, index, request, d['search_timeout'])
            if rec['status'] not in ('ok', 'timeout'):
                required_value(rec, result)
            v = result['value'] if rec['status'] == 'ok' else dict(chosen=None, outcome='search_timeout')
            if v['chosen']:
                s['pending'] = dict(selection_op=index, selection=v, start_metrics=s['metrics'])
                s['search_round'] = 1
            else:
                s['empty_searches'] += 1
                s['search_timeouts'] += rec['status'] == 'timeout'
                # Bound breadth expansion; a timeout retries fresh RNG at base breadth.
                s['search_round'] = 1 if rec['status'] == 'timeout' else min(3, s['search_round']+1)
        else:
            p = s['pending']['selection']['chosen']
            before = state(s['document'])
            request = dict(kind='physics', document=s['document'], chosen=p, config=cfg,
                           seed=seed_for(task['seed'], 'physics', s['attempts']+1))
            rec, result = invoke(folder, index, request, d['physics_timeout'])
            if rec['status'] not in ('ok', 'timeout'):
                required_value(rec, result)
            accepted = False
            v = None
            if rec['status'] == 'ok':
                v = required_value(rec, result)
                accepted = v['accepted']
                if accepted:
                    s['document'], s['metrics'] = v['document'], v['attempted_metrics']
            else:
                s['physics_timeouts'] += 1
            s['excluded'].append(action_key(request['document'], p))
            s['attempts'] += 1
            s['accepted'] += accepted
            after_seconds = s['worker_seconds'] + rec['worker_seconds']
            if s['metrics']['area_percent'] > s['best_metrics']['area_percent'] + 1e-9:
                s.update(best_document=s['document'], best_metrics=s['metrics'],
                         best_seconds=after_seconds, best_attempt=s['attempts'])
                write(folder / 'best_coverage.json', s['best_document'])
            if gap_key(s['metrics']) < gap_key(s['gap_champion_metrics']):
                s['gap_champion_metrics'], s['gap_champion_document'] = s['metrics'], s['document']
                write(folder / 'best_gaps.json', s['gap_champion_document'])
            if v:
                v = {k:value for k,value in v.items() if k!='document'}
            record = dict(number=s['attempts'], selection_op=s['pending']['selection_op'], physics_op=index,
                          start=before, end=state(s['document']), start_metrics=s['pending']['start_metrics'],
                          end_metrics=s['metrics'], accepted=accepted, timed_out=rec['status']=='timeout',
                          cumulative_worker_seconds=after_seconds, chosen=p,
                          physics_seed=request['seed'], result=v)
            write(folder / f'attempt_{s["attempts"]:05d}.json.gz', record)
            s['pending'] = None
        s['worker_seconds'] += rec['worker_seconds']
        s['next_op'] += 1
        save_state(folder, s)
    write(folder / 'best_coverage.json', s['best_document'])
    write(folder / 'final.json', s['document'])
    summary = dict(key=task['key'], status='complete', worker_seconds=s['worker_seconds'],
                   best_metrics=s['best_metrics'], end_metrics=s['metrics'], attempts=s['attempts'],
                   accepted=s['accepted'], samples=len(horizons), extended=task['key'] in m['extension_keys'])
    write(folder / 'summary.json', summary)
    return summary


def tasks_for(m):
    d = m['design']
    parents = [dict(kind='classical', key=f'p{i:03d}', parent=i, seed=seed_for(d['seed'], 'parent', i))
               for i in range(d['lineages'])]
    physics = [dict(kind='physics', key=f'p{p:03d}_h{h:02d}_r{r:02d}', parent=p, horizon=h,
                    replicate=r, seed=seed_for(d['seed'], 'continuation', p, r))
               for p in range(d['lineages']) for h in range(len(d['classical_minutes']))
               for r in range(d['replicates'])]
    random.Random(d['seed']).shuffle(parents)
    random.Random(d['seed']+1).shuffle(physics)
    return parents, physics


def report(out):
    m = read(out / 'manifest.json')
    cache_path = out/'report_cache.json'
    cache = read(cache_path) if cache_path.exists() else {}
    cache_changed = False

    def metadata(path):
        # Handoffs/initial boards are large and immutable after atomic commit.
        # Cache their small metadata, never reload every outline on each tick.
        nonlocal cache_changed
        stat = path.stat()
        key = str(path.relative_to(out))
        signature = [stat.st_mtime_ns, stat.st_size]
        item = cache.get(key)
        if item is None or item['signature'] != signature:
            value = read(path)
            value = {k:v for k,v in value.items() if k!='document'}
            item = dict(signature=signature, value=value)
            cache[key] = item
            cache_changed = True
        return item['value']

    rows, parent_rows = [], []
    totals = dict(worker_seconds=0., operation_cpu_seconds=0., timeout_cpu_unknown_count=0,
                  interrupted_accounting_lower_bound_count=0, operations=0, orchestration_worker_seconds=0.)
    for kind in ('parents', 'physics'):
        for path in (out/kind).glob('*/sessions/*.json'):
            overhead = read(path)['orchestration_worker_seconds']
            totals['worker_seconds'] += overhead
            totals['orchestration_worker_seconds'] += overhead
        for path in sorted((out / kind).glob('*/operations/op_*/receipt.json')):
            rec = read(path)
            totals['worker_seconds'] += rec['worker_seconds']
            totals['operation_cpu_seconds'] += rec['cpu_seconds'] or 0.
            totals['timeout_cpu_unknown_count'] += rec['cpu_seconds'] is None
            totals['interrupted_accounting_lower_bound_count'] += rec['interrupted_accounting_lower_bound']
            totals['operations'] += 1
    for path in sorted((out / 'parents').glob('*/handoff_*.json')):
        p = metadata(path)
        parent_rows.append(dict(parent=p['parent'], classical_minutes=p['nominal_minutes'],
            actual_refinement_seconds=p['actual_refinement_seconds'],
            cumulative_worker_seconds=p['cumulative_worker_seconds'], **metric_columns(p['metrics'])))
    best = None
    for folder in sorted((out / 'physics').glob('p*')):
        if not (folder / 'task.json').exists() or not (folder / 'initial.json').exists():
            continue
        task, initial = read(folder / 'task.json'), metadata(folder / 'initial.json')
        for path in sorted(folder.glob('sample_*.json')):
            p = read(path)
            row = dict(key=task['key'], parent=task['parent'], replicate=task['replicate'],
                classical_minutes=initial['nominal_minutes'], physics_minutes=p['nominal_minutes'],
                parent_worker_seconds=initial['cumulative_worker_seconds'],
                physics_worker_seconds=p['actual_worker_seconds'],
                attempts=p['attempts'], accepted=p['accepted'], empty_searches=p['empty_searches'],
                search_timeouts=p['search_timeouts'], physics_timeouts=p['physics_timeouts'],
                extended=task['key'] in m['extension_keys'], initial_coverage=initial['metrics']['area_percent'],
                **metric_columns(p['best_metrics']))
            rows.append(row)
        cp = folder / 'progress.json'
        if cp.exists():
            s = read(cp)
            if best is None or s['best_metrics']['area_percent'] > best['metrics']['area_percent']:
                best = dict(key=task['key'], metrics=s['best_metrics'], best_attempt=s['best_attempt'])
    for name, data in [('parents.csv', parent_rows), ('samples.csv', rows)]:
        if data:
            temp = out / (name + '.tmp')
            with temp.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
            temp.replace(out / name)
    parent_tasks, physics_tasks = tasks_for(m)
    summary = dict(parent_jobs_complete=sum((out/'parents'/t['key']/'summary.json').exists() for t in parent_tasks),
                   parent_jobs_total=len(parent_tasks),
                   physics_jobs_complete=sum((out/'physics'/t['key']/'summary.json').exists() for t in physics_tasks),
                   physics_jobs_total=len(physics_tasks), samples=len(rows), handoffs=len(parent_rows),
                   costs=totals, best=None if best is None else dict(key=best['key'], metrics=best['metrics']),
                   interpretation='Related roots/checkpoints/replicates are correlated. Descriptive calibration, not independent campaign validation. Audit/transaction overshoot is included at actual times.')
    if best:
        previous = out/'champion.json'
        if not previous.exists() or read(previous) != best:
            shutil.copyfile(out/'physics'/best['key']/'best_coverage.json', out/'best_coverage.json.tmp')
            (out/'best_coverage.json.tmp').replace(out/'best_coverage.json')
            write(previous, best)
    write(out / 'report.json', summary)
    if cache_changed:
        write(cache_path, cache)
    text = (f'Classical parents: {summary["parent_jobs_complete"]}/{len(parent_tasks)} complete\n'
            f'Physics continuations: {summary["physics_jobs_complete"]}/{len(physics_tasks)} complete\n'
            f'Worker hours recorded: {totals["worker_seconds"]/3600:.3f}\n'
            f'Timed samples: {len(rows)}; handoffs: {len(parent_rows)}\n')
    if best:
        text += f'Best verified-clearance coverage: {best["metrics"]["area_percent"]:.6f}% ({best["key"]})\n'
    text += ('Read samples.csv at actual times; a soft nominal horizon may overshoot.\n'
             'Budget sum includes every sampled branch; charge shared parents once in campaign replay.\n'
             'CPU totals omit killed-operation CPU; worker wall time includes their observed cost.\n'
             'Use --verify after stopping. No passing patio or global optimum is implied.\n')
    (out / 'report.txt').write_text(text)
    return summary


def metric_columns(m):
    return dict(coverage=m['area_percent'], interior_gap=m['gaps'].get('largest_interior_inches'),
                edge_gap=m['gaps'].get('largest_edge_inches'), gap_audit_stable=m['gaps']['stable'],
                layout_pass=m['layout_pass'])


def summary_view(s):
    result = {k:v for k,v in s.items() if k not in ('best', 'interpretation')}
    result['best'] = None if not s.get('best') else dict(key=s['best']['key'], **metric_columns(s['best']['metrics']))
    return result


def gap_key(m):
    g = m['gaps']
    if not g['stable']:
        return (1e6, 1e6, -m['area_percent'])
    interior, edge = g.get('largest_interior_inches',1e6), g.get('largest_edge_inches',1e6)
    return max(interior/2, edge/5), interior+edge, -m['area_percent']


def verify(out):
    from common import clearance, restore, state
    from aggressive_policy import eligible, probabilities
    m = verify_pins(out)
    reference = read(out / 'source.json')
    inventory = {str(s['id']): s['outline'] for s in reference['stones']}
    counts = dict(handoffs=0, attempts=0, accepted=0, receipts=0)

    def valid_document(doc, metrics=None):
        check(doc['units'] == reference['units'] and doc['patio'] == reference['patio'], 'Board changed')
        check(len(doc['stones']) == len(inventory) and
              {str(s['id']): s['outline'] for s in doc['stones']} == inventory, 'Original outlines changed')
        check(clearance(doc)['valid'], 'Invalid full-outline clearance')
        if metrics:
            import shapely
            from common import shapes
            actual = 100*sum(g.area for g in shapes(doc).values())/shapely.Polygon(doc['patio']).area
            check(abs(actual-metrics['area_percent']) < 1e-7, 'Coverage differs from original outlines')

    for path in (out / 'parents').glob('*/handoff_*.json'):
        saved = read(path)
        valid_document(saved['document'], saved['metrics'])
        counts['handoffs'] += 1
    for directory in list((out/'parents').glob('p*')) + list((out/'physics').glob('p*')):
        for receipt in directory.glob('operations/op_*/receipt.json'):
            rec = read(receipt)
            check(digest(receipt.parent/'request.json.gz') == rec['request_sha256'], 'Request hash mismatch')
            if rec['result_sha256']:
                check(digest(receipt.parent/'result.json.gz') == rec['result_sha256'], 'Result hash mismatch')
            counts['receipts'] += 1
    for folder in sorted((out/'physics').glob('p*')):
        if not (folder/'initial.json').exists():
            continue
        initial = read(folder/'initial.json')['document']
        previous = state(initial)
        maximum = read(folder/'initial.json')['metrics']['area_percent']
        local_count = 0
        for number, path in enumerate(sorted(folder.glob('attempt_*.json.gz')), 1):
            local_count = number
            r = read(path)
            check(r['number'] == number and r['start'] == previous, 'Broken trajectory chain')
            doc = restore(initial, r['start'])
            check(eligible(doc, r['chosen'], m['physics_config']), 'Ineligible physics action')
            selection = read(folder/'operations'/f'op_{r["selection_op"]:06d}'/'result.json.gz')['value']
            probs, _ = probabilities(selection['candidates'], 'normal', m['physics_config'])
            saved = selection['selection']
            check(selection['candidates'][saved['index']] == r['chosen'] and
                  len(probs) == len(saved['probabilities']) and
                  all(abs(a-b) < 1e-12 for a,b in zip(probs, saved['probabilities'])), 'Selection mismatch')
            check(probs[saved['index']] > 0 and abs(sum(probs)-1) < 1e-10, 'Invalid selection mass')
            if not r['timed_out']:
                expected = (set(r['start']['poses'])-set(r['chosen']['remove'])) | {
                    p['stone'] for p in r['chosen']['placements']}
                check(set(r['result']['attempted_end']['poses']) == expected, 'Attempted inventory mismatch')
            for sid, meta in r['start']['physics'].get('pieces', {}).items():
                if meta.get('locked'):
                    check(r['end']['poses'].get(sid) == r['start']['poses'].get(sid), 'Locked stone moved')
            if r['accepted']:
                result = r['result']
                check(result['settled'] and result['attempted_end'] == r['end'], 'Unsettled accepted endpoint')
                motion = result['final_motion']
                check(motion['max_speed_ips'] < m['physics_config']['max_speed_ips'] and
                      motion['max_angular_speed_dps'] < m['physics_config']['max_angular_speed_dps'], 'Motion gate failed')
                trace = result['phases'][-1]['trace']
                needed = m['physics_config']['stable_samples']
                check(len(trace) >= needed, 'Missing consecutive settling observations')
                for observation in trace[-needed:]:
                    check(observation['clearance']['valid'] and
                          observation['motion']['max_speed_ips'] < m['physics_config']['max_speed_ips'] and
                          observation['motion']['max_angular_speed_dps'] < m['physics_config']['max_angular_speed_dps'],
                          'Pre-stop consecutive settling gate failed')
                valid_document(restore(initial, r['end']), r['end_metrics'])
                counts['accepted'] += 1
            else:
                check(r['end'] == r['start'] and r['end_metrics'] == r['start_metrics'], 'Rollback mismatch')
            previous = r['end']
            maximum = max(maximum, r['end_metrics']['area_percent'])
            counts['attempts'] += 1
        if (folder/'checkpoint.json').exists():
            cp = read(folder/'checkpoint.json')
            check(cp['attempts'] == local_count,
                  'Attempt/checkpoint mismatch; resume before verifying an interrupted transaction')
            check(state(cp['document']) == previous, 'Checkpoint differs from trajectory endpoint')
            check(abs(cp['best_metrics']['area_percent']-maximum) < 1e-7, 'Champion lost')
            valid_document(cp['best_document'], cp['best_metrics'])
            valid_document(cp['gap_champion_document'], cp['gap_champion_metrics'])
            if (folder/'summary.json').exists():
                check(state(read(folder/'final.json')) == previous, 'Final export differs')
                check(state(read(folder/'best_coverage.json')) == state(cp['best_document']), 'Champion export differs')
                for i in range(cp['next_horizon']):
                    sample = read(folder/f'sample_{i:02d}.json')
                    check(sample['actual_worker_seconds']+1e-6 >= sample['nominal_minutes']*60,
                          'Timed sample claims unspent budget')
    counts['scope'] = 'Pins, saved-byte hashes, inventory/board, handoff/accepted/champion original-outline clearance and coverage, eligibility, selection probabilities, chains, rollback, locked poses, consecutive pre-stop settling, exports and time boundaries. Saved gap audits are not recomputed.'
    write(out/'verification.json', counts)
    return counts


def execute(out, m, workers):
    parent_tasks, physics_tasks = tasks_for(m)
    queue = parent_tasks + physics_tasks
    (out/'STOP').unlink(missing_ok=True)

    def stop(signum=None, frame=None):
        (out/'STOP').touch()
        print('\nStopping at bounded operation boundaries; rerun the same command to resume.', flush=True)

    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    active = {}
    errors = []
    last = 0.
    try:
        with futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                        initializer=init_worker) as pool:
            while queue or active:
                if shutil.disk_usage(out).free < m['design']['minimum_free_gib'] * 1024**3:
                    print('Free-disk reserve reached.', flush=True)
                    stop()
                while len(active) < workers and not stopped(out):
                    ready = next((t for t in queue if t['kind']=='classical' or
                        (out/'parents'/f'p{t["parent"]:03d}'/f'handoff_{t["horizon"]:02d}.json').exists()), None)
                    if ready is None:
                        break
                    queue.remove(ready)
                    base = 'parents' if ready['kind']=='classical' else 'physics'
                    if (out/base/ready['key']/'summary.json').exists():
                        continue
                    active[pool.submit(process_job, str(out), ready)] = ready
                if not active:
                    break
                done, _ = futures.wait(active, timeout=2, return_when=futures.FIRST_COMPLETED)
                for future in done:
                    task = active.pop(future)
                    try:
                        result = future.result()
                        print(f'{task["key"]}: {result["status"]}', flush=True)
                    except Exception:
                        error = dict(task=task, traceback=traceback.format_exc())
                        errors.append(error)
                        write(out/'errors'/f'{task["key"]}_{time.time_ns()}.json', error)
                        print(error['traceback'], flush=True)
                        stop()
                if time.monotonic()-last > 30 or done:
                    summary = report(out)
                    print(f'Progress: parents {summary["parent_jobs_complete"]}/{summary["parent_jobs_total"]}; '
                          f'physics {summary["physics_jobs_complete"]}/{summary["physics_jobs_total"]}; '
                          f'{summary["costs"]["worker_seconds"]/3600:.2f} worker-hours; '
                          f'active {", ".join(t["key"] for t in active.values())}', flush=True)
                    last = time.monotonic()
            report(out)
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
    if errors:
        raise RuntimeError('Experiment stopped after a worker error. See errors/; historical records are intact.')
    if not stopped(out):
        check(not queue, 'Unscheduled jobs remain without ready parents; experiment is incomplete')
        result = verify(out)
        write(out/'COMPLETE.json', dict(finished_unix=time.time(), verification=result))
        print(f'Experiment complete and verified. Results: {out / "report.txt"}', flush=True)
    else:
        print(f'Checkpointed. Results so far: {out / "report.txt"}', flush=True)


def minutes(value):
    values = [float(x) for x in value.split(',')]
    if not values or any(not math.isfinite(x) or x < 0 for x in values) or values != sorted(set(values)):
        raise argparse.ArgumentTypeError('Use strictly increasing finite nonnegative comma-separated minutes')
    return values


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--_operation':
        operation_main(sys.argv[2], sys.argv[3])
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--native-python', type=Path, default=DEFAULT_NATIVE)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--lineages', type=int, default=12)
    p.add_argument('--replicates', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261119)
    p.add_argument('--classical-minutes', type=minutes, default=minutes('0,1,4,16'))
    p.add_argument('--physics-minutes', type=minutes, default=minutes('2,8,30'))
    p.add_argument('--extension-minutes', type=float, default=90.)
    p.add_argument('--extension-fraction', type=float, default=.25)
    p.add_argument('--native-seconds', type=float, default=30.)
    p.add_argument('--patch-seconds', type=float, default=10.)
    p.add_argument('--search-timeout', type=float, default=300.)
    p.add_argument('--physics-timeout', type=float, default=300.)
    p.add_argument('--classical-timeout', type=float, default=120.)
    p.add_argument('--audit-timeout', type=float, default=180.)
    p.add_argument('--minimum-free-gib', type=float, default=10.)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--status', action='store_true')
    p.add_argument('--report', action='store_true')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--stop', action='store_true')
    a = p.parse_args()
    out = a.out.resolve()
    if a.stop:
        check((out/'manifest.json').exists(), 'No prepared experiment at this output')
        (out/'STOP').touch()
        print('Safe stop requested.')
        return
    if a.status:
        print(json.dumps(summary_view(read(out/'report.json')), indent=2))
        return
    check(a.workers>0 and a.lineages>0 and a.replicates>0, 'Positive workers/lineages/replicates required')
    for key in ('native_seconds','patch_seconds','search_timeout','physics_timeout','classical_timeout',
                'audit_timeout','extension_minutes','minimum_free_gib'):
        check(math.isfinite(getattr(a,key)) and getattr(a,key)>0, f'Invalid {key}')
    check(math.isfinite(a.extension_fraction) and 0<=a.extension_fraction<=1, 'Invalid extension fraction')
    check(a.physics_minutes[0]>0 and a.extension_minutes>a.physics_minutes[-1], 'Invalid physics horizons')
    check(out != ROOT and out not in ROOT.parents and out != a.source.resolve().parent,
          'Use a dedicated output directory outside the source directory and project ancestors')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This experiment is already running')
        if a.report or a.verify:
            print(json.dumps(verify(out) if a.verify else summary_view(report(out)), indent=2))
            return
        excluded = {'out','source','workers','prepare_only','status','report','verify','stop'}
        design = {k: str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in excluded}
        m = prepare(out, design, a.source.resolve())
        report(out)
        jobs = a.lineages*len(a.classical_minutes)*a.replicates
        base_hours = jobs*a.physics_minutes[-1]/60
        extra_hours = len(m['extension_keys'])*(a.extension_minutes-a.physics_minutes[-1])/60
        print(f'Output: {out}\n{a.lineages} classical lineages, {jobs} physics continuations, '
              f'{len(m["extension_keys"])} random extensions.\n'
              f'Nominal physics allocation: {base_hours+extra_hours:.1f} worker-hours; '
              f'plus classical work, audits and bounded overshoot. {a.workers} workers.\n'
              'Ctrl-C checkpoints; rerun this exact command to resume.', flush=True)
        if not a.prepare_only:
            execute(out, m, a.workers)


if __name__ == '__main__':
    main()
