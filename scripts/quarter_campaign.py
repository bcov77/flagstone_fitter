"""Frozen, foreground experiment matrix for a long quarter-patio campaign.

Standard-library supervisor; numerical imports occur only in isolated workers.
Prepare with --dry-run, run normally, or explicitly --resume the same directory.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

from corner_campaign import atomic_json, identity, matching_process

ROOT = Path(__file__).absolute().parents[1]
RESUME_SCHEMA = 'quarter-resume-v1'
VARIANTS = {
    'energy': dict(objective='energy'),
    'worst': dict(objective='worst'),
    'balanced': dict(objective='balanced'),
    'coverage_guard': dict(objective='balanced', coverage_loss_limit=.005),
    'large_patch': dict(objective='balanced', patch_stones=10, patch_max_stones=16,
                       rebuild_stones=5, translation_radius_ft=.45, maxiter=350),
    'contact': dict(objective='balanced', proposal_strategy='contact'),
    'multi_rebuild': dict(objective='balanced', rebuild_alternatives=3, target_policy='random'),
    'moves_only': dict(objective='balanced', no_swaps=True, no_insert=True, no_rebuild=True),
}


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def utc():
    return datetime.now(timezone.utc).isoformat()


def say(text):
    print(f'[{datetime.now().astimezone():%H:%M:%S}] {text}', flush=True)


def matrix(seconds, seeds=(101, 202, 303)):
    """Matched starts, then independent deepening, then shared Pareto parents."""
    bootstrap = min(120, seconds*.03/len(seeds))
    remaining = max(1, seconds-len(seeds)*bootstrap-min(60, seconds*.01))
    tasks = [dict(phase='bootstrap', variant='bootstrap', seed=seed,
                  seconds=bootstrap, parameters=dict(native_angle_deg=i*25), parent_rule='cold')
             for i, seed in enumerate(seeds)]
    for seed in seeds:
        for name, parameters in VARIANTS.items():
            tasks.append(dict(phase='screen', variant=name, seed=seed,
                              seconds=remaining*.25/(len(seeds)*len(VARIANTS)),
                              parameters=parameters, parent_rule='matched-bootstrap'))
    for name, parameters in VARIANTS.items():
        tasks.append(dict(phase='deepen', variant=name, seed=404,
                          seconds=remaining*.25/len(VARIANTS), parameters=parameters,
                          parent_rule='own-best-certified-gap'))
    for cycle in range(2):
        for name, parameters in VARIANTS.items():
            tasks.append(dict(phase='portfolio', variant=name, seed=505+cycle,
                              seconds=remaining*.5/(2*len(VARIANTS)), parameters=parameters,
                              parent_rule='pareto-gap' if cycle == 0 else 'pareto-center-area'))
    return [dict(number=i, **task) for i, task in enumerate(tasks)]


def validate_cores(cores):
    maximum = os.cpu_count() or 1
    if type(cores) is not int or not 1 <= cores <= maximum:
        raise ValueError(f'--cores must be an integer from 1 through {maximum}')
    return cores


def prepare(output, trial, hours=16, native_python=None, cores=4):
    validate_cores(cores)
    out = Path(output).absolute()
    out.mkdir(parents=True, exist_ok=True)
    if any(p.name != 'console.log' for p in out.iterdir()):
        raise ValueError('Choose a new directory, or explicitly use --resume')
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError('Hours must be positive and finite')
    trial = Path(trial).absolute()
    contract = read(trial/'contract.json')
    if (contract.get('version') != 'quarter-final-layout-v1'
            or contract.get('input_sha256') != hashlib.sha256((trial/'input.json').read_bytes()).hexdigest()
            or contract.get('minimum_clearance_in') != .5
            or contract.get('maximum_empty_circle_diameter_strictly_below_in') != 2
            or contract.get('edge_relaxation') or contract.get('prefix_constraints')):
        raise ValueError('Requires an unchanged strict final-layout trial')
    native = Path(native_python or Path.home()/'miniconda3/envs/patio-packingsolver/bin/python').absolute()
    if not native.is_file():
        raise ValueError(f'Missing native interpreter: {native}')
    snap = out/'snapshot'
    (snap/'scripts').mkdir(parents=True)
    for name in ('quarter_patio.py', 'quarter_native.py', 'quarter_campaign.py',
                 'corner_campaign.py', 'five_stone_edge_start.py', 'run_quarter_campaign.sh'):
        shutil.copy2(ROOT/'scripts'/name, snap/'scripts'/name)
    shutil.copytree(ROOT/'src/flagstone', snap/'src/flagstone', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('pyproject.toml', 'uv.lock'):
        shutil.copy2(ROOT/name, snap/name)
    (out/'trial').mkdir()
    for name in ('input.json', 'contract.json', 'source_inventory.json'):
        if (trial/name).exists():
            shutil.copy2(trial/name, out/'trial'/name)
    plan = dict(version='quarter-campaign-v3-resumable', resume_schema=RESUME_SCHEMA, created_utc=utc(), hours=hours,
                seconds=hours*3600, trial_source=str(trial), trial=str(out/'trial'),
                native_python=str(native), python_executable=sys.executable,
                max_concurrent_tasks=cores, numerical_threads_per_task=1,
                allow_extensions=True,
                variants=VARIANTS, tasks=matrix(hours*3600),
                extension_batch_size=math.lcm(len(VARIANTS), cores),
                extension_rule='Repeat matched-parent waves balanced across variants and worker slots until deadline; no new algorithm variants',
                phase_barriers='Bootstrap, screen, deepen, each portfolio round and each extension wave',
                acceptance='Original outlines, strict half-inch clearances, global gap upper <2in',
                comparison='Screening uses matched bootstrap parents; later phases are adaptive continuation')
    atomic_json(out/'plan.json', plan)
    manifest = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                for folder in (snap, out/'trial') for p in folder.rglob('*') if p.is_file()}
    manifest['plan.json'] = hashlib.sha256((out/'plan.json').read_bytes()).hexdigest()
    atomic_json(out/'manifest.json', manifest)
    atomic_json(out/'state.json', dict(status='prepared', attempts=[], next_task=0,
                                      started_utc=None, deadline_epoch=None, consecutive_crashes=0))
    write_report(out, read(out/'state.json'), plan)
    return plan


def validate_frozen(out):
    manifest = read(out/'manifest.json')
    if not manifest:
        raise ValueError('Missing frozen manifest')
    for name, digest in manifest.items():
        if hashlib.sha256((out/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen input/source changed: {name}')
    pointer = read(out/'recovery.json')
    if pointer:
        revision = (out/pointer['revision']).resolve()
        if not revision.is_relative_to(out.resolve()):
            raise ValueError('Recovery path escapes campaign')
        path = revision/'manifest.json'
        if hashlib.sha256(path.read_bytes()).hexdigest() != pointer['manifest_sha256']:
            raise ValueError('Recovery manifest changed')
        for name, digest in read(path).items():
            if hashlib.sha256((revision/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Recovery source changed: {name}')


def active_snapshot(out):
    pointer = read(Path(out)/'recovery.json')
    return Path(out)/pointer['revision']/'snapshot' if pointer else Path(out)/'snapshot'


def install_recovery(out):
    """Add a versioned runner capsule; preserve original numerical sources/data."""
    out = Path(out).absolute()
    validate_frozen(out)
    active = active_snapshot(out)
    current_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (hashlib.sha256((active/'scripts/quarter_campaign.py').read_bytes()).hexdigest() == current_hash
            and (active/'scripts/quarter_patio.py').read_bytes() == (ROOT/'scripts/quarter_patio.py').read_bytes()):
        return
    with (out/'supervisor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Stop the active supervisor before upgrading recovery')
        revision_root = out/'recovery'
        revision_root.mkdir(exist_ok=True)
        number = 1
        while (revision_root/f'rev_{number:03d}').exists():
            number += 1
        revision = revision_root/f'rev_{number:03d}'
        snapshot = revision/'snapshot'
        (snapshot/'scripts').mkdir(parents=True)
        original = out/'snapshot'
        # Preserve the original objective/geometry/native implementation.
        shutil.copytree(original/'src', snapshot/'src', ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('quarter_native.py', 'five_stone_edge_start.py', 'corner_campaign.py'):
            shutil.copy2(original/'scripts'/name, snapshot/'scripts'/name)
        for name in ('quarter_campaign.py', 'quarter_patio.py', 'run_quarter_campaign.sh'):
            shutil.copy2(ROOT/'scripts'/name, snapshot/'scripts'/name)
        for name in ('pyproject.toml', 'uv.lock'):
            shutil.copy2(original/name, snapshot/name)
        shutil.copy2(out/'state.json', revision/'state_before.json')
        atomic_json(revision/'provenance.json', dict(schema=RESUME_SCHEMA, installed_utc=utc(),
                    original_manifest_sha256=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),
                    change='Checkpoint/resume supervision and worker bookkeeping only; original numerical sources retained'))
        manifest = {str(p.relative_to(revision)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in revision.rglob('*') if p.is_file()}
        atomic_json(revision/'manifest.json', manifest)
        atomic_json(out/'recovery.json', dict(schema=RESUME_SCHEMA, revision=str(revision.relative_to(out)),
                    manifest_sha256=hashlib.sha256((revision/'manifest.json').read_bytes()).hexdigest()))


def pause_remaining(state):
    """Graceful interruption pauses wall budget; legacy interruptions are inferred."""
    if state.get('remaining_seconds') is not None:
        return max(0, state['remaining_seconds'])
    if state.get('deadline_epoch') is None:
        return None
    stamp = state.get('paused_epoch')
    if stamp is None and state.get('status') == 'interrupted' and state.get('updated_utc'):
        stamp = datetime.fromisoformat(state['updated_utc']).timestamp()
    return max(0, state['deadline_epoch']-stamp) if stamp is not None else None


def task_remaining(record):
    total = record.get('total_budget_seconds', record['budget_seconds'])
    return max(0, total-record.get('elapsed_seconds', 0))


def prepare_task_resume(out, record, campaign_deadline):
    """Keep logical task/lineage; archive segments and use a new output directory."""
    remaining = task_remaining(record)
    if remaining <= 2:
        record.update(outcome='budget-exhausted-after-interruption')
        return False
    candidates = [record['directory'], record.get('resume_from')]
    candidates += [s['directory'] for s in reversed(record.get('segments', []))]
    source = next((p for p in candidates if p and any((Path(p)/name).exists()
                  for name in ('checkpoint.json', 'best.json', 'native/patterns.json'))), None)
    source = source or next((p for p in candidates if p and Path(p).is_dir()), record['directory'])
    previous = dict(record)
    previous.pop('segments', None)
    record.setdefault('segments', []).append(previous)
    record.setdefault('total_budget_seconds', record['budget_seconds'])
    record['consumed_seconds'] = record.get('elapsed_seconds', 0)
    original = Path(record.get('original_directory', record['directory']))
    record['original_directory'] = str(original)
    segment = len(record['segments'])
    directory = original.with_name(original.name+f'_resume_{segment:03d}')
    while directory.exists():
        segment += 1
        directory = original.with_name(original.name+f'_resume_{segment:03d}')
    record.update(resume_from=source, directory=str(directory),
                  log=str(Path(out)/f"tasks/task_{record['number']:03d}_resume_{segment:03d}.log"),
                  outcome='running', budget_seconds=min(remaining, campaign_deadline-time.time()),
                  started_epoch=time.time(), deadline_epoch=min(campaign_deadline, time.time()+remaining))
    for key in ('pid', 'identity', 'returncode', 'log_offset', 'launch_error', 'finished_utc', 'result'):
        record.pop(key, None)
    return True


def preflight(native_python):
    """Import-only environment check; never construct or optimize an instance."""
    native = native_python or str(Path.home()/'miniconda3/envs/patio-packingsolver/bin/python')
    checks = [
        [sys.executable, '-c', 'import numpy, scipy, shapely; print("Geometry/scientific environment OK")'],
        [native, '-c', 'import packingsolver.irregular as p; import importlib.metadata as m; '
         'assert hasattr(p.OptimizeParameters(), "use_local_search_direct_trigonometry"); '
         'print("PackingSolver " + m.version("packingsolver") + " import/API OK")']]
    messages = []
    for command in checks:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise RuntimeError('Environment preflight failed: '+result.stderr[-2000:])
        messages.append(result.stdout.strip())
    return messages


def clean_nested(directory):
    """Match recorded start identity before touching a separately owned group."""
    for file in Path(directory).glob('*.owner.json'):
        owner = read(file)
        if owner.get('status') != 'running':
            continue
        pid, expected = owner.get('pid'), owner.get('identity')
        if pid and expected and identity(pid) == expected:
            try:
                if os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        owner['status'] = 'cleaned-after-parent-exit'
        atomic_json(file, owner)
    # A worker can die in the tiny interval between spawning a detached native
    # group and persisting its ownership file. Exact frozen script + unique task
    # path identifies that campaign-owned startup window without trusting a PID.
    directory = Path(directory).absolute()
    root = directory.parent.parent
    script_dirs = {root/'snapshot/scripts', active_snapshot(root)/'scripts'}
    expected = tuple(item for scripts in script_dirs for item in
                     (str(scripts/'quarter_native.py'), str(scripts/'quarter_native2.py'), str(scripts/'quarter_patio.py')+' verify ', str(scripts/'quarter_patio2.py')+' verify '))
    listing = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True)
    for line in listing.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if (len(parts) != 2 or str(directory)+os.sep not in parts[1]
                or not any(script in parts[1] for script in expected)):
            continue
        pid = int(parts[0])
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def stop(process, record):
    pid = process.pid if process else record.get('pid')
    if not pid:
        return
    if process is None and not matching_process(pid, record['directory'], record.get('identity')):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if process:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process:
        process.wait()


def discover_worker(record):
    """Close the spawn-to-PID-save crash window before attempting a relaunch."""
    if record.get('pid') or not record.get('command'):
        return
    script = record['command'][2]
    listing = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True)
    matches = []
    for line in listing.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and script in parts[1] and record['directory'] in parts[1]:
            matches.append(int(parts[0]))
    if len(matches) > 1:
        raise RuntimeError('Multiple workers claim the same task; refusing duplicate launch')
    if matches:
        record.update(pid=matches[0], identity=identity(matches[0]))


def result_for(directory):
    folder = Path(directory)
    run = read(folder/'summary.json') or read(folder/'state.json')
    v = read(folder/'verification.json')
    path = folder/'best.json'
    gap = v.get('gap') or {}
    interval_ok = (all(isinstance(gap.get(k), (int, float)) and math.isfinite(gap[k])
                       for k in ('lower_diameter_in', 'upper_diameter_in'))
                   and 0 <= gap['lower_diameter_in'] <= gap['upper_diameter_in'])
    eligible = (path.is_file() and v.get('reference_matched') is True
                and v.get('clearance_errors') == [] and v.get('gap')
                and interval_ok
                and v.get('layout_sha256') == hashlib.sha256(path.read_bytes()).hexdigest())
    counts, costs = {}, {}
    for r in run.get('rounds', []):
        for name in ('continuous', 'replacement', 'insertion', 'rebuilt'):
            action = r.get(name, {})
            actions = action.get('alternatives', [action])
            for item in actions:
                if item.get('skipped'):
                    continue
                counts[name+'_attempted'] = counts.get(name+'_attempted', 0)+1
                counts[name+'_accepted'] = counts.get(name+'_accepted', 0)+int(item.get('accepted', False))
                costs[name] = costs.get(name, 0)+item.get('elapsed_seconds', 0)
    return dict(run_status=run.get('status', 'missing-result'), initial=run.get('initial'),
                metrics=run.get('best') or run.get('initial'),
                eligible_parent=bool(eligible), verified=bool(eligible and v.get('passed') and gap['upper_diameter_in'] < 2),
                gap=v.get('gap') if eligible else None, clearance_errors=v.get('clearance_errors'),
                layout=str(path) if eligible else None, rounds=len(run.get('rounds', [])),
                cpu_seconds_total=run.get('cpu_seconds_total'), action_counts=counts, action_seconds=costs)


def parents(state):
    return [a for a in state['attempts'] if a.get('result', {}).get('eligible_parent')]


def pareto(attempts):
    def vector(a):
        r = a['result']
        return (r['gap']['upper_diameter_in'], r['metrics']['sampled_bad_center_area_sqft'], -r['metrics']['coverage'])
    return [a for a in attempts if not any(
        all(x <= y for x, y in zip(vector(b), vector(a))) and
        any(x < y for x, y in zip(vector(b), vector(a))) for b in attempts)]


def choose_parent(task, state):
    candidates = parents(state)
    if task['parent_rule'] == 'cold':
        return None
    if task['parent_rule'] == 'matched-bootstrap':
        candidates = [a for a in candidates if a['task']['phase'] == 'bootstrap' and a['task']['seed'] == task['seed']]
    elif task['parent_rule'] == 'own-best-certified-gap':
        candidates = [a for a in candidates if a['task']['variant'] == task['variant']]
    else:
        candidates = pareto(candidates)
    if not candidates:
        return None
    if task['parent_rule'] == 'pareto-center-area':
        return min(candidates, key=lambda a: (a['result']['metrics']['sampled_bad_center_area_sqft'],
                                              a['result']['gap']['upper_diameter_in']))
    return min(candidates, key=lambda a: a['result']['gap']['upper_diameter_in'])


def csv_write(path, rows, fields):
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def write_report(out, state, plan):
    rows = []
    for a in state['attempts']:
        r = a.get('result', {})
        m, gap = r.get('metrics') or {}, r.get('gap') or {}
        initial = r.get('initial') or {}
        row = dict(task=a['number'], phase=a['task']['phase'], variant=a['task']['variant'],
                   seed=a['task']['seed'], outcome=a.get('outcome'), run_status=r.get('run_status'),
                   elapsed_seconds=a.get('elapsed_seconds'), budget_seconds=a.get('budget_seconds'),
                   cpu_seconds=r.get('cpu_seconds_total'), rounds=r.get('rounds'), verified=r.get('verified'),
                   lower_in=gap.get('lower_diameter_in'), upper_in=gap.get('upper_diameter_in'),
                   sampled_max_in=m.get('sampled_max_diameter_in'), bad_center_area_sqft=m.get('sampled_bad_center_area_sqft'),
                   coverage=m.get('coverage'), placed=m.get('placed'), parent=a.get('parent_number'),
                   initial_sampled_max_in=initial.get('sampled_max_diameter_in'),
                   initial_bad_center_area_sqft=initial.get('sampled_bad_center_area_sqft'),
                   initial_coverage=initial.get('coverage'), parent_lower_in=a.get('parent_lower_in'),
                   parent_upper_in=a.get('parent_upper_in'),
                   gap_improvement_lower_in=(a['parent_lower_in']-gap['upper_diameter_in']
                                             if a.get('parent_lower_in') is not None and gap else None),
                   directory=a['directory'])
        row.update(concurrency_limit=a.get('concurrency_limit'), active_at_launch=a.get('active_at_launch'))
        row.update(r.get('action_counts', {}))
        for name, seconds in r.get('action_seconds', {}).items():
            row[name+'_seconds'] = seconds
        rows.append(row)
    fields = ['task', 'phase', 'variant', 'seed', 'parent', 'outcome', 'run_status', 'elapsed_seconds',
              'budget_seconds', 'cpu_seconds', 'rounds', 'verified', 'lower_in', 'upper_in',
              'sampled_max_in', 'bad_center_area_sqft', 'coverage', 'placed', 'directory']
    fields += ['initial_sampled_max_in', 'initial_bad_center_area_sqft', 'initial_coverage',
               'parent_lower_in', 'parent_upper_in', 'gap_improvement_lower_in',
               'concurrency_limit', 'active_at_launch']
    fields += [k for name in ('continuous', 'replacement', 'insertion', 'rebuilt')
               for k in (name+'_attempted', name+'_accepted', name+'_seconds')]
    csv_write(out/'tasks.csv', rows, fields)
    aggregates = []
    for name in ('bootstrap', *plan['variants']):
        subset = [r for r in rows if r['variant'] == name]
        checked = [r for r in subset if r['upper_in'] is not None]
        screen = [r for r in checked if r['phase'] == 'screen']
        improvements = [r['gap_improvement_lower_in'] for r in screen if r['gap_improvement_lower_in'] is not None]
        aggregates.append(dict(variant=name, tasks=len(subset),
                               completed=sum(r['outcome'] == 'completed' for r in subset),
                               crashes=sum(r['outcome'] == 'worker-failed' for r in subset),
                               timeouts=sum(r['outcome'] == 'timed-out' for r in subset),
                               verified=sum(bool(r['verified']) for r in subset),
                               checked=len(checked), best_upper_in=min((r['upper_in'] for r in checked), default=None),
                               screen_checked=len(screen),
                               screen_median_upper_in=statistics.median(r['upper_in'] for r in screen) if screen else None,
                               screen_median_improvement_lower_in=statistics.median(improvements) if improvements else None,
                               elapsed_seconds=sum(r['elapsed_seconds'] or 0 for r in subset)))
    csv_write(out/'variants.csv', aggregates, list(aggregates[0]))
    frontier = pareto(parents(state))
    analysis = dict(status=state['status'], updated_utc=utc(), deadline_epoch=state.get('deadline_epoch'),
                    max_concurrent_tasks=state.get('max_concurrent_tasks', plan.get('max_concurrent_tasks', 1)),
                    peak_active_tasks=state.get('peak_active_tasks', 0), resource_history=state.get('resource_history', []),
                    tasks=rows, variants=aggregates, pareto_task_numbers=[a['number'] for a in frontier],
                    validated_successes=[r for r in rows if r['verified']],
                    interpretation='Only gap upper <2in plus original-reference and clearance checks is success. '
                                   'Screening starts are matched; later adaptive continuations are not cold-start comparisons.')
    atomic_json(out/'analysis_summary.json', analysis)
    lines = ['# Quarter-patio campaign report', '', f"Status: **{state['status']}**. Updated {analysis['updated_utc']}.", '',
             f"CPU budget: **{analysis['max_concurrent_tasks']} single-thread workers**; peak active tasks: {analysis['peak_active_tasks']}.", '',
             'Strict two-inch global gap goal; half-inch clearances; all inventory available; no prefix gates.', '',
             '## Variants', '', '| Variant | Tasks | Checked | Successes | Best upper (in) | Screen median upper (in) | Crashes | Timeouts |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregates:
        upper = f"{r['best_upper_in']:.6f}" if r['best_upper_in'] is not None else '—'
        screen_upper = f"{r['screen_median_upper_in']:.6f}" if r['screen_median_upper_in'] is not None else '—'
        lines.append(f"| {r['variant']} | {r['tasks']} | {r['checked']} | {r['verified']} | {upper} | {screen_upper} | {r['crashes']} | {r['timeouts']} |")
    lines += ['', '## Saved tradeoffs', '', 'Nondominated final layouts across certified gap upper, sampled bad-center area, and coverage:', '']
    for a in frontier:
        r = a['result']
        lines.append(f"- Task {a['number']} ({a['task']['variant']}): upper {r['gap']['upper_diameter_in']:.6f} in; "
                     f"bad-center proxy {r['metrics']['sampled_bad_center_area_sqft']:.4f} sq ft; "
                     f"coverage {100*r['metrics']['coverage']:.2f}%. [Layout]({os.path.relpath(r['layout'], out)}).")
    lines += ['', '## Review notes', '',
              '- Read `tasks.csv` for seeds, parent lineage, CPU/wall time, actions and failures.',
              '- Read `variants.csv` for aggregate outcomes. Use the `screen` rows for matched-parent comparisons.',
              '- Later phases select parents adaptively; do not interpret them as independent cold-start trials.',
              '- A positive `gap_improvement_lower_in` means the final upper is below the parent lower, a certified improvement despite interval uncertainty.',
              '- A timed-out task may retain a checkpoint but is not a certified success unless verification exists and matches its bytes.',
              '- Bootstrap frames are 0/25/50 degrees; poses return to the original patio frame. Seeds shuffle input order, not native optimizer RNG.',
              '- Human visual review and preliminary input measurement limitations remain pending.', '']
    temp = out/'report.tmp'
    temp.write_text('\n'.join(lines))
    temp.replace(out/'report.md')


def command_for(out, plan, record, parent):
    task = record['task']
    command = [plan['python_executable'], '-u', str(active_snapshot(out)/'scripts/quarter_patio.py'), 'run',
               '--trial', str(out/'trial'), '--output', record['directory'],
               '--native-python', plan['native_python'], '--seed', str(task['seed']),
               '--seconds', str(max(.1, record['budget_seconds']-10)),
               '--bootstrap-seconds', str(min(60, record['budget_seconds']*.5)),
               '--rounds', '0' if task['phase'] == 'bootstrap' else '1000000']
    if record.get('resume_from'):
        command += ['--resume-from', record['resume_from']]
    elif parent:
        command += ['--start', parent['result']['layout']]
    for key, value in task['parameters'].items():
        flag = '--'+key.replace('_', '-')
        if value is True:
            command.append(flag)
        else:
            command += [flag, str(value)]
    return command


def task_group(task):
    phase = task['phase']
    if phase == 'portfolio':
        return phase+':'+task['parent_rule']
    if phase == 'extension':
        return phase+':'+str(task['wave'])
    return phase


def supervise(out, *, cores=None, heartbeat=60, emit=say, poll=.5, worker_override=None):
    """Bounded process portfolio with durable ownership and phase barriers."""
    out = Path(out).absolute()
    lock = (out/'supervisor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another supervisor already owns this campaign')
    state, plan = read(out/'state.json'), read(out/'plan.json')
    running = {}
    exit_code = 0
    failure_stop = False

    def save():
        atomic_json(out/'state.json', state)

    def event(kind, **values):
        with (out/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=utc(), event=kind, **values))+'\n')
            f.flush()

    def alive(context):
        process, record = context['process'], context['record']
        return (process.poll() is None if process else
                bool(record.get('pid') and matching_process(record['pid'], record['directory'], record.get('identity'))))

    def attach(record, allow_launch=True):
        # Register before any operation that can be interrupted, so finally
        # knows every task whose detached children may need cleanup.
        context = dict(record=record, process=None, offset=record.get('log_offset', 0))
        running[record['number']] = context
        discover_worker(record)
        if record.get('pid'):
            if matching_process(record['pid'], record['directory'], record.get('identity')):
                event('task-adopted', number=record['number'], pid=record['pid'])
        elif allow_launch and not Path(record['directory']).exists():
            env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                       OPENBLAS_NUM_THREADS='1', OPENBLAS_DEFAULT_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1', MKL_NUM_THREADS='1',
                       VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                       BLIS_NUM_THREADS='1', RAYON_NUM_THREADS='1')
            command = worker_override(record) if callable(worker_override) else worker_override or record['command']
            try:
                with Path(record['log']).open('ab', buffering=0) as log:
                    process = subprocess.Popen(command, cwd=active_snapshot(out), stdout=log,
                                               stderr=subprocess.STDOUT, env=env, start_new_session=True)
                context['process'] = process
                record.update(pid=process.pid, identity=identity(process.pid))
            except OSError as exc:
                record['launch_error'] = str(exc)
        save()
        emit(f"START #{record['number']:03d} {record['task']['phase']}/{record['task']['variant']} "
             f"seed {record['task']['seed']}; {record['budget_seconds']/60:.1f} min; "
             f"parent {record.get('parent_number')}; {len(running)}/{capacity} worker slots")

    def finish(number, timed_out=False, forced_outcome=None):
        nonlocal failure_stop
        context = running[number]
        record, process = context['record'], context['process']
        stop(process, record)
        clean_nested(record['directory'])
        result = result_for(record['directory'])
        rc = process.returncode if process else None
        completed_statuses = ('pilot-completed/incomplete', 'target-geometrically-verified-review-pending',
                              'no-valid-bootstrap/incomplete')
        outcome = forced_outcome or ('timed-out' if timed_out else 'worker-failed' if
                    (rc not in (0, None) or result['run_status'] not in completed_statuses) else 'completed')
        record.update(outcome=outcome, returncode=rc, result=result,
                      elapsed_seconds=record.get('consumed_seconds', 0)+time.time()-record.get('started_epoch', time.time()),
                      finished_utc=utc())
        if forced_outcome is None:
            state['consecutive_crashes'] = state['consecutive_crashes']+1 if outcome == 'worker-failed' else 0
            failure_stop |= state['consecutive_crashes'] >= 3
        event('task-finish', number=number, outcome=outcome, verified=result['verified'])
        del running[number]
        save()
        write_report(out, state, plan)
        emit(f"END #{number:03d}: {outcome}; {plan.get('metric_label', 'checked upper')} "
             f"{(result.get('gap') or {}).get('upper_diameter_in', 'unavailable')} {plan.get('metric_units', 'in')}")

    def stop_all():
        # Signal every group first, then use one shared grace period rather
        # than paying the grace period once per worker at the global deadline.
        for context in running.values():
            record = context['record']
            discover_worker(record)
            if alive(context):
                try:
                    os.killpg(record['pid'], signal.SIGTERM)
                except ProcessLookupError:
                    pass
        until = time.monotonic()+5
        while running and time.monotonic() < until and any(alive(c) for c in running.values()):
            time.sleep(.05)
        for context in running.values():
            record, process = context['record'], context['process']
            if alive(context):
                try:
                    os.killpg(record['pid'], signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process:
                process.wait()
            clean_nested(record['directory'])
            record.update(outcome='interrupted', result=result_for(record['directory']),
                          elapsed_seconds=record.get('consumed_seconds', 0)+time.time()-record.get('started_epoch', time.time()),
                          finished_utc=utc())
            event('task-interrupted', number=record['number'], reason=state['status'])

    try:
        validate_frozen(out)
        capacity = validate_cores(cores if cores is not None else
                                  state.get('max_concurrent_tasks', plan.get('max_concurrent_tasks', 1)))
        history = state.setdefault('resource_history', [])
        if not history or history[-1]['cores'] != capacity:
            history.append(dict(time=utc(), cores=capacity))
            event('cpu-budget', cores=capacity)
        state.update(supervisor_script=str(Path(__file__).absolute()), max_concurrent_tasks=capacity)
        state.setdefault('group_parents', {})
        remaining_paused = pause_remaining(state) if state.get('status') == 'interrupted' else None
        if remaining_paused is not None:
            old_deadline = state['deadline_epoch']
            state['deadline_epoch'] = time.time()+remaining_paused
            state.setdefault('resume_history', []).append(dict(time=utc(), remaining_seconds=remaining_paused,
                  previous_deadline_epoch=old_deadline, deadline_epoch=state['deadline_epoch']))
            state.pop('remaining_seconds', None)
            state.pop('paused_epoch', None)
        if state.get('deadline_epoch') is None:
            state.update(started_utc=utc(), deadline_epoch=time.time()+plan['seconds'])
        state['status'] = 'running'
        save()
        recovered = [a for a in state['attempts'] if a.get('outcome') == 'running']
        for record in recovered:
            attach(record, allow_launch=(time.time() < state['deadline_epoch'] and len(running) < capacity))
        # A lower explicit budget on recovery must not leave more CPU-heavy
        # workers running than requested. All their complete checkpoints survive.
        for number in list(running)[capacity:]:
            finish(number, forced_outcome='interrupted-resource-change')
        pending_resume = [a for a in state['attempts'] if a.get('outcome') in
                          ('interrupted', 'interrupted-resource-change')]
        state['peak_active_tasks'] = max(state.get('peak_active_tasks', 0), len(running))
        next_heartbeat = time.monotonic()+heartbeat
        while time.time() < state['deadline_epoch']-max(2, plan.get('final_reserve_seconds', 2)):
            for number, context in list(running.items()):
                record = context['record']
                try:
                    with Path(record['log']).open(errors='replace') as stream:
                        stream.seek(context['offset'])
                        chunk = stream.read()
                        context['offset'] = record['log_offset'] = stream.tell()
                    for line in chunk.splitlines():
                        emit(f"  {number:03d} | {line}")
                except FileNotFoundError:
                    pass
                is_alive = alive(context)
                timed_out = time.time() >= min(record['deadline_epoch'], state['deadline_epoch'])
                if not is_alive or timed_out:
                    finish(number, timed_out=is_alive and timed_out)
            if failure_stop:
                state['status'] = 'stopped-after-three-infrastructure-failures'
                exit_code = 1
                break

            while len(running) < capacity and time.time() < state['deadline_epoch']-max(2, plan.get('final_reserve_seconds', 2)):
                if shutil.disk_usage(out).free < plan.get('minimum_free_bytes', 0):
                    state['status'] = 'stopped-low-disk-space'
                    exit_code = 1
                    break
                if pending_resume:
                    record = pending_resume[0]
                    group = task_group(record['task'])
                    if running and any(task_group(c['record']['task']) != group for c in running.values()):
                        break
                    pending_resume.pop(0)
                    if not prepare_task_resume(out, record, state['deadline_epoch']):
                        save()
                        continue
                    record.update(concurrency_limit=capacity, active_at_launch=len(running)+1)
                    record['command'] = command_for(out, plan, record, None)
                    save()
                    event('task-resume', number=record['number'], source=record['resume_from'],
                          remaining_seconds=record['budget_seconds'])
                    attach(record)
                    state['peak_active_tasks'] = max(state.get('peak_active_tasks', 0), len(running))
                    save()
                    continue
                i = state['next_task']
                if i < len(plan['tasks']):
                    task = plan['tasks'][i]
                else:
                    if not plan.get('allow_extensions', True):
                        if not running:
                            state['status'] = 'declared-matrix-finished'
                        break
                    if not parents(state):
                        if not running:
                            state['status'] = 'no-verified-parent/incomplete'
                        break
                    names = list(plan['variants'])
                    offset = i-len(plan['tasks'])
                    size = plan.get('extension_batch_size', 32)
                    wave = offset//size
                    name = names[offset % len(names)]
                    task = dict(number=i, phase='extension', variant=name, wave=wave,
                                seed=700+wave*100+offset%size//len(names), seconds=plan.get('extension_seconds', 1800),
                                parameters=plan['variants'][name],
                                parent_rule='pareto-gap' if wave%2 == 0 else 'pareto-center-area')
                group = task_group(task)
                if running and any(task_group(c['record']['task']) != group for c in running.values()):
                    break
                # Every task in a comparison group selects from the same
                # completed-parent snapshot, independent of completion ordering.
                if group not in state['group_parents']:
                    state['group_parents'][group] = [a['number'] for a in parents(state)]
                    event('phase-group-start', group=group, parents=state['group_parents'][group])
                allowed = set(state['group_parents'][group])
                parent_state = dict(attempts=[a for a in state['attempts'] if a['number'] in allowed])
                parent = choose_parent(task, parent_state)
                directory = out/f"tasks/task_{i:03d}_{task['variant']}"
                directory.parent.mkdir(exist_ok=True)
                record = dict(number=i, task=task, directory=str(directory),
                              parent_number=parent['number'] if parent else None,
                              parent_lower_in=parent['result']['gap']['lower_diameter_in'] if parent else None,
                              parent_upper_in=parent['result']['gap']['upper_diameter_in'] if parent else None,
                              outcome='running', budget_seconds=min(task['seconds'], state['deadline_epoch']-time.time()),
                              concurrency_limit=capacity, active_at_launch=len(running)+1,
                              log=str(out/f'tasks/task_{i:03d}.log'))
                state['attempts'].append(record)
                state['next_task'] = i+1
                if task['parent_rule'] != 'cold' and parent is None:
                    record.update(outcome='skipped-no-parent', elapsed_seconds=0)
                    save()
                    continue
                record['command'] = command_for(out, plan, record, parent)
                record['started_epoch'] = time.time()
                record['deadline_epoch'] = min(state['deadline_epoch'], time.time()+record['budget_seconds'])
                save()
                event('task-start', number=i, variant=task['variant'], phase=task['phase'],
                      parent=record['parent_number'], cores=capacity)
                attach(record)
                state['peak_active_tasks'] = max(state.get('peak_active_tasks', 0), len(running))
                save()
            if state['status'] != 'running':
                break
            if time.monotonic() >= next_heartbeat:
                emit(f"HEARTBEAT: {len(running)}/{capacity} workers; "
                     f"campaign {max(0, state['deadline_epoch']-time.time())/3600:.2f} h left")
                for number, context in running.items():
                    live = read(Path(context['record']['directory'])/'state.json')
                    metrics = live.get('best') or live.get('initial') or {}
                    emit(f"  #{number:03d}: {len(live.get('rounds', []))} rounds; "
                         f"sampled worst {metrics.get('sampled_max_diameter_in', 'pending')} in")
                next_heartbeat = time.monotonic()+heartbeat
                save()
                write_report(out, state, plan)
            time.sleep(poll)
        if state['status'] == 'running':
            state['status'] = 'wall-budget-finished'
    except (KeyboardInterrupt, BrokenPipeError):
        state['status'] = 'interrupted'
        exit_code = 130
    except BaseException as exc:
        state.update(status='supervisor-failed', error=f'{type(exc).__name__}: {exc}')
        exit_code = 1
        raise
    finally:
        handlers = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            stop_all()
            if state['status'] == 'interrupted':
                state['paused_epoch'] = time.time()
                state['remaining_seconds'] = max(0, state.get('deadline_epoch', time.time())-time.time())
            state['updated_utc'] = utc()
            save()
            write_report(out, state, plan)
            atomic_json(out/'summary.json', dict(status=state['status'], exit_code=exit_code,
                        tasks=len(state['attempts']), max_concurrent_tasks=state.get('max_concurrent_tasks'),
                        peak_active_tasks=state.get('peak_active_tasks', 0),
                        verified_tasks=sum(a.get('result', {}).get('verified', False) for a in state['attempts']),
                        deadline_epoch=state.get('deadline_epoch'), report=str(out/'report.md')))
        finally:
            for s, handler in handlers.items():
                signal.signal(s, handler)
            lock.close()
    return exit_code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--trial', default=str(ROOT/'examples/quarter_patio_v1'))
    p.add_argument('--hours', type=float, default=16)
    p.add_argument('--cores', type=int, help='Maximum concurrent single-thread workers (new campaigns default to 4)')
    p.add_argument('--native-python')
    p.add_argument('--heartbeat', type=float, default=60)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    if not math.isfinite(args.heartbeat) or args.heartbeat <= 0:
        p.error('Heartbeat must be positive and finite')
    if args.cores is not None:
        try:
            validate_cores(args.cores)
        except ValueError as exc:
            p.error(str(exc))
    out = Path(args.output).absolute()
    if args.report_only:
        write_report(out, read(out/'state.json'), read(out/'plan.json'))
        return
    if args.resume:
        validate_frozen(out)
        if read(out/'plan.json').get('version') not in (
                'quarter-campaign-v1', 'quarter-campaign-v2-parallel', 'quarter-campaign-v3-resumable'):
            p.error('Unsupported campaign version')
        install_recovery(out)
    else:
        for message in preflight(args.native_python):
            say(message)
        prepare(out, args.trial, args.hours, args.native_python, cores=args.cores or min(4, os.cpu_count() or 1))
    if args.dry_run:
        plan = read(out/'plan.json')
        say(f"PREPARED ONLY: {len(plan['tasks'])} planned tasks, {plan['hours']} hours, "
            f"{args.cores or plan['max_concurrent_tasks']} workers, 8 variants; no workers started")
        say(f"Plan: {out/'plan.json'}")
        state = read(out/'state.json')
        interrupted = [a for a in state.get('attempts', []) if a.get('outcome') in ('interrupted', 'interrupted-resource-change')]
        remaining_paused = pause_remaining(state)
        if interrupted:
            say(f"RESUME READY: {sum(a.get('outcome') == 'completed' for a in state['attempts'])} completed tasks retained; "
                f"{len(interrupted)} interrupted tasks recoverable; "
                f"{(remaining_paused or 0)/3600:.2f} campaign hours remaining")
        return
    frozen_supervisor = active_snapshot(out)/'scripts/quarter_campaign.py'
    if Path(__file__).absolute() != frozen_supervisor:
        command = [sys.executable, '-u', str(frozen_supervisor),
                   '--output', str(out), '--resume', '--heartbeat', str(args.heartbeat)]
        if args.cores is not None:
            command += ['--cores', str(args.cores)]
        os.execv(sys.executable, command)
    say(f"Campaign directory: {out}")
    budget = args.cores or read(out/'state.json').get('max_concurrent_tasks', read(out/'plan.json').get('max_concurrent_tasks', 1))
    say(f"Review report: {out/'report.md'}; {read(out/'plan.json')['hours']} hours, strict edges, up to {budget} single-thread workers")
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    raise SystemExit(supervise(out, cores=args.cores, heartbeat=args.heartbeat))


if __name__ == '__main__':
    main()
