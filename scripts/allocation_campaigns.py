#!/usr/bin/env python3
"""Prospective, equal-worker-budget allocation comparison; safe-stop and resume."""
import argparse
import collections
import concurrent.futures as futures
import copy
import fcntl
import json
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

import full_trajectories as f

ROOT = f.ROOT
SCRIPT = Path(__file__).resolve()
DEFAULT_OUT = ROOT/'examples/allocation_campaigns_20261120_v1'
POLICIES = ('broad', 'multiple', 'adaptive')


def design(seed):
    # No historical starts. These capacity fields do not schedule unused jobs.
    return dict(lineages=256, replicates=3, seed=seed, classical_minutes=[0.],
                physics_minutes=[2., 8.], extension_minutes=90., extension_fraction=0.,
                native_seconds=30., native_python=str(f.DEFAULT_NATIVE), patch_seconds=10.,
                search_timeout=300., physics_timeout=300., classical_timeout=120.,
                audit_timeout=180., minimum_free_gib=20.)


def initial_state(policy):
    return dict(policy=policy, next_parent=0, next_action=0, queue=[], pending=None,
                awaiting_promotion=False, candidates=[], decisions=[], complete=False,
                controller_seconds=0.)


def task(kind, parent, replicate=0):
    key = f'p{parent:03d}' if kind == 'classical' else f'p{parent:03d}_h00_r{replicate:02d}'
    return dict(kind=kind, key=key, parent=parent, horizon=0, replicate=replicate)


def promote(candidates, seed, decision_index):
    """Predeclared 80% best coverage / 20% random other candidate rule."""
    ordered = sorted(candidates, key=lambda x: (-x['coverage'], x['key']))
    f.check(len(ordered) > 1, 'Promotion requires a completed probe batch')
    rng = random.Random(f.seed_for(seed, 'promotion', decision_index))
    explore = rng.random() < .2
    chosen = rng.choice(ordered[1:]) if explore else ordered[0]
    return dict(index=decision_index, exploration=explore, chosen=chosen,
                candidates=ordered, rule='80% highest absolute coverage; 20% uniform among other five')


def fill_queue(s, seed):
    """Pure scheduling decisions; persist this state before executing anything."""
    if s['queue']:
        return
    if s['policy'] == 'adaptive' and s['awaiting_promotion']:
        d = promote(s['candidates'], seed, len(s['decisions']))
        s['decisions'].append(d)
        chosen = copy.deepcopy(d['chosen']['task'])
        chosen.update(target_minutes=90., stage='extension')
        s['queue'] = [chosen]
        s['candidates'] = []
        s['awaiting_promotion'] = False
        return
    count = 2 if s['policy'] == 'adaptive' else 1
    for _ in range(count):
        p = s['next_parent']
        f.check(p < 256, 'Parent capacity exceeded')
        s['next_parent'] += 1
        s['queue'].append(task('classical', p))
        replicates = 1 if s['policy'] == 'broad' else 3
        for r in range(replicates):
            t = task('physics', p, r)
            t.update(target_minutes=8. if s['policy']=='adaptive' else 30., stage='probe')
            s['queue'].append(t)
    s['awaiting_promotion'] = s['policy'] == 'adaptive'


def horizons(target):
    return [x for x in (2., 8.) if x < target] + [target]


def costs(out):
    """Authoritative receipts plus per-session overhead, including stopped sessions."""
    total, cpu, unknown = 0., 0., 0
    for path in out.glob('*/p*/operations/op_*/receipt.json'):
        rec = f.read(path)
        total += rec['worker_seconds']
        cpu += rec['cpu_seconds'] or 0.
        unknown += rec['cpu_seconds'] is None
    for path in out.glob('*/p*/sessions/*.json'):
        total += f.read(path)['orchestration_worker_seconds']
    return dict(worker_seconds=total, operation_cpu_seconds=cpu, unknown_cpu_operations=unknown)


def next_action(out, s, seed, budget):
    if s['pending']:
        return s['pending']
    used = costs(out)['worker_seconds'] + s['controller_seconds']
    if used >= budget:
        s['complete'] = True
        return None
    fill_queue(s, seed)
    action = copy.deepcopy(s['queue'][0])
    action['action_id'] = s['next_action']
    action['seed'] = (f.seed_for(seed, 'parent', action['parent']) if action['kind']=='classical'
                      else f.seed_for(seed, 'continuation', action['parent'], action['replicate']))
    if action['kind'] == 'physics':
        # A continuation receives only the remaining campaign allocation.
        previous = out/'physics'/action['key']/'progress.json'
        already = f.read(previous)['worker_seconds'] if previous.exists() else 0.
        target = min(action['target_minutes'], (already + budget-used)/60)
        action['horizons'] = horizons(target)
    action['budget_before_seconds'] = used
    s['pending'] = action
    return action


def run_action(out_string, action):
    out = Path(out_string)
    marker = out/'actions'/f'{action["action_id"]:05d}.json'
    if marker.exists():
        return f.read(marker)
    m = f.read(out/'manifest.json')
    folder = out/('parents' if action['kind']=='classical' else 'physics')/action['key']
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        f._IDLE_STARTED, f._OVERHEAD_SECONDS, f._SOURCE_PATH = time.monotonic(), 0., out/'source.json'
        f.write(folder/'task.json', action)
        try:
            if action['kind'] == 'classical':
                result = f.classical_job(out, folder, action, m)
            else:
                m = copy.deepcopy(m)
                m['design']['physics_minutes'] = action['horizons']
                if (folder/'summary.json').exists():
                    # Preserve the opening summary before continuing its exact checkpoint.
                    archived = folder/'stages'/f'before_action_{action["action_id"]:05d}.json'
                    if not archived.exists():
                        f.write(archived, f.read(folder/'summary.json'))
                    (folder/'summary.json').unlink()
                result = f.physics_job(out, folder, action, m)
        finally:
            if f._IDLE_STARTED is not None:
                f._OVERHEAD_SECONDS += time.monotonic()-f._IDLE_STARTED
            f.write(folder/'sessions'/f'{time.time_ns()}_{os.getpid()}.json',
                    dict(orchestration_worker_seconds=f._OVERHEAD_SECONDS))
            f._IDLE_STARTED = None
        if result['status'] == 'complete':
            f.write(marker, dict(action=action, result=result))
        return dict(action=action, result=result)


def commit_action(s, completed):
    action, result = completed['action'], completed['result']
    if result['status'] != 'complete':
        return
    f.check(s['pending'] == action, 'Completion does not match committed dispatch')
    if s['policy']=='adaptive' and action['kind']=='physics' and action['stage']=='probe':
        s['candidates'].append(dict(key=action['key'], coverage=result['best_metrics']['area_percent'],
                                    task=s['queue'][0]))
    s['queue'].pop(0)
    s['pending'] = None
    s['next_action'] += 1


def campaign_report(out, s, budget):
    c = costs(out)
    c['worker_seconds'] += s['controller_seconds']
    best = None
    for path in list(out.glob('physics/p*/progress.json')) + list(out.glob('parents/p*/handoff_00.json')):
        v = f.read(path)
        metrics = v.get('best_metrics', v.get('metrics'))
        candidate = dict(key=path.parent.name, metrics=metrics)
        if best is None or metrics['area_percent'] > best['metrics']['area_percent']:
            best = candidate
    r = dict(policy=s['policy'], complete=s['complete'], costs=c, budget_seconds=budget,
             overshoot_seconds=max(0., c['worker_seconds']-budget), best=best,
             parents=len(list(out.glob('parents/p*/handoff_00.json'))),
             continuations=len(list(out.glob('physics/p*/initial.json'))),
             actions_complete=s['next_action'], decisions=s['decisions'],
             pending=s['pending'])
    f.write(out/'allocation_report.json', r)
    return r


def prepare(out, hours, repeats, seed, runtime_hours=20., policies=POLICIES):
    f.check(bool(policies) and len(set(policies)) == len(policies)
            and all(p in POLICIES for p in policies), 'Invalid policy selection')
    settings = dict(schema='allocation-campaigns-v1', hours_per_campaign=hours,
                    repeats=repeats, seed=seed, policies=list(policies), runtime_hours=runtime_hours,
                    driver_sha256=f.digest(SCRIPT),
                    promotion='Six 8-minute probes from two fresh parents; extend one to 90 minutes. '
                              '80% highest absolute coverage; 20% uniform among the other five.',
                    accounting='All parent and child receipts and worker/controller overhead charged once; '
                               'last horizon shortened to remaining budget; bounded transaction overshoot retained.',
                    verification='Full post-run integrity verification is additional, separately timed overhead.')
    path = out/'allocation_manifest.json'
    if path.exists():
        f.check(f.read(path) == settings, 'Settings/driver changed; use a new output directory')
    else:
        f.write(path, settings)
        shutil.copyfile(SCRIPT, out/'allocation_campaigns.py')
    campaigns = []
    for repeat in range(repeats):
        for policy in policies:
            started = time.monotonic()
            name = f'{policy}_{repeat:02d}'
            folder = out/name
            m = f.prepare(folder, design(f.seed_for(seed, name)), f.DEFAULT_SOURCE)
            state_path = folder/'allocation_state.json'
            if not state_path.exists():
                f.write(state_path, initial_state(policy))
            s = f.read(state_path)
            s['controller_seconds'] += time.monotonic()-started
            f.write(state_path, s)
            campaigns.append((folder, m['design']['seed']))
    random.Random(seed).shuffle(campaigns)
    return campaigns


def report(out, campaigns, budget):
    started = time.monotonic()
    rows = []
    for folder, _ in campaigns:
        s = f.read(folder/'allocation_state.json')
        r = campaign_report(folder, s, budget)
        rows.append(dict(campaign=folder.name, **r))
    result = dict(campaigns=rows, complete=sum(r['complete'] for r in rows), total=len(rows),
                  worker_hours=sum(r['costs']['worker_seconds'] for r in rows)/3600,
                  verification_complete=(out/'COMPLETE.json').exists())
    f.write(out/'report.json', result)
    lines = [f'Campaigns finished: {result["complete"]}/{len(rows)}',
             f'Worker hours: {result["worker_hours"]:.3f}',
             'Campaign              hours     best coverage    parents / continuations']
    for r in rows:
        best = '-' if r['best'] is None else f'{r["best"]["metrics"]["area_percent"]:.4f}%'
        lines.append(f'{r["campaign"]:20s} {r["costs"]["worker_seconds"]/3600:7.3f} {best:>14s} '
                     f'{r["parents"]:8d} / {r["continuations"]}')
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    overhead = (time.monotonic()-started)/len(campaigns)
    for folder, _ in campaigns:
        s = f.read(folder/'allocation_state.json')
        s['controller_seconds'] += overhead
        f.write(folder/'allocation_state.json', s)
    return result


def execute(out, campaigns, budget, workers, runtime_hours):
    started_unix = time.time()
    deadline = time.monotonic()+runtime_hours*3600
    f.write(out/'launch.json', dict(pid=os.getpid(), started_unix=started_unix,
                                  deadline_unix=started_unix+runtime_hours*3600, workers=workers))
    def stop(*_):
        (out/'STOP').touch()
        for folder, _ in campaigns:
            (folder/'STOP').touch()
        print('Safe stop requested; waiting for bounded numerical operations.', flush=True)
    for folder, _ in campaigns:
        (folder/'STOP').unlink(missing_ok=True)
    (out/'STOP').unlink(missing_ok=True)
    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    ready = collections.deque(campaigns)
    active = {}
    last_report = 0.
    try:
        with futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                        initializer=f.init_worker) as pool:
            while ready or active:
                # Reserve two operation caps plus shutdown time for a safe checkpoint.
                if time.monotonic() >= deadline-660 and not (out/'STOP').exists():
                    print('Wall-clock limit approaching.', flush=True)
                    stop()
                if (out/'STOP').exists() or shutil.disk_usage(out).free < 20*1024**3:
                    stop() if not (out/'STOP').exists() else None
                    ready.clear()
                while ready and len(active)<workers and not (out/'STOP').exists():
                    folder, seed = ready.popleft()
                    started = time.monotonic()
                    s = f.read(folder/'allocation_state.json')
                    action = next_action(folder, s, seed, budget)
                    s['controller_seconds'] += time.monotonic()-started
                    f.write(folder/'allocation_state.json', s)
                    if action:
                        active[pool.submit(run_action, str(folder), action)] = (folder, seed)
                if not active:
                    break
                done, _ = futures.wait(active, timeout=2, return_when=futures.FIRST_COMPLETED)
                for future in done:
                    folder, seed = active.pop(future)
                    started = time.monotonic()
                    s = f.read(folder/'allocation_state.json')
                    try:
                        completion = future.result()
                        commit_action(s, completion)
                        s['controller_seconds'] += time.monotonic()-started
                        f.write(folder/'allocation_state.json', s)
                        print(f'{folder.name}: {completion["action"]["key"]} '
                              f'{completion["result"]["status"]}', flush=True)
                        if not (out/'STOP').exists():
                            ready.append((folder, seed))
                    except Exception:
                        f.write(out/'errors'/f'{folder.name}_{time.time_ns()}.json',
                                dict(traceback=traceback.format_exc()))
                        stop()
                        print(traceback.format_exc(), flush=True)
                if time.monotonic()-last_report > 30:
                    summary = report(out, campaigns, budget)
                    print(f'Progress: {summary["complete"]}/{summary["total"]} campaigns; '
                          f'{summary["worker_hours"]:.2f} worker-hours', flush=True)
                    last_report = time.monotonic()
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
    summary = report(out, campaigns, budget)
    if not (out/'STOP').exists() and summary['complete']==len(campaigns):
        f.write(out/'EXECUTION_COMPLETE.json', dict(finished_unix=time.time(), worker_hours=summary['worker_hours']))
        verify_all(out, campaigns, deadline)
    else:
        print('Checkpointed. Rerun the same command to resume.', flush=True)


def verify_all(out, campaigns, deadline=None):
    def stop(*_):
        (out/'STOP').touch()
    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        _verify_all(out, campaigns, deadline)
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
    manifest = out/'allocation_manifest.json'
    if manifest.exists():
        report(out, campaigns, f.read(manifest)['hours_per_campaign']*3600)


def _verify_all(out, campaigns, deadline=None):
    f.check(all(f.read(folder/'allocation_state.json')['complete'] for folder, _ in campaigns),
            'Full verification requires all campaign budgets to finish')
    results = {}
    for folder, _ in campaigns:
        stamp = folder/'allocation_verification.json'
        if not stamp.exists():
            print(f'Verifying {folder.name} (full geometry and receipts)...', flush=True)
            remaining = None if deadline is None else deadline-time.monotonic()-15
            if remaining is not None and remaining <= 0:
                print('Wall-clock limit reached; simulation finished, final verification pending.', flush=True)
                return
            with (folder/'verification.log').open('ab') as log:
                process = subprocess.Popen([sys.executable, str(SCRIPT), '--_verify-one', str(folder)],
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    if (out/'STOP').exists() or (deadline is not None and time.monotonic()>=deadline-15):
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10)
                        print('Final verification stopped; completed campaign audits are retained.', flush=True)
                        return
                    try:
                        process.wait(timeout=.5)
                    except subprocess.TimeoutExpired:
                        pass
                code = process.returncode
            f.check(code==0, f'Verification failed: {folder}/verification.log')
        results[folder.name] = f.read(stamp)
    f.write(out/'COMPLETE.json', dict(finished_unix=time.time(), verification=results))
    print(f'Complete and verified: {out}', flush=True)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--_verify-one':
        folder = Path(sys.argv[2])
        started = time.monotonic()
        result = f.verify(folder)
        f.write(folder/'allocation_verification.json', dict(result=result,
                verifier_wall_seconds=time.monotonic()-started, finished_unix=time.time()))
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--hours', type=float, default=14.)
    p.add_argument('--repeats', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261120)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--policies', nargs='+', choices=POLICIES, default=list(POLICIES),
                   help='Policies to run; repeats applies to each selected policy')
    p.add_argument('--runtime-hours', type=float, default=20.)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--status', action='store_true')
    p.add_argument('--stop', action='store_true')
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    out = a.out.resolve()
    f.check(a.hours>0 and a.hours<float('inf') and a.repeats>0 and a.workers>0
            and 0<a.runtime_hours<float('inf'), 'Invalid allocation')
    f.check(out!=ROOT and out not in ROOT.parents, 'Use a dedicated output directory')
    if a.stop:
        f.check((out/'allocation_manifest.json').exists(), 'No prepared experiment')
        (out/'STOP').touch()
        for folder in out.glob('*/allocation_state.json'):
            (folder.parent/'STOP').touch()
        print('Safe stop requested.')
        return
    if a.status:
        print((out/'report.txt').read_text())
        return
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaigns = prepare(out, a.hours, a.repeats, a.seed, a.runtime_hours, a.policies)
        report(out, campaigns, a.hours*3600)
        if a.verify:
            (out/'STOP').unlink(missing_ok=True)
            verify_all(out, campaigns, time.monotonic()+a.runtime_hours*3600)
        elif not a.prepare_only:
            execute(out, campaigns, a.hours*3600, a.workers, a.runtime_hours)


if __name__ == '__main__':
    main()
