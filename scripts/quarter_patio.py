"""Prepare and run bounded final-layout quarter-patio pilots in the foreground."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import resource
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from flagstone.phase1 import load_json, save_json, validate, plot
from flagstone.quarter import (GAP, apply_records, check_reference, continuous_patch, measure, native_input,
                               neighborhood, prepare, rank, sample_points, try_replacements,
                               rebuild_patch, verify_saved, write)


def child(command, log, seconds):
    """Bound native/certification work and reap only this job's process group."""
    with Path(log).open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, cwd=ROOT)
        owner_file = Path(log).with_suffix('.owner.json')
        identity = subprocess.run(['ps', '-p', str(process.pid), '-o', 'lstart=', '-o', 'command='],
                                  capture_output=True, text=True).stdout.strip()
        owner = dict(pid=process.pid, identity=identity, command=command, status='running')
        write(owner_file, owner)
        try:
            return dict(outcome='completed' if process.wait(timeout=seconds) == 0 else 'failed',
                        returncode=process.returncode)
        except subprocess.TimeoutExpired:
            return dict(outcome='timed-out', returncode=None)
        finally:
            # Native grandchildren must also be reaped after launcher failure.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            finally:
                # A launcher may exit on TERM before a native grandchild does.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                owner.update(status='reaped', returncode=process.returncode)
                write(owner_file, owner)


def checkpoint(layout, path):
    temp = Path(path).with_suffix('.tmp.json')
    save_json(layout, temp)
    temp.replace(path)


POLICY_FIELDS = ('seed', 'objective', 'target_policy', 'patch_stones', 'patch_max_stones',
                 'rebuild_stones', 'rebuild_alternatives', 'translation_radius_ft', 'maxiter',
                 'proposal_strategy', 'coverage_loss_limit', 'no_continuous', 'no_swaps',
                 'no_insert', 'no_rebuild', 'native_angle_deg')


def save_progress(layout, out, state, args, input_hash, next_round, coverage_floor):
    """One atomic pose/RNG/history commit, independent of the large best.json."""
    write(Path(out)/'checkpoint.json', dict(schema='quarter-checkpoint-v1',
          reference_sha256=input_hash, policy={key: getattr(args, key) for key in POLICY_FIELDS},
          ids=list(layout.ids), poses=layout.poses.tolist(), active=layout.active.tolist(),
          rng_state=layout.snapshot()[2], next_round=next_round, coverage_floor=coverage_floor,
          initial=state['initial'], rounds=state['rounds'], best=state.get('best')))


def restore_progress(layout, directory, args, input_hash):
    """New checkpoints restore RNG; legacy runs retain geometry/round progress."""
    directory = Path(directory)
    previous = json.loads((directory/'state.json').read_text()) if (directory/'state.json').exists() else {}
    path = directory/'checkpoint.json'
    if path.exists():
        data = json.loads(path.read_text())
        if data.get('schema') != 'quarter-checkpoint-v1' or data['reference_sha256'] != input_hash:
            raise ValueError('Resume checkpoint belongs to a different trial')
        if data['policy'] != {key: getattr(args, key) for key in POLICY_FIELDS}:
            raise ValueError('Resume checkpoint search policy changed')
        if data['ids'] != list(layout.ids):
            raise ValueError('Resume checkpoint inventory changed')
        poses, active = np.asarray(data['poses'], float), np.asarray(data['active'])
        if (poses.shape != layout.poses.shape or not np.isfinite(poses).all()
                or active.shape != layout.active.shape or active.dtype != bool):
            raise ValueError('Malformed resume checkpoint')
        layout.poses[:], layout.active[:] = poses, active
        layout.rng.bit_generator.state = data['rng_state']
        progress = dict(initial=data['initial'], rounds=data['rounds'], next_round=data['next_round'],
                        coverage_floor=data['coverage_floor'], fidelity='poses-rounds-and-rng')
    elif (directory/'best.json').exists():
        saved = load_json(directory/'best.json', seed=args.seed)
        check_reference(saved, layout)
        layout.poses[:], layout.active[:] = saved.poses, saved.active
        settings = previous.get('settings', {})
        for key in POLICY_FIELDS:
            if key in settings and settings[key] != getattr(args, key):
                raise ValueError(f'Legacy resume policy changed: {key}')
        initial = previous.get('initial') or measure(layout)
        rounds = previous.get('rounds', [])
        progress = dict(initial=initial, rounds=rounds,
                        next_round=max((r['round']+1 for r in rounds), default=0),
                        coverage_floor=max(0, initial['coverage']-args.coverage_loss_limit)
                        if args.coverage_loss_limit is not None else 0,
                        fidelity='legacy-poses-and-rounds; RNG was not saved, reseeded')
    else:
        return None
    if validate(layout, gap=GAP, edge=GAP):
        raise ValueError('Resume checkpoint violates original-outline clearances')
    progress.update(prior_elapsed_seconds=previous.get('elapsed_seconds', 0),
                    prior_cpu_seconds=previous.get('cpu_seconds_total', 0))
    return progress


def run(args):
    trial = Path(args.trial).resolve()
    contract = json.loads((trial/'contract.json').read_text())
    if contract['version'] != 'quarter-final-layout-v1':
        raise ValueError('Unsupported trial')
    if contract['input_sha256'] != hashlib.sha256((trial/'input.json').read_bytes()).hexdigest():
        raise ValueError('Trial input changed')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cpu_started = time.process_time()
    children_started = resource.getrusage(resource.RUSAGE_CHILDREN)
    deadline = started+args.seconds
    remaining = lambda: max(0, deadline-time.monotonic())
    state = dict(status='running', trial=str(trial), seed=args.seed, settings=vars(args).copy(),
                 jobs=[], rounds=[], acceptance='not-certified', visual_review='pending')
    state['pid'] = os.getpid()
    state['process_identity'] = subprocess.run(
        ['ps', '-p', str(os.getpid()), '-o', 'lstart=', '-o', 'command='],
        capture_output=True, text=True).stdout.strip()
    state['settings'].pop('function', None)
    write(out/'state.json', state)
    (out/'contract.json').write_bytes((trial/'contract.json').read_bytes())
    # Record exact executed source bytes, without claiming dependency isolation.
    snapshot = out/'sources'
    snapshot.mkdir()
    files = [Path(__file__), ROOT/'scripts/quarter_native.py',
             ROOT/'scripts/five_stone_edge_start.py', *sorted((ROOT/'src/flagstone').glob('*.py'))]
    hashes = {}
    for file in files:
        shutil.copyfile(file, snapshot/file.name)
        hashes[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
    write(out/'source_manifest.json', hashes)
    write(out/'environment.json', dict(python=sys.version, executable=sys.executable,
          host_cpus=os.cpu_count(), packages={name: importlib.metadata.version(name)
                                             for name in ('numpy', 'scipy', 'shapely')}))
    layout = load_json(trial/'input.json', seed=args.seed)
    points = sample_points(layout.patio)
    best = None
    resumed = None
    try:
        best_state = None
        checked_patterns = []
        result = dict(outcome='not-needed')
        if args.resume_from:
            resumed = restore_progress(layout, args.resume_from, args, contract['input_sha256'])
            state['resume_from'] = str(Path(args.resume_from).resolve())
        if resumed:
            state.update(mode='checkpoint-resume', initial=resumed['initial'], rounds=resumed['rounds'],
                         resume_fidelity=resumed['fidelity'])
            best, best_state = measure(layout, points), layout.snapshot()
            print(f"Resuming {len(state['rounds'])} completed rounds: {resumed['fidelity']}", flush=True)
        elif args.start:
            supplied = load_json(args.start, seed=args.seed)
            check_reference(supplied, layout)
            if validate(supplied, gap=GAP, edge=GAP):
                raise ValueError('Starting layout has clearance violations')
            checkpoint(supplied, out/'supplied_start.json')
            layout.poses[:], layout.active[:] = supplied.poses, supplied.active
            state['start_sha256'] = hashlib.sha256(Path(args.start).read_bytes()).hexdigest()
            state['mode'] = 'saved-layout-continuation'
            best, best_state = measure(layout, points), layout.snapshot()
        else:
            state['mode'] = 'native-bootstrap'
            write(out/'native_input.json', native_input(layout, args.seed, args.native_angle_deg))
            budget = min(args.bootstrap_seconds, max(1, remaining()-20))
            print(f'Bootstrap: all {len(layout.ids)} stones available; native budget {budget:.1f}s', flush=True)
            saved_patterns = Path(args.resume_from)/'native/patterns.json' if args.resume_from else None
            if saved_patterns and saved_patterns.exists() and json.loads(saved_patterns.read_text()).get('patterns'):
                patterns = saved_patterns
                state['bootstrap_recovered_from'] = str(patterns)
            else:
                result = child([args.native_python, str(ROOT/'scripts/quarter_native.py'),
                                '--input', str(out/'native_input.json'), '--output', str(out/'native'),
                                '--seconds', str(budget)], out/'native.log', min(remaining(), budget+10))
                state['jobs'].append(dict(kind='native', **result))
                patterns = out/'native/patterns.json'
            if patterns.exists():
                for index, pattern in enumerate(json.loads(patterns.read_text())['patterns']):
                    if remaining() < 15:
                        break
                    apply_records(layout, pattern['records'])
                    errors = validate(layout, gap=GAP, edge=GAP)
                    if errors:
                        checked_patterns.append(dict(index=index, clearance_errors=len(errors)))
                        continue
                    metrics = measure(layout, points)
                    checked_patterns.append(dict(index=index, metrics=metrics))
                    if best is None or rank(metrics, args.objective) < rank(best, args.objective):
                        best, best_state = metrics, layout.snapshot()
        write(out/'bootstrap_checks.json', checked_patterns)
        if best_state is None:
            state['status'] = ('bootstrap-worker-failed' if result['outcome'] == 'failed'
                               else 'no-valid-bootstrap/incomplete')
            return state
        layout.restore(best_state)
        state.setdefault('initial', best)
        state['best'] = best
        checkpoint(layout, out/'initial.json')
        first_round = resumed['next_round'] if resumed else 0
        coverage_floor = resumed['coverage_floor'] if resumed else (
            max(0, best['coverage']-args.coverage_loss_limit) if args.coverage_loss_limit is not None else 0)
        save_progress(layout, out, state, args, contract['input_sha256'], first_round, coverage_floor)
        checkpoint(layout, out/'best.json')
        print(json.dumps(dict(stage='bootstrap-checked', **best)), flush=True)
        for round_index in range(first_round, args.rounds):
            if remaining() < 20:
                break
            # Alternate among top separated bad centers to avoid repeatedly
            # polishing one unproductive pocket; neighborhoods may overlap.
            from flagstone.quarter import distances, world
            ds = distances(points, list(world(layout).values()), layout.patio.boundary)
            choices = []
            for index in ds.argsort()[::-1]:
                center = list(points[index].coords[0])
                if all(math.dist(center, other) > .6 for other in choices):
                    choices.append(center)
                if len(choices) == 4:
                    break
            center = choices[round_index % len(choices)]
            if args.target_policy == 'worst':
                center = choices[0]
            elif args.target_policy == 'random':
                bad = np.flatnonzero(ds >= 1/12)
                if len(bad):
                    center = list(points[int(layout.rng.choice(bad))].coords[0])
            patch = neighborhood(layout, center, count=min(args.patch_stones+2*(round_index//2), args.patch_max_stones))
            print(f'Repair {round_index}: {len(patch)} nearby stones, center {center}', flush=True)
            def act(enabled, function, **kwargs):
                if not enabled or remaining() < 16:
                    return dict(accepted=False, skipped=True)
                original = layout.snapshot()
                t0, cpu0 = time.monotonic(), time.process_time()
                result = function(layout, **kwargs)
                if result['accepted'] and result['after']['coverage'] < coverage_floor:
                    rng_state = layout.snapshot()[2]
                    layout.restore(original)
                    layout.rng.bit_generator.state = rng_state
                    result.update(accepted=False, coverage_guard_rejected=True, after=result['before'])
                result.update(elapsed_seconds=time.monotonic()-t0, cpu_seconds=time.process_time()-cpu0)
                return result
            repair = act(not args.no_continuous, continuous_patch, movable=patch, points=points,
                         seconds=min(args.patch_seconds, max(.01, remaining()-15)), maxiter=args.maxiter,
                         radius_ft=args.translation_radius_ft, objective=args.objective)
            swap = act(not args.no_swaps, try_replacements, center=center, points=points,
                       seconds=min(6, max(.01, remaining()-15)), limit=8, objective=args.objective)
            insertion = act(not args.no_insert, rebuild_patch, center=center, points=points, count=0,
                            seconds=min(4, max(.01, remaining()-15)), objective=args.objective,
                            strategy=args.proposal_strategy)
            rebuilds = []
            for alternative in range(args.rebuild_alternatives):
                rebuilds.append(act(not args.no_rebuild, rebuild_patch, center=center, points=points,
                                    count=max(1, args.rebuild_stones+(round_index+alternative)%3-1),
                                    seconds=min(6, max(.01, remaining()-15)), objective=args.objective,
                                    strategy=args.proposal_strategy))
            rebuilt = dict(accepted=any(r['accepted'] for r in rebuilds), alternatives=rebuilds)
            best = measure(layout, points)
            if validate(layout, gap=GAP, edge=GAP):
                raise RuntimeError('Repair promoted a clearance violation')
            event = dict(round=round_index, center=center, continuous=repair, replacement=swap,
                         insertion=insertion, rebuilt=rebuilt, metrics=best)
            state['best'] = best
            state['rounds'].append(event)
            save_progress(layout, out, state, args, contract['input_sha256'], round_index+1, coverage_floor)
            checkpoint(layout, out/'best.json')
            write(out/f'round_{round_index:03}.json', event)
            write(out/'state.json', state)
            print(json.dumps(dict(stage='repair-checked', round=round_index, **best)), flush=True)
        # Certification is independently reloaded, isolated and hard bounded.
        if remaining() > 2:
            print('Checking saved original-outline layout and global gap bounds', flush=True)
            result = child([sys.executable, str(Path(__file__).resolve()), 'verify',
                            '--input', str(out/'best.json'), '--reference', str(trial/'input.json'),
                            '--output', str(out/'verification.json')],
                           out/'verification.log', min(45, remaining()))
            state['jobs'].append(dict(kind='verify', **result))
        verification = out/'verification.json'
        if verification.exists():
            v = json.loads(verification.read_text())
            state['verification'] = v
            state['acceptance'] = 'saved-file-verified' if v['passed'] else 'incomplete'
        state['best'] = best
        state['status'] = ('target-geometrically-verified-review-pending'
                           if state['acceptance'] == 'saved-file-verified' else 'pilot-completed/incomplete')
        return state
    except KeyboardInterrupt:
        state['status'] = 'interrupted/incomplete'
        raise
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        state['segment_elapsed_seconds'] = time.monotonic()-started
        state['elapsed_seconds'] = state['segment_elapsed_seconds']+(resumed or {}).get('prior_elapsed_seconds', 0)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        state['cpu_seconds_total'] = (time.process_time()-cpu_started + children.ru_utime+children.ru_stime
                                      - children_started.ru_utime-children_started.ru_stime
                                      +(resumed or {}).get('prior_cpu_seconds', 0))
        write(out/'state.json', state)
        write(out/'summary.json', state)
        print(json.dumps(dict(status=state['status'], elapsed_seconds=state['elapsed_seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--source', default=str(ROOT/'examples/real_stones_concept/layout.json'))
    p.add_argument('--output', required=True)
    p = sub.add_parser('run')
    p.add_argument('--trial', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seconds', type=float, default=180)
    p.add_argument('--bootstrap-seconds', type=float, default=50)
    p.add_argument('--patch-seconds', type=float, default=12)
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--start', help='Continue a saved final layout in a NEW run directory')
    p.add_argument('--resume-from', help='Restore a prior task checkpoint into this NEW segment directory')
    p.add_argument('--objective', choices=('energy', 'worst', 'balanced'), default='energy')
    p.add_argument('--native-angle-deg', type=float, default=0,
                   help='Rotate native bin frame only; exported poses return to original frame')
    p.add_argument('--target-policy', choices=('rotating', 'worst', 'random'), default='rotating')
    p.add_argument('--patch-stones', type=int, default=6)
    p.add_argument('--patch-max-stones', type=int, default=12)
    p.add_argument('--rebuild-stones', type=int, default=3)
    p.add_argument('--rebuild-alternatives', type=int, default=1)
    p.add_argument('--translation-radius-ft', type=float, default=.25)
    p.add_argument('--maxiter', type=int, default=180)
    p.add_argument('--proposal-strategy', choices=('random', 'contact', 'edge', 'mixed'), default='random')
    p.add_argument('--coverage-loss-limit', type=float, help='Maximum coverage loss from this run start, as a fraction')
    for name in ('continuous', 'swaps', 'insert', 'rebuild'):
        p.add_argument('--no-'+name, action='store_true')
    p.add_argument('--native-python', default=str(Path.home()/'miniconda3/envs/patio-packingsolver/bin/python'))
    p = sub.add_parser('verify')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--reference', required=True)
    p = sub.add_parser('render')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.source, args.output), indent=2))
    elif args.command == 'verify':
        print(json.dumps(verify_saved(args.input, args.output, args.reference), indent=2))
    elif args.command == 'render':
        plot(load_json(args.input), args.output)
    else:
        if (not all(math.isfinite(v) and v > 0 for v in (args.seconds, args.bootstrap_seconds, args.patch_seconds, args.translation_radius_ft))
                or args.rounds < 0 or min(args.patch_stones, args.patch_max_stones, args.rebuild_stones, args.rebuild_alternatives, args.maxiter) < 1
                or args.patch_max_stones < args.patch_stones
                or not math.isfinite(args.native_angle_deg)
                or args.coverage_loss_limit is not None and not 0 <= args.coverage_loss_limit <= 1):
            parser.error('Budgets must be positive finite numbers and rounds nonnegative')
        if args.start and args.resume_from:
            parser.error('Use --start or --resume-from, not both')
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGHUP, interrupted)
        result = run(args)
        if result['status'] in ('failed', 'bootstrap-worker-failed'):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
