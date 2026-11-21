"""Bounded warm-start controller pilot; never resumes the paused experiment.

This first slice evaluates one explicit complete pattern, repairs it, and
expands its movable suffix using prefix witnesses. It does not yet generate
pools, schedule milestone growth, own subprocesses, or support pause/resume.
Run from the repository root. Budgets are checked between synchronous jobs;
maxiter bounds each optimizer, not the duration of a native call or export.
"""
import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time
import shutil
import subprocess

import corner_growth as engine


@dataclass(frozen=True)
class Policy:
    version: str = 'corner-repair-pilot-v3-auxiliary-oracle'
    max_rounds: int = 6
    max_seconds: float = 300
    initial_moving: int = 2
    stall_rounds: int = 3
    improvement_in: float = .01
    translation_radius_ft: float = .15
    angle_radius_radians: float = .15
    maxiter: int = 700
    seed: int = 3100
    jts_jar: str | None = None

    def validate(self):
        for name in ('max_rounds', 'initial_moving', 'stall_rounds', 'maxiter'):
            v = getattr(self, name)
            if type(v) is not int or v < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('max_seconds', 'improvement_in', 'translation_radius_ft', 'angle_radius_radians'):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f'{name} must be finite and positive')


def atomic_json(path, value):
    """Replace the index only after flushing a complete JSON document."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def rank(evaluation):
    """Keep invalid geometry separate; never rank by the optimizer's score."""
    errors = [e for r in evaluation['results'] for e in r['clearance_errors']]
    return (bool(errors), evaluation['maximum_spacing_shortfall_in'],
            evaluation['total_spacing_shortfall_in'],
            evaluation['worst_prefix_upper_in'])


def next_moving(evaluation, current, *, stalled=False):
    """Reopen the earliest implicated placement, then broaden on a stall."""
    history = evaluation['history']
    implicated = {sid for r in evaluation['results'] for sid in r['implicated_ids']}
    earliest = min((i for i, r in enumerate(history) if r['stone_id'] in implicated),
                   default=len(history)-current)
    needed = max(current, len(history)-earliest)
    return min(len(history), max(needed, current*2 if stalled else current))


def run(trial, history_file, output, policy=Policy(), *, witnesses_file=None):
    """Write a separate pilot run; acceptance requires saved-file verification."""
    policy.validate()
    from corner_acceleration import optional_tool, jts_crosscheck
    policy=replace(policy,jts_jar=optional_tool(policy.jts_jar,'jts_jar'))
    started = time.monotonic()
    trial = Path(trial).resolve()
    _, contract, layout = engine.load_trial(trial)
    history = engine.checked_history(layout, contract, json.loads(Path(history_file).read_text()))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = output/'trial'
    frozen.mkdir()
    (frozen/'contract.json').write_bytes((trial/'contract.json').read_bytes())
    if policy.jts_jar:
        jar=frozen/'jts-core.jar';shutil.copy2(policy.jts_jar,jar)
        policy=replace(policy,jts_jar=str(jar))
    engine.write(output/'initial_history.json', history)
    engine.write(output/'policy.json', asdict(policy))
    # Record source files and dependency versions; isolated workers come later.
    repo = Path(__file__).resolve().parents[1]
    files = [Path(contract['input']).resolve(), repo/'pyproject.toml', repo/'uv.lock']
    files += sorted((repo/'src/flagstone').glob('*.py'))
    files += [repo/'scripts/corner_controller.py', repo/'scripts/corner_growth.py',
              repo/'scripts/five_stone_edge_start.py',repo/'scripts/corner_acceleration.py',
              repo/'scripts/CornerGapProbe.java']
    if policy.jts_jar:files.append(Path(policy.jts_jar))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    snapshots = output/'sources'
    snapshots.mkdir()
    for i, p in enumerate(files):
        (snapshots/f'{i:03d}_{p.name}').write_bytes(p.read_bytes())
    contract_hash = hashlib.sha256((frozen/'contract.json').read_bytes()).hexdigest()
    engine.write(output/'environment.json', dict(python=platform.python_version(),
        dependencies={p: importlib.metadata.version(p) for p in ('numpy','scipy','shapely','matplotlib','ortools')},
        source_hashes=hashes, contract_sha256=contract_hash, mode='warm-start-repair-pilot',
        snapshot_files={str(p): f'sources/{i:03d}_{p.name}' for i,p in enumerate(files)}))
    state = dict(status='running', best=None, latest=None, rounds_completed=0,
                 visual_review='pending', render_consistency='not-audited',
                 milestones_advanced=[], policy_version=policy.version)

    def event(rule, **data):
        with (output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), rule=rule, **data), allow_nan=False)+'\n')
            f.flush()
            os.fsync(f.fileno())

    def integrity():
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha for p, sha in hashes.items()):
            raise ValueError('Source integrity changed during the pilot')
        if hashlib.sha256((frozen/'contract.json').read_bytes()).hexdigest() != contract_hash:
            raise ValueError('Frozen contract integrity changed')

    def preserve_evaluation(records, destination):
        integrity()
        evaluation = engine.evaluate_history(frozen, records)
        destination.mkdir(exist_ok=False)
        if policy.jts_jar and time.monotonic()-started < policy.max_seconds:
            try:
                check=jts_crosscheck(layout,contract,evaluation,policy.jts_jar,destination,
                    timeout=max(.01,min(10,policy.max_seconds-(time.monotonic()-started))))
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                check=dict(status='unavailable',error=str(exc),acceptance_authority=False)
            # A real contradiction with a certified bound raises RuntimeError;
            # never hide it as optional-tool unavailability.
            engine.write(destination/'jts_crosscheck.json',check)
            evaluation['auxiliary_witnesses']=check.get('witnesses',[])
        engine.write(destination/'history.json', evaluation['history'])
        engine.write(destination/'evaluation.json', evaluation)
        return evaluation

    def promote(evaluation):
        integrity()
        pending = output/'pending_export'
        # Reconstruct and independently reload physical placements. Any export
        # failure remains a failed attempt and cannot create an accepted index.
        engine.save_verified(frozen, contract, evaluation['history'], pending)
        integrity()
        engine.write(pending/'review_status.json', dict(visual_review='pending', render_consistency='not-audited'))
        manifest = {str(p.relative_to(pending)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(pending.rglob('*')) if p.is_file()}
        engine.write(pending/'manifest.json', manifest)
        pending.rename(output/'accepted')
        state.update(status='saved-file-verified', accepted='accepted')
        event('all-prefix-saved-file-verification-passed', target=len(evaluation['history']))

    try:
        best = preserve_evaluation(history, output/'initial')
        state.update(best='initial', latest='initial')
        atomic_json(output/'state.json', state)
        moving = min(policy.initial_moving, len(history))
        stalls = 0
        # Counterexamples remain useful even when their round regresses. Keep
        # them separately from the pose checkpoint chosen for the next restart.
        witnesses = {}
        def retain_witnesses(items):
            for item in items:
                key = (item['step'], *(round(float(v), 5) for v in item['center']))
                witnesses[key] = dict(step=item['step'], center=item['center'])
            atomic_json(output/'witnesses.json', list(witnesses.values()))
        retain_witnesses(r for r in best['results'] if r['gap_status'] != 'passed')
        retain_witnesses(best.get('auxiliary_witnesses',[]))
        if witnesses_file:
            imported = json.loads(Path(witnesses_file).read_text())
            retain_witnesses(item for item in imported if 1 <= item['step'] <= len(history))
        if time.monotonic()-started >= policy.max_seconds:
            state['status'] = 'budget-exhausted/incomplete'
        elif best['passed']:
            promote(best)
        else:
            for index in range(policy.max_rounds):
                if time.monotonic()-started >= policy.max_seconds:
                    state['status'] = 'budget-exhausted/incomplete'
                    break
                expanded = next_moving(best, moving, stalled=stalls >= policy.stall_rounds)
                event('stall-expand' if stalls >= policy.stall_rounds else 'witness-suffix',
                      previous_moving=moving, moving=expanded, restart=state['best'], round=index)
                if expanded != moving:
                    stalls = 0
                moving = expanded
                integrity()
                job = engine.repair_history(frozen, best['history'], output/f'round_{index:03d}',
                    count=moving, radius=policy.translation_radius_ft,
                    angle_radius=policy.angle_radius_radians, maxiter=policy.maxiter,
                    seed=policy.seed+index,
                    centers=list(witnesses.values()))
                checked = preserve_evaluation(job['history'], output/f'checked_{index:03d}')
                retain_witnesses(job.get('centers', []))
                retain_witnesses(r for r in checked['results'] if r['gap_status'] != 'passed')
                retain_witnesses(checked.get('auxiliary_witnesses',[]))
                state.update(latest=f'checked_{index:03d}', rounds_completed=index+1)
                meaningful = (rank(checked)[:3] < rank(best)[:3] or
                              rank(checked)[:3] == rank(best)[:3] and
                              best['worst_prefix_upper_in']-checked['worst_prefix_upper_in'] >= policy.improvement_in)
                stalls = 0 if meaningful else stalls+1
                if rank(checked) < rank(best):
                    best = checked
                    state['best'] = state['latest']
                    event('best-checked-checkpoint', checkpoint=state['best'])
                if checked['passed']:
                    if time.monotonic()-started >= policy.max_seconds:
                        state['status'] = 'budget-exhausted/incomplete'
                    else:
                        promote(checked)
                    break
                atomic_json(output/'state.json', state)
            else:
                state['status'] = 'round-budget-exhausted/incomplete'
        state['elapsed_seconds'] = time.monotonic()-started
        atomic_json(output/'state.json', state)
        event('run-finished', status=state['status'])
        return state
    except Exception as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                     elapsed_seconds=time.monotonic()-started)
        atomic_json(output/'state.json', state)
        event('integrity-or-job-failure', error=str(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', default='examples/corner_01')
    parser.add_argument('--history', required=True, help='Explicit warm-start history.json')
    parser.add_argument('--output', required=True, help='New pilot directory')
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--maxiter', type=int, default=700)
    parser.add_argument('--jts-jar',default='auto',help='Optional auxiliary JTS jar; never acceptance authority')
    parser.add_argument('--no-jts',action='store_true')
    args = parser.parse_args()
    result = run(args.trial, args.history, args.output,
                 Policy(max_rounds=args.rounds, max_seconds=args.seconds, maxiter=args.maxiter,
                        jts_jar=None if args.no_jts else args.jts_jar))
    print(json.dumps(result, indent=2))
