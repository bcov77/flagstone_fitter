"""Foreground, tee-friendly supervisor for one bounded eleven-to-thirteen run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from corner_campaign import atomic_json, cleanup_workers

REPO = Path(__file__).absolute().parents[1]
DEFAULT_SEED = REPO/'examples/controller_campaign_long_20261112/isolated_run_010/target_11_base_7_enrich_1/preserved/prefix_11'
SOURCES = ('corner_campaign_worker.py','corner_campaign.py','corner_growth_controller.py',
           'corner_controller.py','corner_growth.py','five_stone_edge_start.py',
           'corner_acceleration.py','corner_native.py','CornerGapProbe.java','watch_corner_run.py','run_corner_13.sh')


def read_optional(path):
    try: return json.loads(Path(path).read_text())
    except (OSError,ValueError): return {}


def say(message):
    print(f'[{datetime.now().astimezone():%H:%M:%S}] {message}',flush=True)


def duration(seconds):
    seconds=max(0,int(seconds));minutes,seconds=divmod(seconds,60)
    return f'{minutes}m {seconds:02d}s'


def prepare(output, minutes=90, seed_directory=DEFAULT_SEED, seed=7113,
            native_python='auto', jts_jar='auto'):
    if not math.isfinite(minutes) or minutes<=0: raise ValueError('Minutes must be positive and finite')
    output=Path(output).absolute(); output.mkdir(parents=True,exist_ok=True)
    if any(p.name!='console.log' for p in output.iterdir()):
        raise ValueError('Choose a new output directory; existing run evidence is never overwritten')
    source=Path(seed_directory).absolute()
    history=json.loads((source/'history.json').read_text())
    verification=json.loads((source/'verification.json').read_text())
    if (len(history)!=11 or not verification['passed'] or len(verification['results'])!=11
            or any(not r['passed'] or r['clearance_errors'] or r['upper_diameter_in']>=2
                   for r in verification['results']) or not (source/'layout.json').is_file()):
        raise ValueError('This runner requires the saved-file-verified eleven-stone seed')
    snapshot=output/'snapshot';scripts=snapshot/'scripts';scripts.mkdir(parents=True)
    for name in SOURCES: shutil.copy2(REPO/'scripts'/name,scripts/name)
    shutil.copytree(REPO/'src/flagstone',snapshot/'src/flagstone',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('pyproject.toml','uv.lock'): shutil.copy2(REPO/name,snapshot/name)
    trial=output/'trial';trial.mkdir();shutil.copy2(REPO/'examples/corner_01/contract.json',trial/'contract.json')
    frozen_seed=output/'seed';frozen_seed.mkdir()
    for name in ('history.json','verification.json'):shutil.copy2(source/name,frozen_seed/name)
    policy=dict(version='corner-growth-v3-native-indexed',seconds=minutes*60,proposals=24000,
        seeds_per_parent=3,groups=2,repair_rounds=24,maxiter=700,generate_seconds=180,
        coverage_seconds=180,select_seconds=240,repair_seconds=900,verify_seconds=180,
        workers=2,seed=seed,prior_run=None,native_python=native_python,native_seconds=45,
        native_groups=1,jts_jar=jts_jar)
    spec=dict(trial=str(trial),seed_directory=str(frozen_seed),output=str(output/'run'),
              policy=policy,result_file=str(output/'worker.result.json'))
    atomic_json(output/'run_spec.json',spec)
    atomic_json(output/'source_manifest.json',{
        str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob('*') if p.is_file() and p.name!='console.log'})
    atomic_json(output/'seed_provenance.json',dict(source=str(source),starting_milestone=11,
        physical_layout_sha256=hashlib.sha256((source/'layout.json').read_bytes()).hexdigest(),
        fresh_verification='required in worker before search'))
    return spec


class Tail:
    def __init__(self): self.positions={};self.pending={};self.finished=set()

    def lines(self,path):
        path=Path(path)
        try:
            with path.open(errors='replace') as stream:
                stream.seek(self.positions.get(path,0));chunk=stream.read();self.positions[path]=stream.tell()
        except OSError:return []
        text=self.pending.get(path,'')+chunk
        lines=text.split('\n');self.pending[path]=lines.pop()
        return [line for line in lines if line.strip()]


def show_event(line,emit):
    try:event=json.loads(line)
    except ValueError:emit(f'controller | {line}');return
    if not isinstance(event,dict) or 'rule' not in event:
        emit(f'controller | {line}');return
    rule=event['rule']
    if rule=='job-start':
        emit(f"START #{event['number']:03d} {event['kind']} — limit {duration(event['limit_seconds'])}")
    elif rule=='job-finish':
        emit(f"END   #{event['number']:03d} {event['kind']} — {event['outcome']} in {duration(event['elapsed_seconds'])}")
    elif rule=='milestone-verified':
        emit(f"VERIFIED {event['target']} stones — {event['directory']}")
    else:
        emit(f"{rule}: "+json.dumps({k:v for k,v in event.items() if k not in ('time','rule')}))


def stream_progress(output,tail,emit):
    for line in tail.lines(output/'worker.log'):show_event(line,emit)
    state=read_optional(output/'run/state.json')
    for job in state.get('jobs',[]):
        if job['log'] in tail.finished:continue
        for line in tail.lines(job['log']):emit(f"{job['kind']} #{job['number']:03d} | {line}")
        if job['outcome']!='running':tail.finished.add(job['log'])
    return state


def heartbeat(state,elapsed,seconds,emit):
    active=next((j for j in reversed(state.get('jobs',[])) if j['outcome']=='running'),None)
    text=f"HEARTBEAT — {duration(elapsed)} elapsed, {duration(seconds-elapsed)} left; freshly verified {state.get('highest_verified',0)} stones"
    if active:
        age=time.time()-datetime.fromisoformat(active['started_utc']).timestamp()
        text+=f"; #{active['number']:03d} {active['kind']} running {duration(age)}"
        spec=read_optional(active['job'])
        if active['kind']=='repair':
            folder=Path(spec.get('output','.'));repair=read_optional(folder/'state.json')
            checked=read_optional(folder/(repair.get('best') or 'missing')/'evaluation.json')
            if 'worst_prefix_upper_in' in checked:
                text+=f"; best checked gap upper {checked['worst_prefix_upper_in']:.6f} in"
                text+=f"; {repair.get('rounds_completed',0)} repair rounds completed"
                text+='; clearances '+('failed' if any(r['clearance_errors'] for r in checked['results']) else 'passed')
    emit(text)


def stop_child(process):
    if process.poll() is None:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass


def supervise(output,spec,heartbeat_seconds=30,worker_script=None,emit=say,poll_seconds=.5):
    """Run once; return exit status. No background retry, adoption, or hidden restart."""
    output=Path(output);tail=Tail();started=time.monotonic();next_heartbeat=started+heartbeat_seconds
    seconds=spec['policy']['seconds'];process=None;reason=None;state={};exit_code=1
    worker=worker_script or output/'snapshot/scripts/corner_campaign_worker.py'
    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             PYTHONPATH=os.pathsep.join([str(output/'snapshot/scripts'),str(output/'snapshot/src')]))
    try:
        with (output/'worker.log').open('ab',buffering=0) as log:
            process=subprocess.Popen([sys.executable,'-u',str(worker),'--spec',str(output/'run_spec.json')],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        atomic_json(output/'supervision.json',dict(status='running',pid=process.pid,
            started_utc=datetime.now(timezone.utc).isoformat(),wall_budget_seconds=seconds))
        while True:
            state=stream_progress(output,tail,emit) or state
            if process.poll() is not None:
                result=read_optional(spec['result_file'])
                exit_code=0 if process.returncode==0 and result.get('outcome')=='completed' else 1
                reason='completed' if exit_code==0 else 'worker-failed'
                break
            now=time.monotonic()
            if now-started>=seconds:
                reason='wall-budget-exhausted';exit_code=124;break
            if now>=next_heartbeat:
                heartbeat(state,now-started,seconds,emit);next_heartbeat=now+heartbeat_seconds
            time.sleep(poll_seconds)
    except KeyboardInterrupt:reason='interrupted-by-user';exit_code=130
    except BrokenPipeError:reason='log-pipe-closed';exit_code=130
    finally:
        # Stop scheduling first; then reap separately owned numerical jobs.
        # A second Ctrl-C must not interrupt cleanup and leave an orphan.
        handlers={}
        try:
            for signum in (signal.SIGINT,signal.SIGTERM,*([signal.SIGHUP] if hasattr(signal,'SIGHUP') else [])):
                handlers[signum]=signal.signal(signum,signal.SIG_IGN)
            if process:stop_child(process)
            cleanup_workers(output/'run')
        finally:
            for signum,handler in handlers.items():signal.signal(signum,handler)
        state=read_optional(output/'run/state.json') or state
        summary=dict(status=reason or 'supervisor-failed',exit_code=exit_code,
            worker_returncode=process.returncode if process else None,
            elapsed_seconds=time.monotonic()-started,starting_milestone=11,
            highest_verified_in_run=state.get('highest_verified',0),run_status=state.get('status'),
            milestones=state.get('milestones',{}),visual_review='pending')
        atomic_json(output/'summary.json',summary)
        atomic_json(output/'supervision.json',summary)
    try:
        stream_progress(output,tail,emit)
        emit(f"FINISHED — {summary['status']}; run status: {summary['run_status']}; freshly verified: {summary['highest_verified_in_run']}")
        emit(f"Summary: {output/'summary.json'}")
    except BrokenPipeError:pass
    return exit_code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,help='New wrapper directory; the actual growth run is output/run')
    parser.add_argument('--minutes',type=float,default=90)
    parser.add_argument('--seed',type=int,default=7113)
    parser.add_argument('--seed-directory',default=str(DEFAULT_SEED))
    parser.add_argument('--heartbeat',type=float,default=30,help='Seconds between readable status lines')
    parser.add_argument('--native-python',default='auto');parser.add_argument('--jts-jar',default='auto')
    parser.add_argument('--no-native',action='store_true');parser.add_argument('--no-jts',action='store_true')
    parser.add_argument('--dry-run',action='store_true',help='Freeze and print the plan without launching a search')
    args=parser.parse_args()
    if not math.isfinite(args.heartbeat) or args.heartbeat<=0:parser.error('Heartbeat must be positive and finite')
    os.chdir(REPO)
    spec=prepare(args.output,args.minutes,args.seed_directory,args.seed,
        None if args.no_native else args.native_python,None if args.no_jts else args.jts_jar)
    say(f"WATCHED RUN — eleven → thirteen, up to {duration(spec['policy']['seconds'])}, seed {args.seed}")
    say(f"Directory: {Path(args.output).absolute()}")
    say('Live numerical logs follow. Only VERIFIED promotions count; Ctrl-C stops owned workers.')
    if args.dry_run:
        say('DRY RUN — no worker or search launched. Frozen specification: '+str(Path(args.output).absolute()/'run_spec.json'))
        return 0
    def terminate(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGINT,terminate)
    signal.signal(signal.SIGTERM,terminate)
    if hasattr(signal,'SIGHUP'):signal.signal(signal.SIGHUP,terminate)
    return supervise(Path(args.output).absolute(),spec,args.heartbeat)


if __name__=='__main__':raise SystemExit(main())
