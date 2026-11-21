"""Durable standard-library supervisor for independent controller subprocesses.

Use --resume after a supervisor restart. A lock prevents duplicate supervisors;
a live matching child is adopted, not duplicated. Child failures are logged and
retried with new run directories. Numerical libraries load only in workers.
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
import signal
import shutil
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def utc():return datetime.now(timezone.utc).isoformat()


def identity(pid):
    result=subprocess.run(['ps','-p',str(pid),'-o','lstart=','-o','command='],capture_output=True,text=True)
    return result.stdout.strip() if result.returncode==0 else None


def matching_process(pid,spec,expected=None):
    current=identity(pid)
    return bool(current and str(spec) in current and (expected is None or current==expected))


def cleanup_workers(run_directory):
    """Reap only workers whose command still names the exact recorded job file."""
    path=Path(run_directory)/'state.json'
    if not path.exists():return []
    killed=[]
    try:state=read(path);jobs=state.get('jobs',[])
    except (ValueError,OSError):return []
    for job in jobs:
        if job.get('outcome')!='running':continue
        pid=job.get('pid');spec=job.get('job')
        if pid and spec and matching_process(pid,spec):
            try:
                if os.getpgid(pid)==pid:os.killpg(pid,signal.SIGKILL);killed.append(pid)
            except ProcessLookupError:pass
        job.update(outcome='terminated-after-parent-exit',finished_utc=utc())
    if state.get('status')=='running':state['status']='interrupted-after-parent-exit'
    atomic_json(path,state)
    return killed


def verified_milestones(run_directory):
    path=Path(run_directory)/'state.json'
    if not path.exists():return {}
    try:state=read(path)
    except (ValueError,OSError):return {}
    found={}
    for number,directory in state.get('milestones',{}).items():
        folder=Path(directory)
        try:
            verification=read(folder/'verification.json');history=read(folder/'history.json')
            if (verification['passed'] and len(history)==int(number)==len(verification['results'])
                    and all(r['passed'] and not r['clearance_errors'] and r['upper_diameter_in']<2 for r in verification['results'])
                    and (folder/'layout.json').is_file()):found[number]=str(folder)
        except (KeyError,ValueError,OSError):continue
    return found


def make_policy(seconds,prior,phase,cycle,final):
    return dict(seconds=seconds,proposals=24000,seeds_per_parent=3,groups=3,
        repair_rounds=40 if final else 24,maxiter=700,generate_seconds=180,
        select_seconds=240 if final else 180,repair_seconds=2400 if final else 1200,
        verify_seconds=180,workers=2,seed=6107+phase*100000+cycle*10000,prior_run=prior)


def write_analysis(output,state):
    """Rebuild analysis products from durable evidence; never decide acceptance."""
    rows=[];best={};job_counts={};evaluated=0
    runs=[a['directory'] for a in state['attempts']]
    if state.get('legacy_run'):runs.insert(0,state['legacy_run'])
    for directory in runs:
        path=Path(directory)/'state.json'
        if path.exists():
            try:run=read(path)
            except (OSError,ValueError):continue
            for job in run.get('jobs',[]):
                key=job['kind']+':'+job['outcome'];job_counts[key]=job_counts.get(key,0)+1
                rows.append(dict(run=directory,job=job['number'],kind=job['kind'],outcome=job['outcome'],
                    elapsed_seconds=job.get('elapsed_seconds'),limit_seconds=job.get('limit_seconds'),
                    returncode=job.get('returncode'),log=job['log']))
        for file in Path(directory).glob('**/checked_*/evaluation.json'):
            try:result=read(file)
            except (OSError,ValueError):continue
            evaluated+=1
            if any(r['clearance_errors'] for r in result['results']):continue
            target=str(len(result['history']));upper=result['worst_prefix_upper_in']
            if target not in best or upper<best[target]['upper_in']:
                best[target]=dict(upper_in=upper,evaluation=str(file))
    atomic_json(Path(output)/'analysis_summary.json',dict(status=state['status'],highest_verified=state['highest_verified'],
        job_outcomes=job_counts,checked_rounds=evaluated,best_clearance_valid_bounds_by_target=best,
        attempts=[{k:a.get(k) for k in ('number','phase','outcome','returncode','directory','log','result')} for a in state['attempts']]))
    destination=Path(output)/'jobs.csv';temporary=destination.with_suffix('.tmp')
    with temporary.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['run','job','kind','outcome','elapsed_seconds','limit_seconds','returncode','log'])
        writer.writeheader();writer.writerows(rows)
    temporary.replace(destination)


def extend_contract(trial,destination):
    trial=Path(trial);payload=(trial/'contract.json').read_bytes();contract=json.loads(payload)
    target=len(contract['prefix_regions'])+2
    contract['extended_from']=dict(contract=str(trial/'contract.json'),sha256=hashlib.sha256(payload).hexdigest())
    contract['experiment']=f'controller_extended_through_{target}'
    for n in range(len(contract['prefix_regions'])+1,target+1):
        side=math.sqrt(.8*(n if n%2 else n-1))
        contract['prefix_regions'].append(dict(step=n,width_ft=side,depth_ft=side,nominal_area_sqft=side*side))
    destination=Path(destination);destination.mkdir()
    atomic_json(destination/'contract.json',contract)
    return str(destination)


def supervise_child(spec_path,record,deadline,save,event,observe,poll_seconds=5):
    """Launch/adopt a child and return its outcome without propagating its crash."""
    spec=read(spec_path)
    adopted=record.get('pid') and matching_process(record['pid'],spec_path,record.get('process_identity'))
    process=None
    def result_file():
        path=Path(spec['result_file'])
        try:return read(path) if path.exists() else {}
        except (ValueError,OSError) as exc:
            return dict(outcome='crashed',error='Unreadable child result: '+str(exc))
    if adopted:
        event('child-adopted',pid=record['pid'],spec=str(spec_path))
    else:
        if record.get('pid'):
            result_path=Path(spec['result_file'])
            if result_path.exists():
                result=result_file();record['result']=result;record['returncode']=None
                if result.get('outcome')=='completed':return 'completed'
            event('child-disappeared',pid=record['pid'],spec=str(spec_path))
            cleanup_workers(spec['output'])
            return 'crashed'
        log_path=Path(record['log'])
        try:
            with log_path.open('ab',buffering=0) as log:
                process=subprocess.Popen([record.get('python_executable',sys.executable),record['worker_script'],'--spec',str(spec_path)],
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
                             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        except OSError as exc:
            record['launch_error']=str(exc);save();event('child-launch-failed',error=str(exc));return 'crashed'
        record.update(pid=process.pid,process_identity=identity(process.pid),started_utc=utc())
        save();event('child-started',pid=process.pid,spec=str(spec_path),log=str(log_path))
    last_save=0
    try:
        while True:
            running=(process.poll() is None) if process else matching_process(record['pid'],spec_path,record.get('process_identity'))
            if not running:break
            now=time.time()
            observe(spec['output'])
            if now>=deadline:
                if matching_process(record['pid'],spec_path,record.get('process_identity')):
                    try:os.killpg(record['pid'],signal.SIGTERM)
                    except ProcessLookupError:pass
                    if process:
                        try:process.wait(timeout=3)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    else:
                        time.sleep(1)
                        if matching_process(record['pid'],spec_path,record.get('process_identity')):os.killpg(record['pid'],signal.SIGKILL)
                record['returncode']=process.returncode if process else None
                cleanup_workers(spec['output'])
                return 'timed-out'
            if now-last_save>=30:
                record['heartbeat_utc']=utc();save();last_save=now
            time.sleep(min(poll_seconds,max(.01,deadline-now)))
        record['returncode']=process.returncode if process else None
        result_path=Path(spec['result_file'])
        result=result_file()
        record['result']=result
        cleanup_workers(spec['output'])
        return 'completed' if result.get('outcome')=='completed' and record['returncode'] in (0,None) else 'crashed'
    except BaseException:
        # Supervisor interruption leaves a durable live-child identity for
        # --resume to adopt. It does not kill healthy numerical work.
        record['supervisor_detached_utc']=utc();save();raise


def campaign(output,*,initial_seconds=7200,final_seconds=14400,resume=False,poll_seconds=5):
    output=Path(output).resolve()
    if resume:
        if not output.is_dir():raise ValueError('No campaign directory to resume')
    else:output.mkdir(parents=True,exist_ok=False)
    lock=(output/'supervisor.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another supervisor owns this campaign')
    def event(rule,**data):
        record=dict(time=utc(),rule=rule,**data)
        with (output/'events.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(record),flush=True)
    state_path=output/'supervisor_state.json'
    if state_path.exists():
        if not resume:raise ValueError('Use --resume for an existing supervisor state')
        state=read(state_path)
        if state['status'].startswith('finished'):return state
    else:
        state=dict(schema_version=2,status='running',highest_verified=5,phase_index=0,
            final_retry_used=False,phases=[],attempts=[],consecutive_crashes=0,
            trial=str(Path('examples/corner_01').resolve()),seed_directory=str(Path('examples/corner_milestone_5').resolve()),
            prior_run=str(Path('examples/controller_growth_from5_20261112_v2').resolve()),
            initial_seconds=initial_seconds,final_seconds=final_seconds,visual_review='pending')
        # Import the interrupted pre-isolation campaign without losing evidence
        # or silently resetting its original two-hour deadline.
        legacy=output/'state.json'
        if resume and legacy.exists():
            old=read(legacy);phase=old['phases'][-1]
            for run in phase.get('runs',[]):
                state['legacy_run']=run['directory']
                milestones=verified_milestones(run['directory'])
                if milestones:
                    highest=max(map(int,milestones))
                    if highest>=state['highest_verified']:
                        state.update(highest_verified=highest,seed_directory=milestones[str(highest)],prior_run=run['directory'])
            started=datetime.fromisoformat(phase['started_utc']).timestamp()
            # Do not charge the user numerical-search time for this upgrade.
            paused=max(0,time.time()-(Path(state['legacy_run'])/'state.json').stat().st_mtime) if state.get('legacy_run') else 0
            state['phases']=[dict(index=0,budget_seconds=phase['budget_seconds'],
                deadline_epoch=started+phase['budget_seconds']+paused,upgrade_pause_seconds=paused,started_utc=phase['started_utc'],
                initial_highest=phase['initial_highest'],final_wall_retry=False,
                check_interval_seconds=2400,next_check_epoch=started+2400+paused,status='running')]
            event('legacy-campaign-imported',highest_verified=state['highest_verified'])
    def save():atomic_json(state_path,state)
    snapshot=output/'supervisor_sources';snapshot.mkdir(exist_ok=True)
    scripts=snapshot/'scripts';scripts.mkdir(exist_ok=True)
    # On resume use the already frozen worker launcher; never overwrite it.
    for name in ('corner_campaign.py','corner_campaign_worker.py','corner_growth_controller.py','corner_controller.py','corner_growth.py','five_stone_edge_start.py',
                 'corner_acceleration.py','corner_native.py','CornerGapProbe.java'):
        target=scripts/name
        if not target.exists():target.write_bytes((Path(__file__).parent/name).read_bytes())
    repo=Path(__file__).resolve().parents[1]
    # Keep the venv executable path: resolving its symlink selects the base
    # interpreter and silently drops the numerical environment.
    project_python=repo/'.venv/bin/python'
    state['worker_python']=str(project_python) if project_python.exists() else sys.executable
    if state.get('status')=='infrastructure-blocked':state['consecutive_crashes']=0
    if not (snapshot/'src').exists():
        shutil.copytree(repo/'src',snapshot/'src',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('pyproject.toml','uv.lock'):
        if not (snapshot/name).exists():shutil.copy2(repo/name,snapshot/name)
    worker=scripts/'corner_campaign_worker.py'
    for attempt in state['attempts']:
        attempt['worker_script']=str(worker)
    state.update(status='running',supervisor_pid=os.getpid(),supervisor_started_utc=utc())
    instances=output/'supervisor_instances';instances.mkdir(exist_ok=True)
    instance_source=instances/f'{len(state.get("supervisor_instances",[])):03d}_{os.getpid()}.py'
    instance_source.write_bytes(Path(__file__).read_bytes())
    state.setdefault('supervisor_instances',[]).append(dict(pid=os.getpid(),started_utc=utc(),
        source=str(instance_source),sha256=hashlib.sha256(instance_source.read_bytes()).hexdigest()))
    atomic_json(output/'supervisor_source_manifest.json',{str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob('*') if p.is_file()})
    save();write_analysis(output,state);event('supervisor-started',pid=os.getpid(),highest_verified=state['highest_verified'])
    try:
        while True:
            if len(state['phases'])<=state['phase_index']:
                final=state['final_retry_used'];budget=state['final_seconds'] if final else state['initial_seconds'];now=time.time()
                state['phases'].append(dict(index=state['phase_index'],budget_seconds=budget,deadline_epoch=now+budget,
                    started_utc=utc(),initial_highest=state['highest_verified'],final_wall_retry=final,
                    check_interval_seconds=4800 if final else 2400,next_check_epoch=now+(4800 if final else 2400),status='running'))
                save();event('phase-started',**state['phases'][-1])
            phase=state['phases'][state['phase_index']]
            def observe(directory):
                milestones=verified_milestones(directory)
                if milestones:
                    highest=max(map(int,milestones))
                    if highest>state['highest_verified']:
                        state.update(highest_verified=highest,seed_directory=milestones[str(highest)])
                        save();event('milestone-observed',milestone=highest,directory=state['seed_directory'])
                if time.time()>=phase['next_check_epoch']:
                    write_analysis(output,state)
                    event('scheduled-progress-check',highest_verified=state['highest_verified'],phase=phase['index'],run=directory)
                    phase['next_check_epoch']+=phase['check_interval_seconds'];save()
            active=next((a for a in state['attempts'] if a['outcome']=='running'),None)
            remaining=phase['deadline_epoch']-time.time()
            if remaining<=120 and active is None:
                phase.update(status='finished',highest_verified=state['highest_verified'],finished_utc=utc(),
                    advanced=state['highest_verified']>phase['initial_highest'])
                save();event('phase-finished',**phase)
                if phase['final_wall_retry']:
                    state['status']='finished-after-final-wall-retry';break
                if not phase['advanced']:
                    state['final_retry_used']=True;event('wall-declared',highest_verified=state['highest_verified'])
                state['phase_index']+=1;continue
            if active is None:
                number=len(state['attempts']);run_dir=output/f'isolated_run_{number:03d}'
                spec_path=output/f'attempt_{number:03d}.json'
                policy=make_policy(max(.01,remaining),state['prior_run'],phase['index'],number,phase['final_wall_retry'])
                spec=dict(trial=state['trial'],seed_directory=state['seed_directory'],output=str(run_dir),
                    policy=policy,result_file=str(output/f'attempt_{number:03d}.result.json'))
                atomic_json(spec_path,spec)
                active=dict(number=number,phase=phase['index'],directory=str(run_dir),spec=str(spec_path),
                    log=str(output/f'attempt_{number:03d}.log'),worker_script=str(worker),
                    python_executable=state['worker_python'],outcome='running',created_utc=utc())
                state['attempts'].append(active);save()
            outcome=supervise_child(Path(active['spec']),active,phase['deadline_epoch'],save,event,observe,poll_seconds)
            active.update(outcome=outcome,finished_utc=utc());observe(active['directory'])
            run_state=Path(active['directory'])/'state.json'
            if run_state.exists():state['prior_run']=active['directory']
            state['consecutive_crashes']=state['consecutive_crashes']+1 if outcome=='crashed' else 0
            event('attempt-finished',number=active['number'],outcome=outcome,returncode=active.get('returncode'))
            if state['consecutive_crashes']>=3:
                state['status']='infrastructure-blocked';save();event('repeated-worker-crashes');break
            if outcome=='crashed':time.sleep(min(30,5*state['consecutive_crashes']))
            if outcome=='completed':
                result=active['result']['state']
                if result['status']=='target-geometrically-verified-review-pending':
                    destination=output/f'contract_through_{state["highest_verified"]+2}'
                    state['trial']=extend_contract(state['trial'],destination)
                    state['prior_run']=state['trial']
                    event('harder-contract-declared',trial=state['trial'])
            save()
            write_analysis(output,state)
        save();write_analysis(output,state);return state
    except BaseException as exc:
        state.update(status='supervisor-interrupted',error_type=type(exc).__name__,error=str(exc));save()
        event('supervisor-interrupted',error=str(exc));raise
    finally:
        # Lock release is automatic on a crash as well.
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--initial-seconds',type=float,default=7200)
    parser.add_argument('--final-seconds',type=float,default=14400)
    args=parser.parse_args()
    campaign(args.output,initial_seconds=args.initial_seconds,final_seconds=args.final_seconds,resume=args.resume)
