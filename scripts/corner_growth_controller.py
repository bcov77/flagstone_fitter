"""Bounded autonomous warm-start growth experiment, using only the supplied seed.

Run from the repository root. Workers execute source snapshots, each in an owned
process group with a hard timeout. Human visual review remains pending.
"""
import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import resource

from corner_controller import atomic_json


@dataclass(frozen=True)
class GrowthPolicy:
    version: str = 'corner-growth-v3-native-indexed'
    seconds: float = 1200
    proposals: int = 18000
    seeds_per_parent: int = 2
    groups: int = 2
    repair_rounds: int = 3
    maxiter: int = 350
    generate_seconds: float = 100
    select_seconds: float = 90
    repair_seconds: float = 180
    verify_seconds: float = 90
    workers: int = 2
    seed: int = 6107
    prior_run: str | None = None
    coverage_seconds: float = 180
    native_python: str | None = 'auto'
    native_seconds: float = 45
    native_groups: int = 2
    jts_jar: str | None = 'auto'


def read(path):
    return json.loads(Path(path).read_text())


def read_native_environment(python, probe):
    result=subprocess.run([python,'-c',probe],check=True,capture_output=True,text=True,timeout=10)
    return json.loads(result.stdout)


def worker(job):
    """File-based job interface. Console output is diagnostic, never the API."""
    import corner_growth as c
    from corner_controller import Policy, run as repair
    kind = job['kind']
    out = Path(job['output'])
    if kind == 'generate':
        c.export_pool(job['trial'], job['parent'], out, seed=job['seed'],
                      proposals=job['proposals'], target_override=job['target'])
        return dict(candidates=len(read(out/'candidate_pool.json')['candidates']))
    if kind == 'coverage':
        from corner_acceleration import coverage_diagnostics
        _, contract, layout = c.load_trial(job['trial'])
        result = coverage_diagnostics(layout,contract,read(Path(job['pool'])/'candidate_pool.json'))
        atomic_json(out,result)
        return result
    if kind == 'native':
        from corner_acceleration import native_input
        payload = Path(job['input_file'])
        native_input(job['trial'],job['parent'],job['target'],payload)
        subprocess.run([job['native_python'],str(Path(__file__).with_name('corner_native.py')),
                        '--input',str(payload),'--output',str(out),'--seconds',str(job['seconds']),
                        '--groups',str(job['groups'])],check=True)
        return dict(patterns=len(read(out/'patterns.json')['patterns']))
    if kind == 'native-check':
        from corner_acceleration import check_native_patterns
        return check_native_patterns(job['trial'],job['native_file'],job['parent'],job['target'],out,job['groups'])
    if kind == 'select':
        data = read(Path(job['pool'])/'candidate_pool.json')
        if data['target']-len(data['parent_history']) == 2:
            c.select_pair(job['pool'],out,seconds=job['solve_seconds'],
                          scout=job.get('scout',False),backend='highs' if job.get('scout') else 'cp_sat')
        else:
            c.select_joint_stages(job['pool'],out,seconds=job['solve_seconds'],
                                  solve_limit=min(45,job['solve_seconds']),backend='cp_sat',
                                  workers=job['workers'],raster_spacing=.06)
        trace = read(out/'trace.json')
        return dict(accepted=bool(trace['accepted'] is not None),
                    complete_patterns=sum('chosen' in r for r in trace['iterations']))
    if kind == 'repair':
        return repair(job['trial'],job['history'],out,Policy(max_rounds=job['rounds'],
            max_seconds=job['seconds'],maxiter=job['maxiter'],seed=job['seed'],
            translation_radius_ft=.18,angle_radius_radians=.18,jts_jar=job.get('jts_jar')),witnesses_file=job.get('witnesses_file'))
    if kind == 'preserve':
        root,contract,_ = c.load_trial(job['trial'])
        history = read(job['history'])
        if len(history)<5 or len(history)%2!=1 or len(history)>len(contract['prefix_regions']):
            raise ValueError('Preservation requires an odd declared milestone from five onward')
        out.mkdir()
        for target in range(5,len(history)+1,2):
            c.save_verified(root,contract,history[:target],out/f'prefix_{target}')
        return dict(target=len(history),passed=True)
    raise ValueError('Unknown worker job')


def run(trial, seed_directory, output, policy):
    from corner_acceleration import optional_tool
    policy = replace(policy,native_python=optional_tool(policy.native_python,'native_python'),
                     jts_jar=optional_tool(policy.jts_jar,'jts_jar'))
    if (policy.seconds <= 0 or policy.proposals < 1 or policy.seeds_per_parent < 1
            or policy.native_seconds <= 0 or policy.coverage_seconds <= 0 or policy.native_groups < 1):
        raise ValueError('Invalid growth policy')
    started = time.monotonic()
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    trial = str(Path(trial).resolve())
    seed_directory = Path(seed_directory).resolve()
    seed_count = len(read(seed_directory/'history.json'))
    if seed_count<5 or seed_count%2!=1:
        raise ValueError('Growth needs an odd verified milestone seed')
    repo = Path(__file__).resolve().parents[1]
    snapshot = output/'snapshot'
    (snapshot/'scripts').mkdir(parents=True)
    shutil.copytree(repo/'src/flagstone',snapshot/'src/flagstone',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('corner_growth_controller.py','corner_controller.py','corner_growth.py','five_stone_edge_start.py',
                 'corner_acceleration.py','corner_native.py','CornerGapProbe.java'):
        shutil.copy2(repo/'scripts'/name,snapshot/'scripts'/name)
    for name in ('pyproject.toml','uv.lock'):
        shutil.copy2(repo/name,snapshot/name)
    shutil.copy2(Path(trial)/'contract.json',output/'contract.json')
    shutil.copy2(seed_directory/'history.json',output/'seed_history.json')
    contract = read(output/'contract.json')
    shutil.copy2(contract['input'],output/'source_layout.json')
    # Orientation proposals are allowed inputs; no later winning layout is loaded.
    orientation = Path('examples/edge_packingsolver_seeds_ccw/orientations.json')
    shutil.copy2(orientation,output/'orientations.json')
    atomic_json(output/'policy.json',asdict(policy))
    frozen_files = [p for p in snapshot.rglob('*') if p.is_file()] + [output/'contract.json',
        output/'seed_history.json',output/'source_layout.json',output/'orientations.json',
        Path(trial)/'contract.json',Path(contract['input']).resolve(),orientation.resolve()]
    if policy.jts_jar:
        frozen_jar=output/'jts-core.jar'; shutil.copy2(policy.jts_jar,frozen_jar)
        policy=replace(policy,jts_jar=str(frozen_jar))
        atomic_json(output/'policy.json',asdict(policy))
        frozen_files.append(frozen_jar)
    if policy.native_python:
        probe = ('import json,platform,importlib.metadata; import packingsolver._packingsolver as core; '
                 'print(json.dumps(dict(python=platform.python_version(),'
                 'packingsolver=importlib.metadata.version("packingsolver"),binary=core.__file__)))')
        environment=read_native_environment(policy.native_python,probe)
        atomic_json(output/'native_environment.json',environment)
        frozen_files.extend([Path(environment['binary']),output/'native_environment.json'])
    frozen_files.append(output/'policy.json')
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_files}
    atomic_json(output/'source_manifest.json',hashes)
    state = dict(status='running',mode='warm-start-growth',highest_verified=0,
                 seed_milestone=seed_count,target=len(contract['prefix_regions']),visual_review='pending',jobs=[],milestones={})
    state['execution']=dict(policy_version=policy.version,max_concurrent_jobs=1,
        selector_workers=policy.workers,native_workers=1,host_cpu_count=os.cpu_count(),
        gpu_backend=False,native_enabled=bool(policy.native_python),jts_enabled=bool(policy.jts_jar))
    def event(rule, **data):
        rec = dict(time=datetime.now(timezone.utc).isoformat(),rule=rule,**data)
        with (output/'events.jsonl').open('a') as f:
            f.write(json.dumps(rec)+'\n'); f.flush(); os.fsync(f.fileno())
        print(json.dumps(rec),flush=True)
    def remaining():
        return policy.seconds-(time.monotonic()-started)
    def job(kind, cap, **params):
        if remaining() <= 0:
            return dict(outcome='total-budget-exhausted')
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha for p,sha in hashes.items()):
            raise ValueError('Frozen source/input integrity changed')
        number = len(state['jobs'])
        if kind == 'repair': params['jts_jar']=policy.jts_jar
        spec = dict(kind=kind,trial=trial,**params)
        path = output/f'job_{number:03d}.json'
        atomic_json(path,spec)
        env = dict(os.environ,PYTHONPATH=os.pathsep.join([str(snapshot/'scripts'),str(snapshot/'src')]),
                   OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',
                   PYTHONDONTWRITEBYTECODE='1')
        limit = max(.01,min(cap,remaining()))
        log = output/f'job_{number:03d}.log'
        t = time.monotonic()
        with log.open('w') as f:
            proc = subprocess.Popen([sys.executable,str(snapshot/'scripts/corner_growth_controller.py'),
                '--worker',str(path)],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            rec = dict(number=number,kind=kind,pid=proc.pid,started_utc=datetime.now(timezone.utc).isoformat(),
                       limit_seconds=limit,job=str(path),log=str(log),outcome='running')
            state['jobs'].append(rec)
            atomic_json(output/'state.json',state)
            event('job-start',**rec)
            try:
                code = proc.wait(timeout=limit)
                outcome = 'completed' if code == 0 else 'failed'
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                outcome = 'timed-out'
            except BaseException:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise
            finally:
                # Native/Java helpers inherit this owned group. A launcher can
                # exit before its helpers; reap the group even after wait succeeds.
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        rec.update(outcome=outcome,elapsed_seconds=time.monotonic()-t,returncode=proc.returncode)
        result_path = path.with_suffix('.result.json')
        result = read(result_path) if outcome=='completed' and result_path.exists() else {}
        rec['result'] = result
        atomic_json(output/'state.json',state)
        event('job-finish',number=number,kind=kind,outcome=outcome,elapsed_seconds=rec['elapsed_seconds'])
        return dict(outcome=outcome,**result)
    def preserve(history, destination):
        result = job('preserve',policy.verify_seconds,history=str(history),output=str(destination))
        if result['outcome'] != 'completed':
            raise RuntimeError('Independent prefix preservation failed: '+result['outcome'])
        return destination/f'prefix_{result["target"]}'
    try:
        current = preserve(output/'seed_history.json',output/'verified_seed')
        state.update(highest_verified=seed_count)
        for milestone in range(5,seed_count+1,2):
            state['milestones'][str(milestone)] = str(current.parent/f'prefix_{milestone}')
        for target in range(seed_count+2,len(contract['prefix_regions'])+1,2):
            state['current_target'] = target
            success = False
            incumbent = read(current/'history.json')
            # Resume the controller's own evaluated attempts, never historical
            # winning layouts. Re-evaluate and physically verify before promotion.
            if policy.prior_run:
                prior = Path(policy.prior_run).resolve()
                if read(prior/'contract.json') != contract:
                    raise ValueError('Prior attempts have a different experiment contract')
                retained=[];seen=set()
                for evaluated in prior.glob('**/checked_*/evaluation.json'):
                    evaluation=read(evaluated)
                    if len(evaluation['history'])!=target:continue
                    identity=tuple(r['stone_id'] for r in evaluation['history'])
                    invalid=any(r['clearance_errors'] for r in evaluation['results'])
                    retained.append((invalid,evaluation['worst_prefix_upper_in'],str(evaluated)))
                for _,_,evaluated in sorted(retained):
                    evaluation=read(Path(evaluated))
                    identity=tuple(r['stone_id'] for r in evaluation['history'])
                    if identity in seen:continue
                    seen.add(identity)
                    if len(seen)>policy.groups or remaining()<=policy.verify_seconds:break
                    attempt=output/f'target_{target}_retained_{len(seen)-1}'
                    attempt.mkdir()
                    atomic_json(attempt/'history.json',evaluation['history'])
                    atomic_json(attempt/'provenance.json',dict(evaluation=evaluated))
                    event('repair-retained-controller-checkpoint',target=target,source=evaluated)
                    repaired=attempt/'repair'
                    witnesses=Path(evaluated).parent.parent/'witnesses.json'
                    job('repair',policy.repair_seconds,history=str(attempt/'history.json'),output=str(repaired),
                        rounds=policy.repair_rounds,seconds=max(1,policy.repair_seconds-10),
                        maxiter=policy.maxiter,seed=policy.seed+target*10+len(seen),
                        witnesses_file=str(witnesses) if witnesses.exists() else None)
                    v=repaired/'accepted/verification.json'
                    if v.exists() and read(v)['passed']:
                        current=preserve(repaired/'accepted/history.json',attempt/'preserved')
                        state['highest_verified']=target
                        for milestone in range(5,target+1,2):
                            state['milestones'][str(milestone)]=str(current.parent/f'prefix_{milestone}')
                        event('milestone-verified',target=target,directory=str(current))
                        success=True;break
            if success:continue
            for base_count in range(len(incumbent),0,-2):
                if remaining()<=0:break
                parent = output/f'target_{target}_parent_{base_count}'
                parent.mkdir()
                atomic_json(parent/'history.json',incumbent[:base_count])
                if policy.native_python and remaining() > policy.verify_seconds:
                    native=output/f'target_{target}_base_{base_count}_native'
                    native_input_file=output/f'target_{target}_base_{base_count}_native_input.json'
                    job('native',policy.native_seconds+30,parent=str(parent),target=target,
                        input_file=str(native_input_file),output=str(native),native_python=policy.native_python,
                        seconds=policy.native_seconds,groups=policy.native_groups)
                    # Full native patterns are atomically saved before a timeout;
                    # none is trusted until a separate original-geometry check.
                    patterns=native/'patterns.json'
                    if patterns.exists() and read(patterns)['patterns']:
                        checked=job('native-check',policy.coverage_seconds,parent=str(parent),target=target,
                            native_file=str(patterns),output=str(native/'checked'),groups=policy.native_groups)
                        for index,history_path in enumerate(checked.get('histories',[])):
                            repaired=native/f'repair_{index}'
                            job('repair',policy.repair_seconds,history=history_path,output=str(repaired),
                                rounds=policy.repair_rounds,seconds=max(1,policy.repair_seconds-10),
                                maxiter=policy.maxiter,seed=policy.seed+target*10+index)
                            v=repaired/'accepted/verification.json'
                            if v.exists() and read(v)['passed']:
                                current=preserve(repaired/'accepted/history.json',native/'preserved')
                                state['highest_verified']=target
                                for milestone in range(5,target+1,2):
                                    state['milestones'][str(milestone)]=str(current.parent/f'prefix_{milestone}')
                                event('milestone-verified',target=target,directory=str(current),source='native-proposal')
                                success=True;break
                    if success:break
                for enrichment in range(policy.seeds_per_parent):
                    if remaining()<=0:break
                    attempt = output/f'target_{target}_base_{base_count}_enrich_{enrichment}'
                    attempt.mkdir()
                    event('propose-continuation',target=target,fixed_count=base_count,enrichment=enrichment)
                    gen = job('generate',policy.generate_seconds,parent=str(parent),output=str(attempt/'pool'),
                        target=target,seed=policy.seed+target*100+base_count*10+enrichment,proposals=policy.proposals)
                    pool_file=attempt/'pool/candidate_pool.json'
                    if not pool_file.exists():continue
                    try:read(pool_file)
                    except (ValueError,OSError):
                        event('incomplete-pool-not-salvaged',path=str(pool_file));continue
                    if gen['outcome']!='completed':event('salvage-generated-pool',path=str(pool_file))
                    # Coverage has its own deadline. Finished pools survive a
                    # generation timeout and are never regenerated just for diagnosis.
                    gen=job('coverage',policy.coverage_seconds,pool=str(attempt/'pool'),
                            output=str(attempt/'coverage.json'))
                    if gen['outcome']!='completed' or gen['candidates']<2:continue
                    if gen['zero_cover']:
                        event('zero-cover-enrich' if enrichment+1<policy.seeds_per_parent else 'zero-cover-backtrack',
                              target=target,fixed_count=base_count,zero_centers=len(gen['zero_cover']))
                        continue
                    selection = attempt/'selection'
                    selected = job('select',policy.select_seconds+15,pool=str(attempt/'pool'),output=str(selection),
                                   solve_seconds=policy.select_seconds,workers=policy.workers)
                    verified = selection/'accepted/verification.json'
                    candidate = selection/'accepted' if selected['outcome']=='completed' and verified.exists() and read(verified)['passed'] else None
                    trace_file = selection/'trace.json'
                    if candidate is None and trace_file.exists():
                        data = read(attempt/'pool/candidate_pool.json')
                        patterns = [];seen = set()
                        for rec in read(trace_file)['iterations']:
                            if 'chosen' not in rec:continue
                            identity = tuple(data['candidates'][j]['stone_id'] for j in rec['chosen'])
                            if identity in seen:continue
                            seen.add(identity)
                            score = max((r['upper_diameter_in'] for r in rec.get('prefix_checks',[])),
                                        default=rec.get('upper_diameter_in',100))
                            patterns.append((score,rec))
                        patterns.sort(key=lambda item:item[0])
                        for group,(_,rec) in enumerate(patterns[:policy.groups]):
                            if remaining()<=0:break
                            records = list(data['parent_history'])
                            for j in rec['chosen']:
                                r = data['candidates'][j]
                                records.append(dict(step=len(records)+1,stone_id=r['stone_id'],pose=r['pose']))
                            seed = attempt/f'group_{group}.json'
                            atomic_json(seed,records)
                            repaired = attempt/f'repair_{group}'
                            job('repair',policy.repair_seconds,history=str(seed),output=str(repaired),
                                rounds=policy.repair_rounds,seconds=max(1,policy.repair_seconds-10),
                                maxiter=policy.maxiter,seed=policy.seed+target*10+group)
                            v = repaired/'accepted/verification.json'
                            if v.exists() and read(v)['passed']:
                                candidate = repaired/'accepted';break
                    if candidate is not None:
                        current = preserve(candidate/'history.json',attempt/'preserved')
                        state['highest_verified'] = target
                        for milestone in range(5,target+1,2):
                            if milestone <= target:
                                state['milestones'][str(milestone)] = str(current.parent/f'prefix_{milestone}')
                        event('milestone-verified',target=target,directory=str(current))
                        success = True;break
                if success:break
                event('reopen-earlier-prefix',target=target,previous_fixed=base_count,next_fixed=max(1,base_count-2))
            if not success:break
        state['status'] = ('target-geometrically-verified-review-pending' if state['highest_verified']==len(contract['prefix_regions']) else
                           'budget-exhausted/incomplete' if remaining()<=0 else 'declared-search-exhausted/incomplete')
    except BaseException as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic()-started
        atomic_json(output/'state.json',state)
        event('run-finished',status=state['status'],highest_verified=state['highest_verified'])
    return state


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker')
    ap.add_argument('--trial',default='examples/corner_01')
    ap.add_argument('--seed-directory',default='examples/corner_milestone_5')
    ap.add_argument('--output')
    ap.add_argument('--seconds',type=float,default=1200)
    ap.add_argument('--native-python',default='auto',help='PackingSolver interpreter; auto discovers the existing environment')
    ap.add_argument('--jts-jar',default='auto',help='Optional JTS jar; auto checks PATIO_JTS_JAR and the user cache')
    ap.add_argument('--no-native',action='store_true')
    ap.add_argument('--no-jts',action='store_true')
    args = ap.parse_args()
    if args.worker:
        path = Path(args.worker)
        started=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu=time.process_time()
        result = worker(read(path))
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        result['timing']=dict(wall_seconds=time.monotonic()-started,
            cpu_seconds=time.process_time()-cpu+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
        atomic_json(path.with_suffix('.result.json'),result)
    else:
        run(args.trial,args.seed_directory,args.output,GrowthPolicy(seconds=args.seconds,
            native_python=None if args.no_native else args.native_python,jts_jar=None if args.no_jts else args.jts_jar))
