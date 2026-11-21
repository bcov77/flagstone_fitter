"""Foreground ten-process, indefinitely repeating online patio trajectories."""
import argparse, copy, fcntl, importlib.metadata, json, multiprocessing, os, signal, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
from proposal_search import search as search_proposal, ATTEMPT_POLICY
from common import ROOT,write,sha,seed_for,state,restore,measure,clearance,read,attempt_path
from selector import generate,choose
from physics import relax
HERE=Path(__file__).resolve().parent
DEFAULT_SOURCE=ROOT/'examples/quarter2_20h_ready_20261114/tasks/task_576_escape/best.json'
DEFAULT_OUT=ROOT/'examples/physics_trajectories_20261117_r3'

def mode_for(number):
    return 'golden' if number%10==9 else 'silver' if number%10==0 else 'normal'

def dependencies():
    paths=list(HERE.glob('*.py'))+[ROOT/'physics_patio/patio_physics/engine.py',
        ROOT/'analysis/physics_placement_campaign/future_gap_geometry.py',ROOT/'analysis/physics_placement_campaign/selection_geometry.py',
        *[ROOT/f'src/flagstone/{p}.py' for p in ['quarter_gaps','quarter','phase1']]]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}

def versions():
    return {name:importlib.metadata.version(name) for name in ['numpy','scipy','shapely','pymunk','pyvoronoi','matplotlib']}

def prepare(out,source,config):
    out.mkdir(parents=True,exist_ok=True);path=out/'manifest.json'
    if path.exists():
        m=json.loads(path.read_text())
        if m['dependencies']!=dependencies() or m['versions']!=versions():raise RuntimeError('Code/library hashes changed; preserve this experiment and choose a new --out')
        if m['config']!=config:raise RuntimeError('Config differs; resume with the original config or use a new --out')
        if sha(out/'source.json')!=m['source_sha256']:raise RuntimeError('Source snapshot changed')
        if sha(source)!=m['source_sha256']:raise RuntimeError('Requested source differs from frozen source; use a new --out')
        return m
    raw=source.read_bytes();d=json.loads(raw)
    if d['units']!='ft':raise ValueError('Requires feet')
    check=clearance(d)
    if not check['valid']:raise ValueError(f'Start board fails half-inch clearance: {check}')
    (out/'source.json').write_bytes(raw)
    m=dict(version=1,created_unix=time.time(),source=str(source.resolve()),source_sha256=sha(source),config=config,
        dependencies=dependencies(),versions=versions(),start_metrics=measure(d),
        policy=dict(schedule='Runs 1–8 normal, 9 Golden, 10 Silver, repeating by global run number',
        parallelism='Independent trajectories in separate processes; serial decisions within each trajectory',
        acceptance='Half-inch original-outline clearance and observed settling; no objective-improvement rejection',
        attempt_count=ATTEMPT_POLICY,
        stop='STOP file or Ctrl-C: finish current attempts, checkpoint, stop assigning new trajectories',
        selection='Experimental risk-first heuristics; full conditional probabilities and every evaluated alternative saved'))
    write(path,m);return m

def init_worker():
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    signal.signal(signal.SIGTERM,signal.SIG_IGN)

def trajectory(out,number):
    out=Path(out);m=json.loads((out/'manifest.json').read_text());cfg=m['config'];source=json.loads((out/'source.json').read_text())
    folder=out/'runs'/f'run_{number:06d}';folder.mkdir(parents=True,exist_ok=True);mode=mode_for(number)
    write(folder/'run.json',dict(number=number,mode=mode,seed=seed_for(cfg['seed'],number),attempts=cfg['attempts_per_run']))
    document=source;metrics=m['start_metrics'];history=[];attempted_ids=set()
    for i in range(1,cfg['attempts_per_run']+1):
        path=attempt_path(folder,i)
        if not path.exists():break
        r=read(path)
        if r['number']!=i or r['run']!=number:raise RuntimeError('Invalid saved attempt identity')
        if r['start']!=state(document):raise RuntimeError('Broken trajectory chain')
        document=restore(source,r['end']);metrics=r['end_metrics'];history.append(dict(accepted=r['accepted'],chosen={'placements':r['chosen']['placements']} if r['chosen'] else None,repair_attempted=r['repair_attempted'],wall_seconds=r['wall_seconds']))
        if r.get('chosen'):attempted_ids.update(p['stone'] for p in r['chosen']['placements'])
    for i in range(len(history)+1,cfg['attempts_per_run']+1):
        path=attempt_path(folder,i)
        if (out/'STOP').exists():return dict(number=number,status='checkpointed',attempts=len(history))
        start_time=time.monotonic();start=state(document);rng=np.random.default_rng(seed_for(cfg['seed'],number,i))
        write(folder/'progress.json',dict(run=number,mode=mode,attempt=i,phase='selecting',updated_unix=time.time(),completed=len(history),metrics=metrics))
        pending_path=folder/'pending_attempt.json.gz'
        if pending_path.exists():
            record=read(pending_path)
            if record['number']<=len(history):
                committed=read(attempt_path(folder,record['number']))
                if any(record[k]!=committed[k] for k in ['run','number','start','chosen']):raise RuntimeError('Stale pending record disagrees with committed transaction')
                pending_path.unlink()
            elif record['start']!=start or record['number']!=i or record['run']!=number:
                raise RuntimeError('Pending attempt does not match retained board')
            elif not record.get('chosen'):
                write(folder/f'parent_empty_pending_{i:03d}.json.gz',record)
                pending_path.unlink()
        if not pending_path.exists():
            def progress(round_number):
                write(folder/'progress.json',dict(run=number,mode=mode,attempt=i,phase='selecting',search_round=round_number,updated_unix=time.time(),completed=len(history),metrics=metrics))
            found=search_proposal(folder,i,document,start,cfg,mode,rng,generate,choose,attempted_ids,lambda:(out/'STOP').exists(),progress)
            if found is None:return dict(number=number,status='checkpointed',attempts=len(history))
            pool,diagnostics,chosen,selection=found
            record=dict(run=number,number=i,mode=mode,seed=seed_for(cfg['seed'],number,i),start=start,start_metrics=metrics,
                chosen=chosen,selection=selection,proposal_diagnostics=diagnostics,candidates=pool,
                compute_revision=m.get('compute_revision','proposal_retry_v2'))
            write(pending_path,record)
        else:
            record=read(pending_path)
            chosen=record['chosen']
        if chosen:
            write(folder/'progress.json',dict(run=number,mode=mode,attempt=i,phase='relaxing',updated_unix=time.time(),completed=len(history),metrics=metrics))
            result=relax(document,chosen,cfg,seed_for(cfg['seed'],number,i)+1)
            attempted=result.pop('attempted_document');accepted=result['accepted']
            attempted_metrics=measure(attempted)
            if accepted:document=attempted;metrics=attempted_metrics
            record.update(result,attempted_end=state(attempted),attempted_end_metrics=attempted_metrics)
            attempted_ids.update(p['stone'] for p in chosen['placements'])
        record.update(end=state(document),end_metrics=metrics,wall_seconds=time.monotonic()-start_time)
        # Attempt is the transaction boundary: endpoint, reasons and outcome commit together.
        write(path,record);history.append(dict(accepted=record['accepted'],chosen={'placements':chosen['placements']} if chosen else None,repair_attempted=record['repair_attempted'],wall_seconds=record['wall_seconds']));(folder/'pending_attempt.json.gz').unlink(missing_ok=True)
        write(folder/'progress.json',dict(run=number,mode=mode,attempt=i,phase='checkpointed',updated_unix=time.time(),completed=len(history),metrics=metrics))
    summary=dict(number=number,mode=mode,status='complete',attempts=len(history),accepted=sum(r['accepted'] for r in history),
        doubles_attempted=sum(bool(r['chosen']) and len(r['chosen']['placements'])==2 for r in history),
        repair_attempted=sum(r['repair_attempted'] for r in history),repair_successes=sum(r['repair_attempted'] and r['accepted'] for r in history),
        start_metrics=m['start_metrics'],end_metrics=metrics,wall_seconds=sum(r['wall_seconds'] for r in history))
    write(folder/'final.json',document);write(folder/'summary.json',summary)
    # Plots are derived after numerical checkpoints; failures never erase a trajectory.
    try:
        from report import render_run
        render_run(out,number)
    except Exception:
        write(folder/'render_error.json',dict(traceback=traceback.format_exc()))
    return summary

def validate_config(cfg):
    ints=['attempts_per_run','target_count','piece_budget','angle_step_degrees','coarse_per_target','planning_pool','stable_samples']
    if any(not isinstance(cfg[k],int) or isinstance(cfg[k],bool) or cfg[k]<1 for k in ints):raise ValueError('Positive integer counts required')
    if cfg['normal_best_probability']<0 or cfg['normal_ranked_probability']<0 or cfg['normal_best_probability']+cfg['normal_ranked_probability']>1:raise ValueError('Invalid normal mixture probabilities')
    if not 0<cfg['best_fraction']<=1 or not 0<=cfg['silver_best_probability']<=1:raise ValueError('Invalid best/silver probabilities')
    if not .5<=cfg['repair_gap_inches']<=2:raise ValueError('Invalid gap-cycle target')
    if any(not 0<cfg[k]<=60 for k in ['explore_seconds','quiet_sample_seconds','quiet_max_seconds','repair_inflate_seconds','repair_deflate_seconds']):raise ValueError('Invalid physics durations')
    if cfg['quiet_max_seconds']<cfg['stable_samples']*cfg['quiet_sample_seconds']:raise ValueError('Quiet duration cannot satisfy settling criterion')

def status(out):
    rows=[]
    for path in sorted((out/'runs').glob('run_*/summary.json')):rows.append(json.loads(path.read_text()))
    return dict(completed_runs=len(rows),modes={k:sum(r['mode']==k for r in rows) for k in ['normal','golden','silver']},
        accepted=sum(r['accepted'] for r in rows),attempts=sum(r['attempts'] for r in rows),runs=rows)

def execute(args,cfg):
    out=args.out.resolve()
    if out==ROOT or out==HERE or out in args.source.resolve().parents:raise ValueError('Use a separate output directory outside the source directory and its ancestors')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This output directory is already running')
        m=prepare(out,args.source,cfg)
        (out/'STOP').unlink(missing_ok=True)
        stopping=False
        def stop(signum=None,frame=None):
            nonlocal stopping
            if not stopping:print('\nStopping after each worker finishes its current attempt; checkpoints will resume.',flush=True)
            stopping=True;(out/'STOP').touch()
        old_int=signal.signal(signal.SIGINT,stop);old_term=signal.signal(signal.SIGTERM,stop)
        existing=sorted(int(p.name.split('_')[1]) for p in (out/'runs').glob('run_*') if p.is_dir())
        pending=[n for n in existing if not (out/'runs'/f'run_{n:06d}'/'summary.json').exists()]
        next_number=max(existing,default=0)+1;submitted=0;failed=[];last_status=0
        print(f'Output: {out}\n{args.workers} workers; {cfg["attempts_per_run"]} serial attempts/trajectory; 8 normal + 1 Golden + 1 Silver per ten. Ctrl-C checkpoints.',flush=True)
        try:
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker) as executor:
                futures={}
                while True:
                    if (out/'STOP').exists():stopping=True
                    while not stopping and len(futures)<args.workers and (args.runs is None or submitted<args.runs):
                        if pending:number=pending.pop(0)
                        else:number=next_number;next_number+=1
                        # Reserve identity before submit, so parent failure cannot skip unfinished work.
                        folder=out/'runs'/f'run_{number:06d}';folder.mkdir(parents=True,exist_ok=True)
                        futures[executor.submit(trajectory,str(out),number)]=number;submitted+=1
                    if not futures:break
                    done,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
                    for f in done:
                        n=futures.pop(f)
                        try:
                            r=f.result();print(f'Run {n} ({mode_for(n)}): {r["status"]}, {r["attempts"]} attempts'+(f', {r["accepted"]} accepted, {r["end_metrics"]["area_percent"]:.2f}% area' if 'accepted' in r else ''),flush=True)
                        except Exception:
                            error=dict(run=n,traceback=traceback.format_exc());failed.append(error);write(out/'runs'/f'run_{n:06d}'/'error.json',error);stop();print(error['traceback'],flush=True)
                    if time.monotonic()-last_status>=15:
                        messages=[]
                        for n in sorted(futures.values()):
                            p=out/'runs'/f'run_{n:06d}'/'progress.json'
                            if p.exists():
                                r=json.loads(p.read_text());messages.append(f'{n}:{r["completed"]}/{cfg["attempts_per_run"]} {r["phase"]}')
                        print('Active '+', '.join(messages),flush=True);last_status=time.monotonic()
                    if done:write(out/'summary.json',status(out))
            write(out/'summary.json',status(out))
        finally:
            signal.signal(signal.SIGINT,old_int);signal.signal(signal.SIGTERM,old_term)
        if failed:raise RuntimeError('Worker failed; see run error.json. Fix or resume without altering frozen compute inputs.')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=DEFAULT_OUT);ap.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    ap.add_argument('--config',type=Path,default=HERE/'config.json');ap.add_argument('--workers',type=int,default=10);ap.add_argument('--runs',type=int,help='Optional finite number of trajectories to execute/resume this invocation')
    ap.add_argument('--status',action='store_true');args=ap.parse_args()
    if args.status:
        s=status(args.out);print(json.dumps({k:v for k,v in s.items() if k!='runs'},indent=2));return
    if args.workers<1 or (args.runs is not None and args.runs<1):ap.error('Positive workers/runs required')
    cfg=json.loads(args.config.read_text());validate_config(cfg);execute(args,cfg)

if __name__=='__main__':main()
