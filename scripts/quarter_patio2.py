"""Second-campaign foreground numerical worker; separate from historical workers."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import numpy as np
import shapely
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from flagstone.phase1 import load_json,save_json,validate,_layout_data
from flagstone import quarter_search2 as search
from flagstone.quarter import GAP,world,check_reference,apply_records,native_input,write
from flagstone.quarter_gaps import RULE,verify,gap_candidates
from quarter_patio import child

def checkpoint(layout,path):
    # Retain complete original geometry with compact JSON; source coordinates
    # and float precision are unchanged, while repeated checkpoint I/O is smaller.
    path=Path(path);temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(_layout_data(layout),separators=(',',':'),allow_nan=False))
    temporary.replace(path)


POLICIES=('energy','worst','witness','adaptive_grid','shape_swap','joint_swap','small_repack',
          'large_repack','escape','restarts','batch_cpu','batch_gpu')


def geometry_hash(layout):
    data=[(sid,tuple(map(float,layout.poses[i]))) for i,sid in enumerate(layout.ids) if layout.active[i]]
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def bootstrap(layout,out,args,budget):
    source=out/f'native_{args.bootstrap_index:03d}'
    payload=source.with_suffix('.input.json');data=native_input(layout,args.seed+args.bootstrap_index*1009,args.native_angle_deg);write(payload,data)
    result=child([args.native_python,str(ROOT/'scripts/quarter_native2.py'),'--input',str(payload),
                  '--output',str(source),'--seconds',str(max(.1,budget-3))],source.with_suffix('.log'),max(.1,budget))
    args.bootstrap_index+=1
    path=source/'patterns.json'
    if not path.exists():
        if result['outcome']=='failed':raise RuntimeError('Native worker failed: '+str(source.with_suffix('.log')))
        return False
    best=None;state=None;points=search.points_for(layout)
    for item in json.loads(path.read_text()).get('patterns',[]):
        apply_records(layout,item['records'])
        if validate(layout,gap=GAP,edge=GAP):continue
        metrics=search.measure(layout,points)
        if best is None or search.rank(metrics,args.objective)<search.rank(best,args.objective):best,state=metrics,layout.snapshot()
    if state is None:return False
    layout.restore(state);return True


def run(args):
    out=Path(args.output).absolute();out.mkdir(parents=True,exist_ok=False)
    trial=Path(args.trial);contract=json.loads((trial/'contract.json').read_text())
    if contract.get('gap_rule')!=RULE:raise ValueError('Requires versioned local-circle edge5/interior2 trial')
    if hashlib.sha256((trial/'input.json').read_bytes()).hexdigest()!=contract['input_sha256']:raise ValueError('Trial changed')
    started=time.monotonic();cpu0=time.process_time();children0=resource.getrusage(resource.RUSAGE_CHILDREN)
    deadline=started+args.seconds;remaining=lambda:max(0,deadline-time.monotonic())
    layout=load_json(trial/'input.json',seed=args.seed)
    state=dict(status='running',settings=vars(args).copy(),initial=None,best=None,rounds=[],rule=RULE,
               seed=args.seed,policy=args.policy,visual_review='pending',witnesses=[],bootstrap_index=0,reference_sha256=contract['input_sha256'])
    state['settings'].pop('function',None)
    initial=None;incumbent=None;incumbent_metrics=None;exploration=None;round_index=0;prior_cpu=0;prior_elapsed=0
    write(out/'environment.json',dict(python=sys.version,executable=sys.executable,
          packages={name:importlib.metadata.version(name) for name in ('numpy','scipy','shapely','pyvoronoi','mlx')},
          thread_limits={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}))
    (out/'contract.json').write_bytes((trial/'contract.json').read_bytes())
    write(out/'source_manifest.json',{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
          for path in [Path(__file__),ROOT/'scripts/quarter_native2.py',ROOT/'scripts/quarter_native.py',
                       *sorted((ROOT/'src/flagstone').glob('*.py'))]})
    witness_bank=[];last_witness=-1e9;stalled=0;last_restart=-1e9
    args.bootstrap_index=0
    saved_hash=None;stalled_seconds=0.
    parameters={key:getattr(args,key) for key in ('patch_seconds','native_angle_deg','stall_rounds','stall_seconds','batch_size','batch_score_points')}
    points=search.points_for(layout,shifted=args.policy=='adaptive_grid')
    def commit():
        nonlocal incumbent,incumbent_metrics,saved_hash
        if incumbent is None:return
        exploration=layout.snapshot();layout.restore(incumbent)
        current_hash=geometry_hash(layout)
        if saved_hash!=current_hash or not (out/'best.json').exists():
            checkpoint(layout,out/'best.json');saved_hash=current_hash
        state['best']=incumbent_metrics;state['geometry_sha256']=current_hash
        data=dict(schema='quarter2-checkpoint-v2',rule=RULE,reference_sha256=contract['input_sha256'],
                  policy=args.policy,objective=args.objective,seed=args.seed,search_parameters=parameters,round=round_index,
                  incumbent=dict(poses=incumbent[0].tolist(),active=incumbent[1].tolist()),
                  exploration=dict(poses=exploration[0].tolist(),active=exploration[1].tolist()),
                  rng_state=exploration[2],witnesses=witness_bank,initial=state['initial'],best=incumbent_metrics,
                  rounds=state['rounds'],stalled=stalled,stalled_seconds=stalled_seconds,bootstrap_index=args.bootstrap_index,
                  prior_cpu_seconds=prior_cpu+time.process_time()-cpu0+resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime-children0.ru_utime-children0.ru_stime,
                  prior_elapsed_seconds=prior_elapsed+time.monotonic()-started)
        write(out/'checkpoint.json',data);layout.restore(exploration);write(out/'state.json',state)
    write(out/'state.json',state)
    try:
        resume_checkpoint=Path(args.resume_from)/'checkpoint.json' if args.resume_from else None
        if args.resume_from and resume_checkpoint.exists():
            data=json.loads(resume_checkpoint.read_text())
            if (data['schema']!='quarter2-checkpoint-v2' or data['reference_sha256']!=contract['input_sha256']
                or data.get('search_parameters')!=parameters or data['rule']!=RULE or (data['policy'],data['objective'],data['seed'])!=(args.policy,args.objective,args.seed)):
                raise ValueError('Resume policy/reference mismatch')
            for key in ('incumbent','exploration'):
                poses=np.asarray(data[key]['poses'],float);active=np.asarray(data[key]['active'])
                if poses.shape!=layout.poses.shape or not np.isfinite(poses).all() or active.shape!=layout.active.shape or active.dtype!=bool:raise ValueError('Malformed checkpoint')
                layout.poses[:]=poses;layout.active[:]=active
                if validate(layout,gap=GAP,edge=GAP):raise ValueError('Checkpoint has clearance violations')
                if key=='incumbent':incumbent=layout.snapshot()
            layout.rng.bit_generator.state=data['rng_state'];incumbent=(incumbent[0],incumbent[1],data['rng_state'])
            round_index=data['round'];witness_bank=data['witnesses'];state['initial']=data['initial'];state['rounds']=data['rounds']
            stalled=data['stalled'];stalled_seconds=data['stalled_seconds'];args.bootstrap_index=data['bootstrap_index'];prior_cpu=data['prior_cpu_seconds'];prior_elapsed=data['prior_elapsed_seconds']
            incumbent_metrics=data['best']
            previous_file=Path(args.resume_from)/'state.json'
            previous=json.loads(previous_file.read_text()) if previous_file.exists() else {}
            prior_cpu=max(prior_cpu,previous.get('cpu_seconds_total',0));prior_elapsed=max(prior_elapsed,previous.get('elapsed_seconds',0))
        elif args.resume_from:
            previous=Path(args.resume_from)
            previous_state=json.loads((previous/'state.json').read_text()) if (previous/'state.json').exists() else {}
            if previous_state and (previous_state.get('rule')!=RULE or previous_state.get('policy')!=args.policy
                    or previous_state.get('seed')!=args.seed or previous_state.get('reference_sha256')!=contract['input_sha256']
                    or previous_state.get('settings',{}).get('objective')!=args.objective):
                raise ValueError('Bootstrap recovery policy/reference mismatch')
            prior_cpu=previous_state.get('cpu_seconds_total',0);prior_elapsed=previous_state.get('elapsed_seconds',0)
            recovered=None;recovered_metrics=None
            for pool in sorted(previous.glob('native_*/patterns.json')):
                for item in json.loads(pool.read_text()).get('patterns',[]):
                    apply_records(layout,item['records'])
                    if validate(layout,gap=GAP,edge=GAP):continue
                    metrics=search.measure(layout,points)
                    if recovered_metrics is None or search.rank(metrics,args.objective)<search.rank(recovered_metrics,args.objective):
                        recovered,recovered_metrics=layout.snapshot(),metrics
            if recovered is not None:
                layout.restore(recovered);state['resume_fidelity']='salvaged native patterns; no optimizer state'
            elif not bootstrap(layout,out,args,min(args.bootstrap_seconds,max(1,remaining()-45))):
                state['status']='no-valid-bootstrap/incomplete';return state
        elif args.start:
            supplied=load_json(args.start,seed=args.seed);check_reference(supplied,layout)
            if validate(supplied,gap=GAP,edge=GAP):raise ValueError('Invalid parent clearances')
            layout.poses[:]=supplied.poses;layout.active[:]=supplied.active
        elif not bootstrap(layout,out,args,min(args.bootstrap_seconds,max(1,remaining()-45))):
            state['status']='no-valid-bootstrap/incomplete';return state
        if incumbent is None:
            incumbent=layout.snapshot();incumbent_metrics=search.measure(layout,points)
        state['initial']=state['initial'] or incumbent_metrics;state['best']=incumbent_metrics
        checkpoint(layout,out/'initial.json');commit()
        scorer=None
        # One OS lock serializes GPU scorers across workers; the short screen
        # measures actual contention. A worker never keeps the lock between actions.
        while round_index<args.rounds and remaining()>50:
            loop_started=time.monotonic()
            if args.policy in ('witness','adaptive_grid') and time.monotonic()-last_witness>120 and remaining()>80:
                circles=gap_candidates(layout.patio,list(world(layout).values()))
                ranked=sorted(circles,key=lambda c:c['diameter_in']/(5 if c['kind']=='edge' else 2),reverse=True)
                witness_bank.extend(c['center'] for c in ranked[:12]);witness_bank=witness_bank[-120:]
                last_witness=time.monotonic()
            points=search.points_for(layout,shifted=args.policy=='adaptive_grid',witnesses=witness_bank)
            # Scores must be recomputed on the same point set before comparison.
            if args.policy.startswith('batch_'):
                metrics=incumbent_metrics
            else:
                current=layout.snapshot();layout.restore(incumbent);incumbent_metrics=search.measure(layout,points);layout.restore(current)
                metrics=search.measure(layout,points)
            center=metrics['largest_center']
            if witness_bank and args.policy in ('witness','adaptive_grid'):
                center=witness_bank[round_index%len(witness_bank)]
            original=layout.snapshot();t0=time.monotonic();action={};kind='continuous'
            budget=min(args.patch_seconds,max(.01,remaining()-48))
            if args.policy=='shape_swap':
                kind='replacement';action=search.replacement(layout,center,points,seconds=budget,objective=args.objective)
            elif args.policy=='joint_swap':
                kind='replacement';action=search.replacement(layout,center,points,seconds=budget,group=2+round_index%2,joint=True,objective=args.objective)
            elif args.policy in ('small_repack','large_repack'):
                kind='rebuilt';count=(3+round_index%3) if args.policy=='small_repack' else (6+round_index%5)
                action=search.repack(layout,center,points,seconds=budget,count=count,objective=args.objective)
            elif args.policy.startswith('batch_'):
                kind='continuous'
                if scorer is None:
                    from flagstone.quarter_batch import BatchScorer
                    coords=shapely.get_coordinates(points[::max(1,len(points)//args.batch_score_points)])
                    scorer=BatchScorer(layout,coords,'gpu' if args.policy=='batch_gpu' else 'cpu')
                if args.policy=='batch_gpu':
                    from flagstone.quarter_batch import GPUQueueDeadline
                    scorer.gpu_lock_path=args.gpu_lock;scorer.queue_deadline=deadline-50
                    try:action=search.batch_move(layout,center,points,backend='gpu',scorer=scorer,objective=args.objective,batch_size=args.batch_size)
                    except GPUQueueDeadline:action=dict(accepted=False,skipped=True,reason='gpu-queue-deadline')
                else:action=search.batch_move(layout,center,points,scorer=scorer,objective=args.objective,batch_size=args.batch_size)
            elif args.policy=='restarts' and stalled>=8 and time.monotonic()-last_restart>120 and remaining()>100:
                kind='rebuilt';last_restart=time.monotonic()
                success=bootstrap(layout,out,args,min(30,remaining()-50));action=dict(accepted=success,reason='fresh-global-start')
            else:
                if args.policy=='escape' and stalled>=5:
                    patch=search.neighborhood(layout,center,count=3)
                    trial=layout.snapshot();layout.poses[patch]+=layout.rng.normal(0,[.05,.05,.15],(len(patch),3))
                    if validate(layout,gap=GAP,edge=GAP):
                        rng=layout.snapshot()[2];layout.restore(trial);layout.rng.bit_generator.state=rng
                    action['exploratory']=True
                repair=search.continuous(layout,search.neighborhood(layout,center,count=6 if round_index%3 else 9),points,
                                         seconds=budget,maxiter=240,radius_ft=.35,objective=args.objective)
                action.update(repair)
                # Baselines retain occasional structural actions, instead of
                # accidentally comparing new mechanisms only to movement control.
                if args.policy in ('energy','worst','witness','adaptive_grid','restarts') and stalled>=4 and round_index%5==4 and remaining()>65:
                    structural=search.repack(layout,center,points,seconds=min(8,remaining()-50),count=3,objective=args.objective)
                    action['structural']=structural
            if validate(layout,gap=GAP,edge=GAP):
                rng=layout.snapshot()[2];layout.restore(original);layout.rng.bit_generator.state=rng
                raise RuntimeError('Action returned an invalid layout')
            if args.policy.startswith('batch_'):
                metrics=action.get('after',incumbent_metrics);improved=action.get('accepted',False)
            else:
                metrics=search.measure(layout,points)
                improved=search.rank(metrics,args.objective)<search.rank(incumbent_metrics,args.objective)
            if improved:incumbent=layout.snapshot();incumbent_metrics=metrics;stalled=0;stalled_seconds=0.
            else:
                stalled+=1;stalled_seconds+=time.monotonic()-loop_started
                if args.policy not in ('escape','restarts'):
                    rng=layout.snapshot()[2];layout.restore(incumbent);layout.rng.bit_generator.state=rng
            state['rounds'].append(dict(round=round_index,center=center,**{kind:action},
                                        improved_incumbent=improved,elapsed_seconds=time.monotonic()-t0,
                                        metrics=metrics,stalled=stalled))
            round_index+=1;commit()
            print(json.dumps(dict(round=round_index,policy=args.policy,best_ratio=incumbent_metrics['sampled_worst_ratio'],stalled=stalled)),flush=True)
            if stalled>=args.stall_rounds and stalled_seconds>=args.stall_seconds:
                state['stop_reason']='stall-limit';break
        commit();layout.restore(incumbent)
        if remaining()>2:
            result=child([sys.executable,str(Path(__file__).resolve()),'verify','--input',str(out/'best.json'),
                          '--reference',str(trial/'input.json'),'--output',str(out/'verification.json')],
                         out/'verification.log',min(45,remaining()))
            state['verification_job']=result
        state['status']='pilot-completed/incomplete'
        if (out/'verification.json').exists():
            v=json.loads((out/'verification.json').read_text());state['verification']=v
            if v['passed']:state['status']='target-geometrically-verified-review-pending'
        return state
    except KeyboardInterrupt:
        state['status']='interrupted/incomplete';commit();raise
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        children=resource.getrusage(resource.RUSAGE_CHILDREN)
        state['elapsed_seconds']=prior_elapsed+time.monotonic()-started
        state['cpu_seconds_total']=prior_cpu+time.process_time()-cpu0+children.ru_utime+children.ru_stime-children0.ru_utime-children0.ru_stime
        write(out/'state.json',state);write(out/'summary.json',state)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    v=sub.add_parser('verify');v.add_argument('--input',required=True);v.add_argument('--reference',required=True);v.add_argument('--output',required=True)
    r=sub.add_parser('run')
    for name in ('trial','output'):r.add_argument('--'+name,required=True)
    r.add_argument('--policy',choices=POLICIES,default='energy');r.add_argument('--objective',choices=['energy','worst'],default='energy')
    r.add_argument('--seed',type=int,default=1);r.add_argument('--seconds',type=float,default=180)
    r.add_argument('--bootstrap-seconds',type=float,default=30);r.add_argument('--native-angle-deg',type=float,default=0)
    r.add_argument('--patch-seconds',type=float,default=15);r.add_argument('--rounds',type=int,default=1000000)
    r.add_argument('--batch-size',type=int,default=16384);r.add_argument('--batch-score-points',type=int,default=512)
    r.add_argument('--stall-rounds',type=int,default=60);r.add_argument('--stall-seconds',type=float,default=120);r.add_argument('--start');r.add_argument('--resume-from')
    r.add_argument('--gpu-lock',default='/tmp/flagstone-quarter2-gpu.lock')
    r.add_argument('--native-python',default=str(Path.home()/'miniconda3/envs/patio-packingsolver/bin/python'))
    args=p.parse_args()
    if args.command=='verify':print(json.dumps(verify(args.input,args.reference,args.output)));return
    if (not all(math.isfinite(v) and v>0 for v in (args.seconds,args.bootstrap_seconds,args.patch_seconds,args.stall_seconds))
            or args.rounds<0 or min(args.stall_rounds,args.batch_size,args.batch_score_points)<1 or not math.isfinite(args.native_angle_deg)
            or args.start and args.resume_from):p.error('Invalid budgets or mutually exclusive start/resume')
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
    result=run(args)
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
