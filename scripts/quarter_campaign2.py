"""Prepare/supervise versioned edge5/interior2 experiments using the tested process supervisor."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import sys
import time
import quarter_campaign as c
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from flagstone.quarter_gaps import RULE
from quarter_patio2 import POLICIES

ORIGINAL_RESULT=c.result_for
ORIGINAL_PREPARE=c.prepare


def result_for(directory):
    folder=Path(directory);run=c.read(folder/'summary.json') or c.read(folder/'state.json');v=c.read(folder/'verification.json')
    path=folder/'best.json';gaps=v.get('gaps',{});groups=gaps.get('groups',{})
    eligible=(v.get('rule')==RULE and path.is_file() and v.get('original_geometry') is True
              and v.get('reference_matched') is True and v.get('clearance_errors')==[]
              and v.get('layout_sha256')==hashlib.sha256(path.read_bytes()).hexdigest()
              and gaps.get('stable') is True and set(groups)=={'edge','interior'})
    if eligible:
        for kind,limit in [('edge',5),('interior',2)]:
            g=groups[kind];lo,hi=g.get('lower_diameter_in'),g.get('upper_diameter_in')
            eligible &= (g.get('limit_in')==limit and isinstance(lo,(int,float)) and isinstance(hi,(int,float))
                         and math.isfinite(lo) and math.isfinite(hi) and 0<=lo<=hi)
    counts={};costs={}
    for r in run.get('rounds',[]):
        for name in ('continuous','replacement','insertion','rebuilt'):
            action=r.get(name)
            if action is None:continue
            counts[name+'_attempted']=counts.get(name+'_attempted',0)+1
            counts[name+'_accepted']=counts.get(name+'_accepted',0)+bool(action.get('accepted'))
            costs[name]=costs.get(name,0)+r.get('elapsed_seconds',0)
    lo=max((groups[k]['lower_diameter_in']/limit for k,limit in [('edge',5),('interior',2)]),default=0) if eligible else None
    hi=max((groups[k]['upper_diameter_in']/limit for k,limit in [('edge',5),('interior',2)]),default=0) if eligible else None
    # Legacy supervisor stores generic parent bounds under these field names.
    # For v2 they are DIMENSIONLESS normalized violations, always labeled in report.
    gap=dict(lower_diameter_in=lo,upper_diameter_in=hi) if eligible else None
    return dict(run_status=run.get('status','missing-result'),initial=run.get('initial'),metrics=run.get('best') or run.get('initial'),
                eligible_parent=bool(eligible),verified=bool(eligible and v.get('passed') and hi<=1),gap=gap,groups=groups if eligible else {},
                layout=str(path) if eligible else None,rounds=len(run.get('rounds',[])),cpu_seconds_total=run.get('cpu_seconds_total'),
                action_counts=counts,action_seconds=costs,policy=run.get('policy'),geometry_sha256=run.get('geometry_sha256'),
                stop_reason=run.get('stop_reason'),method=gaps.get('certification'))


def unique(attempts):
    result=[];seen=set()
    for a in attempts:
        key=a['result'].get('geometry_sha256') or a['result'].get('layout')
        if key not in seen:seen.add(key);result.append(a)
    return result


def ranking(state,adaptive=False):
    policies={}
    for a in c.parents(state):
        if a['task']['phase']=='screen':policies.setdefault(a['result']['policy'],[]).append(a['result']['gap']['upper_diameter_in'])
    if adaptive:
        def value(policy):
            runs=[a for a in c.parents(state) if a['result']['policy']==policy and a['task']['phase']!='bootstrap']
            improvement=sum(max(0,(a.get('parent_lower_in') or 0)-a['result']['gap']['upper_diameter_in']) for a in runs)
            hours=sum(a['result'].get('cpu_seconds_total') or 0 for a in runs)/3600
            return (-improvement/max(hours,1e-9),statistics.median(policies[policy]),policy)
        return sorted(policies,key=value)
    return sorted(policies,key=lambda p:(statistics.median(policies[p]),p))


def choose_parent(task,state):
    pool=unique(c.parents(state))
    if task['parent_rule']=='cold':return None
    if task['parent_rule']=='matched-bootstrap':
        pool=[a for a in c.parents(state) if a['task']['phase']=='bootstrap' and a['task']['seed']==task['seed']]
    elif task['phase']=='deepen':
        policies=ranking(state)
        if not policies:return None
        selected=policies[task['finalist']%len(policies)]
        task['resolved_policy']=selected
        pool=[a for a in pool if a['result']['policy']==selected and a['task']['phase']=='screen']
        pool.sort(key=lambda a:a['result']['gap']['upper_diameter_in'])
        return pool[task['replica']%len(pool)] if pool else None
    elif task['phase']=='extension':
        offset=task['number']%10
        if offset<2:
            # Explicit 20% cold restarts, while retaining a parent for lineage.
            task['resolved_policy']='restarts';task['force_cold']=True
        elif offset<4:
            available=ranking(state,adaptive=True)
            if available:task['resolved_policy']=available[(task['seed']+offset)%len(available)]
            pool.sort(key=lambda a:a['result']['gap']['upper_diameter_in'],reverse=True)
            return pool[task['seed']%min(12,len(pool))] if pool else None
        else:
            policies=ranking(state,adaptive=True)
            if policies:task['resolved_policy']=policies[(task['seed']+offset)%min(6,len(policies))]
        # Keep distinct Pareto parents; rotate rather than clone one winner.
        pool=c.pareto(pool)
        pool.sort(key=lambda a:a['result']['gap']['upper_diameter_in'])
        return pool[(task['seed']+offset)%len(pool)] if pool else None
    return min(pool,key=lambda a:a['result']['gap']['upper_diameter_in']) if pool else None


def command_for(out,plan,record,parent):
    task=record['task'];policy=task.get('resolved_policy',task['parameters'].get('policy','energy'))
    command=[plan['python_executable'],'-u',str(c.active_snapshot(out)/'scripts/quarter_patio2.py'),'run',
             '--trial',str(out/'trial'),'--output',record['directory'],'--native-python',plan['native_python'],
             '--seed',str(task['seed']),'--seconds',str(max(.1,record['budget_seconds']-5)),
             '--policy',policy,'--objective','worst' if policy=='worst' else 'energy',
             '--rounds',str(task.get('rounds',0 if task['phase']=='bootstrap' else 1000000)),
             '--patch-seconds',str(task['parameters'].get('patch_seconds',15)),
             '--batch-size',str(task['parameters'].get('batch_size',16384)),
             '--batch-score-points',str(task['parameters'].get('batch_score_points',512)),
             '--gpu-lock',str(out/'gpu.lock'),'--bootstrap-seconds',str(task['parameters'].get('bootstrap_seconds',min(120,max(1,record['budget_seconds']*.5)))),
             '--native-angle-deg',str(task['parameters'].get('native_angle_deg',(task['seed']*17)%90))]
    if record.get('resume_from'):command+=['--resume-from',record['resume_from']]
    elif task.get('saved_start'):command+=['--start',str(out/task['saved_start'])]
    elif parent and not task.get('force_cold'):command+=['--start',parent['result']['layout']]
    return command


def write_report(out,state,plan):
    rows=[]
    for a in state['attempts']:
        r=a.get('result',{});groups=r.get('groups',{});metrics=r.get('metrics') or {}
        row=dict(task=a['number'],phase=a['task']['phase'],policy=r.get('policy') or a['task'].get('resolved_policy',a['task']['variant']),
                 seed=a['task']['seed'],parent=a.get('parent_number'),outcome=a.get('outcome'),
                 cpu_seconds=r.get('cpu_seconds_total'),elapsed_seconds=a.get('elapsed_seconds'),
                 checked=r.get('eligible_parent',False),passed=r.get('verified',False),
                 normalized_upper=(r.get('gap') or {}).get('upper_diameter_in'),
                 edge_upper_in=groups.get('edge',{}).get('upper_diameter_in'),
                 interior_upper_in=groups.get('interior',{}).get('upper_diameter_in'),
                 coverage=metrics.get('coverage'),placed=metrics.get('placed'),rounds=r.get('rounds'),
                 stop_reason=r.get('stop_reason'),geometry_sha256=r.get('geometry_sha256'),directory=a['directory'])
        row.update(r.get('action_counts',{}));rows.append(row)
    fields=list(rows[0]) if rows else ['task','phase','policy','outcome','edge_upper_in','interior_upper_in','normalized_upper']
    fields=list(dict.fromkeys([*fields,*[k for row in rows for k in row]]))
    c.csv_write(out/'tasks.csv',rows,fields)
    aggregates=[]
    for policy in sorted({r['policy'] for r in rows}):
        rr=[r for r in rows if r['policy']==policy];screen=[r['normalized_upper'] for r in rr if r['phase']=='screen' and r['checked']]
        good=[r for r in rr if r['checked']]
        aggregates.append(dict(policy=policy,tasks=len(rr),checked=len(good),passes=sum(bool(r['passed']) for r in rr),
                               best_normalized_upper=min((r['normalized_upper'] for r in good),default=None),
                               matched_screen_median=statistics.median(screen) if screen else None,
                               cpu_hours=sum(r['cpu_seconds'] or 0 for r in rr)/3600))
    c.csv_write(out/'variants.csv',aggregates,list(aggregates[0]) if aggregates else ['policy','tasks','checked','passes'])
    frontier=c.pareto(unique(c.parents(state)))
    analysis=dict(status=state['status'],rule=RULE,tasks=rows,variants=aggregates,
                  pareto_task_numbers=[a['number'] for a in frontier],validated_successes=[r for r in rows if r['passed']],
                  max_concurrent_tasks=state.get('max_concurrent_tasks',plan['max_concurrent_tasks']),
                  peak_active_tasks=state.get('peak_active_tasks',0),updated_utc=c.utc(),
                  bound_method='Dual-scale segment-Voronoi numerical tolerance; not interval-arithmetic proof')
    c.atomic_json(out/'analysis_summary.json',analysis)
    lines=['# Second quarter-patio campaign','',f"Status: **{state['status']}**. Updated {analysis['updated_utc']}.",'',
           'Locally largest empty circles: edge diameter ≤5 in; interior diameter ≤2 in. Half-inch clearances.',
           'Both maxima are independently recomputed from original polygon segments at two integer scales.',
           'Reported bounds include a declared 0.002-inch numerical margin; this is not an interval-arithmetic proof.',
           'Normalized upper = max(edge upper / 5, interior upper / 2); passing requires ≤1 and original-clearance checks.',
           '',f"CPU workers: {analysis['max_concurrent_tasks']}; peak: {analysis['peak_active_tasks']}. GPU actions share one lock.",'',
           '| Policy | Tasks | Checked | Passes | Best normalized upper | Matched screen median | CPU hours |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in aggregates:
        fmt=lambda x:f'{x:.4f}' if x is not None else '—'
        lines.append(f"| {r['policy']} | {r['tasks']} | {r['checked']} | {r['passes']} | {fmt(r['best_normalized_upper'])} | {fmt(r['matched_screen_median'])} | {r['cpu_hours']:.2f} |")
    lines+=['','## Distinct saved tradeoffs','']
    for a in frontier:
        r=a['result'];g=r['groups']
        lines.append(f"- Task {a['number']} ({r['policy']}): edge {g['edge']['upper_diameter_in']:.4f} in; interior {g['interior']['upper_diameter_in']:.4f} in. [Layout]({os.path.relpath(r['layout'],out)}).")
    lines+=['','Only screening rows compare shared starts. Later selection is adaptive. Missing/stale/unstable checks cannot pass.',
            'Some policy combinations are bundled engineering experiments. Human measurement/construction review remains pending.','']
    (out/'report.tmp').write_text('\n'.join(lines));(out/'report.tmp').replace(out/'report.md')


def prepare(out,trial,hours=20,cores=10,starts=(),gpu=True,native_python=None):
    from flagstone.phase1 import load_json,validate
    from flagstone.quarter import check_reference,GAP
    reference=load_json(Path(trial)/'input.json')
    for path in starts:
        supplied=load_json(path);check_reference(supplied,reference)
        if validate(supplied,gap=GAP,edge=GAP):raise ValueError('Saved campaign start has invalid clearances')
    out=Path(out).absolute();plan=ORIGINAL_PREPARE(out,trial,hours=hours,cores=cores,native_python=native_python)
    for name in ('quarter_campaign2.py','quarter_patio2.py','quarter_native2.py','benchmark_quarter_gpu.py','run_quarter_campaign2.sh'):
        shutil.copy2(ROOT/'scripts'/name,out/'snapshot/scripts'/name)
    contract=c.read(out/'trial/contract.json');contract.update(version='quarter-final-layout-v2',gap_rule=RULE,
        edge_circle_diameter_limit_in=5,interior_circle_diameter_limit_in=2,gap_circle_definition='locally largest; flat medial plateaus retained',
        edge_relaxation=True,numerical_tolerance_in=.002)
    contract.pop('maximum_empty_circle_diameter_strictly_below_in',None);c.atomic_json(out/'trial/contract.json',contract)
    (out/'starts').mkdir()
    saved=[]
    for i,path in enumerate(starts):
        target=out/f'starts/start_{i:02d}.json';shutil.copy2(path,target);saved.append(str(target.relative_to(out)))
    policies=[p for p in POLICIES if gpu or p!='batch_gpu']
    tasks=[]
    # Two checked historical parents and four fresh parents enter the matched screen.
    seeds=[101,202,303,404,505,606]
    for i in range(10):
        task=dict(number=len(tasks),phase='bootstrap',variant='bootstrap',seed=seeds[i] if i<6 else 707+i,
                  seconds=1200,parameters=dict(policy='energy',native_angle_deg=i*9),parent_rule='cold')
        if i<len(saved):task['saved_start']=saved[i]
        tasks.append(task)
    for seed in seeds:
        for policy in policies:
            tasks.append(dict(number=len(tasks),phase='screen',variant=policy,seed=seed,seconds=900,
                              parameters=dict(policy=policy),parent_rule='matched-bootstrap'))
    for finalist in range(6):
        for replica in range(5):
            tasks.append(dict(number=len(tasks),phase='deepen',variant=f'finalist_{finalist}',seed=900+replica,
                              finalist=finalist,replica=replica,seconds=3600,parameters={},parent_rule='own-best-certified-gap'))
    plan.update(version='quarter-campaign-v4-local-circles',worker_script='quarter_patio2.py',
                variants={p:dict(policy=p) for p in policies},tasks=tasks,allow_extensions=True,
                extension_batch_size=cores,extension_seconds=1200,
                final_reserve_seconds=min(300,hours*3600*.05),
                acceptance=RULE+'; complete dual-scale numerical audit and exact original clearances',
                gpu_enabled=gpu,gap_rule=RULE,gpu_batch_size=16384,batch_score_points_target=512,proposal='12-policy screen, replicated six finalists, adaptive 20/20/60 restart/diversity/finalist allocation',
                minimum_free_bytes=20*1024**3,numerical_tolerance_in=.002,metric_label='normalized gap ratio',metric_units='')
    c.atomic_json(out/'plan.json',plan)
    if (ROOT/'quarter_campaign2.md').exists():shutil.copy2(ROOT/'quarter_campaign2.md',out/'README.md')
    manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in (out/'snapshot',out/'trial',out/'starts') for p in folder.rglob('*') if p.is_file()}
    manifest['plan.json']=hashlib.sha256((out/'plan.json').read_bytes()).hexdigest()
    if (out/'README.md').exists():manifest['README.md']=hashlib.sha256((out/'README.md').read_bytes()).hexdigest()
    c.atomic_json(out/'manifest.json',manifest)
    write_report(out,c.read(out/'state.json'),plan);return plan


def install_hooks():
    c.result_for=result_for;c.choose_parent=choose_parent;c.write_report=write_report;c.command_for=command_for


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--trial',default=str(ROOT/'examples/quarter_patio_v1'));p.add_argument('--hours',type=float,default=20)
    p.add_argument('--cores',type=int);p.add_argument('--start',action='append',default=[])
    p.add_argument('--no-gpu',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--native-python');p.add_argument('--report-only',action='store_true');p.add_argument('--heartbeat',type=float,default=60)
    args=p.parse_args();out=Path(args.output).absolute()
    if args.cores is not None:c.validate_cores(args.cores)
    if not math.isfinite(args.heartbeat) or args.heartbeat<=0:p.error('Heartbeat must be positive')
    install_hooks()
    if args.report_only:
        if c.read(out/'plan.json').get('version')!='quarter-campaign-v4-local-circles':p.error('Use the historical launcher for an old campaign')
        c.validate_frozen(out);write_report(out,c.read(out/'state.json'),c.read(out/'plan.json'));return
    if args.resume:
        c.validate_frozen(out)
        if c.read(out/'plan.json').get('version')!='quarter-campaign-v4-local-circles':p.error('Use the historical launcher for an old campaign')
    else:
        for message in c.preflight(args.native_python):print(message,flush=True)
        if not args.no_gpu:
            import mlx.core as mx
            if not mx.metal.is_available():raise RuntimeError('Metal GPU unavailable; use --no-gpu')
            print('Metal GPU preflight OK',flush=True)
        prepare(out,args.trial,args.hours,args.cores or min(10,os.cpu_count() or 1),args.start,not args.no_gpu,args.native_python)
    if args.dry_run:
        plan=c.read(out/'plan.json');print(f"PREPARED ONLY: {len(plan['tasks'])} initial tasks, {len(plan['variants'])} variants, {plan['hours']} hours, {args.cores or plan['max_concurrent_tasks']} workers; no workers started");return
    frozen=out/'snapshot/scripts/quarter_campaign2.py'
    if Path(__file__).absolute()!=frozen:
        command=[sys.executable,'-u',str(frozen),'--output',str(out),'--resume','--heartbeat',str(args.heartbeat)]
        if args.cores is not None:command+=['--cores',str(args.cores)]
        os.execv(sys.executable,command)
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
    raise SystemExit(c.supervise(out,cores=args.cores,heartbeat=args.heartbeat))


if __name__=='__main__':main()
