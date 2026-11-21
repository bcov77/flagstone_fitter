"""Try independent Ironnest runs until the wall-time budget expires."""
import json, os, subprocess, sys, time
from pathlib import Path
from flagstone.phase1 import load_json, save_json, validate

root=Path(os.environ.get('PATIO_COMPARISON_DIR','examples/revised_solver_comparison')); source=Path('examples/flagstone_reference/revised_25ft_inventory.json')
seconds=float(sys.argv[1]) if len(sys.argv)>1 else 180
started=time.monotonic(); best=0; records=[]; seed=0
while time.monotonic()-started < seconds:
    seed+=1; path=root/f'ironnest_seed{seed}.json'; remaining=seconds-(time.monotonic()-started)
    env=os.environ.copy(); env.update(PATIO_SOLVER_GAP=str(1/24+.0001),PATIO_SOLVER_BUDGET='20')
    try:
        run=subprocess.run(['/Users/brian/miniconda3/envs/patio-ironnest/bin/python','scripts/solver_worker.py',
                            'ironnest',str(root/'worker_input.json'),str(path),str(seconds),str(seed)],
                           env=env,text=True,capture_output=True,timeout=remaining)
    except subprocess.TimeoutExpired:
        records.append({'seed':seed,'status':'timed_out','remaining_seconds':remaining}); break
    if run.returncode:
        records.append({'seed':seed,'status':'error','stderr':run.stderr[-2000:]}); break
    result=json.loads(path.read_text()); layout=load_json(source); seen=set()
    for i,x,y,a in result['poses']:
        if i in seen or not 0<=i<len(layout.ids): raise ValueError('Invalid ID')
        seen.add(i); layout.active[i]=True; layout.poses[i]=[x,y,a]
    errors=validate(layout,gap=1/24,edge=1/24)
    coverage=sum(layout.shapes[s].area for i,s in enumerate(layout.ids) if layout.active[i])/layout.patio.area
    record={'seed':seed,'status':'completed','valid':not errors,'violations':errors,
            'coverage':coverage,'placed':len(seen),'elapsed_seconds':result['elapsed_seconds']}
    records.append(record)
    if not errors and coverage>best:
        best=coverage; save_json(layout,root/'ironnest_best.layout.json')
    (root/'ironnest_summary.json').write_text(json.dumps({'runs':records,'best_coverage':best},indent=2))
    print(json.dumps({k:v for k,v in record.items() if k!='violations'}),flush=True)
(root/'ironnest_summary.json').write_text(json.dumps({'runs':records,'best_coverage':best,'total_elapsed':time.monotonic()-started},indent=2))
