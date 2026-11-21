"""Run one hour per solver sequentially; SIGTERM stops the active child."""
import json, os, signal, subprocess, sys, time
from pathlib import Path

root=Path('examples/hourly_solver_comparison'); root.mkdir(parents=True,exist_ok=True)
source='examples/flagstone_reference/revised_25ft_inventory.json'
worker='examples/revised_solver_comparison/worker_input.json'
status=root/'status.json'; child=None
def save(state,**kwargs):
    tmp=root/'status.tmp.json';tmp.write_text(json.dumps({'state':state,'updated':time.time(),**kwargs},indent=2));os.replace(tmp,status)
def stop(signum,frame):
    global child
    if child and child.poll() is None:
        child.terminate()
    # Do not call Popen.wait() from a signal handler: it can interrupt a wait
    # already holding Popen's internal lock and deadlock the supervisor.
    save('stopped');os._exit(128+signum)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
env=os.environ.copy()
env.update(PATIO_SOLVER_GAP=str(1/24+.0001),PATIO_SOLVER_EDGE=str(1/24+.0001),PATIO_SOLVER_STRATEGY='ga')
iron_env=env.copy();iron_env.pop('PATIO_SOLVER_STRATEGY');iron_env['PATIO_COMPARISON_DIR']=str(root)
(root/'worker_input.json').write_bytes(Path(worker).read_bytes())
commands=[
 ('packingsolver',['/Users/brian/miniconda3/envs/patio-packingsolver/bin/python','scripts/long_packingsolver.py','--input',source,'--output',str(root/'packingsolver'),'--gap',str(1/24),'--edge',str(1/24),'--seconds','3600'],env),
 ('unesting',['/Users/brian/miniconda3/envs/patio-unesting/bin/python','scripts/solver_worker.py','unesting',worker,str(root/'unesting.json'),'3600','1'],env),
 ('ironnest',['.venv/bin/python','scripts/benchmark_ironnest.py','3600'],iron_env)]
for name,command,variables in commands:
    save('running',solver=name,supervisor_pid=os.getpid())
    with (root/(name+'.log')).open('a') as log:
        child=subprocess.Popen(command,env=variables,stdout=log,stderr=subprocess.STDOUT)
        code=child.wait()
    child=None
    if code:
        save('failed',solver=name,exit_code=code);sys.exit(code)
    if name=='unesting':
        code=subprocess.call(['.venv/bin/python','scripts/evaluate_external_solver.py',source,str(root/'unesting.json')])
        if code: save('failed',solver='unesting_validation',exit_code=code);sys.exit(code)
save('completed')
