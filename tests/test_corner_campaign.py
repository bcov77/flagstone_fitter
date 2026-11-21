"""Campaign deadlines, crash isolation, reattachment and harder contracts."""
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import corner_campaign as campaign
import corner_growth as engine


def test_wall_runs_one_final_retry_with_amended_check_intervals(tmp_path,monkeypatch):
    elapsed=[1000.]
    monkeypatch.setattr(campaign.time,'time',lambda:elapsed[0])
    calls=[]
    def fake_child(spec_path,record,deadline,save,event,observe,poll_seconds):
        calls.append(campaign.read(spec_path)['policy'])
        elapsed[0]=deadline
        record['result']=dict(outcome='completed',state=dict(status='budget-exhausted/incomplete',highest_verified=5))
        return 'completed'
    monkeypatch.setattr(campaign,'supervise_child',fake_child)
    state=campaign.campaign(tmp_path/'campaign',initial_seconds=7200,final_seconds=14400)
    assert len(calls)==2
    assert [r['check_interval_seconds'] for r in state['phases']]==[2400,4800]
    assert [r['budget_seconds'] for r in state['phases']]==[7200,14400]
    assert state['final_retry_used'] and state['status']=='finished-after-final-wall-retry'
    assert all(p['repair_rounds']>3 for p in calls)


def test_worker_crash_does_not_end_campaign(tmp_path,monkeypatch):
    elapsed=[1000.];calls=[]
    monkeypatch.setattr(campaign.time,'time',lambda:elapsed[0])
    monkeypatch.setattr(campaign.time,'sleep',lambda _:None)
    def child(spec,record,deadline,*args):
        calls.append(str(spec))
        if len(calls)==1:
            record['returncode']=-6
            return 'crashed'
        elapsed[0]=deadline
        record['returncode']=0
        record['result']=dict(outcome='completed',state=dict(status='budget-exhausted/incomplete',highest_verified=5))
        return 'completed'
    monkeypatch.setattr(campaign,'supervise_child',child)
    state=campaign.campaign(tmp_path/'run',initial_seconds=500,final_seconds=500)
    assert state['attempts'][0]['outcome']=='crashed'
    assert state['attempts'][1]['outcome']=='completed'
    assert len(set(calls))==len(calls)==3
    assert state['status']=='finished-after-final-wall-retry'


def test_real_subprocess_failure_is_logged_without_raising(tmp_path):
    launcher=tmp_path/'crash.py';launcher.write_text('raise RuntimeError("injected worker crash")\n')
    spec=tmp_path/'job.json'
    campaign.atomic_json(spec,dict(output=str(tmp_path/'result'),result_file=str(tmp_path/'result.json')))
    record=dict(worker_script=str(launcher),log=str(tmp_path/'child.log'))
    outcome=campaign.supervise_child(spec,record,time.time()+5,lambda:None,lambda *a,**k:None,lambda _:None,.01)
    assert outcome=='crashed' and record['returncode']==1
    assert 'injected worker crash' in Path(record['log']).read_text()


def test_resume_adopts_live_child_without_duplicate_launch(tmp_path,monkeypatch):
    launcher=tmp_path/'worker.py'
    launcher.write_text('import sys,json,time,pathlib\np=pathlib.Path(sys.argv[2]); d=json.loads(p.read_text()); time.sleep(.2); pathlib.Path(d["result_file"]).write_text(json.dumps({"outcome":"completed"}))\n')
    spec=tmp_path/'job.json'
    campaign.atomic_json(spec,dict(output=str(tmp_path/'out'),result_file=str(tmp_path/'result.json')))
    child=subprocess.Popen([sys.executable,str(launcher),'--spec',str(spec)],start_new_session=True)
    record=dict(pid=child.pid,process_identity=campaign.identity(child.pid),worker_script=str(launcher),log=str(tmp_path/'child.log'))
    def duplicate(*a,**k):raise AssertionError('Duplicate child launch')
    monkeypatch.setattr(campaign.subprocess,'Popen',duplicate)
    # identity uses subprocess.run, which itself needs Popen; keep that inventory
    # operation independent of the patched launch operation.
    ps_identity=record['process_identity']
    monkeypatch.setattr(campaign,'matching_process',lambda *a,**k:child.poll() is None)
    outcome=campaign.supervise_child(spec,record,time.time()+5,lambda:None,lambda *a,**k:None,lambda _:None,.01)
    child.wait()
    assert outcome=='completed'


def test_extended_contract_preserves_original_regions_and_gates(tmp_path):
    source=Path('examples/corner_01').resolve()
    contract=campaign.extend_contract(source,tmp_path/'extension')
    _,payload,_=engine.load_trial(contract)
    original=campaign.read(source/'contract.json')
    assert payload['prefix_regions'][:13]==original['prefix_regions']
    assert payload['minimum_clearance_in']==.5
    assert payload['maximum_empty_circle_diameter_strictly_below_in']==2
    assert engine.side(15)>engine.side(13)
    payload['prefix_regions'][0]['width_ft']+=.001
    campaign.atomic_json(Path(contract)/'contract.json',payload)
    with pytest.raises(ValueError):engine.load_trial(contract)


def test_crashed_child_cleans_up_only_its_recorded_native_worker(tmp_path):
    run=tmp_path/'run';run.mkdir()
    native_spec=tmp_path/'native_job.json';native_spec.write_text('{}')
    launcher=tmp_path/'launcher.py'
    launcher.write_text('''import subprocess,sys,json,pathlib,os
spec=json.loads(pathlib.Path(sys.argv[2]).read_text())
run=pathlib.Path(spec['output'])
job=run.parent/'native_job.json'
p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',str(job)],start_new_session=True)
(run/'state.json').write_text(json.dumps({'jobs':[{'outcome':'running','pid':p.pid,'job':str(job)}]}))
os._exit(9)
''')
    spec=tmp_path/'attempt.json'
    campaign.atomic_json(spec,dict(output=str(run),result_file=str(tmp_path/'result.json')))
    record=dict(worker_script=str(launcher),log=str(tmp_path/'worker.log'))
    outcome=campaign.supervise_child(spec,record,time.time()+5,lambda:None,lambda *a,**k:None,lambda _:None,.01)
    native=campaign.read(run/'state.json')['jobs'][0]
    assert outcome=='crashed' and record['returncode']==9
    # A killed orphan may briefly be a zombie before the OS reaps it; its
    # command must no longer identify a live native worker.
    for _ in range(30):
        if not campaign.matching_process(native['pid'],native['job']):break
        time.sleep(.01)
    assert not campaign.matching_process(native['pid'],native['job'])
    assert campaign.read(run/'state.json')['jobs'][0]['outcome']=='terminated-after-parent-exit'


def test_malformed_result_is_a_child_failure(tmp_path):
    launcher=tmp_path/'worker.py'
    launcher.write_text('import sys,json,pathlib\ns=json.loads(pathlib.Path(sys.argv[2]).read_text()); pathlib.Path(s["result_file"]).write_text("{broken")\n')
    spec=tmp_path/'spec.json'
    campaign.atomic_json(spec,dict(output=str(tmp_path/'out'),result_file=str(tmp_path/'result.json')))
    record=dict(worker_script=str(launcher),log=str(tmp_path/'log'))
    outcome=campaign.supervise_child(spec,record,time.time()+5,lambda:None,lambda *a,**k:None,lambda _:None,.01)
    assert outcome=='crashed'
    assert 'Unreadable child result' in record['result']['error']
