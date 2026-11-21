"""Foreground logging, bounded execution, and cancellation of separately owned jobs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import watch_corner_run as watch
from corner_campaign import matching_process


def fake_run(tmp_path,mode,seconds=2):
    output=tmp_path/'watched';output.mkdir()
    spec=dict(output=str(output/'run'),policy=dict(seconds=seconds),result_file=str(output/'worker.result.json'))
    watch.atomic_json(output/'run_spec.json',spec)
    worker=tmp_path/'fake_worker.py'
    worker.write_text('''import json,pathlib,sys,time,subprocess
from datetime import datetime,timezone
spec=json.loads(pathlib.Path(sys.argv[2]).read_text())
out=pathlib.Path(spec['output']);out.mkdir()
state={'status':'running','highest_verified':11,'jobs':[],'milestones':{}}
'''+('''
job=out/'job_000.json';job.write_text('{}')
log=out/'job_000.log';log.write_text('helper ready\\n')
helper=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)',str(job)],start_new_session=True)
state['jobs']=[dict(number=0,kind='generate',pid=helper.pid,job=str(job),log=str(log),outcome='running',started_utc=datetime.now(timezone.utc).isoformat())]
(out/'checkpoint.json').write_text('preserve me')
(out/'state.json').write_text(json.dumps(state))
print('controller ready',flush=True)
time.sleep(30)
''' if mode=='slow' else '''
state['status']='declared-search-exhausted/incomplete'
(out/'state.json').write_text(json.dumps(state))
print('visible worker output',flush=True)
pathlib.Path(spec['result_file']).write_text(json.dumps({'outcome':'completed','state':state}))
'''))
    return output,spec,worker


def assert_helper_stopped(output):
    state=watch.read_optional(output/'run/state.json');job=state['jobs'][0]
    for _ in range(50):
        if not matching_process(job['pid'],job['job']):break
        time.sleep(.01)
    assert not matching_process(job['pid'],job['job'])
    assert job['outcome']=='terminated-after-parent-exit'
    assert (output/'run/checkpoint.json').read_text()=='preserve me'


def test_normal_foreground_run_reports_incomplete_honestly(tmp_path):
    output,spec,worker=fake_run(tmp_path,'complete');messages=[]
    assert watch.supervise(output,spec,worker_script=worker,emit=messages.append,poll_seconds=.01)==0
    assert any('visible worker output' in m for m in messages)
    summary=watch.read_optional(output/'summary.json')
    assert summary['run_status']=='declared-search-exhausted/incomplete'
    assert summary['highest_verified_in_run']==11


def test_deadline_stops_controller_and_separately_owned_worker(tmp_path):
    output,spec,worker=fake_run(tmp_path,'slow',seconds=1);messages=[]
    assert watch.supervise(output,spec,worker_script=worker,heartbeat_seconds=.05,
                           emit=messages.append,poll_seconds=.01)==124
    assert any('HEARTBEAT' in m for m in messages)
    assert_helper_stopped(output)


def test_ctrl_c_keeps_checkpoint_and_cleans_up_helpers(tmp_path):
    output,spec,worker=fake_run(tmp_path,'slow')
    def interrupt_on_ready(message):
        if 'helper ready' in message:raise KeyboardInterrupt
    assert watch.supervise(output,spec,worker_script=worker,emit=interrupt_on_ready,poll_seconds=.01)==130
    assert watch.read_optional(output/'summary.json')['status']=='interrupted-by-user'
    assert_helper_stopped(output)


def test_tee_pipe_closing_also_stops_owned_workers(tmp_path):
    output,spec,worker=fake_run(tmp_path,'slow')
    def close_pipe(message):
        if 'helper ready' in message:raise BrokenPipeError
    assert watch.supervise(output,spec,worker_script=worker,emit=close_pipe,poll_seconds=.01)==130
    assert_helper_stopped(output)


def test_shell_dry_run_freezes_plan_and_writes_tee_transcript(tmp_path):
    output=tmp_path/'foreground'
    script=watch.REPO/'scripts/run_corner_13.sh'
    result=subprocess.run(['bash',str(script),str(output),'--dry-run'],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert 'DRY RUN' in result.stdout and 'DRY RUN' in (output/'console.log').read_text()
    spec=watch.read_optional(output/'run_spec.json')
    assert len(watch.read_optional(Path(spec['seed_directory'])/'history.json'))==11
    assert spec['policy']['seconds']==5400 and spec['policy']['seed']==7113
    assert not (output/'run').exists() and not (output/'worker.log').exists()
    original=(output/'run_spec.json').read_bytes()
    rerun=subprocess.run(['bash',str(script),str(output),'--dry-run'],capture_output=True,text=True,timeout=15)
    assert rerun.returncode!=0 and (output/'run_spec.json').read_bytes()==original


def test_tail_waits_for_complete_lines(tmp_path):
    log=tmp_path/'worker.log';log.write_text('first\npart')
    tail=watch.Tail();assert tail.lines(log)==['first']
    with log.open('a') as stream:stream.write('ial\n')
    assert tail.lines(log)==['partial']
