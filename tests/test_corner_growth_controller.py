"""Growth policy and owned-worker deadline behavior; numerical gates tested separately."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import corner_growth_controller as controller


@pytest.mark.parametrize('timeout_after_pool',[False,True])
def test_zero_cover_enriches_then_reopens_without_running_selector(tmp_path,monkeypatch,timeout_after_pool):
    jobs = []
    class CompletedWorker:
        pid = 123456
        returncode = 0
        def __init__(self, command, **kwargs):
            path = Path(command[-1]); spec = controller.read(path); jobs.append(spec)
            self.kind=spec['kind'];self.waited=False
            if spec['kind'] == 'preserve':
                destination = Path(spec['output'])/'prefix_5'
                destination.mkdir(parents=True)
                controller.atomic_json(destination/'history.json',controller.read(spec['history']))
                result = dict(target=5,passed=True)
            else:
                assert spec['kind'] in ('generate','coverage')
                if spec['kind']=='generate':
                    destination=Path(spec['output']);destination.mkdir()
                    controller.atomic_json(destination/'candidate_pool.json',dict(candidates=[]))
                result = dict(candidates=20,zero_cover=[dict(stage=7,center=[1,1])])
            controller.atomic_json(path.with_suffix('.result.json'),result)
        def wait(self,timeout=None):
            if timeout_after_pool and self.kind=='generate' and not self.waited:
                self.waited=True
                raise subprocess.TimeoutExpired('injected-after-pool',timeout)
            return 0
    monkeypatch.setattr(controller.subprocess,'Popen',CompletedWorker)
    monkeypatch.setattr(controller.os,'killpg',lambda *args:None)
    state = controller.run('examples/corner_01','examples/corner_milestone_5',tmp_path/'run',
                           controller.GrowthPolicy(seconds=30,proposals=10,native_python=None,jts_jar=None))
    assert state['highest_verified'] == 5
    assert state['status'] == 'declared-search-exhausted/incomplete'
    assert [len(controller.read(Path(j['parent'])/'history.json')) for j in jobs if j['kind']=='generate'] == [5,5,3,3,1,1]
    assert not any(j['kind']=='select' for j in jobs)
    assert len({j['output'] for j in jobs}) == len(jobs)
    if timeout_after_pool:
        assert all(j['outcome']=='timed-out' for j in state['jobs'] if j['kind']=='generate')
        assert len([j for j in state['jobs'] if j['kind']=='coverage'])==6


def test_deadline_reaps_only_owned_worker_and_records_timeout(tmp_path,monkeypatch):
    popen = subprocess.Popen
    children = []
    def slow_worker(command, **kwargs):
        process = popen([sys.executable,'-c','import time; time.sleep(30)'],**kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(controller.subprocess,'Popen',slow_worker)
    out = tmp_path/'run'
    with pytest.raises(RuntimeError,match='timed-out'):
        controller.run('examples/corner_01','examples/corner_milestone_5',out,
                       controller.GrowthPolicy(seconds=5,verify_seconds=.05,native_python=None,jts_jar=None))
    assert len(children)==1 and children[0].poll() is not None
    state = json.loads((out/'state.json').read_text())
    assert state['status']=='failed'
    assert state['jobs'][0]['outcome']=='timed-out'
    assert state['highest_verified']==0
