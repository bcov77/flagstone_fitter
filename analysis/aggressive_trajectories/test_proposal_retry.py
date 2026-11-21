"""Empty searches, numerical failures and stop/resume cannot spend attempts."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from test_aggressive import board, action, metric, runner as aggressive_runner
from common import read, write, state, attempt_files
from proposal_search import search, search_config
from selector import apply
from shapely.errors import GEOSException

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('baseline_retry_test',HERE.parent/'physics_trajectories/run.py')
baseline_runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline_runner)


def choose(pool,mode,cfg,rng):
    if not pool:return None,{}
    return pool[int(rng.integers(len(pool)))],{}


def test_search_stop_resume_matches_uninterrupted_and_logs_geometry(tmp_path):
    cfg=read(HERE/'config.json');doc=board();start=state(doc)
    seen=[];stopping=False
    def gen(d,c,rng,excluded):
        v=int(rng.integers(100000));seen.append(v)
        if len(seen)==1:raise GEOSException('test numerical failure')
        if len(seen)==2:return [],{}
        return [action(),action('c')],{}
    def progress(n): pass
    rng=np.random.default_rng(123)
    def stop():return stopping and len(seen)>=2
    stopping=True
    assert search(tmp_path,1,doc,start,cfg,'normal',rng,gen,choose,set(),stop,progress) is None
    assert read(tmp_path/'proposal_search_001/checkpoint.json')['next_round']==3
    assert read(tmp_path/'proposal_search_001/round_000001.json.gz')['reason']=='selection_geometry_failure'
    stopping=False
    resumed=search(tmp_path,1,doc,start,cfg,'normal',np.random.default_rng(123),gen,choose,set(),stop,progress)
    original_values=seen[:];seen.clear()
    uninterrupted=search(tmp_path/'other',1,doc,start,cfg,'normal',np.random.default_rng(123),gen,choose,set(),lambda:False,progress)
    assert resumed==uninterrupted and seen==original_values


def test_broadening_keeps_physical_and_edit_rules():
    cfg=read(HERE/'config.json');doc=board()
    a=search_config(cfg,doc,100)
    for k in cfg:
        if k not in ['piece_budget','ordinary_piece_budget','scaled_piece_budget','double_piece_budget','target_count','target_budget','coarse_per_target','scaled_coarse_per_target','refine_per_target','planning_pool']:
            assert a[k]==cfg[k]
    assert a['scaled_piece_budget']==len(doc['stones'])


@pytest.mark.parametrize('runner',[aggressive_runner,baseline_runner],ids=['aggressive','baseline'])
def test_empty_searches_do_not_count_and_pending_proposal_resumes(tmp_path,monkeypatch,runner):
    cfg={**read(HERE/'config.json'),'attempts_per_run':2};d=board()
    write(tmp_path/'source.json',d);write(tmp_path/'manifest.json',dict(config=cfg,start_metrics=metric(d),compute_revision='proposal_retry_v2'))
    monkeypatch.setattr(runner,'measure',metric)
    if runner is aggressive_runner:
        monkeypatch.setattr(runner,'enrich',lambda d,m:m);monkeypatch.setattr(runner,'render_run',lambda *a,**k:None)
    calls=[]
    def gen(doc,c,rng,excluded):
        calls.append(c)
        assert not attempt_files(tmp_path/'runs/run_000001')
        return ([action()] if len(calls)==4 else []),{}
    monkeypatch.setattr(runner,'generate',gen);monkeypatch.setattr(runner,'choose',choose)
    def crash(*a):raise RuntimeError('physics interruption')
    monkeypatch.setattr(runner,'relax',crash)
    with pytest.raises(RuntimeError,match='physics interruption'):runner.trajectory(tmp_path,1)
    folder=tmp_path/'runs/run_000001'
    assert not attempt_files(folder)
    assert len(list((folder/'proposal_search_001').glob('round_*.json.gz')))==3
    assert read(folder/'pending_attempt.json.gz')['selection']['search_round']==4
    def relax(doc,p,c,seed):
        (tmp_path/'STOP').touch()
        return dict(accepted=False,reason='test_physics_rejection',repair_attempted=False,attempted_document=apply(doc,p))
    monkeypatch.setattr(runner,'relax',relax)
    result=runner.trajectory(tmp_path,1)
    assert result['attempts']==1 and len(calls)==4
    record=read(attempt_files(folder)[0]);assert record['chosen'] and record['end']==record['start']


@pytest.mark.parametrize('runner',[aggressive_runner,baseline_runner],ids=['aggressive','baseline'])
def test_permanently_empty_search_can_stop_without_committing(tmp_path,monkeypatch,runner):
    d=board();cfg=read(HERE/'config.json')
    write(tmp_path/'source.json',d);write(tmp_path/'manifest.json',dict(config=cfg,start_metrics=metric(d)))
    def gen(*a,**kw):
        (tmp_path/'STOP').touch();return [],{}
    monkeypatch.setattr(runner,'generate',gen);monkeypatch.setattr(runner,'choose',choose)
    result=runner.trajectory(tmp_path,1)
    assert result['attempts']==0 and result['status']=='checkpointed'
    assert not attempt_files(tmp_path/'runs/run_000001')
    assert not (tmp_path/'runs/run_000001/summary.json').exists()
