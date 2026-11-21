"""Selection, full-size physics, persistent champions and recovery regressions."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'physics_trajectories')]
from common import read,write,state,restore,clearance,attempt_path,attempt_files
from aggressive_geometry import eligible,board_key
from aggressive_policy import probabilities,choose,action_key
from aggressive_physics import relax
from aggressive_archive import archive
from selector import apply
spec=importlib.util.spec_from_file_location('aggressive_runner_test',HERE/'run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

@pytest.fixture
def cfg():return read(HERE/'config.json')

def board():
    def square(s,x,y,active=True,side=.3):return dict(id=s,outline=[[-side/2,-side/2],[side/2,-side/2],[side/2,side/2],[-side/2,side/2]],pose=[x,y,0],active=active)
    return dict(units='ft',patio=[[0,0],[3,0],[3,3],[0,3]],stones=[square('a',.5,.5),square('b',1,1,False),square('c',2,2,False)],physics=dict(settings={},pieces={}))

def metric(d):
    return dict(area_percent=sum(s['active'] for s in d['stones']),clearance=clearance(d),gaps=dict(largest_edge_inches=1,largest_interior_inches=1,numerical_gap_pass=False),quality=dict(broad_void_sqft=1,wall_joint_inches=1,joint_inches=1))

def action(stone='b',pose=None,family='scaled',remove=None):
    return dict(family=family,physics_protocol='free_pulses',remove=remove or [],placements=[dict(stone=stone,pose=pose or [1,1,0])],tier=0,rank_key=[0])

@pytest.mark.parametrize('mode',['normal','golden','silver'])
def test_family_mass_and_conditional_best_sets(cfg,mode):
    pool=[]
    for f in cfg['family_weights']:
        for i in range(8):pool.append(dict(family=f,tier=0 if i<4 else 2,rank_key=[0 if i<4 else 2,i]))
    p,d=probabilities(pool,mode,cfg)
    assert sum(p)==pytest.approx(1)
    for f,w in cfg['family_weights'].items():assert sum(p[i] for i,v in enumerate(pool) if v['family']==f)==pytest.approx(w)
    if mode=='golden':assert all(p[i]==0 for i,v in enumerate(pool) if v['tier']==2)
    selected,record=choose(pool,mode,cfg,np.random.default_rng(7))
    assert record['probability']==p[record['index']]
    assert selected['family']==record['selected_family']

def test_missing_family_fallback_is_renormalized_and_logged(cfg):
    pool=[dict(family='ordinary',tier=0,rank_key=[0]),dict(family='scaled',tier=0,rank_key=[0])]
    p,_=probabilities(pool,'normal',cfg)
    assert p.tolist()==pytest.approx([4/9,5/9])
    _,r=choose(pool,'normal',cfg,np.random.default_rng(0));assert r['unavailable_families']==['compound']
    assert choose([],'normal',cfg,np.random.default_rng(0))[0] is None

@pytest.mark.parametrize('degrees,expected',[(0,False),(44.99,False),(45,True),(315,True),(316,False)])
def test_45_degree_rule_uses_circular_rotation(cfg,degrees,expected):
    assert eligible(board(),action('a',[.6,.5,math.radians(degrees)],remove=['a']),cfg)==expected

def test_locks_no_downsizing_and_duplicate_inventory(cfg):
    d=board();a=action(remove=['a']);assert eligible(d,a,cfg)
    d['physics']['pieces']['a']=dict(locked=True);assert not eligible(d,a,cfg)
    d['physics']['pieces']['a']['locked']=False
    d['stones'][1]['outline']=[[0,0],[.01,0],[0,.01]];assert not eligible(d,a,cfg)
    a['remove']=[];a['placements']*=2;assert not eligible(d,a,cfg)

def test_cache_survives_restoration_and_permits_changed_context():
    d=board();d['stones'][1]['pose']=[9,9,2];restored=restore(board(),state(d));a=action()
    assert board_key(d)==board_key(restored) and action_key(d,a)==action_key(restored,a)
    restored['stones'][0]['pose'][0]+=.1;assert action_key(d,a)!=action_key(restored,a)

def test_free_pulse_real_physics_preserves_full_inventory_and_settles(cfg):
    d=board();a=action();before=copy.deepcopy(d)
    quick={**cfg,'pulse_seconds':[.05,.05],'quiet_max_seconds':2,'quiet_sample_seconds':.25}
    r=relax(d,a,quick,7)
    assert r['accepted'] and r['settled'] and r['clearance']['valid']
    assert r['final_motion']['max_speed_ips']<cfg['max_speed_ips']
    assert [s['outline'] for s in r['attempted_document']['stones']]==[s['outline'] for s in d['stones']]
    assert d==before

def test_aggressive_failure_runs_bounded_repair_and_reports_raw_endpoint(cfg):
    quick={**cfg,'pulse_seconds':[.01],'quiet_max_seconds':.02,'quiet_sample_seconds':.01,'repair_inflate_seconds':.01,'repair_deflate_seconds':.01}
    r=relax(board(),action(pose=[-.8,1,0]),quick,7)
    assert not r['accepted'] and r['repair_attempted']
    assert [p['name'] for p in r['phases']]==['free_pulses','direct','gap_cycle']
    assert not clearance(r['attempted_document'])['valid']

def test_compound_removal_retains_unique_full_size_inventory(cfg):
    d=board();d['stones'][1].update(active=True,pose=[1,1,0]);d['stones'][2]['outline']=[[-.3,-.3],[.3,-.3],[.3,.3],[-.3,.3]]
    a=action('c',[1.5,1.5,0],family='compound',remove=['a','b'])
    assert eligible(d,a,cfg)
    r=relax(d,a,{**cfg,'pulse_seconds':[.01],'quiet_max_seconds':2,'quiet_sample_seconds':.25},4)
    assert r['accepted']
    assert {s['id'] for s in r['attempted_document']['stones'] if s['active']}=={'c'}

def test_archive_preserves_champion_after_later_worse_endpoint(tmp_path):
    d=board();m=metric(d);archive(tmp_path,d,0,m)
    better=apply(d,action());bm=metric(better);archive(tmp_path,better,1,bm)
    worse=copy.deepcopy(d);worse['stones'][0]['pose'][0]+=.1;wm=metric(worse);wm['quality']['broad_void_sqft']=2;archive(tmp_path,worse,2,wm)
    a=read(tmp_path/'archive.json');assert a['best_coverage']['attempt']==1
    assert state(read(tmp_path/'best_coverage.json'))==state(better)

def test_pending_resume_rollback_and_finished_replay(tmp_path,cfg,monkeypatch):
    d=board();cfg={**cfg,'attempts_per_run':3};write(tmp_path/'source.json',d);write(tmp_path/'manifest.json',dict(config=cfg,start_metrics=metric(d)))
    monkeypatch.setattr(runner,'measure',metric);monkeypatch.setattr(runner,'enrich',lambda d,m:m);monkeypatch.setattr(runner,'render_run',lambda *a,**k:None)
    calls=[]
    def gen(doc,cfg,rng,excluded=()):calls.append(set(excluded));return [action()],{}
    monkeypatch.setattr(runner,'generate',gen)
    phase={'crash':True}
    def simulate(doc,p,cfg,seed):
        if phase['crash']:phase['crash']=False;raise RuntimeError('Interrupted after pending selection')
        (tmp_path/'STOP').touch()
        return dict(accepted=False,reason='test_rollback',repair_attempted=False,attempted_document=apply(doc,p))
    monkeypatch.setattr(runner,'relax',simulate)
    with pytest.raises(RuntimeError,match='Interrupted'):runner.trajectory(tmp_path,9)
    assert len(calls)==1 and (tmp_path/'runs/run_000009/pending_attempt.json.gz').exists()
    r=runner.trajectory(tmp_path,9);assert r['status']=='checkpointed' and r['attempts']==1 and len(calls)==1
    first=read(attempt_path(tmp_path/'runs/run_000009',1));assert first['end']==first['start'] and first['attempted_end']!=first['end']
    write(tmp_path/'runs/run_000009/pending_attempt.json.gz',{k:first[k] for k in ['run','number','start','chosen']})
    (tmp_path/'STOP').unlink();runner.trajectory(tmp_path,9)
    assert len(calls)==2 and action_key(d,action()) in calls[-1]
    (tmp_path/'STOP').unlink();done=runner.trajectory(tmp_path,9)
    assert done['attempts']==3 and done['mode']=='golden'
    assert len(attempt_files(tmp_path/'runs/run_000009'))==3
    count=len(calls);runner.trajectory(tmp_path,9);assert len(calls)==count

def test_config_rejects_unsafe_rules(cfg):
    runner.validate_config(cfg)
    for override in [dict(minimum_reorientation_degrees=44),dict(minimum_replacement_area_ratio=.95),dict(family_weights={'ordinary':1}),dict(probe_max_intrusion_fraction=.3)]:
        with pytest.raises(ValueError):runner.validate_config({**cfg,**override})
