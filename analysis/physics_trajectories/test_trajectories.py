"""Behavioral checks for adaptive choices, attempts, rollback, resume and physics."""
import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import clearance,state,restore,write,read,attempt_path,attempt_files
import selector
import run
from physics import relax

@pytest.fixture
def cfg():return json.loads((Path(__file__).parent/'config.json').read_text())

def square(sid,x,y,active=True,side=.3):
    return dict(id=sid,outline=[[-side/2,-side/2],[side/2,-side/2],[side/2,side/2],[-side/2,side/2]],pose=[x,y,0],active=active)

def board():return dict(units='ft',patio=[[0,0],[3,0],[3,3],[0,3]],stones=[square('a',.5,.5),square('b',1,1,False),square('c',2,2,False)])

def test_every_ten_has_eight_normal_one_gold_one_silver():
    for start in [1,11,101]:
        modes=[run.mode_for(i) for i in range(start,start+10)]
        assert modes.count('normal')==8 and modes.count('golden')==modes.count('silver')==1

def test_selection_exact_silver_mixture_and_golden_diversity(cfg):
    pool=[dict(tier=0) for _ in range(5)]+[dict(tier=3) for _ in range(15)]
    normal,best=selector.probabilities(pool,'normal',cfg);gold,_=selector.probabilities(pool,'golden',cfg);silver,_=selector.probabilities(pool,'silver',cfg)
    assert len(best)>1 and not any(gold[5:])
    assert np.allclose(silver,.8*gold+.2*normal)
    assert normal[:5].sum()>.7 and normal[-1]>0
    for v in [normal,gold,silver]:assert sum(v)==pytest.approx(1)

def test_best_tier_limits_golden_even_when_fewer_than_three(cfg):
    p,_=selector.probabilities([dict(tier=0),dict(tier=2),dict(tier=3)],'golden',cfg)
    assert p.tolist()==[1,0,0]

def test_double_distinct_inventory_and_apply_preserves_input():
    d=board();before=copy.deepcopy(d);p=dict(remove=['a'],placements=[dict(stone='b',pose=[1,1,0]),dict(stone='c',pose=[2,2,0])])
    after=selector.apply(d,p)
    assert d==before and {s['id'] for s in after['stones'] if s['active']}=={'b','c'}
    assert clearance(after)['valid'];assert restore(d,state(after))=={**after,'physics':{}}

def test_clearance_rejects_nonoverlap_below_half_inch():
    d=board();d['stones'][1].update(active=True,pose=[.82,.5,0])
    c=clearance(d);assert not c['valid'] and c['overlap_or_outside_sqft']==0

def test_double_pair_gate_cannot_lock_overlapping_new_pieces(cfg):
    import shapely
    local={s['id']:shapely.Polygon(s['outline']) for s in board()['stones']};p=shapely.box(.2,.2,2.5,2.5)
    target=dict(old=None,pocket=p,free=p)
    assert selector.fit([dict(stone='b',pose=[1,1,0]),dict(stone='c',pose=[1,1,0])],target,local,{},shapely.box(0,0,3,3),cfg,min_coverage=0) is None

def test_real_physics_double_is_one_outcome_and_settles(cfg):
    d=board();p=dict(remove=[],placements=[dict(stone='b',pose=[1,1,0]),dict(stone='c',pose=[2,2,0])])
    fast={**cfg,'explore_seconds':.05,'quiet_max_seconds':2,'quiet_sample_seconds':.25,'max_speed_ips':.1,'max_angular_speed_dps':1}
    r=relax(d,p,fast,7)
    assert r['accepted'] and not r['repair_attempted'] and len(r['phases'])==1
    assert len([s for s in r['attempted_document']['stones'] if s['active']])==3
    assert all(not v['locked'] for v in r['attempted_document']['physics']['pieces'].values())

def test_invalid_physics_calls_cycle_then_returns_failure(cfg):
    d=board();p=dict(remove=[],placements=[dict(stone='b',pose=[-.8,1,0])])
    fast={**cfg,'explore_seconds':.01,'quiet_max_seconds':.02,'quiet_sample_seconds':.01,'repair_inflate_seconds':.01,'repair_deflate_seconds':.01}
    r=relax(d,p,fast,8)
    assert not r['accepted'] and r['repair_attempted'] and len(r['phases'])==2
    assert r['reason']=='clearance_failure_rolled_back'

def test_transactions_rollback_double_attempt_count_and_resume(tmp_path,cfg,monkeypatch):
    d=board();out=tmp_path;write(out/'source.json',d)
    # Tiny metrics stub isolates transaction semantics; real geometry/physics have separate checks.
    def metric(doc):return dict(area_percent=sum(s['active'] for s in doc['stones']),gaps={},clearance=clearance(doc))
    cfg={**cfg,'attempts_per_run':3};write(out/'manifest.json',dict(config=cfg,start_metrics=metric(d)))
    pool=[dict(remove=[],placements=[dict(stone='b',pose=[1,1,0]),dict(stone='c',pose=[2,2,0])],tier=0)]
    calls=[]
    def generate(doc,cfg,rng,excluded=()):calls.append(copy.deepcopy(doc));return copy.deepcopy(pool),{}
    monkeypatch.setattr(run,'generate',generate);monkeypatch.setattr(run,'measure',metric)
    def simulation(doc,p,cfg,seed):
        attempted=selector.apply(doc,p)
        return dict(accepted=False,reason='clearance_failure_rolled_back',repair_attempted=True,attempted_document=attempted)
    monkeypatch.setattr(run,'relax',simulation)
    import report
    monkeypatch.setattr(report,'render_run',lambda *args:None)
    summary=run.trajectory(out,1)
    assert summary['attempts']==3 and summary['accepted']==0 and summary['doubles_attempted']==3
    assert all(state(v)==state(d) for v in calls)
    assert state(json.loads((out/'runs/run_000001/final.json').read_text()))==state(d)
    # A finished resume does not generate any more proposals or rerun physics.
    run.trajectory(out,1);assert len(calls)==3
    first=read(attempt_path(out/'runs/run_000001',1))
    assert first['attempted_end']!=first['end'] and first['end']==first['start']

def test_stop_file_leaves_checkpoint_without_starting_attempt(tmp_path,cfg):
    d=board();write(tmp_path/'source.json',d);write(tmp_path/'manifest.json',dict(config=cfg,start_metrics={}))
    (tmp_path/'STOP').touch();r=run.trajectory(tmp_path,10)
    assert r['status']=='checkpointed' and r['attempts']==0

def test_config_rejects_impossible_settling_window(cfg):
    with pytest.raises(ValueError):run.validate_config({**cfg,'quiet_max_seconds':.1})

def test_online_band_cache_matches_frozen_assessment():
    import shapely
    from future_gap_geometry import assess as frozen,reference_stability
    from online_geometry import assess as online
    from common import shapes,GAP
    d=board();bs=shapes(d);patio=shapely.Polygon(d['patio']);p=patio.buffer(-GAP);cfg={**selector.PLAN_CONFIG,'max_filler_regions':0}
    stab=reference_stability(bs,patio,GAP,cfg);refs={}
    args=(bs,'a',refs,patio,p,patio,{},stab,GAP,cfg)
    assert online(*args)==frozen(*args)
    # Nonzero stable-neighbor scores exercise cached band intersections.
    bs['b']=shapely.box(.9,.9,1.2,1.2);stab={'b':dict(score=.8)}
    args=(bs,'a',{'b':bs['b']},patio,p,patio,{},stab,GAP,cfg)
    assert online(*args)==frozen(*args)

def test_acceptance_changes_next_board_and_empty_resume_keeps_exclusions(tmp_path,cfg,monkeypatch):
    d=board();write(tmp_path/'source.json',d)
    def metric(doc):return dict(area_percent=sum(s['active'] for s in doc['stones']),gaps={},clearance=clearance(doc))
    write(tmp_path/'manifest.json',dict(config=cfg,start_metrics=metric(d)))
    seen=[]
    def gen(doc,cfg,rng,excluded=()):
        seen.append((state(doc),set(excluded)))
        if not excluded:return [dict(remove=[],placements=[dict(stone='b',pose=[1,1,0])],tier=0)],{}
        (tmp_path/'STOP').touch()
        return [],{}
    def sim(doc,p,cfg,seed):
        (tmp_path/'STOP').touch()
        return dict(accepted=True,reason='valid_and_observed_settled',repair_attempted=False,attempted_document=selector.apply(doc,p))
    monkeypatch.setattr(run,'generate',gen);monkeypatch.setattr(run,'relax',sim);monkeypatch.setattr(run,'measure',metric)
    import report
    monkeypatch.setattr(report,'render_run',lambda *args:None)
    first=run.trajectory(tmp_path,9);assert first['attempts']==1 and first['status']=='checkpointed'
    (tmp_path/'STOP').unlink();done=run.trajectory(tmp_path,9)
    assert done['attempts']==1 and done['status']=='checkpointed'
    assert len(attempt_files(tmp_path/'runs/run_000009'))==1
    assert all('b' in frame['poses'] and 'b' in excluded for frame,excluded in seen[1:])
    assert not (tmp_path/'runs/run_000009/pending_attempt.json').exists()

def test_area_cannot_move_risk_tier_into_golden_best(cfg):
    pool=[dict(tier=0,area_gain_sqft=.01),dict(tier=3,area_gain_sqft=5)]
    probs,_=selector.probabilities(pool,'golden',cfg)
    assert probs.tolist()==[1,0]

def test_default_board_matches_editor_launcher():
    import ast
    tree=ast.parse((run.ROOT/'physics_patio/patio_physics/server.py').read_text())
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_LAYOUT' for t in n.targets))
    assert isinstance(assignment.value,ast.BinOp)
    assert run.DEFAULT_SOURCE==run.ROOT/ast.literal_eval(assignment.value.right)

def test_atomic_compressed_records_roundtrip_and_ignore_temp(tmp_path):
    payload={'reason':'Corner ½″', 'poses':{'a':[1.,2.,0.]}}
    path=attempt_path(tmp_path,1);write(path,payload)
    assert path.name=='attempt_001.json.gz' and read(path)==payload
    (tmp_path/'attempt_002.json.999.tmp').write_text('incomplete')
    assert attempt_files(tmp_path)==[path]

def test_exact_crash_fixture_uses_empty_set_identity_without_rounding():
    import shapely
    from shapely.errors import GEOSException
    from online_geometry import planning_overlay
    fixture=read(Path(__file__).parent/'fixtures/geos_overlay_failure.json.gz')
    a,b=[shapely.from_wkb(bytes.fromhex(fixture[k])) for k in ('a_wkb','b_wkb')]
    before=(a.wkb,b.wkb)
    assert a.is_valid and b.is_valid and b.is_empty
    with pytest.raises(GEOSException):shapely.intersection(a,b)
    events=[];result=planning_overlay('intersection',a,b,events,'remaining_subject_indent')
    assert result.is_empty and result.area==0 and result.is_valid
    assert events[0]['strategy']=='exact_empty_set_identity' and events[0]['grid_size_ft']==0
    assert (a.wkb,b.wkb)==before

def test_precision_fallback_is_logged_and_does_not_modify_outlines(monkeypatch):
    import shapely
    from shapely.errors import GEOSException
    from online_geometry import planning_overlay
    a,b=shapely.box(0,0,1,1),shapely.box(.5,.5,2,2);before=(a.wkb,b.wkb);exact=shapely.intersection
    def flaky(x,y,**kwargs):
        if 'grid_size' not in kwargs:raise GEOSException('synthetic overlay failure')
        return exact(x,y,**kwargs)
    monkeypatch.setattr(shapely,'intersection',flaky);events=[]
    result=planning_overlay('intersection',a,b,events,'test')
    assert result.area==pytest.approx(.25) and events[0]['grid_size_ft']==1e-10
    assert (a.wkb,b.wkb)==before

def test_unrecoverable_overlay_never_becomes_zero_risk(monkeypatch):
    import shapely
    from shapely.errors import GEOSException
    from online_geometry import planning_overlay
    def fail(*args,**kwargs):raise GEOSException('unrecoverable')
    monkeypatch.setattr(shapely,'intersection',fail)
    with pytest.raises(GEOSException,match='exhausted bounded precision retries'):
        planning_overlay('intersection',shapely.box(0,0,1,1),shapely.box(0,0,2,2),[],'test')

def test_unassessable_selection_is_recorded_without_ending_other_runs(tmp_path,cfg,monkeypatch):
    from shapely.errors import GEOSException
    d=board();write(tmp_path/'source.json',d);write(tmp_path/'manifest.json',dict(config=cfg,start_metrics={}))
    def fail(*args,**kwargs):
        (tmp_path/'STOP').touch();raise GEOSException('planning failure')
    monkeypatch.setattr(run,'generate',fail)
    result=run.trajectory(tmp_path,1);assert result['attempts']==0
    assert not attempt_files(tmp_path/'runs/run_000001')
    r=read(tmp_path/'runs/run_000001/proposal_search_001/round_000001.json.gz')
    assert r['reason']=='selection_geometry_failure'
    assert r['diagnostics']['numerical_failure']['type']=='GEOSException'

def test_versioned_continuation_preserves_original_bytes_and_run_ids(tmp_path,cfg,monkeypatch):
    import continue_run
    from common import sha
    parent=tmp_path/'original';dest=tmp_path/'revision';d=board()
    write(parent/'source.json',d);write(parent/'runs/run_000009/run.json',dict(number=9,mode='golden'))
    record=dict(run=9,number=1,mode='golden',start=state(d),end=state(d),end_metrics={})
    path=attempt_path(parent/'runs/run_000009',1);write(path,record)
    code=tmp_path/'analysis/physics_trajectories/revisions/pre_geos_fix_20261116/compute.py';code.parent.mkdir(parents=True);code.write_text('old code')
    old=dict(source_sha256=sha(parent/'source.json'),versions={'test':'1'},dependencies={'compute.py':sha(code)},config=cfg)
    write(parent/'manifest.json',old);write(parent/'runs/run_000009/error.json',dict(error='old crash'))
    before=(path.read_bytes(),(parent/'manifest.json').read_bytes())
    monkeypatch.setattr(continue_run,'ROOT',tmp_path);monkeypatch.setattr(continue_run,'versions',lambda:{'test':'1'});monkeypatch.setattr(continue_run,'dependencies',lambda:{'compute.py':'new-code-hash'})
    new=continue_run.prepare_continuation(parent,dest)
    assert (path.read_bytes(),(parent/'manifest.json').read_bytes())==before
    assert attempt_path(dest/'runs/run_000009',1).read_bytes()==before[0]
    assert new['continuation']['total_inherited_attempts']==1 and new['dependencies']=={'compute.py':'new-code-hash'}
    assert not (dest/'runs/run_000009/error.json').exists()
    assert (dest/'runs/run_000009/parent_history/error.json').exists()
    assert new['continuation']['inherited_attempt_hashes'][str(path.relative_to(parent))]==sha(path)


def test_online_assessment_allocates_real_filler_and_matches_frozen():
    import shapely
    from future_gap_geometry import assess as frozen
    from online_geometry import assess as online
    from common import GAP
    patio=shapely.box(0,0,1,1)
    inventory={'filler':shapely.box(-.3,-.3,.3,.3)}
    cfg={**selector.PLAN_CONFIG,'filler_min_patch_fraction':.3}
    args=({},None,{},patio,patio,patio,inventory,{},GAP,cfg)
    result=online(*args)
    assert result==frozen(*args)
    assert result['repair_filler_plan']==[dict(stone='filler',pose=[.5,.5,0.])]
    assert result['metrics']['verified_fillable_regions']==1
    assert result['regions'][0]['filler']['status']=='verified_filler_pose'
    assert set(inventory)=={'filler'}


def test_continuation_supports_explicit_revision_and_nested_lineage(tmp_path,cfg,monkeypatch):
    import continue_run
    from common import sha
    parent=tmp_path/'r2';dest=tmp_path/'r3';archive=tmp_path/'archive'
    write(parent/'source.json',board());write(parent/'lineage/parent_manifest.json',{'original':True})
    archive.mkdir();code=archive/'compute.py';code.write_text('r2 code')
    write(parent/'manifest.json',dict(source_sha256=sha(parent/'source.json'),versions={},
        dependencies={'compute.py':sha(code)},config=cfg,compute_revision='r2'))
    monkeypatch.setattr(continue_run,'versions',lambda:{})
    monkeypatch.setattr(continue_run,'dependencies',lambda:{'compute.py':'r3-hash'})
    result=continue_run.prepare_continuation(parent,dest,archive,'filler_math_import_fix_r3')
    assert result['compute_revision']=='filler_math_import_fix_r3'
    assert read(dest/'lineage/parent_lineage/parent_manifest.json')=={'original':True}
    assert (dest/'lineage/parent_code/compute.py').read_bytes()==code.read_bytes()
