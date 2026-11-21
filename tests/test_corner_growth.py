"""Corner domains must not become walls; verify physical saved-file reconstruction."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon, box

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import corner_growth as corner
from flagstone.phase1 import Layout, save_json


def test_two_reporting_fronts_are_not_obstacles():
    patio=box(0,0,5,5)
    # This tiny center domain crosses neither stone nor physical boundary.
    # Treating either reporting edge as a wall would incorrectly certify it.
    result=corner.bounded_gap(patio,[box(.05,.05,.8,.8)],box(.92,.92,.94,.94))
    assert result['lower_diameter_in']>2


def test_actual_corner_frame_and_growth():
    _,contract,layout=corner.load_trial('examples/corner_01')
    assert contract['frame']['traced_interior_angle_degrees']==pytest.approx(89.43509671847862)
    previous=None
    for n in range(1,14):
        domain=corner.region(contract,n).intersection(layout.patio)
        if previous is not None:assert domain.buffer(1e-12).covers(previous)
        if n>=3 and n%2:assert corner.side(n)>corner.side(n-2)
        previous=domain
    assert corner.side(5)==2
    assert corner.side(13)>3.2


@pytest.mark.parametrize("backend",["highs","cp_sat"])
def test_known_feasible_pair_and_physical_verification(tmp_path,monkeypatch,backend):
    shape=box(-.63,-.63,.63,.63)
    small=box(-.3,-.3,.3,.3)
    patio=Polygon([(0,0),(4,0),(4,4),(0,4)])
    layout=Layout(patio,{'1':shape,'2':small,'3':small},np.zeros((3,3)),np.zeros(3,dtype=bool),('1','2','3'),np.random.default_rng(1),units='ft')
    source=tmp_path/'source.json';save_json(layout,source)
    monkeypatch.setattr(corner,'SOURCE',str(source))
    # Production declare uses real vertex6; this fixture has vertex0.
    original_frame=corner.frame
    monkeypatch.setattr(corner,'frame',lambda patio,vertex=0:original_frame(patio,0))
    root=tmp_path/'trial';corner.declare(root)
    contract=json.loads((root/'contract.json').read_text())
    pose=[.63+corner.GAP,.63+corner.GAP,0]
    history=[dict(step=1,stone_id='1',pose=pose)]
    parent=tmp_path/'parent';parent.mkdir();corner.write(parent/'history.json',history)
    pool=tmp_path/'pool';pool.mkdir()
    candidates=[dict(stone_id='2',pose=[1.26+2*corner.GAP+.3,.3+corner.GAP,0]),dict(stone_id='3',pose=[.3+corner.GAP,1.26+2*corner.GAP+.3,0])]
    corner.write(pool/'candidate_pool.json',dict(trial=str(root),parent_history=history,target=3,candidates=candidates))
    out=tmp_path/'selection';corner.select_pair(pool,out,seconds=10,backend=backend)
    verification=json.loads((out/'accepted/verification.json').read_text())
    assert verification['passed'] and len(verification['results'])==3
    assert (out/'accepted/step_3.png').exists()
    assert (out/'accepted/results.csv').exists()
    # Saved history is not authority over physical poses.
    saved=out/'accepted/history.json';records=json.loads(saved.read_text());records[0]['pose'][0]+=.01;corner.write(saved,records)
    with pytest.raises(ValueError,match='disagrees'):corner.verify(out/'accepted')

    # A different predeclared fixture region leaves a large real void. Scout
    # radius levels may propose a pair but must never produce acceptance.
    scout_root=tmp_path/'scout_trial';scout_root.mkdir()
    contract=json.loads((root/'contract.json').read_text())
    for step in (3,4):
        contract['prefix_regions'][step-1].update(width_ft=1.5,depth_ft=1.5)
    corner.write(scout_root/'contract.json',contract)
    payload=json.loads((pool/'candidate_pool.json').read_text());payload['trial']=str(scout_root)
    scout_pool=tmp_path/'scout_pool';scout_pool.mkdir();corner.write(scout_pool/'candidate_pool.json',payload)
    scout_out=tmp_path/'scout_selection';corner.select_pair(scout_pool,scout_out,seconds=10,scout=True)
    trace=json.loads((scout_out/'trace.json').read_text())
    assert trace['accepted'] is None and not (scout_out/'accepted').exists()
    assert any(r.get('lower_diameter_in',0)>2 for r in trace['iterations'])


@pytest.mark.parametrize("backend",["highs","cp_sat"])
def test_joint_stages_reconstruct_all_five_prefixes(tmp_path,monkeypatch,backend):
    shapes={'1':box(-.63,-.63,.63,.63),'2':box(-.3,-.63,.3,.63),
            '3':box(-.63,-.3,.63,.3),'4':box(-.3,-.3,.3,.3),'5':box(-.1,-.1,.1,.1)}
    patio=Polygon([(0,0),(4,0),(4,4),(0,4)])
    layout=Layout(patio,shapes,np.zeros((5,3)),np.zeros(5,dtype=bool),tuple(shapes),np.random.default_rng(1),units='ft')
    source=tmp_path/'source.json';save_json(layout,source)
    monkeypatch.setattr(corner,'SOURCE',str(source))
    original_frame=corner.frame
    monkeypatch.setattr(corner,'frame',lambda patio,vertex=0:original_frame(patio,0))
    root=tmp_path/'trial';corner.declare(root)
    g=corner.GAP;a=.63+g;b=1.26+2*g+.3
    history=[dict(step=1,stone_id='1',pose=[a,a,0])]
    candidates=[dict(stone_id='2',pose=[b,a,0]),dict(stone_id='3',pose=[a,b,0]),
                dict(stone_id='4',pose=[b,b,0]),dict(stone_id='5',pose=[2.1,.3,0])]
    pool=tmp_path/'joint_pool';pool.mkdir()
    corner.write(pool/'candidate_pool.json',dict(trial=str(root),parent_history=history,target=5,candidates=candidates))
    calls=[]
    if backend=='cp_sat':
        from types import SimpleNamespace
        solve=corner.solve_boolean_model
        def initially_unknown(size,constraints,seconds,**kwargs):
            calls.append(seconds)
            if len(calls)==1:return SimpleNamespace(x=None,status=1,message='UNKNOWN')
            return solve(size,constraints,seconds,**kwargs)
        monkeypatch.setattr(corner,'solve_boolean_model',initially_unknown)
    out=tmp_path/'selection';corner.select_joint_stages(pool,out,seconds=10,backend=backend,
        solve_limit=.1 if backend=='cp_sat' else 30)
    if backend=='cp_sat':assert calls[1]>calls[0]
    result=json.loads((out/'accepted/verification.json').read_text())
    assert result['passed'] and len(result['results'])==5
    trace=json.loads((out/'trace.json').read_text())
    last=trace['iterations'][-1]
    assert sorted(last['stage_assignments'])==[0,0,1,1]
    assert len(set(last['stone_ids']))==4
    assert (out/'accepted/step_5.png').exists()
    augmented=tmp_path/'augmented'
    corner.augment_pool(pool,out/'accepted',augmented)
    retained=json.loads((augmented/'candidate_pool.json').read_text())
    assert len(retained['candidates'])==4
    assert len(set(retained['first_stage_hint_indices']))==2
    bad=json.loads((pool/'candidate_pool.json').read_text())
    bad['parent_history'][0]['pose'][0]+=.01
    bad_pool=tmp_path/'bad_pool';bad_pool.mkdir();corner.write(bad_pool/'candidate_pool.json',bad)
    with pytest.raises(ValueError,match='exact fixed parent'):
        corner.augment_pool(bad_pool,out/'accepted',tmp_path/'bad_augmented')


def test_hint_does_not_override_hard_constraints():
    from scipy.optimize import LinearConstraint
    from scipy.sparse import csr_matrix
    constraint=LinearConstraint(csr_matrix([[1.]]),1,np.inf)
    result=corner.solve_boolean_model(1,[constraint],5,backend='cp_sat',hints={0:0})
    assert result.x is not None and result.x[0]==1


def test_three_joint_stages_check_all_seven_prefixes(tmp_path,monkeypatch):
    g=corner.GAP;a=.63+g;b=1.26+2*g+.3
    previous_edge=1.26+2*g+.6;outer_start=previous_edge+g;outer_end=outer_start+.38
    width=previous_edge-g;height=outer_end-g
    shapes={'1':box(-.63,-.63,.63,.63),'2':box(-.3,-.63,.3,.63),
            '3':box(-.63,-.3,.63,.3),'4':box(-.3,-.3,.3,.3),'5':box(-.1,-.1,.1,.1),
            '6':box(-width/2,-.19,width/2,.19),'7':box(-.19,-height/2,.19,height/2)}
    patio=Polygon([(0,0),(4,0),(4,4),(0,4)])
    layout=Layout(patio,shapes,np.zeros((7,3)),np.zeros(7,dtype=bool),tuple(shapes),np.random.default_rng(1),units='ft')
    source=tmp_path/'source.json';save_json(layout,source);monkeypatch.setattr(corner,'SOURCE',str(source))
    original_frame=corner.frame;monkeypatch.setattr(corner,'frame',lambda patio,vertex=0:original_frame(patio,0))
    root=tmp_path/'trial';corner.declare(root)
    history=[dict(step=1,stone_id='1',pose=[a,a,0])]
    candidates=[dict(stone_id='2',pose=[b,a,0]),dict(stone_id='3',pose=[a,b,0]),
                dict(stone_id='4',pose=[b,b,0]),dict(stone_id='5',pose=[3,.3,0]),
                dict(stone_id='6',pose=[(g+previous_edge)/2,(outer_start+outer_end)/2,0]),
                dict(stone_id='7',pose=[(outer_start+outer_end)/2,(g+outer_end)/2,0])]
    pool=tmp_path/'pool';pool.mkdir();corner.write(pool/'candidate_pool.json',dict(trial=str(root),parent_history=history,target=7,candidates=candidates))
    out=tmp_path/'selection';corner.select_joint_stages(pool,out,seconds=10,backend='cp_sat')
    verified=json.loads((out/'accepted/verification.json').read_text());trace=json.loads((out/'trace.json').read_text())
    assert verified['passed'] and len(verified['results'])==7
    assert sorted(trace['iterations'][-1]['stage_assignments'])==[0,0,1,1,2,2]
    assert [r['stage'] for r in trace['iterations'][-1]['prefix_checks']]==[3,5,7]
    assert all((out/'accepted'/f'step_{n}.png').exists() for n in range(1,8))

    # With deliberately sparse initial center probes and a coarse conflict
    # raster, both the exact spacing and gap oracles must cut this pattern.
    broken=tmp_path/'broken_trial';broken.mkdir()
    contract=json.loads((root/'contract.json').read_text())
    for step in (7,8):contract['prefix_regions'][step-1].update(width_ft=2.5,depth_ft=2.5,nominal_area_sqft=6.25)
    corner.write(broken/'contract.json',contract)
    payload=json.loads((pool/'candidate_pool.json').read_text());payload['trial']=str(broken)
    payload['candidates'][4]['pose'][1]-=g-.03
    payload['candidates'][5]['pose'][0]-=g-.03
    broken_pool=tmp_path/'broken_pool';broken_pool.mkdir();corner.write(broken_pool/'candidate_pool.json',payload)
    monkeypatch.setattr(corner,'coverage_samples',lambda patio,domain,spacing=.16:corner.shapely.points([[0,0]]))
    bad_out=tmp_path/'bad_selection';corner.select_joint_stages(broken_pool,bad_out,seconds=10,backend='cp_sat',raster_spacing=.5)
    bad=json.loads((bad_out/'trace.json').read_text())
    first=next(r for r in bad['iterations'] if r.get('conflict'))
    assert not first['passed'] and not first['clearance_passed']
    assert any(r['stage']==7 and r['lower_diameter_in']>2 for r in first['prefix_checks'])
    assert any(r['stage']==7 and r['covering_candidates']==0 for r in bad['cuts'])
    assert bad['accepted'] is None and not (bad_out/'accepted').exists()


def test_cached_boolean_model_adds_new_cuts_without_rebuilding():
    from scipy.optimize import LinearConstraint
    from scipy.sparse import csr_matrix
    base=LinearConstraint(csr_matrix([[1.,1.]]),1,1);state={}
    first=corner.solve_boolean_model(2,[base],5,backend='cp_sat',state=state)
    selected=int(np.flatnonzero(first.x>.5)[0]);row=np.zeros((1,2));row[0,selected]=1
    cut=LinearConstraint(csr_matrix(row),0,0)
    second=corner.solve_boolean_model(2,[base,cut],5,backend='cp_sat',state=state)
    assert second.x[selected]==0 and sum(second.x)==1
    assert second.performance['new_rows']==1 and second.performance['total_rows']==2
    replacement=LinearConstraint(csr_matrix([[1.,1.]]),1,1)
    with pytest.raises(ValueError,match='append-only'):
        corner.solve_boolean_model(2,[replacement,cut],5,backend='cp_sat',state=state)


def test_adaptive_spacing_witness_cuts_nearby_pose_family():
    from itertools import combinations
    polys=np.array([box(0,0,1,1),box(1.03,0,2.03,1),box(1.01,-.1,1.5,1.1),box(4,4,5,5)],dtype=object)
    cover,witness=corner.conflict_clique(polys,0,1)
    assert cover.tolist()==[True,True,True,False]
    assert witness['covering_poses']==3 and not witness['fallback_pair_only']
    for a,b in combinations(np.flatnonzero(cover),2):assert polys[a].distance(polys[b])<.5/12
    with pytest.raises(ValueError,match='true spacing violation'):corner.conflict_clique(polys,0,3)


def test_overlap_margin_guides_recovery_and_preserves_clearance_gate():
    a=box(0,0,1,1)
    deep=box(.8,0,1.8,1);shallow=box(.9,0,1.9,1)
    assert corner.pair_spacing_margin(a,deep)<corner.pair_spacing_margin(a,shallow)<0
    assert corner.pair_spacing_margin(a,box(1.02,0,2.02,1))<0
    assert corner.pair_spacing_margin(a,box(1.05,0,2.05,1))>0


def test_front_and_physical_wall_offset_centres_are_in_coverage_probes():
    patio=box(0,0,4,4);domain=box(0,0,1.35,1.35)
    probes=corner.coverage_samples(patio,domain)
    xy=corner.shapely.get_coordinates(probes)
    assert np.any(np.all(np.isclose(xy,[1.35,1.35],atol=1e-12),axis=1))
    assert np.any(np.isclose(xy[:,0],1/12+1e-6,atol=1e-8)&(xy[:,1]>.2))
    assert corner.shapely.covers(domain,probes).all()
