"""Executable regression fixtures drawn from the human demonstration, not quality labels."""
import json

import numpy as np
import pytest
from shapely.affinity import scale
from shapely.geometry import Polygon
from shapely.ops import unary_union

from study import HERE,ROOT,load,document,replay
from experiments import target_for,proposal,human_style
from selector import apply,coarse,fit


@pytest.fixture(scope='module')
def demo():
    source,final,records,manifest=load()
    cases={c['id']:c for c in json.loads((HERE/'fixtures.json').read_text())}
    return source,final,records,cases


def test_complete_demonstration_physics_replays_frame_by_frame():
    result=replay(write=False)
    assert result['frames']==273
    assert result['maximum_errors']['position_ft']<1e-8
    assert result['maximum_errors']['angle_rad']<1e-8
    assert result['maximum_errors']['velocity_fps']<1e-8
    assert result['maximum_errors']['angular_velocity_rps']<1e-8
    assert result['final_position_error_ft']<1e-8


@pytest.mark.parametrize('case_id',['run_24','run_31'])
def test_multi_removal_and_pair_filling_preserve_original_inventory(demo,case_id):
    source,_,_,cases=demo;case=cases[case_id]
    before=document(source,case['pre']);planned=apply(before,proposal(case))
    active={str(s['id']):s['pose'] for s in planned['stones'] if s['active']}
    expected={sid:p['pose'] for sid,p in case['proposed']['pieces'].items() if p['active']}
    assert active==expected
    original={str(s['id']):s['outline'] for s in source['stones']}
    assert all(s['outline']==original[str(s['id'])] for s in planned['stones'])
    assert len(case['removed'])==2 if case_id=='run_24' else len(case['selected'])==2


def test_rejected_piece_is_available_in_a_different_opening(demo):
    _,final,records,cases=demo
    # Stone 70 is tried at upper left, rejected for 119, then reused lower down.
    later=cases['run_28'];assert later['selected']==['70']
    assert not later['pre']['pieces']['70']['active']
    earlier=cases['run_24'];assert earlier['selected']==['70']
    old=earlier['proposed']['pieces']['70']['pose'];new=later['proposed']['pieces']['70']['pose']
    assert abs(old[1]-new[1])>1
    assert any(s['id']=='70' and s['active'] for s in final['stones'])


def test_orientation_strokes_belong_to_one_placement(demo):
    _,_,records,cases=demo;case=cases['run_16']
    edits=[op for op in case['operations'] if op.get('sid')=='52']
    assert sum(op['action']=='pose' for op in edits)>5
    assert case['selected']==['52']
    assert len(case['pulse_durations'])==3
    # The outcome fixture is the fully oriented full-size proposal, not each 5° UI stroke.
    assert proposal(case)['placements'][0]['pose']==case['proposed']['pieces']['52']['pose']


def test_first_replacement_makes_space_for_the_next_corner_filler(demo):
    from study import shapes
    source,_,_,cases=demo
    first,second=cases['run_09'],cases['run_11']
    filler=shapes(document(source,second['proposed']))['59']
    reduced=scale(filler,.8,.8,origin='centroid')
    original=unary_union([g.buffer(.5/12) for g in shapes(document(source,first['pre'])).values()])
    prepared=unary_union([g.buffer(.5/12) for g in shapes(document(source,second['pre'])).values()])
    assert reduced.intersection(prepared).area < reduced.intersection(original).area*.1
    assert '113' in second['pre']['pieces'] and second['pre']['pieces']['113']['active']


def test_followup_can_substantially_reorient_one_member_of_a_pair(demo):
    import math
    _,_,records,cases=demo
    initial=cases['run_31']['last_stop']['pieces']['41']['pose'][2]
    next_start=next(r for r in records if r.get('action')=='simulation_start' and r['after']['run_number']==32)
    proposed=next_start['after']['pieces']['41']['pose'][2]
    assert abs(math.degrees(math.remainder(proposed-initial,2*math.pi)))>=45


def test_scaled_probe_exposes_the_human_fullsize_upgrade(demo):
    source,_,_,cases=demo;case=cases['run_16'];target,board,patio=target_for(source,case)
    local={str(s['id']):Polygon(s['outline']) for s in source['stones']}
    cfg=json.loads((ROOT/'analysis/physics_trajectories/config.json').read_text())
    assert fit(proposal(case)['placements'],target,local,board,patio,cfg) is None
    reduced={sid:scale(g,.8,.8,origin='centroid') for sid,g in local.items()}
    p=fit(proposal(case)['placements'],target,reduced,board,patio,cfg)
    assert p is not None and p['clearance_intrusion_fraction']==0


@pytest.mark.parametrize('case_id',['run_16','run_28'])
def test_blind_scaled_inventory_search_recalls_human_piece_in_top_ten(demo,case_id):
    source,_,_,cases=demo;case=cases[case_id];target,board,_=target_for(source,case)
    # The search receives the opening and full available inventory, not a preferred ID.
    local={str(s['id']):scale(Polygon(s['outline']),.8,.8,origin='centroid') for s in source['stones']}
    available=[sid for sid in local if sid not in board or sid in case['removed']]
    cfg=json.loads((ROOT/'analysis/physics_trajectories/config.json').read_text())
    cfg.update(piece_budget=len(local),coarse_per_target=1000)
    ranked=coarse(local,available,target['pocket'],target['free'],cfg,np.random.default_rng(0))
    ids=list(dict.fromkeys(sid for _,sid,_ in ranked))
    assert case['selected'][0] in ids[:10]


def test_blind_scaled_operator_installs_a_clearance_valid_fullsize_upgrade(demo):
    import copy
    from study import metrics
    source,_,_,cases=demo;case=cases['run_16'];target,board,patio=target_for(source,case)
    reduced={str(s['id']):scale(Polygon(s['outline']),.8,.8,origin='centroid') for s in source['stones']}
    cfg=json.loads((ROOT/'analysis/physics_trajectories/config.json').read_text())
    available=[sid for sid in reduced if sid not in board or sid in case['removed']]
    ranked=coarse(reduced,available,target['pocket'],target['free'],{**cfg,'piece_budget':len(reduced),'coarse_per_target':1000},np.random.default_rng(0))
    # Pick the first geometry-eligible proposal, without consulting the demonstrated ID.
    _,sid,pose=next(row for row in ranked if fit([dict(stone=row[1],pose=row[2])],target,reduced,board,patio,cfg) is not None)
    derived=copy.deepcopy(case);snap=copy.deepcopy(case['pre'])
    for old in case['removed']:snap['pieces'][old]['active']=False
    snap['pieces'][sid].update(active=True,locked=False,pose=pose,anchor=pose)
    snap['velocities']={key:snap['velocities'].get(key,[0,0,0]) for key,p in snap['pieces'].items() if p['active']}
    snap.update(running=False,drives={},elapsed=0.,accumulator=0.)
    derived.update(selected=[sid],proposed=snap)
    end,result=human_style(source,derived,1701,cfg)
    outcome=metrics(end)
    assert result['settled'] and outcome['clearance_valid']
    assert outcome['coverage']>case['before_metrics']['coverage']+.5
    original={str(s['id']):s['outline'] for s in source['stones']}
    assert all(s['outline']==original[str(s['id'])] for s in end['stones'])
