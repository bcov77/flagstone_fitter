"""Proposal constraints and the human bar counterexample."""
import copy
import json
import math
import sys
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from proposals import eligible,board_key,detect_seams,mobile_seam_candidates
sys.path.insert(0,str(HERE))
from pilot import dominates,rank
from probe_audit import measure
from study import load,document


@pytest.fixture(scope='module')
def demo():
    source,*_=load()
    return source,{c['id']:c for c in json.loads((HERE.parent/'human_trajectory_fa791034/fixtures.json').read_text())},json.loads((HERE/'config.json').read_text())


@pytest.mark.parametrize('angle,ok',[(44.999,False),(45,True),(315,True),(316,False),(360,False)])
def test_deliberate_adjustment_has_circular_45_degree_minimum(demo,angle,ok):
    source,cases,cfg=demo;doc=document(source,cases['run_16']['pre'])
    s=next(s for s in doc['stones'] if s['active']);sid=str(s['id']);pose=[s['pose'][0]+.1,s['pose'][1],s['pose'][2]+math.radians(angle)]
    action=dict(remove=[sid],placements=[dict(stone=sid,pose=pose)])
    assert eligible(doc,action,cfg)==ok


def test_lock_duplicate_inventory_and_large_to_small_guards(demo):
    source,cases,cfg=demo;doc=document(source,cases['run_16']['pre'])
    sid='62';action=dict(remove=[sid],placements=[dict(stone='52',pose=[1,1,0])])
    assert eligible(doc,action,cfg)
    doc['physics']['pieces'][sid]['locked']=True
    assert not eligible(doc,action,cfg)
    doc['physics']['pieces'][sid]['locked']=False
    action['placements']*=2
    assert not eligible(doc,action,cfg)
    action['placements']=action['placements'][:1]
    next(s for s in doc['stones'] if str(s['id'])=='52')['outline']=[[0,0],[.01,0],[0,.01]]
    assert not eligible(doc,action,cfg)


def test_returned_stone_can_be_reused_in_changed_context(demo):
    source,cases,cfg=demo;a=document(source,cases['run_24']['pre']);b=document(source,cases['run_28']['pre'])
    assert board_key(a)!=board_key(b)
    c=cases['run_28'];action=dict(remove=c['removed'],placements=[dict(stone='70',pose=c['proposed']['pieces']['70']['pose'])])
    assert eligible(b,action,cfg)


def test_80_percent_probe_covers_52_but_not_horizontal_bar(demo):
    source,cases,_=demo
    a=measure(source,cases['run_16']);b=measure(source,cases['run_21'])
    assert a['scales']['0.8']['intrusion_fraction']<1e-8
    assert b['scales']['0.8']['intrusion_fraction']>.35
    assert b['scales']['0.7']['intrusion_fraction']>.30
    assert b['recorded_stop_metrics']['overlap_sqin']<1e-7
    assert not b['recorded_stop_metrics']['clearance_valid']


def test_seam_is_detected_without_piece_identity(demo):
    source,cases,cfg=demo;doc=document(source,cases['run_21']['pre']);focus=[2.7,3.2]
    seams=detect_seams(doc,cfg,focus)
    assert any(abs(s['angle'])<math.radians(15) and s['length']>2.5 for s in seams)
    renamed=copy.deepcopy(doc)
    for i,s in enumerate(renamed['stones']):s['id']=f'anonymous_{i}'
    assert [(s['angle'],s['length']) for s in detect_seams(renamed,cfg,focus)]==[(s['angle'],s['length']) for s in seams]


def test_mobility_probe_retrieves_human_bar_without_id_or_pose_bonus(demo):
    source,cases,cfg=demo;doc=document(source,cases['run_21']['pre']);before=copy.deepcopy(doc)
    # Only a neighborhood center is provided, as for the other benchmarks.
    choices=mobile_seam_candidates(doc,cfg,[2.7,3.2])
    assert choices and choices[0]['placements'][0]['stone']=='13'
    assert choices[0]['probe']['intrusion_fraction']<1e-8
    assert choices[0]['probe']['fixed_neighbor_intrusion_fraction']>.2
    assert doc==before  # virtual envelopes never resize actual inventory


def test_root_comparison_does_not_reward_damage_or_force_latest_board():
    root=dict(coverage=75,broad_void_sqft=2,wall_joint_inches=30,joint_inches=100)
    bad=dict(root,coverage=76,broad_void_sqft=1.9,wall_joint_inches=20)
    assert rank(root,root)<rank(bad,root)
    assert not dominates(bad,root) and not dominates(root,bad)
    good=dict(root,coverage=76,broad_void_sqft=1.8)
    assert dominates(good,root)
