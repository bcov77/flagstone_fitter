"""Geometry-only proposal portfolio; no human piece IDs or saved winning poses."""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import shapely
from shapely.affinity import affine_transform,rotate,scale
from shapely.ops import unary_union

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'analysis/physics_trajectories'),str(ROOT/'physics_patio')]
from common import shapes,world,GAP
from selector import coarse,pockets
from future_gap_geometry import joint_profile,walls


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def board_key(doc):
    return canonical(dict(stones=[dict(id=str(s['id']),pose=s['pose'] if s['active'] else None,active=s['active']) for s in doc['stones']],physics=doc.get('physics',{})))


def eligible(doc,action,cfg):
    byid={str(s['id']):s for s in doc['stones']};remove=action['remove'];ids=[p['stone'] for p in action['placements']]
    if len(ids)!=len(set(ids)) or not ids:return False
    if any(s not in byid for s in [*remove,*ids]):return False
    locked={s for s,v in doc.get('physics',{}).get('pieces',{}).items() if v.get('locked')}
    if set(remove)&locked:return False
    for p in action['placements']:
        sid=p['stone']
        if byid[sid]['active']:
            if sid not in remove:return False
            delta=abs(math.degrees(math.remainder(p['pose'][2]-byid[sid]['pose'][2],2*math.pi)))
            if delta+1e-8<cfg['minimum_reorientation_degrees']:return False
    old=sum(shapely.Polygon(byid[s]['outline']).area for s in remove)
    new=sum(shapely.Polygon(byid[s]['outline']).area for s in ids)
    return not remove or new+1e-9>=old*cfg['minimum_replacement_area_ratio']


def joint_summary(board,patio):
    simple={s:g.simplify(.002,preserve_topology=True) for s,g in board.items()};wl=walls(patio)
    total=wall=0.;byid={}
    for sid,g in simple.items():
        near={s:h for s,h in simple.items() if s!=sid and g.distance(h)<1.05/12}
        nearwalls={s:h for s,h in wl.items() if g.distance(h)<1.05/12}
        p=joint_profile(g,near,nearwalls,GAP,dict(boundary_step_ft=.03,joint_excess_inches=.5,joint_angle_degrees=20))
        runs={s:v['longest_run_ft']*12 for s,v in p['by_owner'].items() if v['longest_run_ft']>=.25}
        w=sum(v for s,v in runs.items() if s.startswith('wall_'));t=sum(v*(1 if s.startswith('wall_') else .5) for s,v in runs.items())
        wall+=w;total+=t;byid[sid]=dict(wall=w,total=t,matched_fraction=p['matched_fraction'])
    return dict(wall=wall,total=total,by_id=byid)


def targets(doc,cfg,focus=None,followup=False):
    board=shapes(doc);patio=shapely.Polygon(doc['patio']);free,holes=pockets(patio,board)
    within=lambda g: focus is None or g.distance(shapely.Point(focus))<1.5
    ts=[dict(id=f'insert_{i}',remove=[],old=None,pocket=p,free=free) for i,p in enumerate(holes) if within(p) and p.area>=.12]
    joints=joint_summary(board,patio);locked={s for s,v in doc.get('physics',{}).get('pieces',{}).items() if v.get('locked')}
    ordered=sorted([s for s in board if s not in locked and within(board[s])],key=lambda s:(joints['by_id'][s]['wall']>=6,joints['by_id'][s]['matched_fraction'],board[s].area))
    for sid in ordered[:(1 if followup else max(4,cfg['target_budget']))]:
        others={s:g for s,g in board.items() if s!=sid};f,ps=pockets(patio,others)
        relevant=[p for p in ps if p.intersection(board[sid]).area>board[sid].area*.35]
        if relevant:
            p=max(relevant,key=lambda p:p.intersection(board[sid]).area)
            # A scaled action must address existing excess space, not just replace a good slot.
            existing=p.intersection(free).area
            ts.append(dict(id=f'replace_{sid}',remove=[sid],old=sid,pocket=p,free=f,existing_free_sqft=existing))
    if not followup:
        for a,b in zip(ordered[:3],ordered[1:3]):
            if board[a].distance(board[b])>.25:continue
            f,ps=pockets(patio,{s:g for s,g in board.items() if s not in [a,b]})
            relevant=[p for p in ps if p.intersection(board[a]).area>.2*board[a].area and p.intersection(board[b]).area>.2*board[b].area]
            if relevant:ts.append(dict(id=f'merge_{a}_{b}',remove=[a,b],old=None,pocket=max(relevant,key=lambda p:p.area),free=f,existing_free_sqft=free.intersection(unary_union([board[a],board[b]]).buffer(.5)).area))
    for t in ts:
        distance=0 if focus is None else t['pocket'].distance(shapely.Point(focus))
        t['priority']=(distance,t['pocket'].area>cfg['scaled_max_pocket_sqft'],-t.get('existing_free_sqft',t['pocket'].area),t['pocket'].area)
    ts.sort(key=lambda t:t['priority'])
    # Reserve one replacement opening instead of letting three tiny natural
    # holes consume the entire portfolio. The inventory still decides fit.
    inserts=[t for t in ts if not t['remove']];replaces=[t for t in ts if t['remove']]
    singles=[t for t in replaces if len(t['remove'])==1];merges=[t for t in replaces if len(t['remove'])>1]
    chosen=(inserts[:2]+singles[:1]+merges[:1])[:cfg['target_budget']]
    for t in ts:
        if len(chosen)>=cfg['target_budget']:break
        if not any(t is old for old in chosen):chosen.append(t)
    return chosen

