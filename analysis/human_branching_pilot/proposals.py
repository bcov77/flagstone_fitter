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
    return canonical(dict(stones=[dict(id=str(s['id']),pose=s['pose'],active=s['active']) for s in doc['stones']],physics=doc.get('physics',{})))


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
    for sid in ordered[:(1 if followup else 4)]:
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
    chosen=inserts[:max(1,cfg['target_budget']-1)]+replaces[:1]
    for t in ts:
        if len(chosen)>=cfg['target_budget']:break
        if not any(t is old for old in chosen):chosen.append(t)
    return chosen


def detect_seams(doc,cfg,focus=None):
    board=shapes(doc);segments=[]
    for sid,g in board.items():
        xy=np.asarray(g.simplify(.025,preserve_topology=True).exterior.coords)
        center=np.array(g.representative_point().coords[0])
        for a,b in zip(xy,xy[1:]):
            length=np.linalg.norm(b-a)
            if length<.25:continue
            tangent=(b-a)/length;normal=np.array([-tangent[1],tangent[0]]);mid=(a+b)/2
            if np.dot(center-mid,normal)>0:normal=-normal
            segments.append((sid,a,b,tangent,normal,mid))
    seeds=[]
    for i,(owner,a,b,t,n,mid) in enumerate(segments):
        if focus is not None and np.linalg.norm(mid-np.asarray(focus))>2.5:continue
        for other,c,d,u,m,center in segments[i+1:]:
            if owner==other or abs(np.dot(t,u))<math.cos(math.radians(30)) or np.dot(n,m)>-.5:continue
            separation=float(np.dot(center-mid,n))
            if not .6/12<separation<7/12:continue
            lo=max(min(a@t,b@t),min(c@t,d@t));hi=min(max(a@t,b@t),max(c@t,d@t))
            if hi-lo<.2:continue
            axis=t if t[0]>0 or (abs(t[0])<1e-8 and t[1]>0) else -t
            angle=math.atan2(axis[1],axis[0]);normal=np.array([-axis[1],axis[0]])
            offset=float(((a+b+c+d)/4)@normal)
            interval=sorted([float(lo*np.dot(t,axis)),float(hi*np.dot(t,axis))])
            seeds.append(dict(angle=angle,lo=interval[0],hi=interval[1],offset=offset,width=separation,owners={owner,other}))
    groups=[]
    for seed in sorted(seeds,key=lambda s:s['hi']-s['lo'],reverse=True):
        for group in groups:
            st=np.array([math.cos(seed['angle']),math.sin(seed['angle'])]);sn=np.array([-st[1],st[0]])
            gt=np.array([math.cos(group['angle']),math.sin(group['angle'])]);gn=np.array([-gt[1],gt[0]])
            center=st*((seed['lo']+seed['hi'])/2)+sn*seed['offset']
            ends=[st*seed[k]+sn*seed['offset'] for k in ['lo','hi']];lo,hi=sorted(float(p@gt) for p in ends)
            if abs(math.remainder(seed['angle']-group['angle'],math.pi))<math.radians(25) and abs(float(center@gn)-group['offset'])<3/12 and lo<=group['hi']+.45 and hi>=group['lo']-.45:
                group.update(lo=min(group['lo'],lo),hi=max(group['hi'],hi),width=max(group['width'],seed['width']),owners=group['owners']|seed['owners']);break
        else:groups.append(seed.copy())
    result=[]
    for g in groups:
        # A long seam may cross a gap between two upper neighbors. Extend the
        # opposing-edge seed along compatible physical faces, not only the
        # short overlap of one particular pair of polygon segments.
        gt=np.array([math.cos(g['angle']),math.sin(g['angle'])]);gn=np.array([-gt[1],gt[0]])
        for _ in range(3):
            for sid,a,b,t,n,mid in segments:
                if abs(float(t@gt))<math.cos(math.radians(30)):continue
                side=float(mid@gn)-g['offset']
                if abs(side)>max(4/12,g['width']/2+2/12) or side*float(n@gn)>=0:continue
                lo,hi=sorted([float(a@gt),float(b@gt)])
                if lo>g['hi']+.75 or hi<g['lo']-.75:continue
                lower,upper=min(g['lo'],lo),max(g['hi'],hi)
                if upper-lower>4.5:continue
                g.update(lo=lower,hi=upper,owners=g['owners']|{sid})
        length=g['hi']-g['lo']
        if length*12<cfg['seam_min_length_inches']:continue
        t=np.array([math.cos(g['angle']),math.sin(g['angle'])]);n=np.array([-t[1],t[0]])
        center=t*((g['lo']+g['hi'])/2)+n*g['offset']
        if focus is not None and np.linalg.norm(center-np.asarray(focus))>1.5:continue
        result.append(dict(angle=g['angle'],center=center.tolist(),length=length,width=g['width'],owners=sorted(g['owners'])))
    return sorted(result,key=lambda s:-s['length'])[:3]


def seam_candidates(doc,cfg,rng,focus=None):
    board=shapes(doc);patio=shapely.Polygon(doc['patio']);free=patio.buffer(-GAP).difference(unary_union(list(board.values())).buffer(GAP));result=[]
    simple_free=free.simplify(.002,preserve_topology=True)
    local={str(s['id']):shapely.Polygon(s['outline']).simplify(.002,preserve_topology=True) for s in doc['stones']}
    for seam in detect_seams(doc,cfg,focus):
        for sid,g in local.items():
            if sid in board:continue
            box=list(g.minimum_rotated_rectangle.exterior.coords);edges=[np.array(box[i+1])-np.array(box[i]) for i in range(4)];long=max(edges,key=np.linalg.norm);length=np.linalg.norm(long);width=min(np.linalg.norm(e) for e in edges)
            if length/width<cfg['seam_min_aspect_ratio'] or not .65*seam['length']<=length<=1.6*seam['length'] or width*12>9:continue
            axis=math.atan2(long[1],long[0]);best=None
            t=np.array([math.cos(seam['angle']),math.sin(seam['angle'])]);n=np.array([-t[1],t[0]])
            for reverse in [0,math.pi]:
                for turn in [-10,0,10]:
                    angle=seam['angle']-axis+reverse+math.radians(turn)
                    for along in np.linspace(-max(.2,seam['length']/2-length/2+.25),max(.2,seam['length']/2-length/2+.25),9):
                        for across in [-.15,0,.15]:
                            center=np.asarray(seam['center'])+along*t+across*n;pose=[*map(float,center),float(angle)];full=world(g,pose)
                            if not patio.buffer(-GAP).buffer(1e-8).covers(full):continue
                            aligned=rotate(full,-math.degrees(seam['angle']),origin=tuple(center))
                            for factor in cfg['seam_width_probe_scales']:
                                probe=rotate(scale(aligned,cfg['seam_length_probe_scale'],factor,origin=tuple(center)),math.degrees(seam['angle']),origin=tuple(center))
                                intrusion=probe.difference(simple_free).area/probe.area
                                if intrusion>cfg['probe_max_intrusion_fraction']:continue
                                score=(1-factor,intrusion,abs(length-seam['length']))
                                if best is None or score<best[0]:best=(score,pose,factor,intrusion)
                                break
            if best:
                _,pose,factor,intrusion=best
                result.append(dict(family='seam',remove=[],placements=[dict(stone=sid,pose=pose)],
                                   probe=dict(length_scale=cfg['seam_length_probe_scale'],width_scale=factor,intrusion_fraction=intrusion),
                                   seam=seam,rank=[1-factor,intrusion,-g.area]))
    return sorted(result,key=lambda p:p['rank'])[:8]


def mobile_seam_candidates(doc,cfg,focus=None):
    """Virtual mobility envelope only. All accepted boards retain full outlines.

    Eroding neighbors independently is an optimistic proposal filter: it does
    not imply their required translations are jointly feasible.
    """
    board=shapes(doc);patio=shapely.Polygon(doc['patio']);inside=patio.buffer(-GAP)
    clearance_free=inside.difference(unary_union(list(board.values())).buffer(GAP))
    mobility=cfg['seam_neighbor_mobility_inches']/12
    locked={s for s,v in doc.get('physics',{}).get('pieces',{}).items() if v.get('locked')}
    occupied=unary_union([g.buffer(GAP if sid in locked else GAP-mobility) for sid,g in board.items()])
    mobile_free=inside.difference(occupied).simplify(.002,preserve_topology=True)
    result=[]
    for seam in detect_seams(doc,cfg,focus):
        t=np.array([math.cos(seam['angle']),math.sin(seam['angle'])]);n=np.array([-t[1],t[0]])
        for s in doc['stones']:
            sid=str(s['id'])
            if sid in board:continue
            local=shapely.Polygon(s['outline']);g=local.simplify(.002,preserve_topology=True)
            box=list(g.minimum_rotated_rectangle.exterior.coords);edges=[np.array(b)-np.array(a) for a,b in zip(box,box[1:])];long=max(edges,key=np.linalg.norm);length=np.linalg.norm(long);width=min(np.linalg.norm(e) for e in edges)
            if length/width<cfg['seam_min_aspect_ratio'] or width*12>9 or not .65*seam['length']<=length<=1.6*seam['length']:continue
            axis=math.atan2(long[1],long[0]);best=None
            extent=max(.2,seam['length']/2-length/2+.25)
            for reverse in [0,math.pi]:
                for turn in [-10,0,10]:
                    for along in np.linspace(-extent,extent,9):
                        for across in [-.15,0,.15]:
                            center=np.asarray(seam['center'])+along*t+across*n
                            pose=[*map(float,center),float(seam['angle']-axis+reverse+math.radians(turn))]
                            full=world(g,pose);probe=scale(full,.8,.8,origin='centroid')
                            if not inside.buffer(1e-8).covers(full):continue
                            intr=probe.difference(mobile_free).area/probe.area
                            if intr>cfg['probe_max_intrusion_fraction']:continue
                            fixed_intr=probe.difference(clearance_free).area/probe.area
                            full_intr=full.difference(clearance_free).area/full.area
                            if full_intr>.5:continue
                            score=(fixed_intr,-local.area,intr)
                            if best is None or score<best[0]:best=(score,pose)
            if best:
                score,pose=best
                # Recheck the selected probe on exact outlines and envelopes.
                full=world(local,pose);probe=scale(full,.8,.8,origin='centroid')
                exact_free=inside.difference(occupied)
                intr=probe.difference(exact_free).area/probe.area
                if intr>cfg['probe_max_intrusion_fraction'] or not inside.buffer(1e-8).covers(full):continue
                result.append(dict(family='mobile_seam',remove=[],placements=[dict(stone=sid,pose=pose)],
                                   probe=dict(scale=.8,neighbor_mobility_inches=cfg['seam_neighbor_mobility_inches'],
                                              intrusion_fraction=intr,fixed_neighbor_intrusion_fraction=probe.difference(clearance_free).area/probe.area),
                                   seam=seam,rank=list(score)))
    return sorted(result,key=lambda p:p['rank'])[:8]


def generate(doc,cfg,rng,focus=None,followup=False):
    board=shapes(doc);patio=shapely.Polygon(doc['patio']);local={str(s['id']):shapely.Polygon(s['outline']) for s in doc['stones']};pool=[]
    for target in targets(doc,cfg,focus,followup):
        available=[s for s in local if s not in board or s in target['remove']]
        for factor in [1.,cfg['probe_scale']]:
            if factor<1 and not cfg['scaled_min_pocket_sqft']<=target['pocket'].area<=cfg['scaled_max_pocket_sqft']:continue
            if factor<1 and target['remove'] and target.get('existing_free_sqft',0)<.08:continue
            probe_local={s:scale(g,factor,factor,origin='centroid') for s,g in local.items()}
            ranked=coarse(probe_local,available,target['pocket'],target['free'],cfg,rng)
            count=0;seen=set()
            for cost,sid,pose in ranked:
                if sid in seen:continue
                full=world(local[sid],pose);probe=world(probe_local[sid],pose)
                coverage=probe.intersection(target['pocket']).area/target['pocket'].area
                intr=probe.difference(target['free']).area/probe.area;overhang=probe.difference(target['pocket']).area/probe.area
                if coverage<cfg['probe_min_coverage'] or intr>cfg['probe_max_intrusion_fraction'] or overhang>cfg['probe_max_overhang_fraction']:continue
                if not patio.buffer(-GAP).buffer(1e-8).covers(full):continue
                p=dict(family='ordinary' if factor==1 else 'scaled',remove=target['remove'][:],placements=[dict(stone=sid,pose=pose)],
                       target=target['id'],probe=dict(scale=factor,coverage=coverage,intrusion_fraction=intr,overhang=overhang),
                       rank=[intr,-coverage,-local[sid].area])
                if not eligible(doc,p,cfg):continue
                seen.add(sid);pool.append(p);count+=1
                if count>=4:break
    pool.extend(p for p in seam_candidates(doc,cfg,rng,focus) if eligible(doc,p,cfg))
    pool.extend(p for p in mobile_seam_candidates(doc,cfg,focus) if eligible(doc,p,cfg))
    # Keep distinct pieces/actions; a returned piece is eligible in a changed board/opening.
    unique={}
    for p in sorted(pool,key=lambda p:p['rank']):
        key=canonical(dict(remove=p['remove'],placements=p['placements']))
        if key not in unique:unique[key]=p
    groups={f:[] for f in ['scaled','ordinary','seam','mobile_seam']}
    for p in unique.values():groups[p['family']].append(p)
    answer=[]
    while any(groups.values()):
        for family in groups:
            if groups[family]:answer.append(groups[family].pop(0))
    return answer
