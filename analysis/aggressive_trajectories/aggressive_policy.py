"""Separate proposal regimes and explicit family/within-family probabilities."""
import copy
import math
import numpy as np
import shapely
from shapely.affinity import scale

from common import shapes,world,GAP
import selector as ordinary
from aggressive_geometry import canonical,board_key,eligible,targets


def action_key(doc,p):
    return canonical([board_key(doc),dict(remove=sorted(p['remove']),placements=p['placements'])])


def validate(cfg):
    weights=cfg['family_weights']
    if set(weights)!={'ordinary','scaled','compound'} or any(not math.isfinite(v) or v<=0 for v in weights.values()) or abs(sum(weights.values())-1)>1e-9:raise ValueError('Three positive family weights must sum to one')
    for key in ['ordinary_piece_budget','scaled_piece_budget','scaled_coarse_per_target','target_budget','archive_limit']:
        if type(cfg[key]) is not int or cfg[key]<1:raise ValueError(f'Invalid {key}')
    if not 45<=cfg['minimum_reorientation_degrees']<=180 or cfg['minimum_replacement_area_ratio']<1:raise ValueError('Require >=45 degree adjustments and no net downsizing')
    if not 0<cfg['probe_scale']<1 or not 0<cfg['probe_min_coverage']<=1:raise ValueError('Invalid probe scale/coverage')
    if not 0<=cfg['probe_max_intrusion_fraction']<=.02 or not 0<=cfg['probe_max_overhang_fraction']<=.12:raise ValueError('Invalid probe allowances')
    if not 0<cfg['scaled_min_pocket_sqft']<=cfg['scaled_max_pocket_sqft']:raise ValueError('Invalid medium pocket bounds')
    if not cfg['pulse_seconds'] or any(not math.isfinite(t) or not 0<t<=2 for t in cfg['pulse_seconds']):raise ValueError('Invalid pulse durations')


def generate(doc,cfg,rng,excluded=()):
    local={str(s['id']):shapely.Polygon(s['outline']) for s in doc['stones']};board=shapes(doc);patio=shapely.Polygon(doc['patio']);pool=[]
    base={**cfg,'piece_budget':cfg['ordinary_piece_budget']}
    op,diagnostics=ordinary.generate(doc,base,rng)
    for p in op:
        if not eligible(doc,p,cfg):continue
        p['family']='compound' if len(p['remove'])>1 or len(p['placements'])>1 else 'ordinary'
        p['physics_protocol']='held_release'
        if action_key(doc,p) not in excluded:pool.append(p)
    target_log=[];probe_local={s:scale(g,cfg['probe_scale'],cfg['probe_scale'],origin='centroid') for s,g in local.items()}
    coarse_cfg={**cfg,'piece_budget':cfg['scaled_piece_budget'],'coarse_per_target':cfg['scaled_coarse_per_target']}
    for t in targets(doc,cfg):
        log=dict(id=t['id'],remove=t['remove'],area_sqft=t['pocket'].area,existing_free_sqft=t.get('existing_free_sqft',t['pocket'].area),accepted=0)
        target_log.append(log)
        if not cfg['scaled_min_pocket_sqft']<=t['pocket'].area<=cfg['scaled_max_pocket_sqft']:continue
        if t['remove'] and t.get('existing_free_sqft',0)<.08:continue
        available=[s for s in local if s not in board or s in t['remove']]
        ranked=ordinary.coarse(probe_local,available,t['pocket'],t['free'],coarse_cfg,rng)
        seen=set()
        for cost,sid,pose in ranked:
            sid=str(sid)
            if sid in seen:continue
            full=world(local[sid],pose);probe=world(probe_local[sid],pose)
            coverage=probe.intersection(t['pocket']).area/t['pocket'].area
            intrusion=probe.difference(t['free']).area/probe.area;overhang=probe.difference(t['pocket']).area/probe.area
            if coverage<cfg['probe_min_coverage'] or intrusion>cfg['probe_max_intrusion_fraction'] or overhang>cfg['probe_max_overhang_fraction']:continue
            if not patio.buffer(-GAP).buffer(1e-8).covers(full):continue
            p=dict(family='compound' if len(t['remove'])>1 else 'scaled',physics_protocol='free_pulses',
                   target=t['id'],remove=t['remove'][:],placements=[dict(stone=sid,pose=pose)],tier=0,
                   probe=dict(scale=cfg['probe_scale'],coverage=coverage,intrusion_fraction=intrusion,overhang=overhang),
                   fit=dict(coverage=coverage,clearance_intrusion_fraction=intrusion,outside_pocket_fraction=overhang,
                            scope='80% probe geometry, not full-size fit'),
                   area_gain_sqft=full.area-sum(local[s].area for s in t['remove']),
                   reasons=['Medium problematic opening with well-fitting 80% probe',
                            'Full-size overlap allowed at start; final full-size clearance and settling required',
                            'Probe ranking is a hypothesis; post-physics gap/joint damage is measured separately'])
            p['rank_key']=[0,intrusion,overhang,-coverage,-p['area_gain_sqft']]
            if not eligible(doc,p,cfg) or action_key(doc,p) in excluded:continue
            pool.append(p);seen.add(sid);log['accepted']+=1
            if log['accepted']>=cfg['refine_per_target']:break
    # Distinct opening/piece action mass, never a multiplicity bonus for poses.
    unique={}
    for p in sorted(pool,key=lambda p:p['rank_key']):
        key=(p['family'],tuple(sorted(p['remove'])),tuple(sorted(x['stone'] for x in p['placements'])))
        if key not in unique:unique[key]=p
    pool=sorted(unique.values(),key=lambda p:(p['family'],p['rank_key']))
    return pool,dict(ordinary=diagnostics,scaled_targets=target_log,families={f:sum(p['family']==f for p in pool) for f in cfg['family_weights']},
                     scope='Frozen ordinary planning plus separate original-outline 80% probe search. No seam heuristics.')


def probabilities(pool,mode,cfg):
    if not pool:return np.array([]),{}
    families=[f for f in cfg['family_weights'] if any(p['family']==f for p in pool)]
    total=sum(cfg['family_weights'][f] for f in families);probs=np.zeros(len(pool));details={}
    for family in families:
        ids=sorted([i for i,p in enumerate(pool) if p['family']==family],key=lambda i:pool[i]['rank_key'])
        conditional,best=ordinary.probabilities([pool[i] for i in ids],mode,cfg)
        mass=cfg['family_weights'][family]/total
        probs[ids]=mass*conditional
        details[family]=dict(probability=mass,indices=ids,conditional_probabilities=conditional.tolist(),best_indices=[ids[i] for i in best])
    return probs,details


def choose(pool,mode,cfg,rng):
    probs,details=probabilities(pool,mode,cfg)
    if not len(pool):return None,dict(mode=mode,reason='No eligible candidate in any bounded family search',requested_family_weights=cfg['family_weights'])
    index=int(rng.choice(len(pool),p=probs))
    return pool[index],dict(mode=mode,index=index,probability=float(probs[index]),probabilities=probs.tolist(),families=details,
                            requested_family_weights=cfg['family_weights'],unavailable_families=[f for f in cfg['family_weights'] if f not in details],
                            selected_family=pool[index]['family'],note='Marginal family x within-family probability. Golden/Silver best sets are within family; never a global quality guarantee.')
