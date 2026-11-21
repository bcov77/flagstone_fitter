"""Shrink probes and bounded translation witnesses for full-size near-fits.

A witness checks original outlines at proposed endpoints, not a collision-free
motion path, physics convergence, or the patio's maximum-gap rules.
"""
import math
import numpy as np
from shapely.affinity import scale, translate
from shapely.ops import unary_union


def shrink_probe(geometry, factor):
    if not 0 < factor <= 1:
        raise ValueError('Probe scale must be in (0, 1]')
    return scale(geometry, xfact=factor, yfact=factor, origin='centroid')


def nearfit_checks(full_metrics, probe_gate, wall_clear, config):
    checks = dict(
        probe_shape=probe_gate['status']=='passes_provisional_screen',
        full_physical_overlap=full_metrics['physical_overlap_sqft']/full_metrics['area_sqft'] <= config['physical_overlap_fraction'],
        full_clearance_area=full_metrics['clearance_intrusion_fraction'] <= config['clearance_intrusion_fraction'],
        full_clearance_depth=full_metrics['max_clearance_intrusion_inches'] <= config['clearance_depth_inches'],
        fixed_wall=bool(wall_clear),
    )
    return checks


def translation_witness(candidate, neighbors, patio, gap, config):
    """Find short translations for directly affected neighbors, others fixed.

    A small discrete beam search is deliberately conservative. Failure means no
    witness was found in this search, never that a physics solution is impossible.
    Every returned endpoint is independently checked against ALL other stones.
    """
    tol=1e-8
    result=dict(status='no_bounded_translation_witness', translations_ft={},
                max_neighbor_translation_inches=None, total_neighbor_translation_inches=None,
                method='discrete_translations_endpoint_check', physics_steps=0)
    wall=patio.buffer(-gap)
    if not wall.buffer(tol).covers(candidate):
        return dict(result,reason='candidate_crosses_fixed_wall_clearance')
    affected=sorted(s for s,g in neighbors.items() if candidate.intersects(g) or candidate.distance(g)<gap-tol)
    result['affected_neighbors']=affected
    if set(affected)&set(config.get('locked_neighbors',[])):
        return dict(result,reason='affected_neighbor_locked')
    if len(affected)>config['max_neighbors']:
        return dict(result,reason='too_many_affected_neighbors')
    if not affected:
        return dict(result,status='no_movement_needed',reason=None,
                    max_neighbor_translation_inches=0.,total_neighbor_translation_inches=0.)
    fixed=unary_union([g for s,g in sorted(neighbors.items()) if s not in affected])
    options={}
    maximum=config['max_translation_inches']/12
    radii=np.arange(config['translation_step_inches']/12,maximum+tol,config['translation_step_inches']/12)
    if not len(radii) or abs(radii[-1]-maximum)>tol:
        radii=np.r_[radii,maximum]
    for sid in affected:
        options[sid]=[]
        for radius in radii:
            for angle in np.arange(config['directions'])*2*math.pi/config['directions']:
                dx,dy=float(radius*math.cos(angle)),float(radius*math.sin(angle))
                moved=translate(neighbors[sid],dx,dy)
                if not wall.buffer(tol).covers(moved):
                    continue
                if moved.intersects(candidate) or moved.distance(candidate)<gap-tol:
                    continue
                if not fixed.is_empty and (moved.intersects(fixed) or moved.distance(fixed)<gap-tol):
                    continue
                options[sid].append((radius,dx,dy,moved))
        if not options[sid]:
            return dict(result,reason=f'no_sampled_slack_for_{sid}')
    # Keep alternative directions at each radius so neighbor/neighbor conflicts
    # can be resolved jointly instead of selecting each translation independently.
    beam=[(0.,{})]
    for sid in sorted(affected,key=lambda s:(len(options[s]),s)):
        expanded=[]
        for cost,plan in beam:
            for radius,dx,dy,moved in options[sid]:
                if any(moved.intersects(other[2]) or moved.distance(other[2])<gap-tol for other in plan.values()):
                    continue
                expanded.append((cost+radius**2,{**plan,sid:(dx,dy,moved)}))
        if not expanded:
            return dict(result,reason='neighbor_translations_conflict')
        expanded.sort(key=lambda entry:entry[0])
        beam=expanded[:config['beam_width']]
    plan=beam[0][1]
    # Independent endpoint verification includes stationary and moved neighbors.
    final={s:plan[s][2] if s in plan else g for s,g in neighbors.items()}
    for sid,(dx,dy,moved) in plan.items():
        if not wall.buffer(tol).covers(moved) or moved.intersects(candidate) or moved.distance(candidate)<gap-tol:
            raise AssertionError('Invalid translation endpoint')
        if math.hypot(dx,dy)>maximum+tol:
            raise AssertionError('Translation exceeds movement budget')
        for other,g in final.items():
            if other!=sid and (moved.intersects(g) or moved.distance(g)<gap-tol):
                raise AssertionError('Endpoint violates another stone clearance')
    lengths=[12*math.hypot(dx,dy) for dx,dy,_ in plan.values()]
    return dict(result,status='bounded_translation_witness',reason=None,
                translations_ft={s:[dx,dy] for s,(dx,dy,_) in plan.items()},
                max_neighbor_translation_inches=max(lengths),total_neighbor_translation_inches=sum(lengths))
