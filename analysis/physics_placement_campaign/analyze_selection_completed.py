"""Summarize completed geometry searches; never equate screens with human quality."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import selection_campaign as s
import shapely


def analyze(out):
    manifest=s.load_manifest(out);targets={t['id']:t for t in manifest['targets']}
    paths=[out/'results'/f"{t['id']}.json" for t in manifest['tasks']]
    if not all(p.exists() for p in paths):
        raise RuntimeError('Search is incomplete')
    scales=defaultdict(Counter);failures=defaultdict(Counter);families=defaultdict(Counter)
    ready=set();near=set();original_ready=set();original_near=set()
    scale_actions=defaultdict(set);scale_targets=defaultdict(set);small_moves=set();large_moves=set()
    boards=defaultdict(lambda:dict(ready=set(),near=set()))
    stats=Counter();examples=[];per_target=defaultdict(Counter)
    for path in paths:
        result=json.loads(path.read_text());t=targets[result['task']['target']]
        for index,c in enumerate(result['candidates']):
            stats['poses']+=1;families[t['family']]['poses']+=1
            key=(t['id'],c['stone']);old=c['stone']==t['old']
            if c['fit_gate']['status']=='passes_provisional_screen':
                stats['fixed_neighbor_passing_poses']+=1
                (original_ready if old else ready).add(key)
                if not old:
                    boards[t['context']]['ready'].add(key);per_target[t['id']]['ready']+=1
            if 'shrink_probe' not in c:
                continue
            factor=str(c['shrink_probe']['linear_scale']);z=scales[factor]
            z['probes']+=1
            shape=c['shrink_probe']['fit_gate']['status']=='passes_provisional_screen'
            z['shape_screen_passes']+=int(shape)
            if shape:
                for k,v in c['nearfit_gate']['checks'].items():
                    if not v:failures[factor][k]+=1
                if c.get('space_plan',{}).get('reason'):
                    reason=c['space_plan']['reason']
                    if reason.startswith('no_sampled_slack_for_'):
                        reason='no_sampled_slack_for_neighbor'
                    failures[factor][reason]+=1
            if c['nearfit_gate']['status']!='passes_nearfit_review_screen':
                continue
            stats['nearfit_passing_poses']+=1;families[t['family']]['nearfit_passing_poses']+=1
            z['nearfit_passing_poses']+=1
            if old:
                original_near.add(key);z['original_piece_poses']+=1
                continue
            z['reserve_piece_poses']+=1
            near.add(key);scale_actions[factor].add(key);scale_targets[factor].add(t['id'])
            boards[t['context']]['near'].add(key);per_target[t['id']]['near']+=1
            movement=c['space_plan']['max_neighbor_translation_inches']
            (large_moves if movement>.75+1e-8 else small_moves).add(key)
            stats['reserve_poses_above_old_movement_budget']+=int(movement>.75+1e-8)
            examples.append(dict(task=result['task']['id'],candidate_index=index,target=t['id'],stone=c['stone'],
                                 family=t['family'],context=t['context'],synthetic=bool(t['removed']),
                                 scale=float(factor),max_neighbor_translation_inches=movement,
                                 neighbor_count=len(c['space_plan']['translations_ft']),
                                 probe_edge_tail_inches=c['shrink_probe']['metrics']['edge_excess_q90_inches'],
                                 probe_coverage=c['shrink_probe']['metrics']['coverage'],
                                 physical_overlap_fraction=c['metrics']['physical_overlap_sqft']/c['metrics']['area_sqft']))
    others=set().union(*(v for k,v in scale_actions.items() if k!='0.8'))
    # Synthetic deletion contexts repeat some identical local openings. Preserve
    # action family/original while collapsing matching masks and local neighbors.
    groups=[];group_for={}
    for t in manifest['targets']:
        p=shapely.from_wkt(t['pocket'])
        active=set(s.b.ACTIVE)-set(t['removed'])-({t['old']} if t['old'] else set())
        local=tuple(sorted(sid for sid in active if s.b.ORIGINAL[sid].intersects(p.buffer(.2))))
        matched=None
        for i,(ref,g,ids) in enumerate(groups):
            if t['family']!=ref['family'] or t['old']!=ref['old'] or local!=ids:
                continue
            if max(abs(a-b) for a,b in zip(p.bounds,g.bounds))<1e-5 and p.symmetric_difference(g).area<1e-6:
                matched=i;break
        if matched is None:
            matched=len(groups);groups.append((t,p,local))
        group_for[t['id']]=matched
    collapse=lambda actions:{(group_for[target],stone) for target,stone in actions}
    dr,dn=collapse(ready),collapse(near)
    summary=dict(tasks=len(paths),targets=len(targets),counts=dict(stats),families={k:dict(v) for k,v in families.items()},
        distinct_reserve_actions=dict(fixed_neighbor=len(ready),nearfit=len(near),nearfit_without_fixed_neighbor=len(near-ready),
                                     either=len(ready|near),eighty_only=len(scale_actions['0.8']-others-ready),
                                     witnessed_only_above_old_movement_budget=len(large_moves-small_moves-ready)),
        openings_with_reserve_choices=dict(fixed_neighbor=len({k[0] for k in ready}),nearfit=len({k[0] for k in near}),
                                          either=len({k[0] for k in ready|near})),
        original_piece_actions=dict(fixed_neighbor=len(original_ready),nearfit=len(original_near)),
        repeated_local_opening_adjustment=dict(opening_groups=len(groups),fixed_neighbor_actions=len(dr),
            nearfit_actions=len(dn),additional_nearfit_actions=len(dn-dr),either_actions=len(dr|dn),
            fixed_neighbor_openings=len({k[0] for k in dr}),either_openings=len({k[0] for k in dr|dn}),
            grouped_targets={str(i):[tid for tid,g in group_for.items() if g==i] for i in range(len(groups))}),
        scales={k:dict(v,distinct_reserve_actions=len(scale_actions[k]),openings=len(scale_targets[k])) for k,v in scales.items()},
        shape_passing_failure_counts={k:dict(v) for k,v in failures.items()},
        contexts={k:{group:len(actions) for group,actions in v.items()} for k,v in boards.items()},
        per_target={k:dict(v) for k,v in per_target.items()},
        caveats=['Actions are opening/piece pairs, not independent boards or compatible simultaneous moves.',
                 'Failure counts overlap; missing witnesses are not impossibility proofs.',
                 'Screen passes and geometric endpoints are not human approval, replacement value, or physics outcomes.',
                 'All contexts share one board lineage; insertion openings are synthetic deletions.'])
    dest=out/'completed_analysis';dest.mkdir(exist_ok=True)
    s.atomic_json(dest/'summary.json',summary);s.atomic_json(dest/'nearfit_examples.json',examples)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=s.DEFAULT_OUT)
    analyze(ap.parse_args().out.resolve())
