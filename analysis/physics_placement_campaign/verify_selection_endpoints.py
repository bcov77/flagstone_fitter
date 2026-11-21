"""Independently rebuild full-size near-fit endpoints from saved physical poses."""
import argparse
import itertools
import json
import math
from pathlib import Path

import selection_campaign as s


def verify(out):
    manifest=s.load_manifest(out)
    targets={t['id']:t for t in manifest['targets']}
    paths=[out/'results'/f"{task['id']}.json" for task in manifest['tasks']]
    if not all(p.exists() for p in paths):
        raise RuntimeError('Complete the target/piece searches before final verification')
    verified=[];total=0
    for path in paths:
        result=json.loads(path.read_text());target=targets[result['task']['target']]
        for index,c in enumerate(result['candidates']):
            total+=1
            assert abs(c['metrics']['area_sqft']-s.b.LOCAL[c['stone']].area)<1e-8
            if 'shrink_probe' in c:
                assert abs(c['shrink_probe']['metrics']['area_sqft']/c['metrics']['area_sqft']-c['shrink_probe']['linear_scale']**2)<1e-8
            if c.get('nearfit_gate',{}).get('status')!='passes_nearfit_review_screen':
                continue
            nc=manifest['config']['nearfit'];plan=c['space_plan']
            assert not set(plan['translations_ft'])&set(nc['locked_neighbors'])
            assert len(plan['translations_ft'])<=nc['max_neighbors']
            active=set(s.b.ACTIVE)-set(target['removed'])-({target['old']} if target['old'] else set())
            assert set(plan['translations_ft'])<=active
            for sid,(dx,dy) in plan['translations_ft'].items():
                pose=plan['neighbor_poses'][sid];original=s.b.STONES[sid]['pose']
                assert abs(pose[0]-original[0]-dx)<1e-10 and abs(pose[1]-original[1]-dy)<1e-10
                assert pose[2]==original[2]
            field={sid:s.b.world(sid,plan['neighbor_poses'].get(sid,s.b.STONES[sid]['pose'])) for sid in active}
            assert c['stone'] not in field
            field[c['stone']]=s.b.world(c['stone'],c['pose'])
            minimum=min(g.distance(h) for g,h in itertools.combinations(field.values(),2))
            assert minimum>=s.b.GAP-1e-8
            assert all(s.b.PATIO.buffer(-s.b.GAP).buffer(1e-8).covers(g) for g in field.values())
            maximum=max((12*math.hypot(dx,dy) for dx,dy in plan['translations_ft'].values()),default=0.)
            assert maximum<=nc['max_translation_inches']+1e-8
            verified.append(dict(task=result['task']['id'],candidate_index=index,stone=c['stone'],
                                 probe_scale=c['shrink_probe']['linear_scale'],
                                 minimum_board_pair_clearance_inches=12*minimum,max_translation_inches=maximum))
    s.atomic_json(out/'verification.json',dict(complete=True,full_size_areas_verified=total,
        nearfit_endpoints_verified=verified,physics_steps=0,verifier_sha256=s.digest(__file__),
        note='Independent full-outline endpoint recomputation; no motion paths or maximum-gap rules certified.'))
    print(f'Verified {total} full-size areas and {len(verified)} complete-board endpoint plans.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=s.DEFAULT_OUT)
    args=ap.parse_args()
    verify(args.out.resolve())
