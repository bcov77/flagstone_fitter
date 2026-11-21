"""Small paired physical pilot of saved 80% shape probes; no production mutations."""
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pymunk
import shapely
from shapely.affinity import affine_transform

OUT=Path(__file__).parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'analysis/physics_trajectories'))
from common import clearance,measure,shapes,state,restore,write,sha
from physics import relax,session_for,quiet,motion
from selector import apply
from future_gap_geometry import joint_profile,walls

SETTINGS=dict(gap_inches=.5,piece_hardness=18,locked_hardness=60,wall_hardness=60,
              angular_stiffness=2,translation_stiffness=2,rotation_magnitude=4,
              rotation_force=12,rotation_frequency=1,translation_magnitude=.25,
              translation_force=12,translation_frequency=1,speed=1)


def staged(document,proposal,cfg,seed):
    candidate=apply(document,proposal);meta=candidate.setdefault('physics',{})
    oldlocks={sid:v.get('locked',False) for sid,v in meta.get('pieces',{}).items()}
    original_settings=copy.deepcopy(meta.get('settings',{}))
    for s in candidate['stones']:
        meta.setdefault('pieces',{}).setdefault(str(s['id']),{})['anchor']=s['pose'][:]
    meta['settings']=SETTINGS.copy()
    selected=[p['stone'] for p in proposal['placements']]
    for sid in selected:meta['pieces'][sid]['locked']=True
    session=session_for(candidate,seed)
    geometry={sid:[(shape,[tuple(v) for v in shape.get_vertices()])
                   for shape in session.bodies[sid].shapes] for sid in selected}
    centers={sid:shapely.Polygon(session.pieces[sid].outline).centroid.coords[0] for sid in selected}
    def resize(factor):
        velocities={sid:(tuple(body.velocity),body.angular_velocity) for sid,body in session.bodies.items()}
        for sid,items in geometry.items():
            cx,cy=centers[sid]
            for shape,vertices in items:
                shape.unsafe_set_vertices([(cx+factor*(x-cx),cy+factor*(y-cy)) for x,y in vertices])
            session.space.reindex_shapes_for_body(session.bodies[sid])
        assert velocities=={sid:(tuple(body.velocity),body.angular_velocity) for sid,body in session.bodies.items()},'Growth reset velocities'
    frames=[]
    resize(.8);session.start()
    # Exactly two seconds held, matching the direct protocol: 1.5 s growth + 0.5 s full-size hold.
    for step in range(11):
        factor=.8+.02*step;resize(factor)
        frames.append(dict(scale=factor,seconds=step*.15,poses=state(session.export())['poses']))
        if step<10:session.advance(.15)
    for sid,items in geometry.items():
        for shape,original in items:
            current=[tuple(v) for v in shape.get_vertices()]
            assert np.allclose(current,original,atol=1e-12,rtol=0),'Full-size collider not restored'
    session.advance(.5)
    for sid in selected:session.pieces[sid].locked=oldlocks.get(sid,False)
    # Identical release/rebuild behavior to the frozen direct-relaxation protocol.
    session._build();session._randomize_drives();session.running=True
    settled,trace=quiet(session,cfg);direct=clearance(session.export());repair=False
    phases=[dict(name='direct',settled=settled,trace=trace)]
    if not direct['valid']:
        repair=True
        session.settings.update(gap_inches=cfg['repair_gap_inches'],rotation_magnitude=4,translation_magnitude=.25,
                                rotation_force=12,translation_force=12,angular_stiffness=2,translation_stiffness=2)
        session._build();session._randomize_drives();session.running=True;session._randomize_drives();session.advance(cfg['repair_inflate_seconds'])
        session.settings['gap_inches']=.5;session._build();session._randomize_drives();session.running=True
        session.advance(cfg['repair_deflate_seconds']);settled,trace=quiet(session,cfg)
        phases.append(dict(name='gap_cycle',settled=settled,trace=trace))
    endmotion=motion(session);attempted=session.export();c=clearance(attempted);session.stop()
    attempted['physics']['settings']=original_settings
    for s in attempted['stones']:
        sid=str(s['id']);attempted['physics']['pieces'][sid]['anchor']=s['pose'][:]
        attempted['physics']['pieces'][sid]['locked']=oldlocks.get(sid,False) if s['active'] else False
    return dict(accepted=c['valid'] and settled,settled=settled,clearance=c,
                repair_attempted=repair,attempted_document=attempted,phases=phases,
                final_motion=endmotion,growth_frames=frames,
                full_colliders_restored=True,growth_preserves_velocities=True)


def joint_totals(document,subject):
    board={s:g.simplify(.002,preserve_topology=True) for s,g in shapes(document).items()}
    g=board[subject];neighbors={s:h for s,h in board.items() if s!=subject and h.distance(g)<1.05/12}
    wl={s:h for s,h in walls(shapely.Polygon(document['patio'])).items() if h.distance(g)<1.05/12}
    profile=joint_profile(g,neighbors,wl,.5/12,dict(boundary_step_ft=.03,joint_excess_inches=.5,joint_angle_degrees=20))
    runs={s:v['longest_run_ft']*12 for s,v in profile['by_owner'].items() if v['longest_run_ft']>=.25}
    return dict(total=sum(runs.values()),wall=sum(v for s,v in runs.items() if s.startswith('wall_')),runs=runs)


def main():
    cfg=json.loads((ROOT/'analysis/physics_trajectories/config.json').read_text())
    source=json.loads((ROOT/'examples/physics_trajectories_20261117_r3/source.json').read_text())
    parent=json.loads((ROOT/'analysis/physics_placement_campaign/campaign_v1/source.json').read_text())
    assert state(source)==state(parent)
    assert {str(s['id']):s['outline'] for s in source['stones']}=={str(s['id']):s['outline'] for s in parent['stones']}
    inputs=json.loads((OUT/'candidates.json').read_text())
    desired=[('board_00_replace_60','22'),('board_00_replace_60','77'),('board_00_replace_62','118'),
             ('board_00_replace_62','114'),('board_00_replace_61','58')]
    candidates=[]
    for target,sid in desired:
        options=[c for c in inputs['candidates'] if c['target']==target and c['stone']==sid]
        candidates.append(min(options,key=lambda c:c['probe_metrics']['edge_excess_q90_inches']))
    seeds=[2026111701,2026111702,2026111703]
    provenance=dict(source_sha256=sha(ROOT/'examples/physics_trajectories_20261117_r3/source.json'),
                    candidate_index_sha256=sha(OUT/'candidates.json'),seeds=seeds,config=cfg,
                    code_sha256=sha(__file__),pymunk=pymunk.version,
                    dependencies={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'analysis/physics_trajectories/physics.py',ROOT/'analysis/physics_trajectories/common.py',ROOT/'physics_patio/patio_physics/engine.py']},
                    scope='Five saved real-opening/piece pairs, one selected pose per pair, three paired seeds. No claim of generalization. All neighbors free except a temporarily held subject.')
    write(OUT/'manifest.json',provenance)
    before=measure(source);write(OUT/'baseline_metrics.json',before)
    jobs=[]
    for ci,c in enumerate(candidates,1):
        target=inputs['targets'][c['target']]
        for method in ['direct','growth']:
            for seed in seeds:jobs.append((f'case_{ci:02}_{method}_{seed}',method,c,seed))
    for old in ['60','61','62']:
        for seed in seeds:
            pose=next(s['pose'] for s in source['stones'] if str(s['id'])==old)
            jobs.append((f'control_{old}_{seed}','retain',dict(target=f'board_00_replace_{old}',stone=old,pose=pose),seed))
    started=time.monotonic()
    for ji,(name,method,c,seed) in enumerate(jobs,1):
        path=OUT/'trials'/f'{name}.json.gz'
        if path.exists():continue
        target=inputs['targets'][c['target']];old=target['old'];sid=c['stone']
        proposal=dict(remove=[old],placements=[dict(stone=sid,pose=c['pose'])])
        result=staged(source,proposal,cfg,seed) if method=='growth' else relax(source,proposal,cfg,seed)
        end=result.pop('attempted_document');metrics=measure(end)
        assert metrics['clearance']==clearance(end)
        assert all(s['outline']==next(z['outline'] for z in source['stones'] if str(z['id'])==str(s['id'])) for s in end['stones'])
        b=shapes(source);after=shapes(end);pocket=shapely.from_wkt(target['pocket']);roi=pocket.buffer(.35)
        initial_affected=[s for s,g in b.items() if s!=old and g.distance(shapes(apply(source,proposal))[sid])<.5/12]
        movement={s:math.dist(sourcepose[:2],state(end)['poses'][s][:2])*12 for s,sourcepose in state(source)['poses'].items() if s in state(end)['poses'] and s!=old}
        finalpose=state(end)['poses'][sid]
        angle=abs(math.degrees(math.remainder(finalpose[2]-c['pose'][2],2*math.pi)))
        translation=math.dist(finalpose[:2],c['pose'][:2])*12
        maxlocal=lambda m:max([v['diameter_in'] for v in m['gaps'].get('largest_circles',[]) if roi.covers(shapely.Point(v['center']))],default=None)
        local_before=maxlocal(before);local_after=maxlocal(metrics)
        record=dict(name=name,method=method,target=c['target'],stone=sid,old=old,seed=seed,
                    proposal=proposal,probe_metrics=c.get('probe_metrics'),before_metrics=before,end_metrics=metrics,
                    end=state(end),initial_affected=initial_affected,movement_inches=movement,
                    neighbors_moved_quarter_inch=sum(v>=.25 for v in movement.values()),
                    second_ring_moved=[s for s,v in movement.items() if v>=.25 and s not in initial_affected],
                    max_neighbor_movement_inches=max(movement.values(),default=0),
                    candidate_translation_inches=translation,candidate_rotation_degrees=angle,
                    stayed_at_target=translation<=.5 and angle<=10,
                    local_gap_before=local_before,local_gap_after=local_after,
                    full_pocket_coverage_after=after[sid].intersection(pocket).area/pocket.area,
                    joints_before=joint_totals(source,old),joints_after=joint_totals(end,sid),
                    **result)
        record['useful_screen']=bool(record['accepted'] and record['stayed_at_target'] and
                                     local_before is not None and local_after is not None and local_after<=local_before-.25 and
                                     metrics['gaps']['largest_interior_inches']<=before['gaps']['largest_interior_inches']+.5 and
                                     metrics['gaps']['largest_edge_inches']<=before['gaps']['largest_edge_inches']+.5)
        write(path,record)
        print(f'{ji}/{len(jobs)} {name}: accepted={record["accepted"]} target={record["stayed_at_target"]} useful={record["useful_screen"]} moved={record["neighbors_moved_quarter_inch"]} max={record["max_neighbor_movement_inches"]:.2f}″ · {time.monotonic()-started:.1f}s',flush=True)


if __name__=='__main__':main()
