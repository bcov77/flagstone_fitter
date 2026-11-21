"""Human demonstration fixtures, sequence replay, and bounded operator experiments."""
import copy
import csv
import gzip
import hashlib
import json
import math
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import shapely
from shapely.affinity import affine_transform

HERE=Path(__file__).parent
ROOT=HERE.parents[1]
ARCHIVE=HERE/'input.zip'
sys.path[:0]=[str(ROOT/'physics_patio'),str(ROOT/'analysis/physics_trajectories')]
from patio_physics.engine import PhysicsSession,DT

PROXIES={}


def load():
    with zipfile.ZipFile(ARCHIVE) as z:
        manifest=json.loads(z.read('manifest.json'))
        for name in ['initial.json','timeline.jsonl','final.json']:
            assert hashlib.sha256(z.read(name)).hexdigest()==manifest[name.split('.')[0]+'_sha256']
        source=json.loads(z.read('initial.json'));final=json.loads(z.read('final.json'))
        records=[json.loads(line) for line in z.read('timeline.jsonl').splitlines()]
    assert [r['sequence'] for r in records]==list(range(1,len(records)+1))
    assert hashlib.sha256((ROOT/'physics_patio/patio_physics/engine.py').read_bytes()).hexdigest()==manifest['dependencies']['engine.py']
    return source,final,records,manifest


def tuples(value):
    return tuple(tuples(v) for v in value) if isinstance(value,list) else value


def document(source,snapshot):
    return dict(schema_version=1,units='ft',patio=source['patio'],
                stones=[dict(id=str(s['id']),outline=s['outline'],pose=snapshot['pieces'][str(s['id'])]['pose'][:],
                             active=snapshot['pieces'][str(s['id'])]['active']) for s in source['stones']],
                physics=dict(settings=snapshot['settings'].copy(),pieces={sid:dict(anchor=p['anchor'][:],locked=p['locked']) for sid,p in snapshot['pieces'].items()}))


def session_for(source,snapshot):
    d=document(source,snapshot);active={str(s['id']):s['active'] for s in d['stones']}
    for s in d['stones']:s['active']=False
    session=PhysicsSession(d)
    for sid,p in session.pieces.items():p.active=active[sid]
    session.proxy_cache=PROXIES;session._build()
    session.rng.setstate(tuples(snapshot['rng']));session.drives=copy.deepcopy(snapshot['drives'])
    session.elapsed=snapshot['elapsed'];session.accumulator=snapshot['accumulator'];session.run_number=snapshot['run_number'];session.running=snapshot['running']
    for sid,v in snapshot['velocities'].items():
        session.bodies[sid].velocity=v[:2];session.bodies[sid].angular_velocity=v[2]
    return session


def shapes(doc):
    board={}
    for s in doc['stones']:
        if not s['active']:continue
        x,y,a=s['pose'];c,t=math.cos(a),math.sin(a)
        board[str(s['id'])]=affine_transform(shapely.Polygon(s['outline']),[c,-t,t,c,x,y])
    return board


def metrics(doc):
    b=shapes(doc);p=shapely.Polygon(doc['patio']);items=list(b.items());pairs=[];wall=[];overlap=0.
    for i,(sid,g) in enumerate(items):
        d=g.distance(p.boundary)*12
        if d<.5-1.2e-7 or not p.covers(g):wall.append(sid)
        for tid,h in items[i+1:]:
            d=g.distance(h)*12
            if d<.5-1.2e-7:pairs.append([sid,tid])
            overlap+=g.intersection(h).area
    return dict(coverage=100*shapely.union_all(list(b.values())).intersection(p).area/p.area,
                nominal_coverage=100*sum(g.area for g in b.values())/p.area,active_count=len(b),
                pair_failures=len(pairs),wall_failures=len(wall),overlap_sqin=144*overlap,
                clearance_valid=not pairs and not wall)


def inspect():
    source,final,records,manifest=load()
    episodes=[];pending=[];current=None
    noops=0
    for r in records:
        if r['kind']=='event' and r['action']=='edit':
            pending.append(r)
            if r['before']['pieces']==r['after']['pieces'] and r['before']['settings']==r['after']['settings']:noops+=1
        elif r['kind']=='event' and r['action']=='simulation_start':
            current=dict(start=r,frames=[],edits=pending);pending=[]
        elif r['kind']=='frame':
            assert current is not None
            current['frames'].append(r)
        elif r['kind']=='event' and r['action']=='simulation_stop':
            current['stop']=r;current['duration']=r['simulated_seconds']-current['start']['simulated_seconds']
            episodes.append(current);current=None
    cases=[]
    for i,e in enumerate(episodes):
        placements=[r for r in e['edits'] if r['payload']['edit']['action']=='insert']
        if not placements:continue
        pre=e['edits'][0]['before']
        first=e['start'];end=e['stop'];following=[]
        j=i+1
        while j<len(episodes) and not episodes[j]['edits']:
            following.append(episodes[j]);end=episodes[j]['stop'];j+=1
        selected=[r['payload']['edit']['sid'] for r in placements if first['after']['pieces'][r['payload']['edit']['sid']]['active']]
        removed=[r['payload']['edit']['sid'] for r in e['edits'] if r['payload']['edit']['action']=='delete' and pre['pieces'][r['payload']['edit']['sid']]['active']]
        if not selected:continue
        case=dict(id=f'run_{first["after"]["run_number"]:02d}',run=first['after']['run_number'],
                  pre=pre,proposed=first['after'],first_stop=e['stop']['after'],last_stop=end['after'],
                  selected=selected,removed=removed,
                  operations=[r['payload']['edit'] for r in e['edits']],
                  sequences=[r['sequence'] for r in e['edits']],
                  start_sequence=first['sequence'],stop_sequence=end['sequence'],
                  pulse_durations=[e['duration']]+[q['duration'] for q in following],
                  before_metrics=metrics(document(source,pre)),first_metrics=metrics(document(source,e['stop']['after'])),
                  last_metrics=metrics(document(source,end['after'])))
        cases.append(case)
    evidence=dict(zip_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),manifest=manifest,
                  events=sum(r['kind']=='event' for r in records),frames=sum(r['kind']=='frame' for r in records),
                  edit_actions=dict(Counter(r['payload']['edit']['action'] for r in records if r.get('action')=='edit')),
                  no_op_edits=noops,pulses=len(episodes),positive_pulses=sum(e['duration']>0 for e in episodes),
                  initial=metrics(source),final=metrics(final),
                  actual_peak_translation=max(v['center_inches'] for r in records if r.get('action')=='simulation_stop' for v in r['motion_peaks'].values()),
                  case_count=len(cases),cases=[{k:v for k,v in c.items() if k not in ['pre','proposed','first_stop','last_stop','operations']} for c in cases])
    (HERE/'fixtures.json').write_text(json.dumps(cases,separators=(',',':')))
    (HERE/'inspection.json').write_text(json.dumps(evidence,indent=2))
    print('Summary',json.dumps({k:v for k,v in evidence.items() if k not in ['manifest','cases']}),flush=True)
    for c in cases:print(c['id'],c['removed'],'→',c['selected'],'pulses',list(map(lambda x:round(x,3),c['pulse_durations'])),'area',round(c['before_metrics']['coverage'],2),'→',round(c['last_metrics']['coverage'],2),'clearance',c['last_metrics']['pair_failures'],c['last_metrics']['wall_failures'],flush=True)


def replay(write=True):
    source,final,records,manifest=load()
    initial=records[0]['after'];s=session_for(source,initial)
    maxima=dict(position_ft=0.,angle_rad=0.,velocity_fps=0.,angular_velocity_rps=0.)
    frames=events=0;per_run={};started=time.monotonic()
    for r in records[1:]:
        if r['kind']=='frame':
            steps=round((r['elapsed']-s.elapsed)/DT)
            assert steps>=0
            if steps:s.advance(steps*DT)
            run=str(r['run_number']);err=0.
            for sid,p in r['poses'].items():
                b=s.bodies[sid];position=math.dist(tuple(b.position),p[:2]);angle=abs(b.angle-p[2])
                maxima['position_ft']=max(maxima['position_ft'],position);maxima['angle_rad']=max(maxima['angle_rad'],angle);err=max(err,position)
                v=r['velocities'][sid];maxima['velocity_fps']=max(maxima['velocity_fps'],math.dist(tuple(b.velocity),v[:2]));maxima['angular_velocity_rps']=max(maxima['angular_velocity_rps'],abs(b.angular_velocity-v[2]))
            per_run[run]=max(per_run.get(run,0.),err);frames+=1
        else:
            a=r['action']
            if a=='edit':s.edit(**r['payload']['edit'])
            elif a=='simulation_start':
                s.start()
                assert json.dumps(s.rng.getstate())==json.dumps(r['after']['rng'])
                assert json.dumps(s.drives)==json.dumps(r['after']['drives'])
            elif a=='simulation_stop':s.stop()
            elif a=='undo':s.undo()
            elif a=='recording_finished':s.stop()
            else:raise AssertionError(f'Unhandled action {a}')
            events+=1
    final_error=max(math.dist(s.pieces[str(stone['id'])].pose[:2],stone['pose'][:2]) for stone in final['stones'] if stone['active'])
    result=dict(frames=frames,events=events,maximum_errors=maxima,per_run_position_errors_ft=per_run,
                final_position_error_ft=final_error,seconds=time.monotonic()-started,
                scope='Physics replay with recorded choices, settings and rhythms; not autonomous selection or generalization.')
    if write:(HERE/'replay_verification.json').write_text(json.dumps(result,indent=2))
    print('Replay',json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='replay':replay()
    else:inspect()
