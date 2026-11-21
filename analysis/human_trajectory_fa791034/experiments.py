"""Counterfactual protocols and blind inventory search on demonstration fixtures."""
import argparse
import copy
import json
import math
import time

import numpy as np
import shapely
from shapely.affinity import scale

from study import HERE,ROOT,load,document,session_for,metrics,shapes
from common import state,write,sha
from physics import relax,quiet,motion
from selector import coarse,pockets,fit

CASES=['run_08','run_16','run_28']
SEEDS=[1701,1702,1703]


def broad_void(doc):
    patio=shapely.Polygon(doc['patio']);step=.5/12
    xs=np.arange(step/2,patio.bounds[2],step);ys=np.arange(step/2,patio.bounds[3],step)
    xx,yy=np.meshgrid(xs,ys);points=shapely.points(np.column_stack((xx.ravel(),yy.ravel())))
    board=shapes(doc);tree=shapely.STRtree([*[g.simplify(.002,preserve_topology=True) for g in board.values()],patio.boundary])
    _,dist=tree.query_nearest(points,all_matches=False,return_distance=True)
    return float(np.sum(dist>1/12)*step**2)


def proposal(case):
    return dict(remove=case['removed'],placements=[dict(stone=s,pose=case['proposed']['pieces'][s]['pose']) for s in case['selected']])


def human_style(source,case,seed,cfg):
    s=session_for(source,case['proposed']);s.running=False;s.rng.seed(seed)
    for duration in case['pulse_durations']:
        s.start();s.advance(duration);s.stop()
    exploratory=s.export();before_quiet=metrics(exploratory)
    settled,trace=quiet(s,cfg);v=motion(s);end=s.export();s.stop()
    return end,dict(settled=settled,final_motion=v,quiet_trace=trace,exploration_metrics=before_quiet)


def bench(worker=0,workers=1):
    source,_,_,_=load();cases={c['id']:c for c in json.loads((HERE/'fixtures.json').read_text())}
    cfg=json.loads((ROOT/'analysis/physics_trajectories/config.json').read_text())
    manifest=dict(input_zip_sha256=sha(HERE/'input.zip'),fixtures_sha256=sha(HERE/'fixtures.json'),
                  script_sha256=sha(__file__),engine_sha256=sha(ROOT/'physics_patio/patio_physics/engine.py'),
                  methods=['production','human_style'],cases=CASES,seeds=SEEDS,
                  scope='Recorded piece/pose is supplied. Tests physics protocols, not autonomous choice. Full original outlines determine clearance; no layout-quality success threshold is assumed.')
    write(HERE/'experiment_manifest.json',manifest)
    jobs=[(name,method,seed) for name in CASES for method in ['production','human_style'] for seed in SEEDS]
    started=time.monotonic()
    for index,(name,method,seed) in enumerate(jobs):
        if index%workers!=worker:continue
        dest=HERE/'trials'/f'{name}_{method}_{seed}.json.gz'
        if dest.exists():continue
        case=cases[name];before=document(source,case['pre'])
        if method=='production':
            result=relax(before,proposal(case),cfg,seed);end=result.pop('attempted_document');result.pop('proposed_start',None)
        else:end,result=human_style(source,case,seed,cfg)
        m=metrics(end);post=shapes(end);pre=shapes(before)
        neighbors={sid:12*math.dist(case['pre']['pieces'][sid]['pose'][:2],next(z['pose'] for z in end['stones'] if str(z['id'])==sid)[:2]) for sid in pre if sid in post and sid not in case['selected']}
        result.update(case=name,method=method,seed=seed,end=state(end),metrics=m,
                      accepted=bool(m['clearance_valid'] and result['settled']),
                      broad_void_sqft=broad_void(end),before_metrics=case['before_metrics'],
                      candidate_shifts_inches={sid:12*math.dist(case['proposed']['pieces'][sid]['pose'][:2],next(z['pose'] for z in end['stones'] if str(z['id'])==sid)[:2]) for sid in case['selected']},
                      neighbors_moved_quarter_inch=sum(v>=.25 for v in neighbors.values()),
                      max_neighbor_inches=max(neighbors.values(),default=0),
                      input_case_sha256=sha(HERE/'fixtures.json'))
        write(dest,result)
        print(f'{index+1}/{len(jobs)} {name} {method} {seed}: valid={result["accepted"]} coverage={m["coverage"]:.3f} pair/wall={m["pair_failures"]}/{m["wall_failures"]} · {time.monotonic()-started:.1f}s',flush=True)


def target_for(source,case):
    before=document(source,case['pre']);board=shapes(before);patio=shapely.Polygon(before['patio'])
    remaining={s:g for s,g in board.items() if s not in case['removed']}
    free,patches=pockets(patio,remaining)
    desired=shapely.union_all([shapes(document(source,case['proposed']))[sid] for sid in case['selected']])
    if not patches:return None
    pocket=max(patches,key=lambda p:p.intersection(desired).area)
    return dict(id=case['id'],old=case['removed'][0] if len(case['removed'])==1 else None,pocket=pocket,free=free),board,patio


def search():
    source,_,_,_=load();cases={c['id']:c for c in json.loads((HERE/'fixtures.json').read_text())}
    local={str(s['id']):shapely.Polygon(s['outline']) for s in source['stones']}
    cfg=json.loads((ROOT/'analysis/physics_trajectories/config.json').read_text())
    results=[]
    for name in ['run_08','run_16','run_28']:
        case=cases[name];value=target_for(source,case)
        if value is None:
            results.append(dict(case=name,status='No pocket found'));continue
        target,board,patio=value;human=case['selected'][0]
        old_fit=fit(proposal(case)['placements'],target,local,board,patio,cfg)
        by_scale={}
        for factor in [1.,.8]:
            scaled={s:scale(g,factor,factor,origin='centroid') for s,g in local.items()}
            probe=fit(proposal(case)['placements'],target,scaled,board,patio,cfg)
            # Blind inventory pool: all currently unused pieces, with no human-ID bonus.
            available=[s for s in local if s not in board or s in case['removed']]
            config={**cfg,'piece_budget':len(local),'coarse_per_target':1000}
            ranked=coarse(scaled,available,target['pocket'],target['free'],config,np.random.default_rng(0))
            unique=[]
            for cost,sid,pose in ranked:
                if sid not in unique:unique.append(sid)
            matching=[(i+1,cost,pose) for i,(cost,sid,pose) in enumerate(ranked) if sid==human]
            by_scale[str(factor)]=dict(human_proposed_pose_passes_current_fit=probe is not None,
                                       human_proposed_pose_fit=probe,
                                       human_piece_rank=unique.index(human)+1 if human in unique else None,
                                       human_best_pose=matching[0] if matching else None,
                                       available_piece_count=len(available),searched_piece_count=len(unique),
                                       top10=unique[:10],top30=unique[:30])
        results.append(dict(case=name,human_piece=human,removed=case['removed'],pocket_area_sqft=target['pocket'].area,
                            full_size_current_gate_pass=old_fit is not None,scales=by_scale,
                            caveat='Research retrieval on the demonstration itself. Coarse rank is not human fit quality; no held-out validation.'))
        print(json.dumps(results[-1]),flush=True)
    write(HERE/'proposal_retrieval.json',results)


def polish():
    source,final,records,_=load();cfg=json.loads((ROOT/'analysis/physics_trajectories/config.json').read_text())
    cases={c['id']:c for c in json.loads((HERE/'fixtures.json').read_text())}
    snapshots=[('final',records[-1]['after'])]+[(name+'_before',cases[name]['pre']) for name in CASES]
    for name,snapshot in snapshots:
        session=session_for(source,snapshot);session.running=False
        settled,trace=quiet(session,cfg);v=motion(session);end=session.export();session.stop()
        result=dict(name=name,settled=settled,final_motion=v,trace=trace,metrics=metrics(end),
                    broad_void_sqft=broad_void(end),end=state(end),
                    source='Recorded final or pre-move state, ordinary production quiet protocol, no insertion.')
        write(HERE/'polish'/f'{name}.json.gz',result)
        print('Polish',name,result['metrics'],'settled',settled,flush=True)


def blind():
    """Evaluate all top-ten distinct retrieved pieces, without choosing by human ID."""
    source,_,_,_=load();cases={c['id']:c for c in json.loads((HERE/'fixtures.json').read_text())}
    case=cases['run_16'];target,board,patio=target_for(source,case)
    cfg=json.loads((ROOT/'analysis/physics_trajectories/config.json').read_text())
    reduced={str(s['id']):scale(shapely.Polygon(s['outline']),.8,.8,origin='centroid') for s in source['stones']}
    available=[sid for sid in reduced if sid not in board or sid in case['removed']]
    config={**cfg,'piece_budget':len(reduced),'coarse_per_target':1000}
    write(HERE/'blind_manifest.json',dict(script_sha256=sha(__file__),input_zip_sha256=sha(HERE/'input.zip'),
                                         fixtures_sha256=sha(HERE/'fixtures.json'),case='run_16',seed=1701,
                                         budget=10,scale=.8,inventory_policy='All currently unused pieces; no human-ID preference'))
    ranked=coarse(reduced,available,target['pocket'],target['free'],config,np.random.default_rng(0))
    choices=[];seen=set()
    for cost,sid,pose in ranked:
        if sid in seen:continue
        seen.add(sid);choices.append((cost,sid,pose))
        if len(choices)==10:break
    for rank,(cost,sid,pose) in enumerate(choices,1):
        dest=HERE/'blind_trials'/f'rank_{rank:02d}_{sid}.json.gz'
        if dest.exists():continue
        placement=[dict(stone=sid,pose=pose)]
        probe=fit(placement,target,reduced,board,patio,cfg)
        row=dict(rank=rank,stone=sid,pose=pose,coarse_cost=cost,probe_pass=probe is not None,
                 input_case='run_16',seed=1701,scope='Demonstration-board prototype; no human-ID preference or held-out validation.')
        if probe is not None:
            candidate=copy.deepcopy(case)
            snapshot=copy.deepcopy(case['pre'])
            for old in case['removed']:snapshot['pieces'][old]['active']=False
            snapshot['pieces'][sid].update(active=True,locked=False,pose=pose[:],anchor=pose[:])
            snapshot['velocities']={key:snapshot['velocities'].get(key,[0,0,0]) for key,p in snapshot['pieces'].items() if p['active']}
            snapshot.update(running=False,drives={},elapsed=0.,accumulator=0.)
            candidate.update(selected=[sid],proposed=snapshot)
            end,result=human_style(source,candidate,1701,cfg)
            row.update(result,end=state(end),metrics=metrics(end),broad_void_sqft=broad_void(end))
            row['accepted']=row['settled'] and row['metrics']['clearance_valid']
        write(dest,row)
        print('Blind',rank,sid,'probe',row['probe_pass'],'accepted',row.get('accepted'),'area',row.get('metrics',{}).get('coverage'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['bench','search','polish','blind']);parser.add_argument('--worker',type=int,default=0);parser.add_argument('--workers',type=int,default=1);a=parser.parse_args()
    if a.mode=='bench':bench(a.worker,a.workers)
    elif a.mode=='search':search()
    elif a.mode=='polish':polish()
    else:blind()
