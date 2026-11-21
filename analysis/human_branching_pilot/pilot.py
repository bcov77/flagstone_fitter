"""Bounded branching research pilot. Never promotes an unsettled/invalid board."""
import argparse
import copy
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'human_trajectory_fa791034'))
from study import load,document,metrics
from experiments import broad_void
from proposals import canonical,board_key,generate,eligible,joint_summary,shapes,shapely
from physics import session_for,quiet,motion
from selector import apply


def evaluate(doc):
    m=metrics(doc);j=joint_summary(shapes(doc),shapely.Polygon(doc['patio']))
    return dict(**m,broad_void_sqft=broad_void(doc),wall_joint_inches=j['wall'],joint_inches=j['total'])


def rank(m,root):
    # Preserve wall fits before optimizing voids and area; compare to root so
    # repeated small losses cannot quietly accumulate down a branch.
    return (round(max(0,root['wall_joint_inches']-m['wall_joint_inches'])/3),
            round(m['broad_void_sqft'],3),-round(m['joint_inches'],1),-m['coverage'])


def dominates(a,b):
    av=[-a['coverage'],a['broad_void_sqft'],-a['wall_joint_inches'],-a['joint_inches']]
    bv=[-b['coverage'],b['broad_void_sqft'],-b['wall_joint_inches'],-b['joint_inches']]
    return all(x<=y+1e-8 for x,y in zip(av,bv)) and any(x<y-1e-8 for x,y in zip(av,bv))


def run_physics(doc,action,cfg,seed):
    if action is not None and not eligible(doc,action,cfg):raise ValueError('Ineligible deliberate edit')
    d=apply(doc,action) if action is not None else copy.deepcopy(doc)
    meta=d.setdefault('physics',{});original=copy.deepcopy(meta)
    for s in d['stones']:
        sid=str(s['id']);meta.setdefault('pieces',{}).setdefault(sid,{})['anchor']=s['pose'][:]
    meta['settings'].update(gap_inches=.5,piece_hardness=60,wall_hardness=80,
                           angular_stiffness=2,translation_stiffness=2,
                           **{k:cfg[k] for k in ['rotation_magnitude','rotation_force','rotation_frequency',
                                                'translation_magnitude','translation_force','translation_frequency']})
    session=session_for(d,seed)
    if action is not None:
        for duration in cfg['pulse_seconds']:
            session.start();session.advance(duration);session.stop()
    settled,trace=quiet(session,cfg);velocity=motion(session);end=session.export();session.stop()
    end['physics']=original
    for s in end['stones']:end['physics']['pieces'][str(s['id'])]['anchor']=s['pose'][:]
    m=evaluate(end)
    return end,dict(settled=settled,accepted=bool(settled and m['clearance_valid']),metrics=m,
                    quiet_trace=trace,motion=velocity)


def search(doc,cfg,focus=None):
    raw=evaluate(doc);root,result=run_physics(doc,None,cfg,cfg['seed'])
    # Root is explicitly the quiet control. Raw recorded pose is also retained.
    nodes=[dict(id=0,parent=None,depth=0,action=None,document=root,**result)]
    frontier=[nodes[0]];trials=[];cache=set()
    for depth in range(1,cfg['depth']+1):
        children=[]
        for parent in frontier:
            proposals=generate(parent['document'],cfg,np.random.default_rng(cfg['seed']+parent['id']),focus,depth>1)
            limit=cfg['root_physics_budget'] if depth==1 else cfg['followup_physics_budget']
            for action in proposals[:limit]:
                key=canonical([board_key(parent['document']),action])
                if key in cache:continue
                cache.add(key);seed=cfg['seed']+len(trials)+1
                end,r=run_physics(parent['document'],action,cfg,seed)
                trial=dict(id=len(trials)+1,parent=parent['id'],depth=depth,action=action,seed=seed,document=end,**r)
                trials.append(trial)
                print(f'depth {depth} {action["family"]} {[p["stone"] for p in action["placements"]]} accepted={r["accepted"]} coverage={r["metrics"]["coverage"]:.3f}',flush=True)
                if r['accepted']:
                    node={**trial,'id':len(nodes)};nodes.append(node);children.append(node)
        frontier=sorted(children,key=lambda n:rank(n['metrics'],nodes[0]['metrics']))[:cfg['beam_width']]
        if not frontier:break
    valid=[n for n in nodes if n['accepted']]
    archive=[n['id'] for n in valid if not any(dominates(o['metrics'],n['metrics']) for o in valid)]
    best=min(valid,key=lambda n:rank(n['metrics'],nodes[0]['metrics']))['id'] if valid else None
    area=max(valid,key=lambda n:n['metrics']['coverage'])['id'] if valid else None
    return dict(raw_metrics=raw,nodes=nodes,trials=trials,pareto_archive=archive,best_quality=best,best_coverage=area,
                scope='Neighborhood focus only is supplied on demonstration cases; no human piece ID/pose. '
                      'Preserves all valid intermediate nodes, including root, and rolls back failed trials. '
                      'Approximate void/joint ranking; no complete dual-scale gap certification.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('case',choices=['run_08','run_16','run_21','original']);args=parser.parse_args()
    cfg=json.loads((HERE/'config.json').read_text());source,*_=load();focus=None
    if args.case=='original':doc=source
    else:
        case=next(c for c in json.loads((HERE.parent/'human_trajectory_fa791034/fixtures.json').read_text()) if c['id']==args.case)
        doc=document(source,case['pre']);focus=np.mean([case['proposed']['pieces'][s]['pose'][:2] for s in case['selected']],axis=0).tolist()
    dest=HERE/args.case;dest.mkdir(exist_ok=True)
    manifest=dict(case=args.case,focus=focus,config=cfg,source=board_key(doc),
                  files={str(p.relative_to(HERE.parent.parent)):canonical(p.read_text()) for p in
                         [Path(__file__),HERE/'proposals.py',HERE.parent/'physics_trajectories/physics.py',HERE.parent.parent/'physics_patio/patio_physics/engine.py']})
    if (dest/'manifest.json').exists():
        assert json.loads((dest/'manifest.json').read_text())==manifest,'Refusing stale resumption'
        if (dest/'result.json.gz').exists():print('Already complete');return
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result=search(doc,cfg,focus)
    with gzip.open(dest/'result.json.gz','wt') as f:json.dump(result,f)
    summary={k:result[k] for k in ['raw_metrics','pareto_archive','best_quality','best_coverage','scope']}
    summary.update(attempts=len(result['trials']),accepted=sum(t['accepted'] for t in result['trials']),
                   nodes=[{k:n[k] for k in ['id','parent','depth','action','accepted','metrics']} for n in result['nodes']])
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for label in ['best_quality','best_coverage']:
        if result[label] is not None:(dest/f'{label}.json').write_text(json.dumps(result['nodes'][result[label]]['document'])+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
