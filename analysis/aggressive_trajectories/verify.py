"""Independent full-outline acceptance/chain/selection/archive verification."""
import argparse
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'physics_trajectories'))
from common import read,write,sha,restore,state,clearance,attempt_files
from aggressive_policy import eligible,probabilities,action_key


def verify(out):
    out=Path(out);m=read(out/'manifest.json');source=read(out/'source.json');cfg=m['config']
    assert sha(out/'source.json')==m['source_sha256']
    for name,digest in m['dependencies'].items():assert sha(HERE.parents[1]/name)==digest,name
    inherited=m.get('continuation',{}).get('inherited_attempt_hashes',{})
    for name,digest in inherited.items():assert sha(out/name)==digest
    count=kept=runs=0;families={f:dict(attempts=0,accepted=0) for f in cfg['family_weights']}
    for folder in sorted((out/'runs').glob('run_*')):
        previous=state(source);records=[];tried=set()
        for i,path in enumerate(attempt_files(folder),1):
            r=read(path);assert r['number']==i and r['start']==previous
            assert r['run']==int(folder.name.split('_')[1])
            if m.get('compute_revision')=='proposal_retry_v2' and str(path.relative_to(out)) not in inherited:
                assert r.get('compute_revision')=='proposal_retry_v2' and r['chosen'] is not None
            if r['chosen']:
                p=r['chosen'];doc=restore(source,r['start']);assert eligible(doc,p,cfg)
                key=action_key(doc,p);assert key not in tried;tried.add(key)
                probabilities_now,_=probabilities(r['candidates'],r['mode'],cfg);saved=r['selection']['probabilities']
                assert len(saved)==len(probabilities_now) and all(abs(a-b)<1e-12 for a,b in zip(saved,probabilities_now))
                index=r['selection']['index'];assert r['candidates'][index]==p and saved[index]>0
                assert math.isclose(sum(saved),1,abs_tol=1e-10) and saved[index]==r['selection']['probability']
                families[p['family']]['attempts']+=1;families[p['family']]['accepted']+=r['accepted']
                old={s for s in r['start']['poses']};expected=(old-set(p['remove']))|{v['stone'] for v in p['placements']}
                assert set(r['attempted_end']['poses'])==expected
            if r['accepted']:
                assert r['settled'] and clearance(restore(source,r['end']))['valid'] and r['attempted_end']==r['end']
                assert r['final_motion']['max_speed_ips']<cfg['max_speed_ips']
                assert r['final_motion']['max_angular_speed_dps']<cfg['max_angular_speed_dps'];kept+=1
            else:assert r['end']==r['start'] and r['end_metrics']==r['start_metrics']
            previous=r['end'];records.append(r);count+=1
        if (folder/'summary.json').exists():
            assert len(records)==cfg['attempts_per_run']
            assert state(read(folder/'final.json'))==previous;runs+=1
        a=read(folder/'archive.json')
        for label in ['best_coverage','best_quality']:
            item=a[label]
            assert state(read(folder/f'{label}.json'))==item['frame']
            assert clearance(restore(source,item['frame']))['valid']
            assert item['attempt']==0 or any(r['end']==item['frame'] for r in records)
        assert a['best_coverage']['metrics']['area_percent']>=max([m['start_metrics']['area_percent']]+[r['end_metrics']['area_percent'] for r in records])-1e-9
    result=dict(runs_verified=runs,attempts_verified=count,accepted_clearances_verified=kept,families=families,
                scope='Compute/source hashes, continuity, rollback, original-outline clearance, settling, inventory transitions, >=45-degree edits, no downsizing, contextual reuse, marginal probabilities and champion persistence. Stored gap audits not recomputed.')
    write(out/'trajectory_verification.json',result);return result


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);print(json.dumps(verify(p.parse_args().out),indent=2))
