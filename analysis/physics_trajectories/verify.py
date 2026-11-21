"""Independently recheck saved trajectory chains, choices and accepted clearances."""
import argparse,json,math
from pathlib import Path
from common import restore,state,clearance,write,sha,read,attempt_files


def verify(out):
    out=Path(out);source=json.loads((out/'source.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    assert sha(out/'source.json')==manifest['source_sha256']
    inherited=manifest.get('continuation',{}).get('inherited_attempt_hashes',{})
    for name,expected in inherited.items():assert sha(out/name)==expected
    attempts=0;accepted=0;finished=0
    for folder in sorted((out/'runs').glob('run_*')):
        number=int(folder.name.split('_')[1]);previous=state(source);used=set();records=[]
        for index,path in enumerate(attempt_files(folder),1):
            r=read(path);assert r['number']==index and r['run']==number
            if manifest.get('compute_revision') and str(path.relative_to(out)) not in inherited:
                assert r.get('compute_revision')==manifest['compute_revision']
                if manifest['compute_revision']=='proposal_retry_v2':assert r['chosen'] is not None
            assert r['start']==previous
            expected='golden' if number%10==9 else 'silver' if number%10==0 else 'normal'
            assert r['mode']==expected
            if r['chosen']:
                p=r['chosen'];ids=[x['stone'] for x in p['placements']]
                assert 1<=len(ids)<=2 and len(set(ids))==len(ids) and not (set(ids)&used)
                used.update(ids)
                probs=r['selection']['probabilities'];pick=r['selection']['index']
                assert all(0<=v<=1 for v in probs) and math.isclose(sum(probs),1,abs_tol=1e-10)
                assert probs[pick]>0 and probs[pick]==r['selection']['probability']
                assert r['candidates'][pick]==p
            if r['accepted']:
                assert r['settled'] and clearance(restore(source,r['end']))['valid']
                assert r['attempted_end']==r['end'];accepted+=1
            else:assert r['end']==r['start'] and r['end_metrics']==r['start_metrics']
            doc=restore(source,r['end']);board_count=sum(s['active'] for s in doc['stones'])
            assert r['end_metrics']['active_count']==board_count
            # Original geometry and inventory come only from immutable source + rigid poses.
            original={str(s['id']):s['outline'] for s in source['stones']}
            assert all(s['outline']==original[str(s['id'])] for s in doc['stones'])
            previous=r['end'];attempts+=1;records.append(r)
        if (folder/'summary.json').exists():
            s=json.loads((folder/'summary.json').read_text());assert len(records)==manifest['config']['attempts_per_run']==s['attempts']
            assert sum(r['accepted'] for r in records)==s['accepted'];assert state(json.loads((folder/'final.json').read_text()))==previous;finished+=1
    result=dict(runs_verified=finished,attempts_verified=attempts,accepted_clearances_verified=accepted,
        scope='Trajectory continuity, rejected rollback, selected-piece uniqueness, conditional probabilities, original inventory and accepted endpoint clearance; gap audits retained from endpoints, not recomputed')
    write(out/'trajectory_verification.json',result);print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);verify(ap.parse_args().out)
