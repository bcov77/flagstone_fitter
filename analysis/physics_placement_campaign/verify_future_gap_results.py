"""Independent endpoint/inventory checks for v5 continuation witnesses."""
import argparse
import json
from pathlib import Path
import shapely
import future_gap_campaign as job


def verify(out):
    m=job.load(out);targets={t['id']:t for t in m['targets']};plans=0;fillers=0
    for task in m['tasks']:
        path=out/'results'/f"{task['id']}.json"
        if not path.exists():raise RuntimeError('Complete the run before final verification')
        r=json.loads(path.read_text());t=targets[task['target']];c=r['candidate'];a=r['assessment']
        board=job.board_for(t,c);roi=shapely.from_wkt(t['roi'])
        assert all(v>=-1e-10 for v in a['metrics'].values())
        assert all(0<=v['score']<=1 for v in r['provisional_stability'].values())
        for region in a['regions']:
            geometry=shapely.from_wkt(region['polygon'])
            assert geometry.difference(roi).area<1e-6
        used=set()
        for item in a['repair_filler_plan']:
            sid=item['stone'];assert sid not in used and sid not in board and sid not in t['removed']
            g=job.v4.b.world(sid,item['pose'])
            assert abs(g.area-job.v4.b.LOCAL[sid].area)<1e-8
            assert job.v4.b.PATIO.buffer(-job.v4.b.GAP).buffer(1e-8).covers(g)
            assert all(g.distance(h)>=job.v4.b.GAP-1e-8 and not g.intersects(h) for h in board.values())
            board[sid]=g;used.add(sid);fillers+=1
        plans+=int(bool(used))
    result=dict(assessments_verified=len(m['tasks']),filler_plans_verified=plans,distinct_allocations_verified=fillers,
                verifier_sha256=job.sha(__file__),physics_steps=0,
                scope='Filler endpoints, inventory uniqueness, ROI locality and immutable source geometry; no motion-path or quality certificate')
    job.v4.atomic_json(out/'verification.json',result);print(result)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=job.DEFAULT_OUT)
    verify(ap.parse_args().out.resolve())
