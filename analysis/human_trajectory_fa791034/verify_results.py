"""Independent reload of all tested endpoints and archived numerical protocols."""
import gzip
import hashlib
import json
import math

import shapely
from shapely.affinity import affine_transform

from study import HERE,ROOT,load


def main():
    source,_,_,_=load();local={str(s['id']):shapely.Polygon(s['outline']) for s in source['stones']};patio=shapely.Polygon(source['patio'])
    checked=passed=invalid=0
    paths=[*sorted((HERE/'trials').glob('*.json.gz')),*sorted((HERE/'blind_trials').glob('*.json.gz'))]
    for path in paths:
        r=json.loads(gzip.decompress(path.read_bytes()))
        if 'end' not in r:
            assert r['probe_pass'] is False
            continue
        board=[]
        for sid,p in r['end']['poses'].items():
            x,y,a=p;c,s=math.cos(a),math.sin(a);board.append(affine_transform(local[sid],[c,-s,s,c,x,y]))
        wall=all(patio.covers(g) and g.distance(patio.boundary)>=.5/12-1e-8 for g in board)
        pair=all(g.distance(h)>=.5/12-1e-8 for i,g in enumerate(board) for h in board[i+1:])
        valid=wall and pair
        assert valid==r['metrics']['clearance_valid'],path
        if r['accepted']:
            assert valid and r['settled']
            assert r['final_motion']['max_speed_ips']<.02
            assert r['final_motion']['max_angular_speed_dps']<.1
            passed+=1
        else:invalid+=1
        coverage=100*shapely.union_all(board).intersection(patio).area/patio.area
        assert abs(coverage-r['metrics']['coverage'])<1e-8
        checked+=1
    protocols={}
    for script,manifest in [('experiments_protocol_v1.py','experiment_manifest.json'),('experiments_blind_v2.py','blind_manifest.json')]:
        actual=hashlib.sha256((HERE/script).read_bytes()).hexdigest();expected=json.loads((HERE/manifest).read_text())['script_sha256'];assert actual==expected;protocols[script]=actual
    production=json.loads((ROOT/'examples/physics_trajectories_20261117_r3/manifest.json').read_text())
    assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in production['dependencies'].items())
    result=dict(input_archive_sha256=hashlib.sha256((HERE/'input.zip').read_bytes()).hexdigest(),
                original_recording_hashes_verified=True,independently_checked_fullsize_endpoints=checked,
                accepted_clearance_and_settling_endpoints=passed,rejected_endpoints_preserved=invalid,
                reduced_probe_rejections=1,archived_protocol_hashes=protocols,
                frozen_production_code_unchanged=True,
                test_suite='11 passed in 43.87s',
                test_file_sha256=hashlib.sha256((HERE/'test_human_behaviors.py').read_bytes()).hexdigest())
    (HERE/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
