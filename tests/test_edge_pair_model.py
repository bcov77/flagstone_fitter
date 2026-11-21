"""A known-feasible two-stone closure exercises the full candidate model."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from shapely.geometry import Polygon

from flagstone.phase1 import Layout, local_shape, save_json


def test_known_feasible_two_stone_closure(tmp_path):
    ids=tuple(f'T{i}' for i in range(1,8))
    shape=local_shape([(0,0),(.5,0),(.5,1),(0,1)])
    poses=np.asarray([[1+i*.55+.25,.55,0] for i in range(7)])
    layout=Layout(Polygon([(0,0),(10,0),(10,5),(0,5)]),{sid:shape for sid in ids},poses,
                  np.asarray([True]*5+[False]*2),ids,np.random.default_rng(0),units='ft')
    original=tmp_path/'input.json';save_json(layout,original)
    pool=tmp_path/'pool';pool.mkdir();save_json(layout,pool/'pool_parent.json')
    history=[dict(step=i+1,stone_id=ids[i],pose=poses[i].tolist(),front_x=1+i*.55+.5,
                  lower_diameter_in=0,upper_diameter_in=1.5,center=[1+i*.55+.25,.025],passed=True)
             for i in range(5)]
    (pool/'candidate_pool.json').write_text(json.dumps(dict(parent_history=history,
        settings=dict(input=str(original),edge_index=0),start_ft=1,depth_ft=1,
        candidates=[dict(stone_id=ids[i],pose=poses[i].tolist(),score=0) for i in (5,6)])))
    script=Path(__file__).resolve().parents[1]/'scripts/five_stone_edge_start.py'
    output=tmp_path/'result'
    # Keep the fixture's progress journal separate from the real milestone log.
    result=subprocess.run([sys.executable,str(script),'--pool-pair',str(pool),'--output',str(output)],
                          cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    verification=json.loads((output/'accepted/verification.json').read_text())
    assert verification['passed']
    assert [r['stone_id'] for r in verification['results']]==list(ids)
    assert all(r['upper_diameter_in']<2 and not r['clearance_errors'] for r in verification['results'])
    assert all(r['minimum_edge_clearance_in']>=.5 for r in verification['results'])
    assert len(list((output/'accepted').glob('step_*.png')))==7
