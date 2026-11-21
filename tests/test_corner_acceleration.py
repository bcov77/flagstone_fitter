"""Native proposals and approximate oracles cannot bypass original acceptance."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import corner_acceleration as helpers
import corner_growth as engine
from flagstone.phase1 import Layout, save_json, transformed


@pytest.fixture
def trial(tmp_path,monkeypatch):
    shapes={'1':box(-.63,-.63,.63,.63),'2':box(-.3,-.3,.3,.3),'3':box(-.3,-.3,.3,.3)}
    layout=Layout(Polygon([(0,0),(4,0),(4,4),(0,4)]),shapes,np.zeros((3,3)),np.zeros(3,bool),
                  tuple(shapes),np.random.default_rng(1),units='ft')
    source=tmp_path/'source.json';save_json(layout,source)
    monkeypatch.setattr(engine,'SOURCE',str(source))
    frame=engine.frame;monkeypatch.setattr(engine,'frame',lambda patio,vertex=0:frame(patio,0))
    root=tmp_path/'trial';engine.declare(root);g=engine.GAP
    history=[dict(step=1,stone_id='1',pose=[.63+g,.63+g,0]),
             dict(step=2,stone_id='2',pose=[1.26+2*g+.3,.3+g,0]),
             dict(step=3,stone_id='3',pose=[.3+g,1.26+2*g+.3,0])]
    parent=tmp_path/'parent';parent.mkdir();engine.write(parent/'history.json',history[:1])
    return root,parent,history


@pytest.mark.parametrize('empty',[False,True])
def test_indexed_coverage_matches_exhaustive_strict_distance(trial,monkeypatch,empty):
    root,parent,history=trial;_,contract,layout=engine.load_trial(root)
    candidates=[] if empty else [dict(stone_id=r['stone_id'],pose=r['pose']) for r in history[1:]]
    # Include exactly-on-threshold and neighboring points as well as interior points.
    g=transformed(layout.shapes['2'],history[1]['pose']);a,b,d,e=g.bounds
    pts=shapely.points([[d+1/12,e],[d+1/12+1e-8,e],[d+1/12-1e-8,e],[.8,.8],[3,3]])
    monkeypatch.setattr(engine,'coverage_samples',lambda *a,**k:pts)
    data=dict(parent_history=history[:1],target=3,candidates=candidates)
    result=helpers.coverage_diagnostics(layout,contract,data)
    obstacles=unary_union([layout.patio.boundary,transformed(layout.shapes['1'],history[0]['pose'])])
    polygons=[transformed(layout.shapes[r['stone_id']],r['pose']) for r in candidates]
    reference=[dict(stage=3,center=[p.x,p.y]) for p in pts if p.distance(obstacles)>=1/12 and
               (not polygons or not (shapely.distance(polygons,p)<1/12).any())]
    assert result['zero_cover']==reference


def test_native_group_rechecks_every_prefix_and_does_not_export_acceptance(trial,tmp_path):
    root,parent,history=trial
    file=tmp_path/'patterns.json'
    engine.write(file,dict(parent_history=history[:1],target=3,patterns=[dict(records=history[1:])]))
    result=helpers.check_native_patterns(root,file,parent,3,tmp_path/'checked',2)
    assert result['histories'] and not result['acceptance_authority']
    for path in result['histories']:
        evaluated=json.loads(Path(path).with_name('evaluation.json').read_text())
        assert len(evaluated['results'])==3
    assert not list((tmp_path/'checked').glob('**/accepted'))
    payload=json.loads(file.read_text());payload['patterns'][0]['records'][0]['stone_id']='1';engine.write(file,payload)
    with pytest.raises(ValueError):helpers.check_native_patterns(root,file,parent,3,tmp_path/'bad',2)


def test_native_uses_physical_bin_and_search_only_focus(trial,tmp_path):
    root,parent,history=trial
    data=helpers.native_input(root,parent,3,tmp_path/'input.json')
    _,_,layout=engine.load_trial(root)
    assert Polygon(data['patio']).equals(layout.patio)
    assert len(data['fixed'])==1
    assert {r['stone_id'] for r in data['items']}=={'2','3'}
    assert data['proposal_focus_padding_ft']>0


def test_jts_candidate_never_grants_a_pass_and_bad_radius_is_rejected(trial,tmp_path,monkeypatch):
    root,parent,history=trial;_,contract,layout=engine.load_trial(root)
    evaluation=engine.evaluate_history(root,history)
    rows=[]
    for n,r in enumerate(evaluation['results'],1):
        rows.append(f'{n}\t{r["center"][0]}\t{r["center"][1]}\t{r["lower_diameter_in"]/24}\t0.01')
    def run(command,**kwargs):
        return SimpleNamespace(stdout='\n'.join(rows)+'\n')
    monkeypatch.setattr(helpers.subprocess,'run',run)
    out=tmp_path/'evaluation';out.mkdir()
    result=helpers.jts_crosscheck(layout,contract,evaluation,'unused.jar',out)
    assert result['status']=='checked' and result['acceptance_authority'] is False
    assert 'passed' not in result
    corrupted=json.loads(json.dumps(evaluation));corrupted['results'][0]['upper_diameter_in']=0
    with pytest.raises(RuntimeError,match='contradicts'):
        helpers.jts_crosscheck(layout,contract,corrupted,'unused.jar',out)
    fields=rows[0].split('\t');fields[3]='999';rows[0]='\t'.join(fields)
    with pytest.raises(ValueError,match='radius disagrees'):
        helpers.jts_crosscheck(layout,contract,evaluation,'unused.jar',out)


def test_exact_gate_witness_overrides_even_a_tiny_certificate_contradiction(trial,tmp_path,monkeypatch):
    root,parent,history=trial;_,contract,layout=engine.load_trial(root)
    history[0]['pose']=[2,2,0]
    evaluation=engine.evaluate_history(root,history)
    evaluation['results'][0].update(passed=True,upper_diameter_in=2-1e-8)
    # Original geometry gives exactly a one-inch radius here. A comparison
    # tolerance must not permit a certified pass in the presence of this witness.
    rows=[f'{n}\t{1/12}\t{1/12}\t{1/12}\t0' for n in range(1,4)]
    monkeypatch.setattr(helpers.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='\n'.join(rows)))
    out=tmp_path/'evaluation';out.mkdir()
    with pytest.raises(RuntimeError,match='contradicts'):
        helpers.jts_crosscheck(layout,contract,evaluation,'unused.jar',out)
