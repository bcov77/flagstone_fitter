"""Continuation integrity and exact cache behavior, including pending work."""
import fcntl
import importlib.util
import pytest
from core import *
import planner,store
from continue_run import prepare_continuation
from test_frontier import toy,metrics,proposal


def parent_output(path):
 doc=toy();cfg=config();cfg['roots']=[62];cfg['warm_start']=False
 node=dict(id='r000062_original',root=62,kind='original',depth=0,sequence=0,
           parent=None,frame=state(doc),metrics=metrics(doc))
 write(path/'source.json',doc);write(path/'nodes'/f'{node["id"]}.json.gz',node)
 meta=dict(number=1,root=62,parent=node['id'],depth=1,operator='insert',phase=0,
           seed=123,parent_geometry=geometry_key(node['frame']),excluded=[])
 write(path/'jobs/job_00000001/job.json',meta)
 write(path/'jobs/job_00000001/selected.json.gz',dict(chosen=proposal(),physics_seed=456))
 write(path/'jobs/job_00000001/progress.json',dict(phase='checkpointed'))
 write(path/'manifest.json',dict(config=cfg,dependencies={},inputs={},versions=versions(),
       source_sha256=sha(path/'source.json'),seed_nodes=[node['id']]))
 (path/'STOP').touch()
 return path


def test_continuation_preserves_pending_selection_and_permits_progress(tmp_path):
 parent=parent_output(tmp_path/'parent');dest=tmp_path/'continuation'
 selected='jobs/job_00000001/selected.json.gz';digest=sha(parent/selected)
 result=prepare_continuation(parent,dest)
 assert result['jobs']==1 and result['committed']==0 and result['pending']==1
 assert sha(dest/selected)==sha(parent/selected)==digest
 assert read(dest/selected)['physics_seed']==456
 assert read(dest/'jobs/job_00000001/job.json')==read(parent/'jobs/job_00000001/job.json')
 write(dest/'jobs/job_00000001/progress.json',dict(phase='physics'))
 store.preflight(dest)
 write(dest/selected,dict(chosen=proposal(),physics_seed=999))
 with pytest.raises(RuntimeError,match='Inherited checkpoint changed'):store.preflight(dest)
 assert sha(parent/selected)==digest


def test_continuation_refuses_live_parent_and_existing_output(tmp_path):
 parent=parent_output(tmp_path/'parent');dest=tmp_path/'continuation'
 with (parent/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with pytest.raises(RuntimeError,match='stop the parent'):prepare_continuation(parent,dest)
 assert not dest.exists()
 dest.mkdir()
 with pytest.raises(ValueError,match='new continuation'):prepare_continuation(parent,dest)


def test_scaled_target_cache_retains_exact_outputs_and_builds_board_once(monkeypatch):
 cfg=config()['physics'];patio=shapely.box(0,0,3,3)
 local={'1':shapely.box(-.3,-.3,.3,.3),'2':shapely.box(-.4,-.4,.4,.4)}
 doc=dict(patio=list(patio.exterior.coords),stones=[dict(id=s,outline=list(g.exterior.coords),
      active=s=='1',pose=[1,1,0]) for s,g in local.items()],physics=dict(pieces={}))
 target=dict(remove=['1'],id='replace_1',pocket=shapely.box(.575,.575,1.425,1.425),free=patio.buffer(-GAP))
 placements=[dict(stone='2',pose=[1,1,0])]
 spec=importlib.util.spec_from_file_location('original_planner_for_test',ROOT/'analysis/frontier_search/planner.py')
 original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
 reference=[original.proposal(doc,target,placements,local,{},patio,cfg,.8) for _ in range(3)]
 assert reference[0] is not None
 calls=[];real_shapes=planner.shapes
 def counted(doc):calls.append(1);return real_shapes(doc)
 monkeypatch.setattr(planner,'shapes',counted)
 actual=[planner.proposal(doc,target,placements,local,{},patio,cfg,.8) for _ in range(3)]
 assert actual==reference and len(calls)==1
