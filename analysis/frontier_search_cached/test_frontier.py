"""Recovery, diversity, candidate sampling, and hard-constraint regression checks."""
import copy,os
import pytest
from core import *
import worker,store
from planner import coarse_poses
from run import exclusions

def toy():
 return dict(units='ft',patio=[[0,0],[3,0],[3,3],[0,3]],stones=[dict(id='1',outline=[[-.25,-.25],[.25,-.25],[.25,.25],[-.25,.25]],pose=[1,1,0],active=True),dict(id='2',outline=[[-.15,-.15],[.15,-.15],[.15,.15],[-.15,.15]],pose=[2,2,0],active=False)],physics=dict(settings={},pieces={'1':dict(locked=False,anchor=[1,1,0]),'2':dict(locked=False,anchor=[2,2,0])}))

def metrics(doc):
 return dict(area_percent=sum(shapely.Polygon(s['outline']).area for s in doc['stones'] if s['active'])/9*100,clearance=dict(valid=True,errors=[]),gaps=dict(stable=True,largest_interior_inches=1.,largest_edge_inches=1.),quality=dict(broad_void_sqft=1.,wall_joint_inches=10.))

def proposal():return dict(operator='placement',family='ordinary',physics_protocol='held_release',remove=[],placements=[dict(stone='2',pose=[2,2,0])],rank_key=[0])

@pytest.fixture
def saved(tmp_path):
 cfg=config();cfg['roots']=[62];cfg['warm_start']=False;doc=toy();node=dict(id='r000062_original',root=62,kind='original',depth=0,sequence=0,parent=None,frame=state(doc),metrics=metrics(doc));meta=dict(number=1,root=62,parent=node['id'],depth=1,operator='insert',phase=0,seed=123,parent_geometry=geometry_key(node['frame']))
 write(tmp_path/'manifest.json',dict(config=cfg));write(tmp_path/'source.json',doc);write(tmp_path/'nodes'/f'{node["id"]}.json.gz',node);write(tmp_path/'jobs/job_00000001/job.json',meta)
 return tmp_path,cfg,doc,node,meta


def fake_relax(accepted,calls):
 def run(doc,p,cfg,seed):
  calls.append((copy.deepcopy(p),seed));end=apply(doc,p);c=dict(valid=accepted,errors=[] if accepted else ['synthetic']);trace=[dict(clearance=c,motion=dict(max_speed_ips=0.,max_angular_speed_dps=0.))]*2
  return dict(accepted=accepted,attempted_document=end,proposed_start=state(end),reason='synthetic',settled=accepted,clearance=c,phases=[dict(name='direct',trace=trace)],final_motion=dict(max_speed_ips=0.,max_angular_speed_dps=0.),physics_seed=seed)
 return run


def test_selected_checkpoint_resumes_exact_action_and_seed_without_research(saved,monkeypatch):
 out,cfg,doc,node,meta=saved;calls=[]
 monkeypatch.setattr(worker,'generate',lambda *args:([proposal()],{}));monkeypatch.setattr(worker,'relax',fake_relax(True,calls));monkeypatch.setattr(worker,'evaluate',metrics)
 assert worker.perform(out,1,os.getppid(),True)=='checkpointed'
 path=out/'jobs/job_00000001/selected.json.gz';digest=sha(path);selected=read(path)
 assert not (out/'jobs/job_00000001/result.json.gz').exists() and not calls
 monkeypatch.setattr(worker,'generate',lambda *args:pytest.fail('Pending selection must not be regenerated'))
 assert worker.perform(out,1,os.getppid())=='complete'
 assert sha(path)==digest and calls==[(selected['chosen'],selected['physics_seed'])]
 r=read(out/'jobs/job_00000001/result.json.gz');assert r['accepted'] and r['start']==node['frame']
 assert worker.perform(out,1,os.getppid())=='complete' and len(calls)==1


def test_rejection_restores_exact_parent_and_preserves_raw_endpoint(saved,monkeypatch):
 out,*_=saved;calls=[]
 monkeypatch.setattr(worker,'generate',lambda *args:([proposal()],{}));monkeypatch.setattr(worker,'relax',fake_relax(False,calls));monkeypatch.setattr(worker,'evaluate',metrics)
 worker.perform(out,1,os.getppid());r=read(out/'jobs/job_00000001/result.json.gz')
 assert r['end']==r['start'] and r['attempted_end']!=r['start'] and not r['accepted'] and r['physics_evaluated']


def test_empty_search_is_not_a_physics_attempt(saved,monkeypatch):
 out,*_=saved
 monkeypatch.setattr(worker,'generate',lambda *args:([],{'scope':'bounded'}));monkeypatch.setattr(worker,'relax',lambda *args:pytest.fail('No proposal must not run physics'))
 worker.perform(out,1,os.getppid());r=read(out/'jobs/job_00000001/result.json.gz')
 assert r['status']=='no_candidate' and not r['physics_evaluated']


def test_stop_before_search_leaves_job_pending(saved,monkeypatch):
 out,*_=saved;(out/'STOP').touch();monkeypatch.setattr(worker,'generate',lambda *args:pytest.fail('STOP must checkpoint'))
 assert worker.perform(out,1,os.getppid())=='checkpointed'
 assert not (out/'jobs/job_00000001/result.json.gz').exists()


def test_contextual_cache_caps_repeats_without_global_piece_ban(saved):
 out,cfg,doc,node,meta=saved;p=proposal();jobs={}
 for number in [1,2]:
  j={**meta,'number':number};jobs[number]=j;write(out/'jobs'/f'job_{number:08d}'/'selected.json.gz',dict(chosen=p))
 next_job={**meta,'number':3}
 assert exclusions(out,next_job,jobs,cfg)==[action_key(p)]
 next_job['parent_geometry']='a changed board';assert exclusions(out,next_job,jobs,cfg)==[]


def test_rounding_keys_are_circular_and_do_not_change_physical_poses():
 f=dict(poses={'1':[1,1,math.radians(359.99)]});g=copy.deepcopy(f);g['poses']['1'][2]=0
 before=copy.deepcopy(f);assert geometry_key(f)==geometry_key(g);assert f==before
 p=proposal();q=copy.deepcopy(p);p['placements'][0]['pose'][2]=2*math.pi
 assert action_key(p)==action_key(q)


def test_multiple_fft_positions_survive_per_piece():
 cfg=dict(raster_step_ft=.05,angle_step_degrees=90,piece_budget=1,poses_per_angle=3,poses_per_piece=8,coarse_limit=24)
 pocket=shapely.box(0,0,1,1);local={'a':shapely.box(-.15,-.15,.15,.15)}
 poses=coarse_poses(local,['a'],pocket,pocket,cfg,np.random.default_rng(4),lambda:False,min_fraction=.01)
 assert len(poses)>2 and len({tuple(p[2][:2]) for p in poses})>1


def test_compound_neighbor_edit_keeps_45_degree_rule():
 doc=toy();p=dict(remove=['1'],placements=[dict(stone='1',pose=[1,1,math.radians(44)]),dict(stone='2',pose=[2,2,0])]);cfg=config()['physics']
 assert not eligible(doc,p,cfg)
 p['placements'][0]['pose'][2]=math.radians(45);assert eligible(doc,p,cfg)
 doc['physics']['pieces']['1']['locked']=True;assert not eligible(doc,p,cfg)


def test_frontier_preserves_roots_champions_and_new_setup_alternatives():
 original=dict(id='root',kind='original',frame=state(toy()),metrics=metrics(toy()),sequence=0)
 nodes=[original]
 for i in range(20):
  n=copy.deepcopy(original);n.update(id=f'n{i}',kind='search',sequence=i+1);n['frame']['poses']['1'][0]+=.02*(i+1);n['metrics']['area_percent']+=.001*(i+1);nodes.append(n)
 nodes[3]['metrics']['area_percent']+=1;nodes[4]['metrics']['gaps']['largest_interior_inches']=.1
 recent=nodes[-1];recent['frame']['poses']['2']=[2,2,0]
 retained=frontier(nodes,8);ids={n['id'] for n in retained}
 assert {'root',nodes[3]['id'],nodes[4]['id'],recent['id']}<=ids and len(retained)<=8


def test_interrupted_preparation_can_resume_without_changing_inputs(tmp_path,monkeypatch):
 batch=tmp_path/'batch';out=tmp_path/'out';doc=toy();cfg=config();cfg['roots']=[62];cfg['warm_start']=False
 write(batch/'source.json',doc);write(batch/'runs/run_000062/final.json',doc);write(batch/'runs/run_000062/summary.json',dict(end_metrics=metrics(doc)))
 monkeypatch.setattr(store,'BATCH',batch);real_read=store.read
 def interrupted(path):
  if str(path).endswith('summary.json'):raise RuntimeError('simulated interruption')
  return real_read(path)
 monkeypatch.setattr(store,'read',interrupted)
 with pytest.raises(RuntimeError,match='simulated interruption'):store.prepare(out,cfg)
 monkeypatch.setattr(store,'read',real_read);m=store.prepare(out,cfg)
 assert m['seed_nodes']==['r000062_original'] and (out/'STOP').exists()
 store.preflight(out,cfg)


def test_audit_failure_keeps_selection_for_deterministic_recovery(saved,monkeypatch):
 out,*_=saved;calls=[]
 monkeypatch.setattr(worker,'generate',lambda *args:([proposal()],{}));monkeypatch.setattr(worker,'relax',fake_relax(True,calls))
 def failed_audit(doc):raise RuntimeError('simulated audit interruption')
 monkeypatch.setattr(worker,'evaluate',failed_audit)
 with pytest.raises(RuntimeError,match='audit interruption'):worker.perform(out,1,os.getppid())
 selected=read(out/'jobs/job_00000001/selected.json.gz');assert not (out/'jobs/job_00000001/result.json.gz').exists()
 monkeypatch.setattr(worker,'generate',lambda *args:pytest.fail('Saved selection must survive audit failure'));monkeypatch.setattr(worker,'evaluate',metrics)
 worker.perform(out,1,os.getppid());assert calls==[(selected['chosen'],selected['physics_seed'])]*2
 assert read(out/'jobs/job_00000001/result.json.gz')['physics_evaluated']


def test_configuration_cannot_relax_physical_prerequisites(tmp_path):
 cfg=read(HERE/'config.json');cfg['physics']={'max_intrusion_fraction':.1};path=tmp_path/'config.json';write(path,cfg)
 with pytest.raises(ValueError,match='cannot be weakened'):config(path)


def test_geometry_failure_rotates_search_without_consuming_physics(saved,monkeypatch):
 out,*_=saved
 def failure(*args):raise shapely.errors.GEOSException('synthetic topology failure')
 monkeypatch.setattr(worker,'generate',failure);monkeypatch.setattr(worker,'relax',lambda *args:pytest.fail('Geometry failure must not run physics'))
 worker.perform(out,1,os.getppid());r=read(out/'jobs/job_00000001/result.json.gz')
 assert r['status']=='geometry_error' and not r['physics_evaluated'] and 'topology' in r['diagnostics']['error']


def test_allocation_snapshots_exclusions_in_one_atomic_job_write(saved,monkeypatch):
 out,cfg,doc,node,meta=saved;writes=[];real_write=store.write
 def watched(path,value):writes.append((str(path),copy.deepcopy(value)));return real_write(path,value)
 monkeypatch.setattr(store,'write',watched);jobs={}
 allocated=store.allocate(out,62,{node['id']:node},jobs,cfg,exclusion_provider=lambda meta,previous:['reserved-action'])
 assert len(writes)==1 and writes[0][1]['excluded']==['reserved-action']
 assert read(out/'jobs/job_00000001/job.json')==allocated
