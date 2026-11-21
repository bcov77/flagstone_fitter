"""New-rule geometry, structural rollback, GPU parity, and frozen scheduling."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from shapely.geometry import box
from flagstone.phase1 import Layout,save_json,validate
from flagstone.quarter import apply_records,prepare
from flagstone import quarter_search2 as search
from flagstone.quarter_gaps import audit,verify,RULE
from flagstone.quarter_batch import BatchScorer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import quarter_campaign as legacy
import quarter_campaign2 as campaign


def fixture_layout():
    return Layout(box(0,0,1,1),{'a':box(-.45,-.3,.45,.3)},np.array([[.5,.35,0.]]),
                  np.ones(1,bool),('a',),np.random.default_rng(1),units='ft')


def test_edge_allowance_really_accepts_a_gap_larger_than_two_inches():
    layout=fixture_layout();assert not validate(layout,gap=.5/12,edge=.5/12)
    result=audit(layout.patio,[box(.05,.05,.95,.65)])
    assert result['stable'] and result['numerical_pass'] and result['certified_pass']
    assert 4.19<result['groups']['edge']['upper_diameter_in']<4.21
    assert result['groups']['interior']['upper_diameter_in']==0


def test_interior_ring_gap_fails_while_edge_clearances_pass():
    stones=[box(.05,.05,.875,.875),box(1.125,.05,1.95,.875),
            box(.05,1.125,.875,1.95),box(1.125,1.125,1.95,1.95)]
    result=audit(box(0,0,2,2),stones)
    assert not result['certified_pass'] and result['groups']['interior']['lower_diameter_in']>2
    assert result['groups']['interior']['center'] is not None


def test_edge_five_inch_failure_and_corners():
    result=audit(box(0,0,1,1),[box(.05,.05,.95,.5)])
    assert result['groups']['edge']['lower_diameter_in']>5
    assert not result['certified_pass']


def test_saved_verifier_checks_original_shapes_hash_and_clearances(tmp_path):
    layout=fixture_layout();path=tmp_path/'layout.json';save_json(layout,path)
    result=verify(path,path,tmp_path/'verification.json')
    assert result['passed'] and result['rule']==RULE and result['original_geometry']
    assert result['layout_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    changed=json.loads(path.read_text());changed['patio'][0][0]-=.1
    bad=tmp_path/'tampered.json';bad.write_text(json.dumps(changed))
    with pytest.raises(ValueError):verify(bad,path,tmp_path/'bad_verification.json')
    layout.poses[0,0]=.02;save_json(layout,bad)
    result=verify(bad,path,tmp_path/'overlap_verification.json')
    assert not result['passed'] and result['clearance_errors']


def test_sampling_witnesses_and_shifted_grids_are_inside_domain():
    layout=fixture_layout();normal=search.points_for(layout);shifted=search.points_for(layout,shifted=True,witnesses=[[.5,.9],[20,20]])
    assert len(shifted)>len(normal)
    assert any(abs(p.x-.5)<1e-10 and abs(p.y-.9)<1e-10 for p in shifted)
    assert all(layout.patio.covers(p) for p in shifted)


def test_unrepairable_joint_replacement_restores_incumbent_and_advances_rng():
    layout=fixture_layout()
    layout=Layout(layout.patio,{'a':layout.shapes['a'],'big':box(-2,-2,2,2)},np.array([[.5,.35,0.],[0,0,0]]),
                  np.array([True,False]),('a','big'),np.random.default_rng(1),units='ft')
    original=layout.snapshot();result=search.replacement(layout,[.5,.7],search.points_for(layout),seconds=.15,joint=True)
    assert not result['accepted'] and result['invalid_starts']>0
    np.testing.assert_array_equal(layout.poses,original[0]);np.testing.assert_array_equal(layout.active,original[1])
    assert layout.snapshot()[2]!=original[2]


def test_continuous_reports_actual_optimizer_termination():
    layout=fixture_layout();result=search.continuous(layout,[0],search.points_for(layout),seconds=2,maxiter=20)
    assert not validate(layout,gap=.5/12,edge=.5/12)
    assert result['reason'] in ('optimizer-returned','time-limit')
    if result['reason']=='optimizer-returned':assert set(result['termination'])=={'status','message','success','nfev'}


def test_cpu_gpu_scores_agree_on_identical_candidate_batch():
    pytest.importorskip('mlx.core')
    layout=fixture_layout();rng=np.random.default_rng(8);points=rng.uniform(0,1,(64,2))
    poses=np.broadcast_to(layout.poses,(16,*layout.poses.shape)).copy()+rng.normal(0,.02,(16,*layout.poses.shape))
    a=BatchScorer(layout,points,'cpu').evaluate(poses,layout.active)
    b=BatchScorer(layout,points,'gpu').evaluate(poses,layout.active)
    np.testing.assert_allclose(a,b,atol=2e-5,rtol=2e-5)
    assert np.argmin(a)==np.argmin(b)


def test_new_campaign_matrix_freezes_contract_sources_and_saved_parents(tmp_path):
    source=tmp_path/'source.json';original=fixture_layout();original.patio=box(0,0,2,2);save_json(original,source);trial=tmp_path/'trial';prepare(source,trial)
    saved=tmp_path/'saved.json';parent=fixture_layout();parent.patio=__import__('flagstone.phase1',fromlist=['load_json']).load_json(trial/'input.json').patio;save_json(parent,saved)
    out=tmp_path/'campaign';plan=campaign.prepare(out,trial,hours=20,cores=10,starts=[saved,saved])
    legacy.validate_frozen(out)
    assert len(plan['tasks'])==112
    assert len([t for t in plan['tasks'] if t['phase']=='screen'])==72
    assert len([t for t in plan['tasks'] if t['phase']=='deepen'])==30
    assert plan['seconds']==20*3600 and plan['extension_seconds']==1200
    assert legacy.read(out/'state.json')['status']=='prepared'
    assert legacy.read(out/'trial/contract.json')['gap_rule']==RULE
    for name in ('quarter_patio2.py','quarter_campaign2.py','run_quarter_campaign2.sh'):
        assert (out/'snapshot/scripts'/name).is_file()
    assert not (out/'tasks').exists()


def test_campaign_rejects_stale_or_partial_certificates(tmp_path):
    layout=fixture_layout();folder=tmp_path/'task';folder.mkdir();path=folder/'best.json';save_json(layout,path)
    metrics=search.measure(layout,search.points_for(layout));legacy.atomic_json(folder/'summary.json',dict(policy='energy',initial=metrics,best=metrics,rounds=[]))
    v=verify(path,path,folder/'verification.json')
    result=campaign.result_for(folder);assert result['verified'] and result['eligible_parent']
    v['gaps']['stable']=False;legacy.atomic_json(folder/'verification.json',v)
    assert not campaign.result_for(folder)['eligible_parent']
    v['gaps']['stable']=True;v['gaps']['groups']['interior']['limit_in']=5;legacy.atomic_json(folder/'verification.json',v)
    assert not campaign.result_for(folder)['verified']


def test_shape_replacement_can_accept_a_larger_reserve_stone():
    layout=Layout(box(0,0,2,2),{'a':box(-.15,-.15,.15,.15),'b':box(-.35,-.35,.35,.35)},
                  np.array([[.5,.5,0.],[0,0,0.]]),np.array([True,False]),('a','b'),np.random.default_rng(3),units='ft')
    result=search.replacement(layout,[.6,.6],search.points_for(layout),seconds=1.5)
    assert result['accepted'] and layout.active.tolist()==[False,True]
    assert not validate(layout,gap=.5/12,edge=.5/12)
    assert search.rank(result['after'])<search.rank(result['before'])


def test_continuous_can_repair_an_initially_overlapping_group():
    layout=Layout(box(0,0,2,2),{'a':box(-.2,-.2,.2,.2),'b':box(-.2,-.2,.2,.2)},
                  np.array([[.5,.5,0.],[.87,.5,0.]]),np.array([True,True]),('a','b'),np.random.default_rng(4),units='ft')
    assert validate(layout,gap=.5/12,edge=.5/12)
    result=search.continuous(layout,[0,1],search.points_for(layout),seconds=2,maxiter=220,radius_ft=.4)
    assert result['accepted'] and not validate(layout,gap=.5/12,edge=.5/12)


def test_flat_medial_corridor_keeps_interior_gap_candidates():
    result=audit(box(0,0,2,2),[box(.25,.25,.875,1.75),box(1.125,.25,1.75,1.75)])
    assert result['groups']['interior']['lower_diameter_in']>2.9
    assert not result['certified_pass']


def prepare_synthetic_campaign(tmp_path):
    source=tmp_path/'source.json';original=fixture_layout();original.patio=box(0,0,2,2);save_json(original,source)
    trial=tmp_path/'trial';prepare(source,trial)
    parent=fixture_layout();parent.patio=legacy.read(trial/'input.json')['patio']
    from flagstone.phase1 import load_json
    parent.patio=load_json(trial/'input.json').patio
    saved=tmp_path/'saved.json';save_json(parent,saved)
    out=tmp_path/'campaign';campaign.prepare(out,trial,hours=.1,cores=2,starts=[saved,saved])
    return out


def test_real_frozen_new_campaign_runs_parallel_workers_and_certifies(tmp_path):
    import subprocess
    out=prepare_synthetic_campaign(tmp_path);plan=legacy.read(out/'plan.json')
    plan.update(seconds=30,final_reserve_seconds=2,allow_extensions=False,tasks=[
        dict(number=i,phase='bootstrap',variant='bootstrap',seed=101+i,seconds=15,rounds=0,
             parameters=dict(policy='energy'),saved_start='starts/start_00.json',parent_rule='cold') for i in range(2)])
    legacy.atomic_json(out/'plan.json',plan);manifest=legacy.read(out/'manifest.json')
    manifest['plan.json']=hashlib.sha256((out/'plan.json').read_bytes()).hexdigest();legacy.atomic_json(out/'manifest.json',manifest)
    result=subprocess.run([sys.executable,str(out/'snapshot/scripts/quarter_campaign2.py'),'--output',str(out),'--resume','--cores','2'],
                          capture_output=True,text=True,timeout=25)
    assert result.returncode==0,result.stdout+result.stderr
    state=legacy.read(out/'state.json');assert state['peak_active_tasks']==2
    assert all(a['outcome']=='completed' and a['result']['verified'] for a in state['attempts'])
    report=legacy.read(out/'analysis_summary.json');assert len(report['validated_successes'])==2


def test_new_worker_interruption_and_rng_resume(tmp_path):
    import subprocess,time,signal
    out=prepare_synthetic_campaign(tmp_path);worker=out/'snapshot/scripts/quarter_patio2.py';first=tmp_path/'first';second=tmp_path/'second'
    common=[sys.executable,str(worker),'run','--trial',str(out/'trial'),'--seed','77','--policy','batch_cpu','--patch-seconds','2']
    process=subprocess.Popen([*common,'--output',str(first),'--start',str(out/'starts/start_00.json'),
                              '--seconds','120'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        until=time.monotonic()+10
        while time.monotonic()<until:
            checkpoint=legacy.read(first/'checkpoint.json')
            if checkpoint.get('round',0)>=1:break
            time.sleep(.03)
        assert checkpoint.get('round',0)>=1
        process.send_signal(signal.SIGTERM);process.wait(timeout=10)
        saved=legacy.read(first/'checkpoint.json')
        assert legacy.read(first/'summary.json')['status']=='interrupted/incomplete'
        result=subprocess.run([*common,'--output',str(second),'--resume-from',str(first),'--seconds','10','--rounds','0'],
                              capture_output=True,text=True,timeout=15)
        assert result.returncode==0,result.stdout+result.stderr
        restored=legacy.read(second/'checkpoint.json')
        assert restored['round']==saved['round'] and restored['rng_state']==saved['rng_state']
        assert restored['incumbent']==saved['incumbent'] and restored['witnesses']==saved['witnesses']
    finally:
        if process.poll() is None:process.kill();process.wait()


def test_adaptive_policy_order_uses_checked_progress_per_cpu_hour():
    def record(number,policy,upper,parent,cpu,phase='screen'):
        return dict(number=number,task=dict(phase=phase),parent_lower_in=parent,
                    result=dict(eligible_parent=True,policy=policy,gap=dict(upper_diameter_in=upper),cpu_seconds_total=cpu))
    state=dict(attempts=[record(0,'energy',2.5,3,3600),record(1,'worst',2.6,3,3600),
                         record(2,'worst',2.0,2.6,100,phase='extension')])
    assert campaign.ranking(state)==['energy','worst']
    assert campaign.ranking(state,adaptive=True)==['worst','energy']


def test_interrupted_bootstrap_salvages_native_patterns_without_new_native_job(tmp_path):
    import subprocess
    out=prepare_synthetic_campaign(tmp_path);source=tmp_path/'interrupted';source.mkdir()
    trial=legacy.read(out/'trial/contract.json');parent=__import__('flagstone.phase1',fromlist=['load_json']).load_json(out/'starts/start_00.json')
    legacy.atomic_json(source/'state.json',dict(rule=RULE,policy='energy',seed=77,reference_sha256=trial['input_sha256'],
                         settings=dict(objective='energy'),cpu_seconds_total=2.,elapsed_seconds=3.))
    pool=source/'native_000';pool.mkdir();legacy.atomic_json(pool/'patterns.json',dict(patterns=[dict(records=[dict(stone_id='a',pose=parent.poses[0].tolist())])]))
    resumed=tmp_path/'resumed';result=subprocess.run([sys.executable,str(out/'snapshot/scripts/quarter_patio2.py'),'run',
        '--trial',str(out/'trial'),'--output',str(resumed),'--resume-from',str(source),'--seconds','10','--rounds','0','--seed','77'],
        capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
    summary=legacy.read(resumed/'summary.json')
    assert summary['resume_fidelity'].startswith('salvaged native patterns')
    assert summary['cpu_seconds_total']>=2 and summary['elapsed_seconds']>=3
    assert not list(resumed.glob('native_*'))
    assert campaign.result_for(resumed)['verified']


def test_incremental_patch_validation_and_metrics_match_full_original_geometry():
    from flagstone.quarter_trial import PatchTrials
    layout=Layout(box(0,0,2,2),{'a':box(-.2,-.2,.2,.2),'b':box(-.2,-.2,.2,.2),'c':box(-.2,-.2,.2,.2)},
                  np.array([[.5,.5,0.],[1.3,.5,0.],[.5,1.3,0.]]),np.ones(3,bool),('a','b','c'),np.random.default_rng(4),units='ft')
    original=layout.snapshot();points=search.points_for(layout);oracle=PatchTrials(layout,[0,1],points);rng=np.random.default_rng(18)
    for _ in range(24):
        poses=original[0].copy();poses[:2]+=rng.normal(0,[.3,.3,.4],(2,3))
        layout.poses[:]=poses
        assert oracle.valid(poses)==(not validate(layout,gap=.5/12,edge=.5/12))
        a,b=oracle.measure(poses),search.measure(layout,points)
        for key in ['sampled_max_diameter_in','sampled_energy','sampled_worst_ratio','sampled_bad_center_area_sqft']:
            assert a[key]==pytest.approx(b[key],abs=1e-11)
    bad=original[0].copy();bad[2,0]+=.1
    with pytest.raises(ValueError,match='Fixed poses changed'):oracle.valid(bad)


def test_gpu_queue_timeout_releases_lock_and_keeps_layout(tmp_path):
    import fcntl,time
    from flagstone.quarter_batch import GPUQueueDeadline
    pytest.importorskip('mlx.core')
    layout=fixture_layout();original=layout.snapshot();scorer=BatchScorer(layout,np.array([[.5,.8],[.2,.2]]),'gpu')
    scorer.gpu_lock_path=str(tmp_path/'gpu.lock');scorer.queue_deadline=time.monotonic()-.1
    with open(scorer.gpu_lock_path,'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with pytest.raises(GPUQueueDeadline):scorer.evaluate(layout.poses[None,:,:],layout.active)
    assert len(scorer.evaluate(layout.poses[None,:,:],layout.active))==1
    np.testing.assert_array_equal(layout.poses,original[0])


def test_patch_cache_invalidates_after_incumbent_pose_changes():
    layout=fixture_layout();points=search.points_for(layout);scorer=BatchScorer(layout,np.array([[.5,.8],[.2,.2]]),'cpu')
    search.batch_move(layout,[.5,.8],points,scorer=scorer,batch_size=16)
    previous=scorer.trial_cache[0];layout.poses[0,0]+=.001
    assert not validate(layout,gap=.5/12,edge=.5/12)
    expected=search.measure(layout,points)
    result=search.batch_move(layout,[.5,.8],points,scorer=scorer,batch_size=16)
    assert scorer.trial_cache[0]!=previous
    assert result['before']['sampled_energy']==pytest.approx(expected['sampled_energy'],abs=1e-11)


def test_new_report_command_refuses_to_rewrite_historical_campaign(tmp_path):
    import subprocess
    source=tmp_path/'source.json';save_json(fixture_layout(),source);trial=tmp_path/'trial';prepare(source,trial)
    out=tmp_path/'old_campaign';legacy.prepare(out,trial,hours=.1,cores=1,native_python=sys.executable)
    before={name:(out/name).read_bytes() for name in ['tasks.csv','variants.csv','report.md','analysis_summary.json']}
    result=subprocess.run([sys.executable,str(Path(campaign.__file__)),'--output',str(out),'--report-only'],capture_output=True,text=True,timeout=10)
    assert result.returncode!=0 and 'historical launcher' in result.stderr
    assert all((out/name).read_bytes()==data for name,data in before.items())
