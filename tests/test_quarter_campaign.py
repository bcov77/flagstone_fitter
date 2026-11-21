"""Campaign scheduling and ownership tests; no real numerical search is launched."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import quarter_campaign as c
from flagstone.phase1 import Layout, save_json
from flagstone.quarter import prepare


@pytest.fixture
def prepared(tmp_path):
    x = Layout(box(0, 0, 2, 2), {'a': box(-.4, -.4, .4, .4)}, np.zeros((1, 3)),
               np.zeros(1, bool), ('a',), np.random.default_rng(1), units='ft')
    source = tmp_path/'source.json'
    save_json(x, source)
    trial = tmp_path/'trial'
    prepare(source, trial)
    out = tmp_path/'campaign'
    c.prepare(out, trial, hours=.1, native_python=sys.executable, cores=1)
    return out


def change_plan(out, tasks, seconds=30):
    plan = c.read(out/'plan.json')
    plan.update(tasks=tasks, seconds=seconds, allow_extensions=False)
    c.atomic_json(out/'plan.json', plan)
    manifest = c.read(out/'manifest.json')
    manifest['plan.json'] = hashlib.sha256((out/'plan.json').read_bytes()).hexdigest()
    c.atomic_json(out/'manifest.json', manifest)
    return plan


def task(i, seconds=3):
    return dict(number=i, phase='bootstrap', variant='bootstrap', seed=101+i,
                seconds=seconds, parameters={}, parent_rule='cold')


def fake_worker(tmp_path):
    path = tmp_path/'fake_worker.py'
    path.write_text('''import hashlib, json, os, sys, time
from pathlib import Path
out=Path(sys.argv[1]); mode=sys.argv[2]; out.mkdir(parents=True,exist_ok=True)
(out/'timing_start').write_text(str(time.monotonic()))
(out/'threads.json').write_text(json.dumps({k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}))
with (out/'starts').open('a') as f: f.write(str(os.getpid())+'\\n')
if mode=='crash': raise RuntimeError('simulated native crash')
if mode=='sleep': time.sleep(20)
if mode=='brief': time.sleep(.35)
if mode=='hold': time.sleep(1)
data=b'fixture-layout'; (out/'best.json').write_bytes(data)
metrics={'sampled_max_diameter_in':3.1,'sampled_bad_center_area_sqft':.2,'coverage':.7,'placed':1}
summary={'status':'pilot-completed/incomplete','initial':metrics,'best':metrics,'rounds':[],'cpu_seconds_total':.01}
(out/'summary.json').write_text(json.dumps(summary))
v={'layout_sha256':hashlib.sha256(data).hexdigest(),'reference_matched':True,'clearance_errors':[],
   'passed':False,'gap':{'lower_diameter_in':3.2,'upper_diameter_in':3.3}}
if mode=='better': v['gap']={'lower_diameter_in':2.2,'upper_diameter_in':2.3}
(out/'verification.json').write_text(json.dumps(v))
(out/'timing_end').write_text(str(time.monotonic()))
''')
    return path


def test_plan_covers_variants_matched_seeds_and_full_budget():
    tasks = c.matrix(16*3600)
    assert len(tasks) == 51
    assert len([t for t in tasks if t['phase'] == 'screen']) == 24
    assert set(t['variant'] for t in tasks if t['phase'] == 'screen') == set(c.VARIANTS)
    for seed in (101, 202, 303):
        assert len([t for t in tasks if t['phase'] == 'screen' and t['seed'] == seed]) == 8
    assert sum(t['seconds'] for t in tasks) <= 16*3600
    assert sum(t['seconds'] for t in tasks) > 15.9*3600


def test_prepare_is_read_only_to_sources_and_launches_nothing(prepared):
    assert c.read(prepared/'state.json')['status'] == 'prepared'
    assert not (prepared/'tasks').exists()
    c.validate_frozen(prepared)
    (prepared/'snapshot/scripts/quarter_native.py').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        c.validate_frozen(prepared)


def test_crashes_are_contained_and_later_tasks_run(prepared, tmp_path):
    change_plan(prepared, [task(i) for i in range(3)])
    worker = fake_worker(tmp_path)
    code = c.supervise(prepared, poll=.02, heartbeat=100, emit=lambda m: None,
                       worker_override=lambda a: [sys.executable, str(worker), a['directory'],
                                                   'crash' if a['number'] < 2 else 'good'])
    assert code == 0
    state = c.read(prepared/'state.json')
    assert [a['outcome'] for a in state['attempts']] == ['worker-failed', 'worker-failed', 'completed']
    assert (prepared/'tasks.csv').exists() and (prepared/'variants.csv').exists()
    report = c.read(prepared/'analysis_summary.json')
    assert not report['validated_successes']
    assert report['tasks'][-1]['upper_in'] == 3.3


def test_three_consecutive_crashes_stop_explicitly(prepared, tmp_path):
    change_plan(prepared, [task(i) for i in range(4)])
    worker = fake_worker(tmp_path)
    code = c.supervise(prepared, poll=.02, emit=lambda m: None,
                       worker_override=lambda a: [sys.executable, str(worker), a['directory'], 'crash'])
    assert code == 1
    state = c.read(prepared/'state.json')
    assert len(state['attempts']) == 3
    assert state['status'] == 'stopped-after-three-infrastructure-failures'


def test_task_timeout_reaps_owned_worker(prepared, tmp_path):
    change_plan(prepared, [task(0, seconds=.2)])
    worker = fake_worker(tmp_path)
    c.supervise(prepared, poll=.02, emit=lambda m: None,
                worker_override=lambda a: [sys.executable, str(worker), a['directory'], 'sleep'])
    attempt = c.read(prepared/'state.json')['attempts'][0]
    assert attempt['outcome'] == 'timed-out'
    with pytest.raises(ProcessLookupError):
        os.kill(attempt['pid'], 0)


def test_resume_adopts_live_worker_without_duplicate(prepared, tmp_path):
    plan = change_plan(prepared, [task(0, seconds=3)])
    worker = fake_worker(tmp_path)
    directory = str(prepared/'tasks/task_000_bootstrap')
    logpath = prepared/'tasks/task_000.log'
    logpath.parent.mkdir()
    with logpath.open('wb') as log:
        process = subprocess.Popen([sys.executable, str(worker), directory, 'brief'],
                                   stdout=log, stderr=log, start_new_session=True)
    state = c.read(prepared/'state.json')
    now = time.time()
    record = dict(number=0, task=plan['tasks'][0], directory=directory, log=str(logpath),
                  outcome='running', pid=process.pid, identity=c.identity(process.pid),
                  started_epoch=now, deadline_epoch=now+3, budget_seconds=3)
    state.update(status='running', attempts=[record], next_task=1, started_utc=c.utc(), deadline_epoch=now+10)
    c.atomic_json(prepared/'state.json', state)
    try:
        c.supervise(prepared, poll=.02, emit=lambda m: None)
        assert c.read(prepared/'state.json')['attempts'][0]['outcome'] == 'completed'
        assert len((Path(directory)/'starts').read_text().splitlines()) == 1
        assert 'task-adopted' in (prepared/'events.jsonl').read_text()
    finally:
        process.wait(timeout=5)


def test_expired_resume_cleans_live_owned_task(prepared, tmp_path):
    plan = change_plan(prepared, [task(0)])
    worker = fake_worker(tmp_path)
    directory = str(prepared/'tasks/task_000_bootstrap')
    Path(directory).parent.mkdir()
    process = subprocess.Popen([sys.executable, str(worker), directory, 'sleep'], start_new_session=True)
    state = c.read(prepared/'state.json')
    now = time.time()
    state.update(status='running', next_task=1, started_utc=c.utc(), deadline_epoch=now-1,
                 attempts=[dict(number=0, task=plan['tasks'][0], directory=directory,
                                log=str(prepared/'tasks/task_000.log'), outcome='running', pid=process.pid,
                                identity=c.identity(process.pid), started_epoch=now-5,
                                deadline_epoch=now-1, budget_seconds=3)])
    c.atomic_json(prepared/'state.json', state)
    try:
        c.supervise(prepared, poll=.02, emit=lambda m: None)
        process.wait(timeout=5)
        assert process.returncode != 0
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_report_requires_matching_certificate_bytes_and_strict_gate(tmp_path):
    worker = fake_worker(tmp_path)
    out = tmp_path/'result'
    subprocess.run([sys.executable, str(worker), str(out), 'good'], check=True)
    v = c.read(out/'verification.json')
    v['passed'] = True
    c.atomic_json(out/'verification.json', v)
    assert c.result_for(out)['eligible_parent']
    assert not c.result_for(out)['verified']
    v['gap']['upper_diameter_in'] = 2
    v['gap']['lower_diameter_in'] = 1.9
    c.atomic_json(out/'verification.json', v)
    assert not c.result_for(out)['verified']
    (out/'best.json').write_text('changed')
    assert not c.result_for(out)['eligible_parent']


def test_screen_parent_is_matched_even_if_other_seed_is_better():
    def a(number, seed, upper):
        return dict(number=number, task=dict(phase='bootstrap', seed=seed, variant='bootstrap'),
                    result=dict(eligible_parent=True, gap=dict(upper_diameter_in=upper),
                                metrics=dict(sampled_bad_center_area_sqft=1, coverage=.7)))
    state = dict(attempts=[a(0, 101, 5), a(1, 202, 3)])
    assert c.choose_parent(dict(parent_rule='matched-bootstrap', seed=101), state)['number'] == 0


def test_objectives_preserve_different_tradeoffs():
    from flagstone.quarter import rank
    a = dict(sampled_energy=.1, sampled_max_diameter_in=4, coverage=.7, sampled_bad_center_area_sqft=2)
    b = dict(sampled_energy=.2, sampled_max_diameter_in=3, coverage=.7, sampled_bad_center_area_sqft=1)
    assert rank(a, 'energy') < rank(b, 'energy')
    assert rank(b, 'worst') < rank(a, 'worst')
    assert rank(b, 'balanced') < rank(a, 'balanced')


def test_spawn_crash_window_discovers_existing_worker(tmp_path):
    worker = fake_worker(tmp_path)
    directory = str(tmp_path/'window')
    command = [sys.executable, '-u', str(worker), directory, 'sleep']
    process = subprocess.Popen(command, start_new_session=True)
    record = dict(command=command, directory=directory)
    try:
        c.discover_worker(record)
        assert record['pid'] == process.pid
        assert c.matching_process(record['pid'], directory, record['identity'])
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def test_nested_cleanup_rejects_wrong_process_identity(tmp_path):
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'], start_new_session=True)
    owner = tmp_path/'native.owner.json'
    c.atomic_json(owner, dict(pid=process.pid, identity='wrong-start-identity', status='running'))
    try:
        c.clean_nested(tmp_path)
        assert process.poll() is None
        c.atomic_json(owner, dict(pid=process.pid, identity=c.identity(process.pid), status='running'))
        c.clean_nested(tmp_path)
        process.wait(timeout=3)
        assert process.returncode != 0
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_rotated_bootstrap_bin_and_inverse_pose_preserve_geometry():
    from quarter_native import original_pose
    from flagstone import quarter as q
    from shapely.affinity import rotate, translate
    from shapely.geometry import Polygon
    x = Layout(box(0, 0, 2, 2), {'a': box(-.4, -.4, .4, .4)}, np.zeros((1, 3)),
               np.zeros(1, bool), ('a',), np.random.default_rng(1), units='ft')
    payload = q.native_input(x, 101, angle_deg=25)
    transform = payload['bin_transform']
    restored = translate(Polygon(payload['patio']), xoff=-transform['offset_ft'][0], yoff=-transform['offset_ft'][1])
    restored = rotate(restored, -transform['angle_rad'], origin=(0, 0), use_radians=True)
    assert restored.hausdorff_distance(x.patio) < 1e-12
    assert Polygon(payload['patio']).exterior.is_ccw
    item = dict(x=700, y=900, angle=63)
    native_shape = q.transformed(x.shapes['a'], [.7, .9, np.deg2rad(63)])
    expected = translate(native_shape, xoff=-transform['offset_ft'][0], yoff=-transform['offset_ft'][1])
    expected = rotate(expected, -transform['angle_rad'], origin=(0, 0), use_radians=True)
    actual = q.transformed(x.shapes['a'], original_pose(item, transform))
    assert actual.hausdorff_distance(expected) < 1e-12


def test_duplicate_supervisor_cannot_modify_state(prepared):
    import fcntl
    before = (prepared/'state.json').read_bytes()
    with (prepared/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='already owns'):
            c.supervise(prepared, emit=lambda m: None)
    assert (prepared/'state.json').read_bytes() == before


def test_supervisor_ctrl_c_reaps_worker_and_preserves_report(prepared, tmp_path):
    change_plan(prepared, [task(0, seconds=20)])
    worker = fake_worker(tmp_path)
    code = f'''import sys
sys.path.insert(0,{str(Path(c.__file__).parent)!r})
import quarter_campaign as c
raise SystemExit(c.supervise({str(prepared)!r},poll=.02,emit=lambda m:None,
    worker_override=lambda a:[sys.executable,{str(worker)!r},a['directory'],'sleep']))
'''
    process = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    marker = prepared/'tasks/task_000_bootstrap/starts'
    try:
        deadline = time.monotonic()+5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert marker.exists()
        pid = int(marker.read_text().strip())
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=8) == 130
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert c.read(prepared/'state.json')['status'] == 'interrupted'
        assert (prepared/'report.md').exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_dead_worker_native_group_is_cleaned(prepared, tmp_path):
    change_plan(prepared, [task(0)])
    worker = tmp_path/'crashing_owner.py'
    worker.write_text('''import json, os, subprocess, sys
from pathlib import Path
out=Path(sys.argv[1]); out.mkdir(parents=True)
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
identity=subprocess.run(['ps','-p',str(child.pid),'-o','lstart=','-o','command='],capture_output=True,text=True).stdout.strip()
(out/'native.owner.json').write_text(json.dumps({'pid':child.pid,'identity':identity,'status':'running'}))
os._exit(9)
''')
    c.supervise(prepared, poll=.02, emit=lambda m: None,
                worker_override=lambda a: [sys.executable, str(worker), a['directory']])
    owner = c.read(prepared/'tasks/task_000_bootstrap/native.owner.json')
    assert owner['status'] == 'cleaned-after-parent-exit'
    # A reparented process can briefly remain a zombie until launchd reaps it.
    result = subprocess.run(['ps', '-p', str(owner['pid']), '-o', 'stat='], capture_output=True, text=True)
    assert not result.stdout.strip() or result.stdout.strip().startswith('Z')


def test_installed_native_rotated_tiny_instance_preserves_original_clearances(tmp_path):
    """Tiny synthetic API smoke check; never submits a real-inventory campaign."""
    from flagstone import quarter as q
    native = Path.home()/'miniconda3/envs/patio-packingsolver/bin/python'
    if not native.exists():
        pytest.skip('Optional native solver environment unavailable')
    x = Layout(box(0, 0, 2, 2), {'a': box(-.4, -.4, .4, .4), 'b': box(-.2, -.2, .2, .2)},
               np.zeros((2, 3)), np.zeros(2, bool), ('a', 'b'), np.random.default_rng(1), units='ft')
    source = tmp_path/'native_input.json'
    q.write(source, q.native_input(x, 101, angle_deg=25))
    out = tmp_path/'native_output'
    result = subprocess.run([str(native), str(Path(c.__file__).with_name('quarter_native.py')),
                             '--input', str(source), '--output', str(out), '--seconds', '1'],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    payload = q.json.loads((out/'patterns.json').read_text())
    assert payload['patterns']
    for pattern in payload['patterns']:
        q.apply_records(x, pattern['records'])
        assert not q.validate(x, gap=q.GAP, edge=q.GAP)


def test_cli_hands_supervision_to_frozen_source_without_launching_tasks(prepared):
    change_plan(prepared, [], seconds=0)
    result = subprocess.run([sys.executable, str(Path(c.__file__)), '--output', str(prepared), '--resume', '--cores', '2'],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    state = c.read(prepared/'state.json')
    assert state['supervisor_script'] == str(prepared/'snapshot/scripts/quarter_campaign.py')
    assert state['max_concurrent_tasks'] == 2
    c.validate_frozen(prepared)
    assert not state['attempts']


def interval_peak(attempts):
    events = []
    for a in attempts:
        folder = Path(a['directory'])
        events += [(float((folder/'timing_start').read_text()), 1),
                   (float((folder/'timing_end').read_text()), -1)]
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    return peak


@pytest.mark.parametrize('cores,count', [(2, 5), (10, 12)])
def test_parallel_workers_actually_overlap_and_obey_cpu_budget(prepared, tmp_path, cores, count):
    change_plan(prepared, [task(i, seconds=5) for i in range(count)], seconds=20)
    worker = fake_worker(tmp_path)
    code = c.supervise(prepared, cores=cores, poll=.01, emit=lambda m: None,
                       worker_override=lambda a: [sys.executable, str(worker), a['directory'], 'hold' if cores == 10 else 'brief'])
    state = c.read(prepared/'state.json')
    assert code == 0
    assert state['peak_active_tasks'] == cores
    assert interval_peak(state['attempts']) == cores
    assert all(a['active_at_launch'] <= cores for a in state['attempts'])
    for a in state['attempts']:
        assert c.read(Path(a['directory'])/'threads.json') == {
            'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'VECLIB_MAXIMUM_THREADS':'1'}


def test_parallel_phase_barriers_wait_for_completed_comparison_group(prepared, tmp_path):
    tasks = [task(0), task(1),
             dict(task(2), phase='screen', variant='energy', seed=101, parent_rule='matched-bootstrap'),
             dict(task(3), phase='screen', variant='worst', seed=101, parent_rule='matched-bootstrap'),
             dict(task(4), phase='deepen', variant='energy', parent_rule='own-best-certified-gap')]
    change_plan(prepared, tasks)
    worker = fake_worker(tmp_path)
    c.supervise(prepared, cores=2, poll=.01, emit=lambda m: None,
                worker_override=lambda a: [sys.executable, str(worker), a['directory'], 'brief'])
    attempts = c.read(prepared/'state.json')['attempts']
    def times(indices, name):
        return [float((Path(attempts[i]['directory'])/name).read_text()) for i in indices]
    assert min(times((2, 3), 'timing_start')) >= max(times((0, 1), 'timing_end'))
    assert min(times((4,), 'timing_start')) >= max(times((2, 3), 'timing_end'))
    assert [a.get('parent_number') for a in attempts[2:]] == [0, 0, 2]


def test_slowly_dispatched_portfolio_uses_frozen_parent_set(prepared, tmp_path):
    tasks = [task(0),
             dict(task(1), phase='portfolio', variant='energy', parent_rule='pareto-gap'),
             dict(task(2), phase='portfolio', variant='worst', parent_rule='pareto-gap')]
    change_plan(prepared, tasks)
    worker = fake_worker(tmp_path)
    c.supervise(prepared, cores=1, poll=.01, emit=lambda m: None,
                worker_override=lambda a: [sys.executable, str(worker), a['directory'],
                                            'better' if a['number'] == 1 else 'good'])
    state = c.read(prepared/'state.json')
    assert state['attempts'][1]['result']['gap']['upper_diameter_in'] == 2.3
    assert state['attempts'][2]['parent_number'] == 0
    assert state['group_parents']['portfolio:pareto-gap'] == [0]


def test_resume_adopts_multiple_workers_without_duplicates(prepared, tmp_path):
    plan = change_plan(prepared, [task(0), task(1)])
    worker = fake_worker(tmp_path)
    records, processes = [], []
    now = time.time()
    (prepared/'tasks').mkdir()
    for i in range(2):
        directory = str(prepared/f'tasks/task_{i:03d}_bootstrap')
        logpath = prepared/f'tasks/task_{i:03d}.log'
        with logpath.open('wb') as log:
            process = subprocess.Popen([sys.executable, str(worker), directory, 'brief'],
                                       stdout=log, stderr=log, start_new_session=True)
        processes.append(process)
        records.append(dict(number=i, task=plan['tasks'][i], directory=directory, log=str(logpath),
                            outcome='running', pid=process.pid, identity=c.identity(process.pid),
                            started_epoch=now, deadline_epoch=now+5, budget_seconds=5))
    state = c.read(prepared/'state.json')
    state.update(status='running', attempts=records, next_task=2, started_utc=c.utc(), deadline_epoch=now+10)
    c.atomic_json(prepared/'state.json', state)
    try:
        c.supervise(prepared, cores=2, poll=.01, emit=lambda m: None)
        for a in c.read(prepared/'state.json')['attempts']:
            assert a['outcome'] == 'completed'
            assert len((Path(a['directory'])/'starts').read_text().splitlines()) == 1
    finally:
        for process in processes:
            process.wait(timeout=5)


def test_parallel_ctrl_c_reaps_all_workers(prepared, tmp_path):
    change_plan(prepared, [task(i, seconds=20) for i in range(3)])
    worker = fake_worker(tmp_path)
    code = f'''import sys
sys.path.insert(0,{str(Path(c.__file__).parent)!r})
import quarter_campaign as c
raise SystemExit(c.supervise({str(prepared)!r},cores=2,poll=.01,emit=lambda m:None,
    worker_override=lambda a:[sys.executable,{str(worker)!r},a['directory'],'sleep']))
'''
    process = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    markers = [prepared/f'tasks/task_{i:03d}_bootstrap/starts' for i in range(2)]
    try:
        deadline = time.monotonic()+5
        while not all(p.exists() for p in markers) and time.monotonic() < deadline:
            time.sleep(.02)
        assert all(p.exists() for p in markers)
        pids = [int(p.read_text().strip()) for p in markers]
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=8) == 130
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        state = c.read(prepared/'state.json')
        assert len(state['attempts']) == 2
        assert all(a['outcome'] == 'interrupted' for a in state['attempts'])
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize('cores', [0, -1, (os.cpu_count() or 1)+1])
def test_invalid_cpu_budgets_rejected_before_preparation(tmp_path, cores):
    out = tmp_path/'invalid'
    result = subprocess.run([sys.executable, str(Path(c.__file__)), '--output', str(out),
                             '--cores', str(cores), '--dry-run'], capture_output=True, text=True)
    assert result.returncode == 2
    assert '--cores must be' in result.stderr
    assert not out.exists()


def test_old_snapshot_gets_additive_recovery_without_rewriting_sources(prepared):
    plan = c.read(prepared/'plan.json')
    plan['version'] = 'quarter-campaign-v1'
    c.atomic_json(prepared/'plan.json', plan)
    manifest = c.read(prepared/'manifest.json')
    manifest['plan.json'] = hashlib.sha256((prepared/'plan.json').read_bytes()).hexdigest()
    legacy = prepared/'snapshot/scripts/quarter_campaign.py'
    legacy.write_text('# old frozen supervisor\n')
    manifest['snapshot/scripts/quarter_campaign.py'] = hashlib.sha256(legacy.read_bytes()).hexdigest()
    c.atomic_json(prepared/'manifest.json', manifest)
    before = (prepared/'state.json').read_bytes()
    original_manifest = (prepared/'manifest.json').read_bytes()
    result = subprocess.run([sys.executable, str(Path(c.__file__)), '--output', str(prepared),
                             '--cores', '2', '--resume', '--dry-run'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert legacy.read_text() == '# old frozen supervisor\n'
    assert (prepared/'manifest.json').read_bytes() == original_manifest
    assert (prepared/'state.json').read_bytes() == before
    assert c.read(prepared/'recovery.json')['schema'] == c.RESUME_SCHEMA
    assert (c.active_snapshot(prepared)/'src/flagstone/quarter.py').read_bytes() == (
        prepared/'snapshot/src/flagstone/quarter.py').read_bytes()
    c.validate_frozen(prepared)


def test_lower_cpu_budget_on_recovery_reaps_excess_live_workers(prepared, tmp_path):
    plan = change_plan(prepared, [task(0), task(1)])
    worker = fake_worker(tmp_path)
    records, processes = [], []
    now = time.time()
    (prepared/'tasks').mkdir()
    for i in range(2):
        directory = str(prepared/f'tasks/task_{i:03d}_bootstrap')
        logpath = prepared/f'tasks/task_{i:03d}.log'
        with logpath.open('wb') as log:
            process = subprocess.Popen([sys.executable, str(worker), directory, 'sleep'],
                                       stdout=log, stderr=log, start_new_session=True)
        processes.append(process)
        records.append(dict(number=i, task=plan['tasks'][i], directory=directory, log=str(logpath),
                            outcome='running', pid=process.pid, identity=c.identity(process.pid),
                            started_epoch=now, deadline_epoch=now+5, budget_seconds=5))
    state = c.read(prepared/'state.json')
    state.update(status='running', attempts=records, next_task=2, started_utc=c.utc(), deadline_epoch=now+1,
                 max_concurrent_tasks=2, peak_active_tasks=2, resource_history=[dict(time=c.utc(), cores=2)])
    c.atomic_json(prepared/'state.json', state)
    try:
        c.supervise(prepared, cores=1, poll=.01, emit=lambda m: None)
        final = c.read(prepared/'state.json')
        assert final['max_concurrent_tasks'] == 1
        assert final['attempts'][1]['outcome'] == 'interrupted-resource-change'
        assert [r['cores'] for r in final['resource_history']] == [2, 1]
        for process in processes:
            process.wait(timeout=5)
            assert process.returncode != 0
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def test_interrupted_tasks_resume_without_repeating_completed_tasks(prepared, tmp_path):
    change_plan(prepared, [task(i, seconds=8) for i in range(3)])
    worker = fake_worker(tmp_path)
    def interrupt(message):
        if message.startswith('START #001'):
            raise KeyboardInterrupt
    override = lambda a: [sys.executable, str(worker), a['directory'], 'good']
    assert c.supervise(prepared, cores=1, poll=.01, emit=interrupt, worker_override=override) == 130
    first = c.read(prepared/'state.json')
    assert first['attempts'][0]['outcome'] == 'completed'
    old_completed = first['attempts'][0]
    old_interrupted = first['attempts'][1]
    assert old_interrupted['outcome'] == 'interrupted'
    assert c.supervise(prepared, cores=1, poll=.01, emit=lambda m: None, worker_override=override) == 0
    final = c.read(prepared/'state.json')
    assert len(final['attempts']) == 3
    assert final['attempts'][0] == old_completed
    resumed = final['attempts'][1]
    assert resumed['outcome'] == 'completed'
    assert resumed['number'] == old_interrupted['number']
    assert resumed['directory'] != old_interrupted['directory']
    assert len(resumed['segments']) == 1
    assert resumed['total_budget_seconds'] == 8
    assert resumed['budget_seconds'] < 8
    assert len((Path(old_completed['directory'])/'starts').read_text().splitlines()) == 1


def test_resume_clock_credits_paused_time_without_resetting_full_budget(prepared):
    change_plan(prepared, [], seconds=1000)
    state = c.read(prepared/'state.json')
    now = time.time()
    state.update(status='interrupted', started_utc=c.utc(), deadline_epoch=now-50,
                 updated_utc=c.datetime.fromtimestamp(now-100, c.timezone.utc).isoformat())
    c.atomic_json(prepared/'state.json', state)
    assert c.pause_remaining(state) == pytest.approx(50, abs=.01)
    c.supervise(prepared, emit=lambda m: None, poll=.01)
    final = c.read(prepared/'state.json')
    assert now+49 < final['deadline_epoch'] < now+52
    assert final['resume_history'][-1]['remaining_seconds'] == pytest.approx(50, abs=.01)


def test_task_budget_and_checkpoint_chain_survive_repeated_startup_interruption(tmp_path):
    folder = tmp_path/'tasks/task_001_energy'
    folder.mkdir(parents=True)
    (folder/'best.json').write_text('saved')
    record = dict(number=1, directory=str(folder), budget_seconds=20, elapsed_seconds=5,
                  outcome='interrupted')
    assert c.prepare_task_resume(tmp_path, record, time.time()+100)
    assert record['budget_seconds'] == pytest.approx(15, abs=.01)
    assert record['resume_from'] == str(folder)
    # The new segment never reached disk before the next interruption.
    record.update(outcome='interrupted', elapsed_seconds=8)
    assert c.prepare_task_resume(tmp_path, record, time.time()+100)
    assert record['budget_seconds'] == pytest.approx(12, abs=.01)
    assert record['resume_from'] == str(folder)
    assert record['total_budget_seconds'] == 20
    assert len(record['segments']) == 2


def test_low_disk_guard_preserves_campaign_without_launching_worker(prepared):
    plan = c.read(prepared/'plan.json')
    plan['minimum_free_bytes'] = 10**30
    c.atomic_json(prepared/'plan.json', plan)
    manifest = c.read(prepared/'manifest.json')
    manifest['plan.json'] = hashlib.sha256((prepared/'plan.json').read_bytes()).hexdigest()
    c.atomic_json(prepared/'manifest.json', manifest)
    assert c.supervise(prepared, poll=.02, emit=lambda m: None) == 1
    state = c.read(prepared/'state.json')
    assert state['status'] == 'stopped-low-disk-space' and state['attempts'] == []
