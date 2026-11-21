"""Real foreground subprocess checks using tiny deterministic native stand-ins."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box

from flagstone.phase1 import Layout, save_json
from flagstone.quarter import prepare
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import quarter_patio as worker

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'scripts/quarter_patio.py'


@pytest.fixture
def tiny_trial(tmp_path):
    x = Layout(box(0, 0, 2, 2), {'a': box(-.4, -.4, .4, .4)},
               np.zeros((1, 3)), np.zeros(1, bool), ('a',), np.random.default_rng(1), units='ft')
    source = tmp_path/'source.json'
    save_json(x, source)
    trial = tmp_path/'trial'
    prepare(source, trial)
    return trial


def fake_native(tmp_path, text):
    path = tmp_path/'native'
    path.write_text(f'#!{sys.executable}\n'+text)
    path.chmod(0o755)
    return path


def command(trial, out, native):
    return [sys.executable, str(SCRIPT), 'run', '--trial', str(trial), '--output', str(out),
            '--native-python', str(native), '--seconds', '30', '--bootstrap-seconds', '2', '--rounds', '0']


def test_foreground_pipeline_certifies_original_final_layout(tiny_trial, tmp_path):
    native = fake_native(tmp_path, """import json, sys
from pathlib import Path
out=Path(sys.argv[sys.argv.index('--output')+1]); out.mkdir()
(out/'patterns.json').write_text(json.dumps({'patterns':[{'records':[{'stone_id':'a','pose':[.5,.5,0]}]}]}))
""")
    out = tmp_path/'run'
    result = subprocess.run(command(tiny_trial, out, native), cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    summary = json.loads((out/'summary.json').read_text())
    assert summary['status'] == 'target-geometrically-verified-review-pending'
    assert summary['verification']['reference_matched']
    assert summary['verification']['gap']['upper_diameter_in'] < 2
    assert (out/'best.json').exists()


def test_native_crash_is_infrastructure_failure(tiny_trial, tmp_path):
    native = fake_native(tmp_path, 'raise RuntimeError("simulated native crash")\n')
    out = tmp_path/'run'
    result = subprocess.run(command(tiny_trial, out, native), cwd=ROOT, capture_output=True, timeout=15)
    # Summary distinguishes execution failure from geometric incompleteness.
    summary = json.loads((out/'summary.json').read_text())
    assert summary['status'] == 'bootstrap-worker-failed'
    assert result.returncode != 0
    assert summary['jobs'][0]['outcome'] == 'failed'
    assert not (out/'best.json').exists()


def test_ctrl_c_reaps_native_and_records_interruption(tiny_trial, tmp_path):
    marker = tmp_path/'native.pid'
    native = fake_native(tmp_path, f"""import os, time
from pathlib import Path
Path({str(marker)!r}).write_text(str(os.getpid()))
time.sleep(30)
""")
    out = tmp_path/'run'
    proc = subprocess.Popen(command(tiny_trial, out, native), cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic()+5
        while not marker.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert marker.exists()
        pid = int(marker.read_text())
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=5) != 0
        summary = json.loads((out/'summary.json').read_text())
        assert summary['status'] == 'interrupted/incomplete'
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def policy(**changes):
    values = dict(seed=1, objective='energy', target_policy='rotating', patch_stones=6,
                  patch_max_stones=12, rebuild_stones=3, rebuild_alternatives=1,
                  translation_radius_ft=.25, maxiter=180, proposal_strategy='random',
                  coverage_loss_limit=None, no_continuous=False, no_swaps=False,
                  no_insert=False, no_rebuild=False, native_angle_deg=0)
    values.update(changes)
    return SimpleNamespace(**values)


def saved_checkpoint(tiny_trial, tmp_path):
    from flagstone import quarter as q
    x = q.load_json(tiny_trial/'input.json')
    q.apply_records(x, [dict(stone_id='a', pose=[.5, .5, 0])])
    x.rng.random(7)
    old = tmp_path/'old_task'
    old.mkdir()
    metrics = q.measure(x)
    state = dict(initial=metrics, best=metrics, rounds=[dict(round=0), dict(round=1)],
                 elapsed_seconds=12, cpu_seconds_total=9)
    q.write(old/'state.json', state)
    digest = json.loads((tiny_trial/'contract.json').read_text())['input_sha256']
    worker.save_progress(x, old, state, policy(), digest, 2, .5)
    save_json(x, old/'best.json')
    return x, old, digest


def test_atomic_resume_restores_rng_rounds_and_coverage_floor(tiny_trial, tmp_path):
    from flagstone import quarter as q
    original, old, digest = saved_checkpoint(tiny_trial, tmp_path)
    # A full layout write can lag behind the atomic checkpoint at interruption.
    (old/'best.json').write_text('unfinished legacy layout')
    restored = q.load_json(tiny_trial/'input.json', seed=999)
    result = worker.restore_progress(restored, old, policy(), digest)
    assert result['next_round'] == 2 and result['coverage_floor'] == .5
    assert result['prior_elapsed_seconds'] == 12
    np.testing.assert_array_equal(restored.poses, original.poses)
    np.testing.assert_array_equal(restored.rng.random(8), original.rng.random(8))


def test_resume_cli_skips_bootstrap_and_keeps_committed_rounds(tiny_trial, tmp_path):
    _, old, _ = saved_checkpoint(tiny_trial, tmp_path)
    out = tmp_path/'resumed'
    result = subprocess.run([sys.executable, str(SCRIPT), 'run', '--trial', str(tiny_trial),
                             '--output', str(out), '--resume-from', str(old), '--seconds', '30',
                             '--rounds', '2', '--native-python', '/must/not/launch/native'],
                            cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    summary = json.loads((out/'summary.json').read_text())
    assert summary['mode'] == 'checkpoint-resume'
    assert len(summary['rounds']) == 2
    assert summary['verification']['passed']
    assert not (out/'native_input.json').exists()
    assert summary['elapsed_seconds'] >= 12
    assert summary['cpu_seconds_total'] >= 9


@pytest.mark.parametrize('change', ['policy', 'trial'])
def test_resume_rejects_incompatible_checkpoint(tiny_trial, tmp_path, change):
    from flagstone import quarter as q
    _, old, digest = saved_checkpoint(tiny_trial, tmp_path)
    restored = q.load_json(tiny_trial/'input.json')
    with pytest.raises(ValueError):
        worker.restore_progress(restored, old, policy(objective='worst') if change == 'policy' else policy(),
                                'wrong-input-hash' if change == 'trial' else digest)
    assert not restored.active.any()


def test_legacy_resume_preserves_geometry_round_position_and_original_floor(tiny_trial, tmp_path):
    from flagstone import quarter as q
    original, old, digest = saved_checkpoint(tiny_trial, tmp_path)
    (old/'checkpoint.json').unlink()
    state = json.loads((old/'state.json').read_text())
    state['settings'] = vars(policy(coverage_loss_limit=.005))
    state['initial']['coverage'] = .645
    q.write(old/'state.json', state)
    restored = q.load_json(tiny_trial/'input.json')
    result = worker.restore_progress(restored, old, policy(coverage_loss_limit=.005), digest)
    assert result['next_round'] == 2
    assert result['coverage_floor'] == pytest.approx(.64)
    assert 'RNG was not saved' in result['fidelity']
    np.testing.assert_array_equal(restored.poses, original.poses)
