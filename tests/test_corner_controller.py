"""Behavioral contracts for the first, explicitly warm-start controller slice."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from shapely.geometry import Polygon, box

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import corner_growth as engine
import corner_controller as controller
from flagstone.phase1 import Layout, save_json


@pytest.fixture
def trial(tmp_path, monkeypatch):
    shapes = {'1': box(-.63,-.63,.63,.63), '2': box(-.3,-.3,.3,.3),
              '3': box(-.3,-.3,.3,.3)}
    layout = Layout(Polygon([(0,0),(4,0),(4,4),(0,4)]), shapes, np.zeros((3,3)),
                    np.zeros(3,dtype=bool), tuple(shapes), np.random.default_rng(1), units='ft')
    source = tmp_path/'source.json'
    save_json(layout, source)
    monkeypatch.setattr(engine, 'SOURCE', str(source))
    original = engine.frame
    monkeypatch.setattr(engine, 'frame', lambda patio, vertex=0: original(patio, 0))
    root = tmp_path/'trial'
    engine.declare(root)
    g = engine.GAP
    history = [dict(step=1, stone_id='1', pose=[.63+g,.63+g,0]),
               dict(step=2, stone_id='2', pose=[1.26+2*g+.3,.3+g,0]),
               dict(step=3, stone_id='3', pose=[.3+g,1.26+2*g+.3,0])]
    return root, history


def test_final_pass_cannot_hide_failed_first_prefix(trial):
    root, history = trial
    history[0], history[1] = history[1], history[0]
    for n,r in enumerate(history,1):
        r['step'] = n
    result = engine.evaluate_history(root, history)
    assert result['results'][-1]['passed']
    assert result['results'][0]['gap_status'] == 'failed'
    assert not result['passed']


def test_both_oracles_run_on_colliding_pattern(trial):
    root, history = trial
    history[0]['pose'] = [.9,.9,0]
    history[1]['pose'] = history[0]['pose'].copy()
    history[2]['pose'] = [3.5,3.5,0]
    result = engine.evaluate_history(root, history)
    assert result['status'] == 'repairable-invalid'
    assert any(r['clearance_errors'] for r in result['results'])
    assert result['results'][-1]['gap_status'] == 'failed'
    assert '1' in result['results'][-1]['implicated_ids']
    assert controller.next_moving(result, 1) == 3


def test_straddling_interval_is_unresolved(trial, monkeypatch):
    root, history = trial
    monkeypatch.setattr(engine, 'bounded_gap', lambda *a, **k: dict(
        passed=False, lower_diameter_in=1.999, upper_diameter_in=2.001, center=[.1,.1], cells=4))
    result = engine.evaluate_history(root, history)
    assert not result['passed']
    assert all(r['gap_status'] == 'unresolved' for r in result['results'])


def test_repair_job_keeps_fixed_poses_and_separate_ranges(trial, tmp_path):
    root, history = trial
    before = json.loads(json.dumps(history))
    result = engine.repair_history(root, history, tmp_path/'round', count=1,
                                   radius=.02, angle_radius=.005, maxiter=8, seed=None)
    assert history == before
    assert [r['pose'] for r in result['history'][:2]] == [r['pose'] for r in before[:2]]
    assert result['movable_ids'] == ['3']
    assert abs(result['history'][2]['pose'][2]) <= .005+1e-10
    assert (tmp_path/'round/repair_result.json').is_file()
    assert not (tmp_path/'round/accepted').exists()


def test_regression_restarts_best_and_exhausts_honestly(trial, tmp_path, monkeypatch):
    root, history = trial
    history[0], history[1] = history[1], history[0]
    for n,r in enumerate(history,1):
        r['step'] = n
    history_file = tmp_path/'seed.json'
    engine.write(history_file, history)
    calls = []
    def worsening_round(directory, records, output, **kwargs):
        calls.append(json.loads(json.dumps(records)))
        worse = json.loads(json.dumps(records))
        worse[1]['pose'] = worse[0]['pose'].copy()
        return dict(history=worse, objective=0)  # A misleading objective cannot promote.
    monkeypatch.setattr(engine, 'repair_history', worsening_round)
    out = tmp_path/'run'
    result = controller.run(root, history_file, out, controller.Policy(max_rounds=2))
    assert result['best'] == 'initial'
    assert calls[0] == calls[1] == history
    assert result['status'] == 'round-budget-exhausted/incomplete'
    assert not (out/'accepted').exists()
    assert json.loads((out/'state.json').read_text()) == result


def test_promotion_requires_reloaded_files_and_keeps_review_pending(trial, tmp_path):
    root, history = trial
    seed = tmp_path/'seed.json'
    engine.write(seed, history)
    out = tmp_path/'run'
    result = controller.run(root, seed, out)
    assert result['status'] == 'saved-file-verified'
    assert result['visual_review'] == 'pending'
    assert result['milestones_advanced'] == []
    assert json.loads((out/'accepted/verification.json').read_text())['passed']
    manifest = json.loads((out/'accepted/manifest.json').read_text())
    assert 'layout.json' in manifest and 'regions.geojson' in manifest
    assert not (out/'pending_export').exists()


def test_partial_export_cannot_promote(trial, tmp_path, monkeypatch):
    root, history = trial
    seed = tmp_path/'seed.json'
    engine.write(seed, history)
    def broken_export(*args):
        Path(args[-1]).mkdir()
        raise ValueError('Saved physical geometry differs from original source')
    monkeypatch.setattr(engine, 'save_verified', broken_export)
    out = tmp_path/'run'
    with pytest.raises(ValueError, match='geometry differs'):
        controller.run(root, seed, out)
    assert json.loads((out/'state.json').read_text())['status'] == 'failed'
    assert not (out/'accepted').exists()


@pytest.mark.parametrize('milestone', [5,7,9,11])
def test_historical_original_geometry_certificates(milestone):
    history = json.loads(Path(f'examples/corner_milestone_{milestone}/history.json').read_text())
    result = engine.evaluate_history('examples/corner_01', history)
    assert result['passed']
    assert all(r['upper_diameter_in'] < 2 and not r['clearance_errors'] for r in result['results'])


def test_real_controller_repairs_current_spacing_failure(trial, tmp_path):
    root, history = trial
    history[1]['pose'][0] -= .02
    seed = tmp_path/'seed.json'
    engine.write(seed, history)
    assert engine.evaluate_history(root, history)['status'] == 'repairable-invalid'
    out = tmp_path/'run'
    result = controller.run(root, seed, out, controller.Policy(
        max_rounds=2, maxiter=100, translation_radius_ft=.06, angle_radius_radians=.01))
    assert result['status'] == 'saved-file-verified'
    assert result['rounds_completed'] >= 1
    assert json.loads((out/'accepted/verification.json').read_text())['passed']


def test_resumed_checkpoint_does_not_depend_on_ancestor_eligibility(trial, tmp_path, monkeypatch):
    root, history = trial
    pool = tmp_path/'pool'
    pool.mkdir()
    engine.write(pool/'candidate_pool.json', dict(trial=str(root), parent_history=history[:1],
                 target=3, candidates=history[1:]))
    selection = tmp_path/'selection'
    selection.mkdir()
    engine.write(selection/'settings.json', dict(pool=str(pool)))
    engine.write(selection/'trace.json', dict(iterations=[dict(chosen=[0,1],
                 stone_ids=['2','3'], conflict=True)]))
    resume = tmp_path/'resume'
    resume.mkdir()
    engine.write(resume/'history.json', history)
    engine.write(resume/'repair_result.json', dict(centers=[]))
    calls = []
    def one_round(directory, records, output, **kwargs):
        calls.append(records)
        return dict(history=records, centers=[], passed=False, objective=2.1)
    monkeypatch.setattr(engine, 'repair_history', one_round)
    engine.repair_pairs(selection, tmp_path/'repair', resume=resume, rounds=1)
    assert len(calls) == 1
    assert calls[0] == history


def test_elapsed_budget_preserves_evaluation_without_starting_export(trial, tmp_path):
    root, history = trial
    seed = tmp_path/'seed.json'
    engine.write(seed, history)
    out = tmp_path/'run'
    result = controller.run(root, seed, out, controller.Policy(max_seconds=1e-9))
    assert result['status'] == 'budget-exhausted/incomplete'
    assert result['rounds_completed'] == 0
    assert (out/'initial/evaluation.json').is_file()
    assert not (out/'accepted').exists()


def test_regressed_round_keeps_its_gap_witness_for_best_checkpoint_restart(trial,tmp_path,monkeypatch):
    root,history=trial
    history[0],history[1]=history[1],history[0]
    for n,r in enumerate(history,1):r['step']=n
    seed=tmp_path/'seed.json';engine.write(seed,history)
    calls=[]
    witness=dict(step=3,center=[1.11,1.22])
    def round_job(directory,records,output,**kwargs):
        calls.append(kwargs['centers'])
        worse=json.loads(json.dumps(records));worse[1]['pose']=worse[0]['pose'].copy()
        return dict(history=worse,centers=[witness])
    monkeypatch.setattr(engine,'repair_history',round_job)
    result=controller.run(root,seed,tmp_path/'run',controller.Policy(max_rounds=2))
    assert result['best']=='initial'
    assert witness not in calls[0]
    assert witness in calls[1]
    assert witness in json.loads((tmp_path/'run/witnesses.json').read_text())
