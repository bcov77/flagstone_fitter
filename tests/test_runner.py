import numpy as np
import pytest
from flagstone.generate import inventory
from flagstone.runner import start_rebuild, continue_rebuild
from flagstone.search import greedy_insert
from flagstone.phase1 import validate


@pytest.mark.parametrize('restore_removed', [False, True])
def test_round_boundary_resume_matches_uninterrupted(tmp_path, restore_removed):
    x = inventory(seed=11, counts=(2, 4, 8), areas=(.04, .02, .01))
    greedy_insert(x, proposals=3)
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    start_rebuild(x, a, patch_size=2, proposals=4, strategy='random', restore_removed=restore_removed)
    start_rebuild(x, b, patch_size=2, proposals=4, strategy='random', restore_removed=restore_removed)
    expected, state_a = continue_rebuild(a, 5)
    _, intermediate = continue_rebuild(b, 2)
    actual, state_b = continue_rebuild(b, 5)
    assert intermediate['rounds'] == 2
    assert {k:v for k,v in state_a.items() if k != 'run_id'} == {k:v for k,v in state_b.items() if k != 'run_id'}
    assert np.allclose(actual.poses, expected.poses, atol=1e-12)
    assert np.array_equal(actual.active, expected.active)
    assert actual.rng.random() == expected.rng.random()
    assert not validate(actual)
    with pytest.raises(ValueError):
        continue_rebuild(b, 4)


def test_interrupted_round_leaves_previous_checkpoint(tmp_path, monkeypatch):
    import flagstone.runner as runner
    x = inventory(seed=7, counts=(2, 3), areas=(.04, .02))
    greedy_insert(x, proposals=5)
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    start_rebuild(x, a, proposals=4, strategy='random')
    start_rebuild(x, b, proposals=4, strategy='random')
    expected, state = continue_rebuild(a, 3)
    original = runner.rebuild_search
    def interrupted(layout, **kwargs):
        layout.rng.random()
        layout.poses[0, 0] += 10
        raise KeyboardInterrupt()
    monkeypatch.setattr(runner, 'rebuild_search', interrupted)
    with pytest.raises(KeyboardInterrupt):
        continue_rebuild(b, 3)
    monkeypatch.setattr(runner, 'rebuild_search', original)
    actual, resumed = continue_rebuild(b, 3)
    assert {k:v for k,v in resumed.items() if k != 'run_id'} == {k:v for k,v in state.items() if k != 'run_id'}
    assert np.array_equal(actual.poses, expected.poses)
    assert np.array_equal(actual.active, expected.active)


def test_durable_insertion_matches_uninterrupted_and_interrupted(tmp_path, monkeypatch):
    from flagstone.runner import start_insertion, continue_insertion
    import flagstone.search as search
    x = inventory(seed=22, counts=(2, 4, 8), areas=(.04, .02, .01))
    direct = inventory(seed=22, counts=(2, 4, 8), areas=(.04, .02, .01))
    greedy_insert(direct, proposals=5, strategy='mixed', grid_resolution=6)
    path = tmp_path / 'insert.json'
    start_insertion(x, path, proposals=5, strategy='mixed', grid_resolution=6)
    _, state = continue_insertion(path, max_stones=3)
    assert state['next_index'] == 3
    original = search.greedy_insert
    def interrupted(layout, **kwargs):
        layout.rng.random()
        layout.poses[0, 0] = 999
        raise KeyboardInterrupt()
    monkeypatch.setattr(search, 'greedy_insert', interrupted)
    with pytest.raises(KeyboardInterrupt):
        continue_insertion(path)
    monkeypatch.setattr(search, 'greedy_insert', original)
    actual, state = continue_insertion(path)
    assert state['next_index'] == len(x.ids)
    assert np.array_equal(actual.poses, direct.poses)
    assert np.array_equal(actual.active, direct.active)
    assert actual.rng.random() == direct.rng.random()
    assert not validate(actual)


@pytest.mark.parametrize('strategy', ['corner', 'contact', 'edge', 'wall'])
def test_proposal_strategy_survives_checkpoint_resume(tmp_path, strategy):
    from flagstone.runner import start_insertion, continue_insertion
    x = inventory(seed=3, counts=(5,), areas=(.01,))
    direct = inventory(seed=3, counts=(5,), areas=(.01,))
    greedy_insert(direct, proposals=4, strategy=strategy, grid_resolution=6)
    path = tmp_path / 'corner.json'
    start_insertion(x, path, proposals=4, strategy=strategy, grid_resolution=6)
    continue_insertion(path, max_stones=2)
    resumed, state = continue_insertion(path)
    assert state['config']['strategy'] == strategy
    assert np.array_equal(resumed.poses, direct.poses)
    assert np.array_equal(resumed.active, direct.active)


def test_checkpoint_preserves_frozen_ids(tmp_path):
    x = inventory(seed=3, counts=(4,), areas=(.01,))
    x.active[0] = True
    x.poses[0] = [.5, .5, .2]
    fixed = x.poses[0].copy()
    path = tmp_path / 'frozen.json'
    start_rebuild(x, path, frozen_ids=[x.ids[0]], proposals=3, strategy='random')
    resumed, state = continue_rebuild(path, 2)
    assert state['config']['frozen_ids'] == [x.ids[0]]
    assert resumed.active[0] and np.array_equal(resumed.poses[0], fixed)


def test_batched_insertion_checkpoint_matches_uninterrupted(tmp_path):
    from flagstone.runner import start_insertion, continue_insertion
    from flagstone.search import batched_random_insert
    x = inventory(seed=27, counts=(12,), areas=(.01,))
    direct = inventory(seed=27, counts=(12,), areas=(.01,))
    batched_random_insert(direct, proposals=7, batch_size=4)
    path = tmp_path / 'batch.json'
    start_insertion(x, path, proposals=7, batch_size=4)
    continue_insertion(path, max_stones=3)
    resumed, state = continue_insertion(path)
    assert state['config']['batch_size'] == 4
    assert np.array_equal(resumed.poses, direct.poses)
    assert np.array_equal(resumed.active, direct.active)
    assert resumed.rng.random() == direct.rng.random()


def test_run_id_persists_between_checkpoints(tmp_path):
    import json
    from flagstone.runner import start_insertion, continue_insertion
    x = inventory(seed=15, counts=(3,), areas=(.01,))
    path = tmp_path / 'run.json'
    state = start_insertion(x, path, proposals=3)
    first = json.loads(path.read_text())
    _, resumed = continue_insertion(path, max_stones=1)
    second = json.loads(path.read_text())
    assert state['run_id'] == resumed['run_id'] == first['run_id'] == second['run_id']


def test_local_patch_checkpoint_replays_identically(tmp_path):
    x = inventory(seed=13, counts=(3,6), areas=(.04,.01))
    greedy_insert(x, proposals=3)
    a, b = tmp_path/'a.json', tmp_path/'b.json'
    opts = dict(strategy='edge', patch_size=2, proposals=5, patch_padding=.1, max_new_stones=4)
    start_rebuild(x, a, **opts); start_rebuild(x, b, **opts)
    expected, sa = continue_rebuild(a, 3)
    continue_rebuild(b, 1)
    actual, sb = continue_rebuild(b, 3)
    assert np.array_equal(expected.poses, actual.poses) and np.array_equal(expected.active, actual.active)
    assert sa['config']['patch_padding'] == sb['config']['patch_padding'] == .1
    assert sa['config']['max_new_stones'] == sb['config']['max_new_stones'] == 4
    assert not validate(actual)


def test_neighborhood_relaxation_checkpoint_matches_direct(tmp_path):
    from flagstone.runner import start_insertion, continue_insertion
    x = inventory(seed=11, counts=(5,), areas=(.03,))
    direct = inventory(seed=11, counts=(5,), areas=(.03,))
    options = dict(proposals=4, strategy='edge', relaxation_steps=2, local_relaxation=True)
    greedy_insert(direct, **options)
    path = tmp_path/'local.json'
    start_insertion(x, path, **options)
    continue_insertion(path, max_stones=2)
    actual, state = continue_insertion(path)
    assert state['config']['local_relaxation'] is True
    assert np.allclose(actual.poses, direct.poses, atol=1e-12)
    assert np.array_equal(actual.active, direct.active) and not validate(actual)


def test_selective_relaxation_settings_and_counts_resume(tmp_path):
    from flagstone.runner import start_insertion, continue_insertion
    x = inventory(seed=8, counts=(6,), areas=(.04,))
    direct = inventory(seed=8, counts=(6,), areas=(.04,))
    opts = dict(proposals=5, strategy='edge', relaxation_steps=2, local_relaxation=True, max_relaxation_penalty=.2)
    expected = greedy_insert(direct, **opts)
    path = tmp_path/'selective.json'
    start_insertion(x, path, **opts)
    continue_insertion(path, max_stones=2)
    resumed, state = continue_insertion(path)
    assert state['config']['max_relaxation_penalty'] == .2
    assert state['relaxation_attempts'] == expected.relaxation_attempts
    assert state['relaxation_skipped'] == expected.relaxation_skipped
    assert state['relaxation_attempts'] == state['relaxation_successes'] + state['relaxation_failures']
    assert np.allclose(resumed.poses, direct.poses) and np.array_equal(resumed.active, direct.active)


def test_clearance_mobility_policy_checkpoint_resume(tmp_path):
    from flagstone.runner import start_insertion, continue_insertion
    a = inventory(seed=88, counts=(6,), areas=(.025,))
    b = inventory(seed=88, counts=(6,), areas=(.025,))
    opts = dict(strategy='edge', proposals=4, relaxation_steps=2, local_relaxation=True,
                max_relaxation_penalty=.2, mobility_policy='clearance')
    direct = greedy_insert(a, **opts)
    path = tmp_path/'run.json'
    start_insertion(b, path, **opts)
    continue_insertion(path, max_stones=2)
    resumed, state = continue_insertion(path)
    assert state['config']['mobility_policy'] == 'clearance'
    assert state['mobile_stone_visits'] == direct.mobile_stone_visits
    assert np.allclose(a.poses, resumed.poses, atol=1e-12)
    assert np.array_equal(a.active, resumed.active) and not validate(resumed)


def test_secondary_rebuild_checkpoint_resume_matches_uninterrupted(tmp_path):
    x = inventory(seed=21, counts=(4,), areas=(.01,))
    greedy_insert(x, proposals=5)
    a,b = tmp_path/'a.json', tmp_path/'b.json'
    opts = dict(strategy='random', proposals=3, patch_size=1,
                secondary_resolution=12, secondary_broad_distance=.12, secondary_edge_strip_width=.03)
    start_rebuild(x,a,**opts); start_rebuild(x,b,**opts)
    expected, sa = continue_rebuild(a,3)
    continue_rebuild(b,1)
    actual, sb = continue_rebuild(b,3)
    assert np.array_equal(expected.poses, actual.poses) and np.array_equal(expected.active, actual.active)
    assert sa['secondary_score'] == sb['secondary_score']
    assert sa['secondary_improvements'] == sb['secondary_improvements']
    assert all(b >= a for a,b in zip(sb['best_history'], sb['best_history'][1:]))
    assert not validate(actual)


def test_patch_size_benchmark_preserves_input_and_validates_results(tmp_path):
    import json
    import subprocess
    import sys
    from flagstone.generate import cut_witness
    from flagstone.runner import start_rebuild
    source = tmp_path / 'source.json'
    start_rebuild(cut_witness(), source, edge=.009)
    before = source.read_bytes()
    out = tmp_path / 'comparison'
    subprocess.run([sys.executable, 'scripts/benchmark_patch_size.py', '--input', str(source),
                    '--output', str(out), '--seconds', '0', '--seeds', '1', '--sizes', '2', '3'],
                   check=True, capture_output=True)
    report = json.loads((out / 'metrics.json').read_text())
    assert source.read_bytes() == before
    assert len(report['runs']) == 2
    assert all(r['rounds'] == 0 and r['violations'] == 0 for r in report['runs'])
