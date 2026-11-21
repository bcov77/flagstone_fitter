import numpy as np
import pytest
from flagstone.exploration import start_exploration, continue_exploration
from flagstone.generate import inventory
from flagstone.search import greedy_insert
from flagstone.phase1 import validate


def test_exploration_retains_best_when_current_loses_coverage(tmp_path):
    x = inventory(seed=5, counts=(3,), areas=(.01,))
    greedy_insert(x, proposals=20)
    initial = x.poses.copy(); initial_active = x.active.copy()
    path = tmp_path / 'chain.json'
    start_exploration(x, path, proposals=0, patch_size=1, max_coverage_drop=.02)
    current, best, state = continue_exploration(path, 1)
    assert state['current_history'][-1] < state['best_history'][-1]
    assert np.array_equal(best.poses, initial) and np.array_equal(best.active, initial_active)
    assert not validate(current) and not validate(best)


@pytest.mark.parametrize('local', [False, True])
def test_exploration_checkpoint_resume_matches_uninterrupted(tmp_path, local):
    x = inventory(seed=12, counts=(2, 4, 6), areas=(.04, .02, .01))
    greedy_insert(x, proposals=4)
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    opts = dict(proposals=3, patch_size=2, strategy='random', max_coverage_drop=.04)
    if local:
        opts.update(patch_padding=.08, max_new_stones=4)
    start_exploration(x, a, **opts); start_exploration(x, b, **opts)
    ca, ba, sa = continue_exploration(a, 5)
    continue_exploration(b, 2)
    cb, bb, sb = continue_exploration(b, 5)
    assert {k:v for k,v in sa.items() if k != 'run_id'} == {k:v for k,v in sb.items() if k != 'run_id'}
    assert np.array_equal(ca.poses, cb.poses) and np.array_equal(ca.active, cb.active)
    assert np.array_equal(ba.poses, bb.poses) and np.array_equal(ba.active, bb.active)
    assert all(b >= a for a, b in zip(sa['best_history'], sa['best_history'][1:]))
    assert not validate(ca) and not validate(ba)
    assert sa['config']['patch_padding'] == (.08 if local else None)
    assert sa['config']['max_new_stones'] == (4 if local else None)


def test_exploration_interrupt_restores_checkpoint_and_frozen(tmp_path, monkeypatch):
    import pytest
    import flagstone.exploration as module
    x = inventory(seed=6, counts=(5,), areas=(.01,))
    greedy_insert(x, proposals=15)
    frozen = x.ids[0]
    assert x.active[0]
    fixed = x.poses[0].copy()
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    opts = dict(proposals=5, patch_size=2, strategy='random', frozen_ids=[frozen], max_coverage_drop=.03,
                patch_padding=.08, max_new_stones=3)
    start_exploration(x, a, **opts); start_exploration(x, b, **opts)
    ca, ba, sa = continue_exploration(a, 4)
    original = module.greedy_insert
    def interrupt(layout, **kwargs):
        if kwargs.get('stone_ids') == ():
            return original(layout, **kwargs)
        layout.rng.random()
        layout.poses[1, 0] += 5
        raise KeyboardInterrupt()
    monkeypatch.setattr(module, 'greedy_insert', interrupt)
    with pytest.raises(KeyboardInterrupt):
        continue_exploration(b, 4)
    monkeypatch.setattr(module, 'greedy_insert', original)
    cb, bb, sb = continue_exploration(b, 4)
    assert {k:v for k,v in sa.items() if k != 'run_id'} == {k:v for k,v in sb.items() if k != 'run_id'}
    assert np.array_equal(ca.poses, cb.poses) and np.array_equal(ba.poses, bb.poses)
    assert np.array_equal(cb.poses[0], fixed) and np.array_equal(bb.poses[0], fixed)
    assert cb.active[0] and bb.active[0]


def test_exploration_reconstructs_incumbent_for_standard_generators(tmp_path):
    for cls in (np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937,
                np.random.Philox, np.random.SFC64):
        x = inventory(seed=4, counts=(4,), areas=(.01,))
        x.rng = np.random.Generator(cls(9))
        greedy_insert(x, proposals=5)
        path = tmp_path / (cls.__name__ + '.json')
        start_exploration(x, path, strategy='random', proposals=1, patch_size=1, max_coverage_drop=.02)
        current, best, state = continue_exploration(path, 2)
        assert type(current.rng.bit_generator) is cls and type(best.rng.bit_generator) is cls
        assert np.array_equal(current.rng.random(5), best.rng.random(5))
        assert not validate(current) and not validate(best)
        assert state['rounds'] == 2


def test_legacy_exploration_checkpoint_without_local_settings_resumes(tmp_path):
    from flagstone.checkpoint import load_checkpoint, save_checkpoint
    x = inventory(seed=7, counts=(4,), areas=(.01,))
    greedy_insert(x, proposals=4)
    path = tmp_path / 'legacy.json'
    start_exploration(x, path, proposals=2, strategy='random')
    layout, state = load_checkpoint(path)
    state['config'].pop('patch_padding'); state['config'].pop('max_new_stones')
    save_checkpoint(layout, path, state)
    current, best, resumed = continue_exploration(path, 1)
    assert resumed['rounds'] == 1 and not validate(current) and not validate(best)


def test_exploration_rejects_invalid_patch_limits(tmp_path):
    x = inventory(seed=3, counts=(2,), areas=(.01,))
    for options in ({'patch_padding': -1}, {'patch_padding': float('nan')},
                    {'max_new_stones': -1}, {'max_new_stones': 1.5},
                    {'nudge_probability': 2}, {'nudge_probability': .5},
                    {'nudge_translation': -1}, {'nudge_angle': float('nan')}):
        with pytest.raises(ValueError):
            start_exploration(x, tmp_path / 'bad.json', **options)
    assert not (tmp_path / 'bad.json').exists()


def test_checkpointed_nudges_keep_best_and_frozen_stones(tmp_path):
    from flagstone.phase1 import synthetic
    x = synthetic(seed=13); x.active[2:] = False
    before = x.poses.copy()
    opts = dict(strategy='random', proposals=0, patch_size=1, frozen_ids=[x.ids[0]],
                patch_padding=.05, max_new_stones=0, nudge_probability=1.,
                nudge_translation=.005, nudge_angle=.01)
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    start_exploration(x, a, **opts); start_exploration(x, b, **opts)
    ca, ba, sa = continue_exploration(a, 8)
    continue_exploration(b, 3)
    cb, bb, sb = continue_exploration(b, 8)
    assert {k:v for k,v in sa.items() if k != 'run_id'} == {k:v for k,v in sb.items() if k != 'run_id'}
    assert sa['nudge_attempts'] == 8 and sa['nudge_valid'] > 0
    assert np.array_equal(ca.poses, cb.poses) and np.array_equal(ba.poses, bb.poses)
    assert np.array_equal(ca.poses[0], before[0]) and np.array_equal(ba.poses, before)
    assert not np.array_equal(ca.poses[1], before[1])
    assert not validate(ca) and not validate(ba)


def test_invalid_nudges_roll_back_and_preserve_incumbent(tmp_path):
    from flagstone.phase1 import synthetic
    x = synthetic(seed=7); x.active[2:] = False
    path = tmp_path / 'run.json'
    start_exploration(x, path, strategy='random', proposals=0, nudge_probability=1.,
                      nudge_translation=2., nudge_angle=.1)
    current, best, state = continue_exploration(path, 10)
    assert state['nudge_attempts'] == 10 and state['nudge_valid'] < 10
    assert not validate(current) and not validate(best)
    assert np.array_equal(best.poses, x.poses)


def test_nudge_round_can_add_stone_and_update_best(tmp_path):
    x = inventory(seed=44, counts=(2,), areas=(.01,))
    x.active[0] = True; x.poses[0] = [.25,.25,0]
    path = tmp_path / 'run.json'
    start_exploration(x, path, strategy='random', proposals=30, max_new_stones=1,
                      nudge_probability=1., nudge_translation=.005)
    current, best, state = continue_exploration(path, 1)
    assert current.active.all() and best.active.all()
    assert state['nudge_valid'] == 1 and state['improvements'] == 1
    assert state['best_history'][-1] > state['best_history'][0]
    assert not validate(current) and not validate(best)


def test_interrupted_nudge_replays_from_checkpoint(tmp_path, monkeypatch):
    import flagstone.exploration as module
    from flagstone.phase1 import synthetic
    x = synthetic(seed=10); x.active[2:] = False
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    opts = dict(strategy='random', proposals=0, nudge_probability=1.,
                nudge_translation=.001, patch_padding=.08, max_new_stones=0)
    start_exploration(x, a, **opts); start_exploration(x, b, **opts)
    expected, best_expected, _ = continue_exploration(a, 2)
    before = b.read_bytes()
    original = module._patch_scope
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(module, '_patch_scope', interrupt)
    with pytest.raises(KeyboardInterrupt):
        continue_exploration(b, 2)
    assert b.read_bytes() == before
    monkeypatch.setattr(module, '_patch_scope', original)
    current, best, _ = continue_exploration(b, 2)
    assert np.array_equal(current.poses, expected.poses)
    assert np.array_equal(best.poses, best_expected.poses)
    assert current.rng.random() == expected.rng.random()


def test_zero_nudge_preserves_legacy_random_trajectory(tmp_path):
    from flagstone.checkpoint import load_checkpoint, save_checkpoint
    x = inventory(seed=17, counts=(6,), areas=(.01,)); greedy_insert(x, proposals=4)
    current_path, legacy_path = tmp_path / 'current.json', tmp_path / 'legacy.json'
    start_exploration(x, current_path, strategy='random', proposals=3)
    start_exploration(x, legacy_path, strategy='random', proposals=3)
    layout, state = load_checkpoint(legacy_path)
    for key in ('nudge_probability', 'nudge_translation', 'nudge_angle'):
        state['config'].pop(key)
    save_checkpoint(layout, legacy_path, state)
    current, best, _ = continue_exploration(current_path, 4)
    legacy, legacy_best, _ = continue_exploration(legacy_path, 4)
    assert np.array_equal(current.poses, legacy.poses) and np.array_equal(current.active, legacy.active)
    assert np.array_equal(best.poses, legacy_best.poses) and np.array_equal(best.active, legacy_best.active)
    assert np.array_equal(current.rng.random(5), legacy.rng.random(5))
