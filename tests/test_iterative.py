import numpy as np
import pytest
from flagstone.generate import inventory
from flagstone.iterative import start_iterative, continue_iterative
from flagstone.phase1 import validate
from flagstone.search import greedy_insert


def test_iterative_resume_matches_fixed_pass_search(tmp_path):
    a = inventory(seed=98, counts=(3,7,10), areas=(.03,.012,.003))
    direct = inventory(seed=98, counts=(3,7,10), areas=(.03,.012,.003))
    path = tmp_path/'iterative.json'
    start_iterative(a, path, proposals=4, strategy='edge')
    continue_iterative(path, 3, max_stones=5)
    actual, state = continue_iterative(path, 3)
    for _ in range(3):
        greedy_insert(direct, proposals=4, strategy='edge')
    assert np.array_equal(actual.poses, direct.poses)
    assert np.array_equal(actual.active, direct.active)
    assert actual.rng.random() == direct.rng.random()
    assert not validate(actual)
    assert all(b['coverage'] >= a['coverage'] for a,b in zip(state['improvements'],state['improvements'][1:]))


def test_iterative_interrupt_replays_stone_from_checkpoint(tmp_path, monkeypatch):
    import flagstone.iterative as module
    a = inventory(seed=8, counts=(5,), areas=(.02,))
    path = tmp_path/'run.json'
    start_iterative(a, path, proposals=3)
    before = path.read_bytes()
    original = module.greedy_insert
    def fail(layout, **kwargs):
        layout.rng.random()
        layout.poses[0,0] = 999
        raise KeyboardInterrupt()
    monkeypatch.setattr(module, 'greedy_insert', fail)
    with pytest.raises(KeyboardInterrupt):
        continue_iterative(path, 1)
    assert path.read_bytes() == before
    monkeypatch.setattr(module, 'greedy_insert', original)
    result, state = continue_iterative(path, 1)
    assert state['completed_passes'] == 1 and not validate(result)


def test_iterative_zero_budget_does_not_advance(tmp_path):
    x = inventory(counts=(2,), areas=(.01,))
    path = tmp_path/'run.json'; start_iterative(x, path)
    before = path.read_bytes()
    _, state = continue_iterative(path, 1, max_seconds=0)
    assert state['stones_tried'] == 0 and path.read_bytes() == before


def test_iterative_keeps_frozen_stones_and_uncertainty(tmp_path):
    from flagstone.uncertainty import parse_uncertainty_model
    x = inventory(seed=5, counts=(5,), areas=(.01,))
    x.active[0] = True; x.poses[0] = [.5,.5,.2]
    fixed = x.poses[0].copy()
    model = {'uncertainty_version':1, 'units':x.units, 'patio':.005,
             'stones':{sid:{'shape':.001,'translation':.001,'rotation':.01} for sid in x.ids}}
    margins, patio = parse_uncertainty_model(x, model)
    path = tmp_path/'model.json'
    start_iterative(x, path, proposals=4, frozen_ids=[x.ids[0]], stone_margins=margins,
                    patio_margin=patio, uncertainty_model=model)
    continue_iterative(path, 2, max_stones=2)
    final, state = continue_iterative(path, 2)
    assert state['uncertainty_model'] == model
    assert final.active[0] and np.array_equal(final.poses[0], fixed)
    assert not validate(final, stone_margins=margins, patio_margin=patio)


@pytest.mark.parametrize('bit_generator', [np.random.MT19937, np.random.Philox])
def test_iterative_resume_with_array_state_generator(tmp_path, bit_generator):
    layout = inventory(seed=3, counts=(7,), areas=(.01,))
    layout.rng = np.random.Generator(bit_generator(57))
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    start_iterative(layout, first, proposals=3)
    start_iterative(layout, second, proposals=3)
    expected, _ = continue_iterative(first, 2)
    continue_iterative(second, 2, max_stones=3)
    resumed, _ = continue_iterative(second, 2)
    assert type(resumed.rng.bit_generator) is bit_generator
    assert np.array_equal(resumed.poses, expected.poses)
    assert np.array_equal(resumed.active, expected.active)
    assert np.array_equal(resumed.rng.random(10), expected.rng.random(10))
    assert not validate(resumed)


def test_geometry_cache_does_not_change_search_trajectory(tmp_path, monkeypatch):
    import flagstone.iterative as iterative
    x = inventory(seed=73, counts=(3,7), areas=(.03,.01), vertex_profile='varied')
    a, b = tmp_path / 'cached.json', tmp_path / 'uncached.json'
    start_iterative(x, a, proposals=4, strategy='edge')
    start_iterative(x, b, proposals=4, strategy='edge')
    cached, cached_state = continue_iterative(a, 3)
    original = iterative.save_checkpoint
    def uncached(layout, path, metadata, **kwargs):
        return original(layout, path, metadata)
    monkeypatch.setattr(iterative, 'save_checkpoint', uncached)
    plain, plain_state = continue_iterative(b, 3)
    assert {k:v for k,v in cached_state.items() if k != 'run_id'} == {k:v for k,v in plain_state.items() if k != 'run_id'}
    assert np.array_equal(cached.poses, plain.poses) and np.array_equal(cached.active, plain.active)
    assert np.array_equal(cached.rng.random(5), plain.rng.random(5))


@pytest.mark.parametrize('relaxation_steps', [0, 1])
def test_prevalidated_loop_matches_full_validation_trajectory(tmp_path, monkeypatch, relaxation_steps):
    import flagstone.iterative as iterative
    x = inventory(seed=86, counts=(3,7), areas=(.03,.01), vertex_profile='varied')
    x.active[0] = True; x.poses[0] = [.5,.5,0]
    margins = {sid:.001 for sid in x.ids}
    options = dict(proposals=5, strategy='wall', frozen_ids=[x.ids[0]], stone_margins=margins,
                   patio_margin=.001, relaxation_steps=relaxation_steps,
                   local_relaxation=bool(relaxation_steps))
    a, b = tmp_path / 'fast.json', tmp_path / 'reference.json'
    start_iterative(x, a, **options)
    start_iterative(x, b, **options)
    fast, fast_state = continue_iterative(a, 3)
    original = iterative.greedy_insert
    def fully_checked(layout, *args, **kwargs):
        kwargs.pop('_validation_token', None)
        return original(layout, *args, **kwargs)
    monkeypatch.setattr(iterative, 'greedy_insert', fully_checked)
    reference, ref_state = continue_iterative(b, 3)
    assert {k:v for k,v in fast_state.items() if k != 'run_id'} == {k:v for k,v in ref_state.items() if k != 'run_id'}
    assert np.array_equal(fast.poses, reference.poses) and np.array_equal(fast.active, reference.active)
    assert np.array_equal(fast.rng.random(5), reference.rng.random(5))
    assert np.array_equal(fast.poses[0], [.5,.5,0])
    assert not validate(fast, stone_margins=margins, patio_margin=.001)
