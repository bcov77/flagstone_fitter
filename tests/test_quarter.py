"""Final-layout behavior: inventory choice, spatial repair and honest gates."""
import json

import numpy as np
import pytest
from shapely.geometry import box

from flagstone.phase1 import Layout, save_json, validate
from flagstone import quarter as q


def layout(shapes=None):
    shapes = shapes or {'a': box(-.4, -.4, .4, .4), 'b': box(-.2, -.2, .2, .2)}
    return Layout(box(0, 0, 2, 2), shapes, np.zeros((len(shapes), 3)),
                  np.zeros(len(shapes), bool), tuple(shapes), np.random.default_rng(1), units='ft')


def test_quarter_uses_all_original_shapes_but_does_not_require_all_placed(tmp_path):
    x = layout()
    x.active[:] = True
    source = tmp_path/'source.json'
    save_json(x, source)
    before = source.read_bytes()
    contract = q.prepare(source, tmp_path/'trial')
    result = q.load_json(tmp_path/'trial/input.json')
    assert result.patio.area == pytest.approx(x.patio.area/4)
    assert result.ids == x.ids and not result.active.any()
    assert all(result.shapes[s].equals_exact(x.shapes[s], 0) for s in x.ids)
    assert source.read_bytes() == before
    assert contract['unused_stones_allowed'] and not contract['prefix_constraints']


@pytest.mark.parametrize('records', [
    [{'stone_id': 'missing', 'pose': [0, 0, 0]}],
    [{'stone_id': 'a', 'pose': [0, 0, float('nan')]}],
    [{'stone_id': 'a', 'pose': [0, 0, 0], 'mirror': True}],
    [{'stone_id': 'a', 'pose': [0, 0, 0]}]*2,
])
def test_bad_native_record_cannot_partially_mutate_layout(records):
    x = layout()
    before = x.snapshot()
    with pytest.raises(ValueError):
        q.apply_records(x, records)
    np.testing.assert_array_equal(x.poses, before[0])
    np.testing.assert_array_equal(x.active, before[1])


def test_neighborhood_depends_on_geometry_not_inventory_order():
    x = layout()
    q.apply_records(x, [{'stone_id': 'a', 'pose': [1.5, 1.5, 0]},
                        {'stone_id': 'b', 'pose': [.3, .3, 0]}])
    assert q.neighborhood(x, [0, 0], 1) == [1]


def test_native_offers_entire_unused_inventory_and_encloses_source():
    x = layout()
    payload = q.native_input(x, 3)
    assert {i['stone_id'] for i in payload['items']} == set(x.ids)
    for item in payload['items']:
        polygon = q.shapely.Polygon(item['outline'])
        assert polygon.covers(x.shapes[item['stone_id']])
        assert polygon.exterior.is_ccw


@pytest.mark.parametrize('objective', ['energy', 'worst', 'balanced'])
def test_real_patch_improves_gap_objective_with_arbitrary_fixed_set(objective):
    x = layout({'a': box(-.35, -.35, .35, .35), 'b': box(-.35, -.35, .35, .35)})
    q.apply_records(x, [{'stone_id': 'a', 'pose': [.45, .45, 0]},
                        {'stone_id': 'b', 'pose': [1.5, .45, 0]}])
    fixed = x.poses[1].copy()
    before = q.measure(x)
    result = q.continuous_patch(x, [0], q.sample_points(x.patio), seconds=3, maxiter=100, radius_ft=.5, objective=objective)
    assert result['accepted']
    assert q.rank(q.measure(x), objective) < q.rank(before, objective)
    assert not validate(x, gap=q.GAP, edge=q.GAP)
    np.testing.assert_array_equal(x.poses[1], fixed)


def test_failed_replacements_restore_layout_and_keep_reserve():
    x = layout({'a': box(-.35, -.35, .35, .35), 'b': box(-2, -2, 2, 2)})
    q.apply_records(x, [{'stone_id': 'a', 'pose': [.5, .5, 0]}])
    before = x.snapshot()
    result = q.try_replacements(x, [.9, .9], q.sample_points(x.patio), seconds=2)
    assert not result['accepted'] and result['trials'] > 0
    np.testing.assert_array_equal(x.poses, before[0])
    np.testing.assert_array_equal(x.active, before[1])


def test_saved_final_layout_failure_cannot_be_promoted(tmp_path):
    x = layout()
    q.apply_records(x, [{'stone_id': 'a', 'pose': [.5, .5, 0]}])
    path = tmp_path/'layout.json'
    save_json(x, path)
    result = q.verify_saved(path, tmp_path/'verification.json')
    assert not result['passed']
    assert not result['clearance_errors']
    assert result['gap']['lower_diameter_in'] > 2
    assert json.loads((tmp_path/'verification.json').read_text()) == result


def test_saved_geometry_cannot_replace_inventory_to_claim_success(tmp_path):
    x = layout()
    reference = tmp_path/'reference.json'
    save_json(x, reference)
    changed = json.loads(reference.read_text())
    changed['patio'] = [[0, 0], [3, 0], [3, 3], [0, 3]]
    saved = tmp_path/'tampered.json'
    saved.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='changed'):
        q.verify_saved(saved, tmp_path/'verification.json', reference)
    assert not (tmp_path/'verification.json').exists()


def test_certifies_final_layout_without_any_prefix_contract(tmp_path):
    x = layout({'a': box(-.4, -.4, .4, .4), 'reserve': box(-.2, -.2, .2, .2)})
    x.patio = box(0, 0, 1, 1)
    q.apply_records(x, [{'stone_id': 'a', 'pose': [.5, .5, 0]}])
    path = tmp_path/'layout.json'
    save_json(x, path)
    result = q.verify_saved(path, tmp_path/'verification.json', path)
    assert result['passed'] and result['reference_matched']
    assert result['gap']['upper_diameter_in'] < 2
    assert not x.active[1]


def test_replacement_exception_restores_every_pose_and_flag(monkeypatch):
    x = layout()
    q.apply_records(x, [{'stone_id': 'a', 'pose': [.5, .5, 0]}])
    original = x.snapshot()
    def fail(*args, **kwargs):
        raise RuntimeError('validation interrupted')
    monkeypatch.setattr(q, 'validate', fail)
    with pytest.raises(RuntimeError):
        q.try_replacements(x, [.5, .5], q.sample_points(x.patio))
    np.testing.assert_array_equal(x.active, original[1])
    np.testing.assert_array_equal(x.poses, original[0])


def test_destroy_rebuild_rejection_and_exception_restore_fixed_and_removed(monkeypatch):
    from flagstone import search
    x = layout()
    q.apply_records(x, [{'stone_id': 'a', 'pose': [.5, .5, 0]}])
    original = x.snapshot()
    def fail(layout, **kwargs):
        layout.poses[:] = 42
        raise RuntimeError('insertion interrupted')
    monkeypatch.setattr(search, 'greedy_insert', fail)
    with pytest.raises(RuntimeError):
        q.rebuild_patch(x, [.5, .5], q.sample_points(x.patio), count=1)
    np.testing.assert_array_equal(x.active, original[1])
    np.testing.assert_array_equal(x.poses, original[0])


@pytest.mark.parametrize('strategy', ['random', 'contact'])
def test_reserve_insertion_improves_gap_and_preserves_existing_pose(strategy):
    x = layout({'a': box(-.35, -.35, .35, .35), 'b': box(-.3, -.3, .3, .3)})
    q.apply_records(x, [{'stone_id': 'a', 'pose': [.5, .5, 0]}])
    fixed = x.poses[0].copy()
    result = q.rebuild_patch(x, [1.4, 1.4], q.sample_points(x.patio), count=0, seconds=2, proposals=80, strategy=strategy)
    assert result['accepted'] and x.active[1]
    assert q.rank(result['after']) < q.rank(result['before'])
    np.testing.assert_array_equal(x.poses[0], fixed)
    assert not validate(x, gap=q.GAP, edge=q.GAP)
