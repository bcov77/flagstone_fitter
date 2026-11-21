import math
import numpy as np
import pytest
from flagstone.phase1 import synthetic, transformed, validate
from flagstone.uncertainty import StoneUncertainty, stone_margins


def test_stone_error_bounds_and_pair_edge_margins():
    x = synthetic()
    x.active[2:] = False
    bounds = StoneUncertainty(shape=.005, translation=.01, rotation=.1)
    margins = stone_margins(x, {x.ids[0]: bounds, x.ids[1]: bounds})
    radius = np.linalg.norm(np.asarray(x.shapes[x.ids[0]].exterior.coords), axis=1).max()
    assert np.isclose(margins[x.ids[0]], .015 + 2*radius*math.sin(.05))
    assert not validate(x, gap=.01, edge=.01)
    violations = validate(x, gap=.01, edge=.01, stone_margins=margins, patio_margin=.005)
    pair = [v for v in violations if v['kind'] == 'pair_clearance'][0]
    assert np.isclose(pair['required'], .01 + sum(margins.values()))
    edge = [v for v in violations if v['kind'] == 'boundary_clearance'][0]
    assert np.isclose(edge['required'], .015 + margins[edge['stone_ids'][0]])


def test_variable_margin_filter_matches_exhaustive():
    from flagstone.generate import inventory
    x = inventory(seed=9, counts=(20,), areas=(.004,))
    x.active[:] = True
    x.poses[:, :2] = x.rng.uniform(.1, .9, (20, 2))
    margins = {sid: float(x.rng.uniform(0, .05)) for sid in x.ids}
    placed = [transformed(x.shapes[sid], x.poses[i]) for i, sid in enumerate(x.ids)]
    expected = {(x.ids[i], x.ids[j]) for i, a in enumerate(placed) for j, b in enumerate(placed)
                if j > i and (a.intersection(b).area > 0 or a.distance(b) < .01 + margins[x.ids[i]] + margins[x.ids[j]] - 1e-9)}
    found = {tuple(v['stone_ids']) for v in validate(x, gap=.01, edge=0, stone_margins=margins) if v['kind'] == 'pair_clearance'}
    assert found == expected


def test_uncertainty_inputs_are_explicit_and_bounded():
    x = synthetic()
    for bad in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            StoneUncertainty(rotation=bad)
    with pytest.raises(ValueError):
        stone_margins(x, {'unknown': StoneUncertainty()})
    with pytest.raises(ValueError):
        validate(x, stone_margins={'unknown': .1})
    # Rotation bounds beyond pi cannot move a point more than its diameter.
    assert np.isclose(StoneUncertainty(rotation=10).margin(x.shapes[x.ids[0]]),
                      2*np.linalg.norm(np.asarray(x.shapes[x.ids[0]].exterior.coords), axis=1).max())


def test_search_and_checkpoint_respect_margins(tmp_path):
    from flagstone.generate import inventory
    from flagstone.runner import start_insertion, continue_insertion
    from flagstone.search import greedy_insert
    x = inventory(seed=23, counts=(10,), areas=(.02,))
    margins = {sid: .025 for sid in x.ids}
    greedy_insert(x, gap=.01, edge=.01, proposals=30, stone_margins=margins, patio_margin=.02)
    assert x.active.any()
    assert not validate(x, gap=.01, edge=.01, stone_margins=margins, patio_margin=.02)
    y = inventory(seed=23, counts=(10,), areas=(.02,))
    path = tmp_path / 'margin-checkpoint.json'
    start_insertion(y, path, proposals=30, stone_margins=margins, patio_margin=.02)
    continue_insertion(path, max_stones=3)
    resumed, state = continue_insertion(path)
    assert state['config']['stone_margins'] == margins
    assert state['config']['patio_margin'] == .02
    assert np.array_equal(x.poses, resumed.poses)
    assert np.array_equal(x.active, resumed.active)
    assert not validate(resumed, gap=.01, edge=.01, stone_margins=margins, patio_margin=.02)


def test_relaxation_separates_margin_only_violation():
    from flagstone.relaxation import energy, relax
    x = synthetic()
    x.active[2:] = False
    margins = {x.ids[0]: .025, x.ids[1]: .025}
    assert not validate(x)
    assert validate(x, stone_margins=margins)
    initial = energy(x, stone_margins=margins)
    result = relax(x, stone_margins=margins, mobility=[0, 1] + [0] * 14, max_steps=50)
    assert result.reason == 'valid' and result.final_energy < initial
    assert not validate(x, stone_margins=margins)


def test_relaxation_corrects_extra_edge_margin():
    from flagstone.relaxation import relax
    x = synthetic()
    x.active[1:] = False
    margins = {x.ids[0]: .04}
    assert not validate(x)
    assert validate(x, stone_margins=margins, patio_margin=.01)
    result = relax(x, stone_margins=margins, patio_margin=.01, max_steps=50)
    assert result.reason == 'valid'
    assert not validate(x, stone_margins=margins, patio_margin=.01)


def test_margin_energy_matches_unfiltered_pair_geometry():
    from flagstone.generate import inventory
    from flagstone.relaxation import energy
    x = inventory(seed=2, counts=(6,), areas=(.01,))
    x.active[:] = True
    x.poses[:, :2] = x.rng.uniform(.1, .9, (6, 2))
    margins = {sid: float(x.rng.uniform(0, .04)) for sid in x.ids}
    g = [transformed(x.shapes[sid], x.poses[i]) for i, sid in enumerate(x.ids)]
    expected = sum(shape.difference(x.patio.buffer(-(.01+.02+margins[sid]))).area for sid, shape in zip(x.ids, g))
    for i, a in enumerate(g):
        for j in range(i+1, len(g)):
            b = g[j]
            expected += a.buffer(.005+margins[x.ids[i]]).intersection(b.buffer(.005+margins[x.ids[j]])).area
            expected += a.intersection(b).area
    assert np.isclose(energy(x, stone_margins=margins, patio_margin=.02), expected/x.patio.area)


def test_explicit_uncertainty_model_requires_all_components_and_matching_units():
    from flagstone.uncertainty import parse_uncertainty_model, verify_uncertainty_model
    x = synthetic()
    model = {'uncertainty_version': 1, 'units': 'normalized', 'patio': .01,
             'stones': {sid: {'shape': .002, 'translation': .003, 'rotation': .01} for sid in x.ids}}
    margins, patio = parse_uncertainty_model(x, model)
    assert set(margins) == set(x.ids) and patio == .01
    verify_uncertainty_model(x, model, margins, patio)
    with pytest.raises(ValueError, match='match'):
        verify_uncertainty_model(x, model, {**margins, x.ids[0]: 0}, patio)
    incomplete = {**model, 'stones': dict(model['stones'])}
    del incomplete['stones'][x.ids[0]]
    with pytest.raises(ValueError, match='every stone'):
        parse_uncertainty_model(x, incomplete)
    with pytest.raises(ValueError, match='units'):
        parse_uncertainty_model(x, {**model, 'units': 'ft'})
    bad = {**model, 'stones': dict(model['stones'])}
    bad['stones'][x.ids[0]] = {'shape': 0, 'translation': 0, 'rotation': True}
    with pytest.raises(ValueError, match='nonnegative'):
        parse_uncertainty_model(x, bad)


def test_small_explicit_bounds_cannot_be_verified_as_zero(tmp_path):
    import pytest
    from flagstone.iterative import start_iterative
    from flagstone.placement import placement_manifest
    from flagstone.uncertainty import parse_uncertainty_model, verify_uncertainty_model
    x = synthetic()
    x.active[:] = False
    model = {'uncertainty_version': 1, 'units': x.units, 'patio': 2e-16,
             'stones': {sid: {'shape': 5e-16, 'translation': 0, 'rotation': 0} for sid in x.ids}}
    margins, patio = parse_uncertainty_model(x, model)
    verify_uncertainty_model(x, model, margins, patio)
    with pytest.raises(ValueError, match='patio margin'):
        verify_uncertainty_model(x, model, margins, 0)
    missing = {sid: 0. for sid in x.ids}
    with pytest.raises(ValueError, match='stone margins'):
        verify_uncertainty_model(x, model, missing, patio)
    with pytest.raises(ValueError, match='stone margins'):
        start_iterative(x, tmp_path / 'bad.json', stone_margins=missing, patio_margin=patio, uncertainty_model=model)
    with pytest.raises(ValueError, match='patio margin'):
        placement_manifest(x, gap=.01, edge=.01, stone_margins=margins, patio_margin=0, uncertainty_model=model)
    assert not (tmp_path / 'bad.json').exists()
