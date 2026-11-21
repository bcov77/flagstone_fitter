"""Cross-scale checks use lengths and numeric tolerance in the same units."""
import numpy as np
import pytest
from flagstone.batch import validate_batch
from flagstone.insertion import _InsertionOracle
from flagstone.phase1 import Layout, local_shape, polygon, validate


@pytest.mark.parametrize('scale', [1e-6, 1., 1e6])
@pytest.mark.parametrize('case,expected_valid', [
    ('clear', True), ('within_tolerance', True), ('gap_violation', False),
    ('overlap', False), ('edge_violation', False), ('outside', False),
])
def test_clearance_consistent_across_scales_and_checkers(scale, case, expected_valid):
    square = local_shape(np.array([[-.05,-.05], [.05,-.05], [.05,.05], [-.05,.05]]) * scale)
    patio = polygon(np.array([[0,0], [1,0], [1,1], [0,1]]) * scale)
    poses = np.array([[.3,.5,0], [.421,.5,0]], dtype=float)
    if case == 'within_tolerance': poses[1,0] = .4199995
    if case == 'gap_violation': poses[1,0] = .419998
    if case == 'overlap': poses[1,0] = .399
    if case == 'edge_violation': poses[1,0] = .079
    if case == 'outside': poses[1,0] = .049
    poses[:,:2] *= scale
    layout = Layout(patio, {'A':square, 'B':square}, poses, np.array([True,True]),
                    ('A','B'), np.random.default_rng(3))
    options = dict(gap=.02 * scale, edge=.03 * scale, tolerance=1e-6 * scale)
    scalar = validate(layout, **options)
    batched = validate_batch(layout, poses[None], layout.active[None], **options)[0]
    assert bool(scalar) is not expected_valid
    assert batched == scalar
    layout.active[1] = False
    oracle = _InsertionOracle(layout, **options)
    assert oracle.can_insert(1, poses[1]) is expected_valid


@pytest.mark.parametrize('scale', [1e-6, 1., 1e6])
@pytest.mark.parametrize('case,expected_valid', [
    ('clear', True), ('pair_within_tolerance', True), ('pair_margin_violation', False),
    ('edge_margin_violation', False),
])
def test_uncertainty_margins_scale_with_geometry(scale, case, expected_valid):
    square = local_shape(np.array([[-.05,-.05], [.05,-.05], [.05,.05], [-.05,.05]]) * scale)
    patio = polygon(np.array([[0,0], [1,0], [1,1], [0,1]]) * scale)
    poses = np.array([[.3,.5,0], [.424,.5,0]])
    if case == 'pair_within_tolerance': poses[1,0] = .4229995
    if case == 'pair_margin_violation': poses[1,0] = .422998
    if case == 'edge_margin_violation': poses[1,0] = .085
    poses[:,:2] *= scale
    layout = Layout(patio, {'A':square, 'B':square}, poses, np.array([True,True]),
                    ('A','B'), np.random.default_rng(3))
    options = dict(gap=.02 * scale, edge=.03 * scale, tolerance=1e-6 * scale,
                   stone_margins={'A': .001 * scale, 'B': .002 * scale}, patio_margin=.004 * scale)
    scalar = validate(layout, **options)
    assert bool(scalar) is not expected_valid
    assert validate_batch(layout, poses[None], layout.active[None], **options)[0] == scalar
    layout.active[1] = False
    assert _InsertionOracle(layout, **options).can_insert(1, poses[1]) is expected_valid


def test_relaxation_step_reduction_is_scale_consistent():
    from flagstone.relaxation import relax
    results = []
    for scale in (1e-6, 1., 1e6):
        patio = polygon(np.array([[0,0], [1,0], [1,1], [0,1]]) * scale)
        square = local_shape(np.array([[-.095,-.095],[.095,-.095],[.095,.095],[-.095,.095]]) * scale)
        poses = np.array([[.2,.5,0],[.38,.5,0]]) * [scale,scale,1]
        layout = Layout(patio, {'A':square, 'B':square}, poses, np.array([True,True]),
                        ('A','B'), np.random.default_rng(1))
        result = relax(layout, gap=.01*scale, edge=.01*scale, tolerance=1e-9*scale,
                       step=.8*scale, max_steps=30, mobility=[0,1])
        assert result.reason == 'valid' and not result.violations
        assert not validate(layout, gap=.01*scale, edge=.01*scale, tolerance=1e-9*scale)
        results.append((result.steps, layout.poses[1,0]/scale))
    assert all(steps == results[0][0] and abs(x-results[0][1]) < 1e-12 for steps, x in results)
