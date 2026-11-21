from flagstone.diagnostics import empty_space
from flagstone.phase1 import synthetic


def test_empty_space_diagnostics_and_empty_inventory():
    x = synthetic()
    result = empty_space(x, resolution=20)
    assert result['empty_samples'] > 0
    assert result['max_distance_to_stone'] > 0
    x.active[:] = False
    result = empty_space(x, resolution=20)
    assert result['empty_samples'] == 400
    assert result['max_distance_to_stone'] is None


def test_physical_metrics():
    import numpy as np
    from flagstone.diagnostics import layout_metrics
    x = synthetic()
    m = layout_metrics(x)
    assert m['placed_count'] == 16 and m['unused_count'] == 0
    assert np.isclose(m['coverage'], 16 * .19 ** 2)
    assert np.isclose(m['minimum_pair_distance'], .055)
    assert np.isclose(m['minimum_edge_distance'], .035)
    x.poses[0, 0] = -1
    m = layout_metrics(x)
    assert m['outside_stone_ids'] == ['S00'] and m['minimum_edge_distance'] is None


def test_edge_and_interior_sampling_and_broad_area():
    x = synthetic()
    result = empty_space(x, resolution=40, edge_strip_width=.1, broad_distance=.04)
    interior, edge = result['interior'], result['edge_strip']
    assert interior['samples'] > 0 and edge['samples'] > 0
    assert interior['samples'] + edge['samples'] == result['empty_samples']
    assert 0 <= interior['broad_area'] <= interior['estimated_area']
    assert 0 <= edge['broad_area'] <= edge['estimated_area']


def test_uncovered_and_unused_area_with_overlap():
    import numpy as np
    from flagstone.diagnostics import layout_metrics
    x = synthetic()
    x.active[2:] = False
    x.poses[1] = x.poses[0]
    m = layout_metrics(x)
    assert np.isclose(m['uncovered_area'], 1 - .19**2)
    assert np.isclose(m['unused_area'], 14 * .19**2)
    assert np.isclose(m['smallest_unused_area'], .19**2)


def test_neighbor_boundary_sampling_for_known_gap():
    import numpy as np
    import pytest
    from flagstone.diagnostics import sampled_neighbor_distances
    x = synthetic()
    x.active[2:] = False
    result = sampled_neighbor_distances(x, spacing=.01, max_distance=.06)
    assert result['matched_samples'] > 0
    assert result['unmatched_samples'] > 0
    assert np.isclose(result['minimum_distance'], .055)
    assert result['maximum_distance'] <= .06
    assert result['matched_samples'] + result['unmatched_samples'] == result['total_samples']
    empty = sampled_neighbor_distances(x, spacing=.01, max_distance=.05)
    assert empty['matched_samples'] == 0 and empty['distance_quantiles'] is None
    with pytest.raises(ValueError):
        sampled_neighbor_distances(x, spacing=0, max_distance=.1)


def test_neighbor_sampling_handles_empty_inventory():
    from flagstone.diagnostics import sampled_neighbor_distances
    x = synthetic()
    x.active[:] = False
    result = sampled_neighbor_distances(x, spacing=.01, max_distance=.1)
    assert result['total_samples'] == 0 and result['minimum_distance'] is None
def test_sampled_voids_measures_empty_circles_and_boundary():
    import pytest
    from flagstone.phase1 import synthetic
    from flagstone.diagnostics import sampled_voids
    x=synthetic(); x.active[:]=False
    before=sampled_voids(x,spacing=.1,diameter=.4)
    assert before['maximum_diameter']==pytest.approx(.9)
    assert before['broad_samples']>0
    x.active[0]=True; x.poses[0]=[.5,.5,0]
    after=sampled_voids(x,spacing=.1,diameter=.4)
    assert after['maximum_diameter']<before['maximum_diameter']
    assert after['broad_core_area']<before['broad_core_area']
    with pytest.raises(ValueError): sampled_voids(x,spacing=0,diameter=.4)
