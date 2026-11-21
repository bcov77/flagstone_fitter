import numpy as np
from flagstone.phase1 import Layout, local_shape, polygon
from flagstone.scoring import score_layout, compare_scores


def layout_at(x, y):
    patio=polygon([(0,0),(1,0),(1,1),(0,1)])
    shape=local_shape([(-.1,-.1),(.1,-.1),(.1,.1),(-.1,.1)])
    return Layout(patio, {'a':shape}, [[x,y,0]], [True], ('a',), np.random.default_rng(0))


def test_validity_and_coverage_dominate_sampled_quality():
    center=score_layout(layout_at(.5,.5), gap=0, edge=0, resolution=20, broad_distance=.3)
    corner=score_layout(layout_at(.15,.15), gap=0, edge=0, resolution=20, broad_distance=.3)
    assert center['coverage'] == corner['coverage']
    assert compare_scores(center, corner) > 0
    invalid=score_layout(layout_at(-.2,.5), gap=0, edge=0, resolution=20, broad_distance=.3)
    assert not invalid['valid'] and invalid['coverage'] is None
    assert compare_scores(center,invalid) > 0 and compare_scores(invalid,center) < 0


def test_equal_coverage_rebuild_accepts_better_empty_space(monkeypatch):
    import flagstone.search as search
    x = layout_at(.15,.15)
    initial = score_layout(x, gap=0, edge=0, resolution=20, broad_distance=.3)
    def move_to_center(layout, **kwargs):
        layout.active[0] = True
        layout.poses[0] = [.5,.5,0]
        return search.SearchResult(1, 1, search.coverage(layout), 0, [])
    monkeypatch.setattr(search, 'greedy_insert', move_to_center)
    result = search.rebuild_search(x, rounds=1, patch_size=1, gap=0, edge=0,
                                   secondary_resolution=20, secondary_broad_distance=.3)
    assert result.secondary_improvements == 1 and result.improvements == 0
    assert result.coverage == initial['coverage'] and result.best_history[0] == result.best_history[-1]
    assert np.allclose(x.poses[0], [.5,.5,0])
    assert compare_scores(score_layout(x, gap=0, edge=0, resolution=20, broad_distance=.3), initial) > 0


def test_comparing_incompatible_secondary_settings_is_rejected():
    import pytest
    x = layout_at(.5,.5)
    a=score_layout(x, gap=0, edge=0, resolution=20, broad_distance=.3)
    b=score_layout(x, gap=0, edge=0, resolution=30, broad_distance=.3)
    with pytest.raises(ValueError, match='settings'):
        compare_scores(a,b)


def test_secondary_scores_scale_with_units():
    from shapely.affinity import scale
    a=layout_at(.5,.5)
    factor=100
    b=Layout(scale(a.patio, xfact=factor, yfact=factor, origin=(0,0)),
             {'a':scale(a.shapes['a'], xfact=factor, yfact=factor, origin=(0,0))},
             [[50,50,0]], [True], ('a',), np.random.default_rng(0), units='cm')
    x=score_layout(a,gap=0,edge=0,resolution=20,broad_distance=.3,edge_strip_width=.1)
    y=score_layout(b,gap=0,edge=0,resolution=20,broad_distance=30,edge_strip_width=10)
    assert np.isclose(x['coverage'],y['coverage'])
    assert np.isclose(x['secondary']['interior_broad_area_fraction'],y['secondary']['interior_broad_area_fraction'])
    assert np.isclose(x['secondary']['interior_max_distance']*factor,y['secondary']['interior_max_distance'])
