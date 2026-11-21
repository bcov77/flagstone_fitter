"""Protect exact gap geometry when changing its nearest-distance acceleration."""
from unittest.mock import patch

import numpy as np
import shapely
from shapely.geometry import Polygon, box

from flagstone import quarter_gaps as gaps


def polygon_tree_distances(stones, points):
    distances = np.full(len(points), np.inf)
    if stones and len(points):
        pairs, ds = shapely.STRtree(stones).query_nearest(points, return_distance=True, all_matches=False)
        distances[pairs[0]] = ds
    return distances


def test_exact_distances_for_concavities_holes_overlaps_and_boundaries():
    stones = [Polygon([(0,0),(3,0),(3,1),(1,1),(1,3),(0,3)]),
              Polygon([(4,0),(8,0),(8,4),(4,4)], holes=[[(5,1),(7,1),(7,3),(5,3)]]),
              box(2.5, .5, 4.5, 2)]
    rng = np.random.default_rng(72)
    xy = np.vstack([rng.uniform([-1,-1], [9,5], (2000,2)),
                    [[0,0],[3,.5],[2,2],[6,2],[5,2],[5,1],[4,2],[4.25,1]]])
    points = shapely.points(xy)
    np.testing.assert_array_equal(gaps._stone_distances(stones, points),
                                  polygon_tree_distances(stones, points))
    assert gaps._stone_distances(stones, shapely.points([[6,2]]))[0] == 1


def test_empty_points_and_empty_inventory():
    assert gaps._stone_distances([box(0,0,1,1)], shapely.points(np.empty((0,2)))).shape == (0,)
    assert np.isinf(gaps._stone_distances([], shapely.points([[0,0],[1,1]]))).all()


def test_complete_dual_scale_audit_matches_polygon_tree_reference():
    patio = box(0,0,4,4)
    stones = [box(.1,.1,1.8,1.8), box(2.1,.1,3.9,1.8),
              Polygon([(.1,2.1),(1.8,2.1),(1.8,3),(.9,3),(.9,3.9),(.1,3.9)]),
              box(2.1,2.1,3.9,3.9)]
    result = gaps.audit(patio, stones)
    with patch.object(gaps, '_stone_distances', polygon_tree_distances):
        reference = gaps.audit(patio, stones)
    assert result == reference
