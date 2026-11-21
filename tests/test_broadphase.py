import numpy as np
import pytest
from flagstone.broadphase import PaddedGeometry, candidate_pairs
from flagstone.generate import inventory
from flagstone.phase1 import transformed


def test_batched_filter_never_misses_exact_contacts():
    x = inventory(seed=14, counts=(30,), areas=(.012,), vertex_profile='detailed')
    g = PaddedGeometry.from_layout(x)
    assert not g.vertices.flags.writeable and not g.vertex_mask.flags.writeable
    rng = np.random.default_rng(9)
    poses = np.zeros((5, len(x.ids), 3))
    poses[:, :, :2] = rng.uniform(-.2, 1.2, (5, len(x.ids), 2))
    poses[:, :, 2] = rng.uniform(-np.pi, np.pi, (5, len(x.ids)))
    poses[0, 1] = poses[0, 0]  # overlapping/containing silhouettes
    active = rng.random((5, len(x.ids))) > .2
    active[0, :2] = True
    margins = rng.uniform(0, .03, (5, len(x.ids)))
    candidates = candidate_pairs(g, poses, active, gap=.02, margins=margins)
    for b in range(5):
        placed = [transformed(x.shapes[sid], poses[b, i]) for i, sid in enumerate(x.ids)]
        found = set(map(tuple, candidates[b]))
        for i in range(len(x.ids)):
            for j in range(i+1, len(x.ids)):
                if active[b, i] and active[b, j] and placed[i].distance(placed[j]) <= .02 + margins[b, i] + margins[b, j]:
                    assert (i, j) in found
        assert all(active[b, i] and active[b, j] and i < j for i, j in found)


def test_near_clearance_and_input_validation():
    x = inventory(seed=1, counts=(2,), areas=(.01,))
    g = PaddedGeometry.from_layout(x)
    poses = np.zeros((1, 2, 3)); active = np.ones((1, 2), bool)
    poses[0, 1, 0] = .3
    assert candidate_pairs(g, poses, active, gap=0)[0].shape == (0, 2)
    poses[0, 1] = poses[0, 0]
    assert tuple(candidate_pairs(g, poses, active)[0][0]) == (0, 1)
    with pytest.raises(ValueError):
        candidate_pairs(g, poses, active.astype(int))
    with pytest.raises(ValueError):
        candidate_pairs(g, poses, active, margins=[0, -1])


def test_near_clearance_pair_at_large_coordinates():
    from flagstone.phase1 import Layout, local_shape, polygon
    square = local_shape([(-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)])
    x = Layout(polygon([(0,0),(2e9,0),(2e9,2e9),(0,2e9)]), {'a':square, 'b':square}, [[1e9,1e9,0],[1e9+1.1,1e9,0]], [True, True], ('a','b'), np.random.default_rng(0))
    geometry = PaddedGeometry.from_layout(x)
    found = candidate_pairs(geometry, x.poses[None], x.active[None], gap=.1)
    assert (0, 1) in set(map(tuple, found[0]))
