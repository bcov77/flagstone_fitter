import numpy as np
from flagstone.generate import inventory
from flagstone.search import greedy_insert
from flagstone.phase1 import validate


def test_search_is_valid_and_reproducible():
    a = inventory(seed=4, counts=(4,), areas=(.01,))
    b = inventory(seed=4, counts=(4,), areas=(.01,))
    result = greedy_insert(a, proposals=20)
    other = greedy_insert(b, proposals=20)
    assert result.accepted > 0 and result.coverage > 0
    assert not validate(a)
    assert result.accepted == other.accepted
    assert np.array_equal(a.poses, b.poses)
    assert np.array_equal(a.active, b.active)


def test_failed_search_restores_geometry():
    x = inventory(counts=(1,), areas=(20,))
    poses = x.poses.copy(); active = x.active.copy()
    result = greedy_insert(x, proposals=3)
    assert result.accepted == 0
    assert np.array_equal(x.poses, poses) and np.array_equal(x.active, active)


def test_mixed_search_is_valid_and_reproducible():
    a = inventory(seed=8, counts=(4,), areas=(.01,))
    b = inventory(seed=8, counts=(4,), areas=(.01,))
    greedy_insert(a, proposals=10, strategy='mixed', grid_resolution=6)
    greedy_insert(b, proposals=10, strategy='mixed', grid_resolution=6)
    assert not validate(a)
    assert np.array_equal(a.poses, b.poses)


def test_rebuild_replaces_smaller_stone_and_tracks_best():
    from flagstone.search import rebuild_search, coverage
    a = inventory(seed=3, counts=(1, 1), areas=(.08, .01), mixed=False)
    a.active[1] = True
    a.poses[1] = [.5, .5, 0]
    initial = coverage(a)
    result = rebuild_search(a, rounds=3, patch_size=1, proposals=20, strategy='random')
    assert result.coverage > initial
    assert result.improvements >= 1
    assert all(b >= a for a, b in zip(result.best_history, result.best_history[1:]))
    assert not validate(a)


def test_rebuild_rolls_back_non_improving_patch():
    from flagstone.search import rebuild_search
    a = inventory(seed=5, counts=(1,), areas=(.01,))
    a.active[0] = True
    a.poses[0] = [.5, .5, 0]
    poses, active = a.poses.copy(), a.active.copy()
    result = rebuild_search(a, rounds=2, proposals=0)
    assert result.improvements == 0
    assert np.array_equal(a.poses, poses) and np.array_equal(a.active, active)


def test_zero_time_budget_does_not_change_layout():
    a = inventory(counts=(2,), areas=(.01,))
    poses = a.poses.copy()
    result = greedy_insert(a, proposals=100, time_limit=0)
    assert result.attempts == 0 and not a.active.any()
    assert np.array_equal(a.poses, poses)


def test_proposal_strategies_are_valid_and_reproducible():
    from flagstone.search import STRATEGIES
    for strategy in STRATEGIES:
        a = inventory(seed=12, counts=(3,), areas=(.01,))
        b = inventory(seed=12, counts=(3,), areas=(.01,))
        greedy_insert(a, proposals=6, strategy=strategy, grid_resolution=6)
        greedy_insert(b, proposals=6, strategy=strategy, grid_resolution=6)
        assert not validate(a)
        assert np.array_equal(a.active, b.active) and np.array_equal(a.poses, b.poses)


def test_interrupted_proposal_restores_layout(monkeypatch):
    import pytest
    import flagstone.search as search
    a = inventory(seed=4, counts=(1,), areas=(.01,))
    poses, flags = a.poses.copy(), a.active.copy()
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(search, 'relax', interrupt)
    with pytest.raises(KeyboardInterrupt):
        greedy_insert(a, proposals=1, relaxation_steps=1, use_prefilter=False)
    assert np.array_equal(a.poses, poses) and np.array_equal(a.active, flags)


def test_frozen_stones_survive_insertion_and_rebuilding():
    from flagstone.search import rebuild_search
    x = inventory(seed=7, counts=(5,), areas=(.02,))
    x.active[0] = True
    x.poses[0] = [.5, .5, .3]
    fixed = x.poses[0].copy()
    greedy_insert(x, proposals=3, relaxation_steps=2, frozen_ids=[x.ids[0]])
    rebuild_search(x, rounds=3, patch_size=3, proposals=3, strategy='random', frozen_ids=[x.ids[0]])
    assert x.active[0] and np.array_equal(x.poses[0], fixed)
    assert not validate(x)
    all_frozen = [sid for i, sid in enumerate(x.ids) if x.active[i]]
    assert rebuild_search(x, rounds=2, frozen_ids=all_frozen).rounds == 0


def test_frozen_ids_must_be_existing_placed_stones():
    import pytest
    x = inventory(counts=(1,), areas=(.01,))
    with pytest.raises(ValueError, match='active'):
        greedy_insert(x, frozen_ids=x.ids)
    with pytest.raises(ValueError, match='unknown'):
        greedy_insert(x, frozen_ids=['missing'])


def test_batched_insertion_is_valid_and_reproducible():
    from flagstone.search import batched_random_insert
    a = inventory(seed=10, counts=(8,), areas=(.02,), vertex_profile='detailed')
    b = inventory(seed=10, counts=(8,), areas=(.02,), vertex_profile='detailed')
    margins = {sid: .01 for sid in a.ids}
    ra = batched_random_insert(a, proposals=16, batch_size=4, stone_margins=margins, patio_margin=.01)
    rb = batched_random_insert(b, proposals=16, batch_size=4, stone_margins=margins, patio_margin=.01)
    assert ra.accepted > 0 and ra.accepted == rb.accepted
    assert np.array_equal(a.poses, b.poses) and np.array_equal(a.active, b.active)
    assert not validate(a, stone_margins=margins, patio_margin=.01)


def test_batch_size_one_matches_scalar_random():
    from flagstone.search import batched_random_insert
    a = inventory(seed=4, counts=(5,), areas=(.01,))
    b = inventory(seed=4, counts=(5,), areas=(.01,))
    batched_random_insert(a, proposals=10, batch_size=1)
    greedy_insert(b, proposals=10, strategy='random')
    assert np.array_equal(a.poses, b.poses) and np.array_equal(a.active, b.active)
    assert a.rng.random() == b.rng.random()


def test_batch_failure_does_not_change_geometry():
    from flagstone.search import batched_random_insert
    a = inventory(counts=(1,), areas=(20,))
    poses, active = a.poses.copy(), a.active.copy()
    result = batched_random_insert(a, proposals=5, batch_size=3)
    assert result.attempts == 5 and result.accepted == 0
    assert np.array_equal(a.poses, poses) and np.array_equal(a.active, active)


def test_batched_insertion_scalar_recheck_blocks_bad_batch_results(monkeypatch):
    import flagstone.batch as batch
    from flagstone.search import batched_random_insert
    x = inventory(counts=(1,), areas=(20,))
    monkeypatch.setattr(batch, 'validate_batch', lambda template, poses, active, **kwargs: [[] for _ in poses])
    result = batched_random_insert(x, proposals=4, batch_size=2)
    assert result.accepted == 0 and not x.active.any()
    assert not validate(x)


def test_contact_proposals_respect_anchor_clearance_and_masks():
    from flagstone.search import _contact_pose
    from flagstone.phase1 import transformed
    x = inventory(seed=14, counts=(3,), areas=(.02,))
    anchor = transformed(x.shapes[x.ids[0]], [.5, .5, .3])
    for _ in range(30):
        angle = x.rng.uniform(-np.pi, np.pi)
        px, py = _contact_pose(x.shapes[x.ids[1]], anchor, angle, .03, x.rng)
        candidate = transformed(x.shapes[x.ids[1]], [px, py, angle])
        assert candidate.distance(anchor) >= .03 - 1e-10
    x.active[0] = True; x.poses[0] = [.5,.5,.3]
    fixed = x.poses[0].copy()
    result = greedy_insert(x, strategy='contact', proposals=20, frozen_ids=[x.ids[0]])
    assert result.accepted and np.array_equal(x.poses[0], fixed) and not validate(x)


def test_edge_proposals_align_rectangles_and_remain_reproducible():
    from flagstone.phase1 import local_shape, transformed
    from flagstone.search import _edge_pose
    shape = local_shape([(-.1,-.05),(.1,-.05),(.1,.05),(-.1,.05)])
    anchor = transformed(shape, [.5,.5,.3])
    rng = np.random.default_rng(9)
    for _ in range(20):
        px, py, angle = _edge_pose(shape, anchor, .02, rng)
        candidate = transformed(shape, [px, py, angle])
        assert candidate.distance(anchor) >= .02 - 1e-9
    a = inventory(seed=17, counts=(6,), areas=(.02,))
    b = inventory(seed=17, counts=(6,), areas=(.02,))
    greedy_insert(a, strategy='edge', proposals=10)
    greedy_insert(b, strategy='edge', proposals=10)
    assert np.array_equal(a.poses, b.poses) and np.array_equal(a.active, b.active)
    assert not validate(a)


def test_region_constrains_new_centers_across_strategies():
    from flagstone.search import STRATEGIES
    for strategy in STRATEGIES:
        x = inventory(seed=16, counts=(5,), areas=(.003,))
        bounds = (.15, .15, .45, .45)
        greedy_insert(x, proposals=20, strategy=strategy, proposal_bounds=bounds, grid_resolution=6)
        assert x.active.any() and not validate(x)
        assert np.all((x.poses[x.active,0] >= bounds[0]) & (x.poses[x.active,0] <= bounds[2]))
        assert np.all((x.poses[x.active,1] >= bounds[1]) & (x.poses[x.active,1] <= bounds[3]))


def test_local_rebuild_increases_coverage_and_keeps_frozen():
    from flagstone.search import rebuild_search, coverage
    x = inventory(seed=3, counts=(1, 2), areas=(.08, .01), mixed=False)
    x.active[1:] = True
    x.poses[1] = [.25, .25, 0]; x.poses[2] = [.75, .75, 0]
    frozen = x.poses[2].copy(); before = coverage(x)
    result = rebuild_search(x, rounds=5, patch_size=1, proposals=30, strategy='random',
                            patch_padding=.3, max_new_stones=1, frozen_ids=[x.ids[2]])
    assert result.coverage > before and result.improvements >= 1
    assert x.active[2] and np.array_equal(x.poses[2], frozen)
    assert not validate(x)


def test_neighborhood_search_rejects_unseen_global_collision(monkeypatch):
    import flagstone.search as search
    x = inventory(seed=2, counts=(2,), areas=(.01,))
    x.active[0] = True; x.poses[0] = [.5, .5, 0]
    before = x.poses.copy()
    def bad_local(layout, **kwargs):
        layout.poses[1] = layout.poses[0]
    monkeypatch.setattr(search, 'relax_neighborhood', bad_local)
    result = greedy_insert(x, proposals=3, relaxation_steps=1, local_relaxation=True, use_prefilter=False)
    assert result.accepted == 0 and np.array_equal(x.poses, before)
    assert x.active.tolist() == [True, False] and not validate(x)


def test_clearance_mobility_search_preserves_frozen_and_validity():
    x = inventory(seed=7, counts=(8,), areas=(.02,))
    x.active[0] = True; x.poses[0] = [.5,.5,0]
    fixed = x.poses[0].copy()
    result = greedy_insert(x, proposals=8, strategy='edge', relaxation_steps=2,
                           local_relaxation=True, max_relaxation_penalty=.2,
                           mobility_policy='clearance', frozen_ids=[x.ids[0]])
    assert x.active[0] and np.array_equal(x.poses[0], fixed)
    assert result.mobile_stone_visits >= result.relaxation_attempts
    assert not result.violations and not validate(x)


def test_rebuild_can_restore_original_removed_pose(monkeypatch):
    import flagstone.search as search
    from flagstone.phase1 import Layout, local_shape, polygon
    def make_layout():
        big = local_shape([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]])
        small = local_shape([[-.05,-.05],[.05,-.05],[.05,.05],[-.05,.05]])
        return Layout(polygon([(0,0),(1,0),(1,1),(0,1)]), {'A':big,'B':small},
                      np.array([[.25,.25,0],[0,0,0]]), np.array([True,False]),
                      ('A','B'), np.random.default_rng(4))
    def insert_only_small(layout, **kwargs):
        layout.poses[1] = [.75,.75,0]; layout.active[1] = True
        return search.SearchResult(1, 1, search.coverage(layout), 0, [])
    monkeypatch.setattr(search, 'greedy_insert', insert_only_small)
    without = make_layout()
    search.rebuild_search(without, rounds=1, patch_size=1, restore_removed=False)
    assert without.active.tolist() == [True,False]
    with_fallback = make_layout()
    result = search.rebuild_search(with_fallback, rounds=1, patch_size=1, restore_removed=True)
    assert with_fallback.active.tolist() == [True,True] and result.improvements == 1
    assert np.array_equal(with_fallback.poses[0], [.25,.25,0])
    assert not validate(with_fallback)


def test_rebuild_restoration_does_not_overlap_new_stone(monkeypatch):
    import flagstone.search as search
    from flagstone.phase1 import Layout, local_shape, polygon
    big = local_shape([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]])
    small = local_shape([[-.05,-.05],[.05,-.05],[.05,.05],[-.05,.05]])
    layout = Layout(polygon([(0,0),(1,0),(1,1),(0,1)]), {'A':big,'B':small},
                    np.array([[.5,.5,0],[0,0,0]]), np.array([True,False]),
                    ('A','B'), np.random.default_rng(5))
    old_poses = layout.poses.copy()
    def insert_on_old_pose(x, **kwargs):
        x.poses[1] = [.5,.5,0]; x.active[1] = True
        return search.SearchResult(1, 1, search.coverage(x), 0, [])
    monkeypatch.setattr(search, 'greedy_insert', insert_on_old_pose)
    result = search.rebuild_search(layout, rounds=1, patch_size=1, restore_removed=True)
    assert result.improvements == 0 and layout.active.tolist() == [True,False]
    assert np.array_equal(layout.poses, old_poses) and not validate(layout)


def test_wall_proposals_touch_interior_side_with_clearance():
    from flagstone.phase1 import local_shape, polygon, transformed
    from flagstone.search import _wall_pose
    shape = local_shape([[-.1,-.05],[.1,-.05],[.1,.05],[-.1,.05]])
    for vertices in ([(0,0),(1,0),(1,1),(0,1)], [(0,1),(1,1),(1,0),(0,0)]):
        patio = polygon(vertices)
        rng = np.random.default_rng(91)
        valid = 0
        for _ in range(100):
            world = transformed(shape, _wall_pose(shape, patio, .02, rng))
            if patio.buffer(-.02).covers(world):
                valid += 1
                assert patio.covers(world)
                assert .02 <= world.distance(patio.boundary) < .02000001
        assert valid > 50


def test_wall_proposal_uses_edge_and_uncertainty_margins(monkeypatch):
    import flagstone.search as search
    x = inventory(seed=3, counts=(1,), areas=(.01,))
    used = []
    def proposed(shape, patio, clearance, rng):
        used.append(clearance)
        return .5, .5, 0
    monkeypatch.setattr(search, '_wall_pose', proposed)
    result = search.greedy_insert(x, strategy='wall', proposals=1, edge=.02,
                                  patio_margin=.003, stone_margins={x.ids[0]:.004})
    assert len(used) == 1 and np.isclose(used[0], .027)
    assert result.accepted == 1 and not result.violations
    assert not validate(x, edge=.02, patio_margin=.003, stone_margins={x.ids[0]:.004})
