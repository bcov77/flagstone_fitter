import numpy as np
from flagstone.generate import benchmark_edge_cases, inventory
from flagstone.insertion import _InsertionOracle
from flagstone.phase1 import validate
from flagstone.search import greedy_insert, STRATEGIES


def test_insertion_precheck_matches_full_validator_on_random_proposals():
    x = benchmark_edge_cases(seed=4)['concave_witness']
    margins = {sid: .002 * (i % 3) for i, sid in enumerate(x.ids)}
    assert not validate(x, stone_margins=margins, patio_margin=.002)
    indices = np.flatnonzero(~x.active)
    rng = np.random.default_rng(42)
    for gap in (0., .01, .03):
        assert not validate(x, gap=gap, edge=.01, stone_margins=margins, patio_margin=.002)
        oracle = _InsertionOracle(x, gap=gap, edge=.01, stone_margins=margins, patio_margin=.002)
        for _ in range(100):
            i = int(rng.choice(indices))
            pose = [*rng.uniform(-.1, 1.1, 2), rng.uniform(-np.pi, np.pi)]
            fast = oracle.can_insert(i, pose)
            old = x.poses[i].copy()
            x.poses[i] = pose; x.active[i] = True
            exact = not validate(x, gap=gap, edge=.01, stone_margins=margins, patio_margin=.002)
            x.active[i] = False; x.poses[i] = old
            assert fast == exact


def test_prefilter_keeps_seeded_search_trajectory():
    for strategy in STRATEGIES:
        a = inventory(seed=13, counts=(10,), areas=(.02,), vertex_profile='detailed')
        b = inventory(seed=13, counts=(10,), areas=(.02,), vertex_profile='detailed')
        ra = greedy_insert(a, proposals=8, strategy=strategy, grid_resolution=8, use_prefilter=True)
        rb = greedy_insert(b, proposals=8, strategy=strategy, grid_resolution=8, use_prefilter=False)
        assert ra.attempts == rb.attempts and ra.accepted == rb.accepted
        assert np.array_equal(a.poses, b.poses) and np.array_equal(a.active, b.active)
        assert a.rng.random() == b.rng.random()


def test_precheck_near_clearance_and_zero_gap_overlap():
    from flagstone.phase1 import synthetic
    x = synthetic()
    x.active[1:] = False
    oracle = _InsertionOracle(x, gap=.01, edge=.01)
    assert oracle.can_insert(1, [.13+.19+.01, .13, 0])
    assert not oracle.can_insert(1, [.13+.19+.009, .13, 0])
    zero = _InsertionOracle(x, gap=0, edge=0)
    assert zero.can_insert(1, [.13+.19, .13, 0])
    assert not zero.can_insert(1, [.13+.19-.000001, .13, 0])


def test_prefilter_with_relaxation_keeps_seeded_trajectory():
    for local in (False, True):
        a = inventory(seed=35, counts=(5,), areas=(.03,))
        b = inventory(seed=35, counts=(5,), areas=(.03,))
        options = dict(proposals=4, strategy='edge', relaxation_steps=2, local_relaxation=local)
        ra = greedy_insert(a, use_prefilter=True, **options)
        rb = greedy_insert(b, use_prefilter=False, **options)
        assert ra.accepted == rb.accepted and ra.attempts == rb.attempts
        assert np.allclose(a.poses, b.poses, atol=1e-12) and np.array_equal(a.active, b.active)
        assert a.rng.random() == b.rng.random()


def test_relaxation_penalty_distinguishes_shallow_and_deep_overlap():
    from flagstone.phase1 import synthetic
    x = synthetic()
    x.active[1:] = False
    oracle = _InsertionOracle(x, gap=.01, edge=.01)
    clear = oracle.relaxation_penalty(1, [.45,.13,0])
    shallow = oracle.relaxation_penalty(1, [.13+.18,.13,0])
    deep = oracle.relaxation_penalty(1, [.13,.13,0])
    assert clear == 0 and 0 < shallow < deep


def test_selective_relaxation_skips_deep_cases_and_keeps_validity():
    from flagstone.search import greedy_insert
    x = inventory(seed=10, counts=(4,), areas=(.04,))
    result = greedy_insert(x, proposals=10, strategy='random', relaxation_steps=2,
                           local_relaxation=True, max_relaxation_penalty=0)
    assert result.relaxation_skipped > 0
    assert result.relaxation_attempts + result.relaxation_skipped <= result.attempts
    assert not validate(x)


def test_clearance_neighbor_selection_matches_exact_distances():
    from flagstone.phase1 import synthetic, transformed
    x = synthetic()
    x.active[3:] = False
    x.active[1] = False
    margins = {sid:.003 * (i % 3) for i,sid in enumerate(x.ids)}
    oracle = _InsertionOracle(x, gap=.01, edge=.01, stone_margins=margins)
    rng = np.random.default_rng(71)
    for _ in range(100):
        pose = [*rng.uniform(0,1,2), rng.uniform(-np.pi,np.pi)]
        candidate = transformed(x.shapes[x.ids[1]], pose)
        expected = [j for j,sid in enumerate(x.ids) if x.active[j] and
                    candidate.distance(transformed(x.shapes[sid], x.poses[j])) <= .01 + margins[x.ids[1]] + margins[sid] + .02]
        assert oracle.nearby_indices(1, pose, extra=.02) == expected


def test_insertion_precheck_respects_explicit_tolerance():
    from flagstone.phase1 import synthetic
    x = synthetic(); x.active[1:] = False
    pose = [.13 + .19 + .0099995, .13, 0]
    strict = _InsertionOracle(x, gap=.01, edge=.01, tolerance=0)
    relaxed = _InsertionOracle(x, gap=.01, edge=.01, tolerance=1e-6)
    assert not strict.can_insert(1, pose) and relaxed.can_insert(1, pose)
    x.poses[1] = pose; x.active[1] = True
    assert validate(x, gap=.01, edge=.01, tolerance=0)
    assert not validate(x, gap=.01, edge=.01, tolerance=1e-6)


def test_distance_first_contact_check_matches_full_intersection():
    from flagstone.phase1 import Layout, local_shape, polygon, transformed
    square = local_shape([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]])
    small = local_shape([[-.02,-.02],[.02,-.02],[.02,.02],[-.02,.02]])
    x = Layout(polygon([(0,0),(1,0),(1,1),(0,1)]), {'A':square,'B':small},
               [[.5,.5,0],[0,0,0]], [True,False], ('A','B'), np.random.default_rng(3))
    rng = np.random.default_rng(190)
    poses = [[.5,.5,0], [.62,.5,0], [.61,.5,0], [.621,.5,0]]
    poses.extend([[*rng.uniform(.3,.7,2), rng.uniform(-np.pi,np.pi)] for _ in range(100)])
    fixed = transformed(square, x.poses[0])
    for gap, tolerance in ((0,0), (.01,0), (.01,.02)):
        oracle = _InsertionOracle(x, gap=gap, edge=0, tolerance=tolerance)
        for pose in poses:
            candidate = transformed(small, pose)
            expected = candidate.intersection(fixed).area > 0 or candidate.distance(fixed) < gap-tolerance
            assert oracle.can_insert(1, pose) is not expected
            x.poses[1] = pose; x.active[1] = True
            violations = validate(x, gap=gap, edge=0, tolerance=tolerance)
            x.active[1] = False
            assert any(v['kind'] == 'pair_clearance' for v in violations) is bool(expected)


def test_prevalidated_insertion_still_rejects_bad_precheck(monkeypatch):
    import flagstone.search as search
    x = inventory(seed=4, counts=(2,), areas=(.01,))
    x.active[0] = True; x.poses[0] = [.5,.5,0]
    before = x.poses.copy()
    monkeypatch.setattr(search._InsertionOracle, 'can_insert', lambda *args: True)
    monkeypatch.setattr(search, '_edge_pose', lambda *args: tuple(x.poses[0]))
    result = search.greedy_insert(x, strategy='edge', proposals=1,
                                  _validation_token=search._VALIDATED_INPUT)
    assert result.accepted == 0 and x.active.tolist() == [True,False]
    assert np.array_equal(x.poses, before) and not validate(x)
