import numpy as np
from flagstone.phase1 import synthetic, validate
from flagstone.relaxation import relax


def test_relax_overlap_with_frozen_stone():
    x = synthetic()
    x.active[2:] = False
    x.poses[1, :2] = x.poses[0, :2] + [.15, 0]
    frozen = x.poses[0].copy()
    result = relax(x, mobility=np.array([0, 1] + [0] * 14), max_steps=100)
    assert result.final_energy < result.initial_energy
    assert np.array_equal(x.poses[0], frozen)
    assert not validate(x)


def test_valid_state_unchanged_and_wrapped_tether():
    from flagstone.relaxation import energy
    x = synthetic()
    original = x.poses.copy()
    result = relax(x)
    assert result.reason == 'valid' and result.steps == 0
    assert np.array_equal(x.poses, original)
    reference = x.poses.copy()
    reference[0, 2] = 2 * np.pi
    assert energy(x, reference=reference, tethers=np.ones(len(x.ids))) < 1e-20


def test_tether_increases_penalty_for_displacement():
    from flagstone.relaxation import energy
    x = synthetic()
    reference = x.poses.copy()
    x.poses[0, 0] += .01
    assert energy(x, reference=reference, tethers=np.ones(len(x.ids))) > energy(x)


def test_rotation_into_narrow_slot():
    from flagstone.phase1 import Layout, polygon, local_shape
    shape = local_shape([(-.3, -.05), (.3, -.05), (.3, .05), (-.3, .05)])
    x = Layout(polygon([(0, 0), (.4, 0), (.4, .8), (0, .8)]), {'s': shape}, [[.2, .4, .5]], [True], ('s',), np.random.default_rng(1))
    result = relax(x, step=.01)
    assert result.reason == 'valid' and not result.violations
    assert abs(x.poses[0, 2] - .5) > .3
    assert result.final_energy < result.initial_energy


def test_failure_is_explicit_and_pressure_preserves_valid_state():
    x = synthetic()
    x.poses[1] = x.poses[0]
    result = relax(x, mobility=np.zeros(len(x.ids)), max_steps=2)
    assert result.violations and result.reason == 'step_limit'
    x = synthetic()
    result = relax(x, pressure=[1000, 0], max_steps=1, step=1)
    assert result.reason == 'restored_valid' and not validate(x)


def test_concave_notch_is_not_filled_by_convex_hull():
    from flagstone.phase1 import Layout, polygon, local_shape
    from flagstone.relaxation import energy
    notch = local_shape([(0, 0), (.6, 0), (.6, .2), (.2, .2), (.2, .6), (0, .6)])
    small = local_shape([(-.1, -.1), (.1, -.1), (.1, .1), (-.1, .1)])
    # Account for the centroid translation used to center the L shape.
    c = polygon([(0, 0), (.6, 0), (.6, .2), (.2, .2), (.2, .6), (0, .6)]).centroid
    x = Layout(polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), {'L': notch, 's': small}, [[c.x+.1, c.y+.1, 0], [.5, .5, 0]], [True, True], ('L', 's'), np.random.default_rng(0))
    assert not validate(x, gap=.01, edge=.01)
    assert energy(x, gap=.01, edge=.01) == 0


def test_bounded_restarts_escape_containment_and_keep_frozen():
    from flagstone.phase1 import Layout, polygon, local_shape
    from flagstone.relaxation import relax_with_restarts
    big = local_shape([(-.2, -.2), (.2, -.2), (.2, .2), (-.2, .2)])
    small = local_shape([(-.05, -.05), (.05, -.05), (.05, .05), (-.05, .05)])
    def make():
        return Layout(polygon([(0,0), (1,0), (1,1), (0,1)]), {'b':big, 's':small}, [[.5,.5,0], [.5,.5,0]], [True,True], ('b','s'), np.random.default_rng(42))
    a, b = make(), make()
    frozen = a.poses[0].copy()
    result = relax_with_restarts(a, restarts=10, translation_noise=.35, mobility=[0, 1], max_steps=20)
    other = relax_with_restarts(b, restarts=10, translation_noise=.35, mobility=[0, 1], max_steps=20)
    assert result.reason == 'valid' and result.restarts_used > 0
    assert np.array_equal(a.poses[0], frozen)
    assert np.array_equal(a.poses, b.poses) and result == other
    assert not validate(a)


def test_energy_bounds_filter_matches_full_pair_evaluation():
    from flagstone.generate import inventory
    from flagstone.phase1 import transformed
    from flagstone.relaxation import energy
    x = inventory(seed=5, counts=(10,), areas=(.02,))
    x.active[:] = True
    x.poses[:, :2] = x.rng.uniform(.1, .9, (10, 2))
    x.poses[:, 2] = x.rng.uniform(-np.pi, np.pi, 10)
    shapes = [transformed(x.shapes[s], x.poses[i]) for i, s in enumerate(x.ids)]
    for gap in (0, .04):
        inset = x.patio.buffer(-.01)
        expected = sum(g.difference(inset).area for g in shapes)
        for i, a in enumerate(shapes):
            for b in shapes[i+1:]:
                aa = a.buffer(gap/2) if gap else a
                bb = b.buffer(gap/2) if gap else b
                expected += aa.intersection(bb).area + a.intersection(b).area
        assert np.isclose(energy(x, gap=gap), expected / x.patio.area)


def test_local_trial_energy_matches_full_recalculation():
    from flagstone.generate import inventory
    from flagstone.relaxation import _LocalObjective, energy
    for seed in range(3):
        x = inventory(seed=seed, counts=(15,), areas=(.02,), mixed=True)
        x.active[:] = True
        x.active[7] = False
        x.poses[:, :2] = x.rng.uniform(.05, .95, (len(x.ids), 2))
        x.poses[:, 2] = x.rng.uniform(-np.pi, np.pi, len(x.ids))
        reference = x.poses.copy()
        reference[:, 2] += .2
        tethers = x.rng.uniform(0, 1, len(x.ids))
        pressure = np.array([.1, -.2])
        margins = {sid: float(x.rng.uniform(0, .03)) for sid in x.ids}
        options = dict(gap=.02, edge=.01, reference=reference, tethers=tethers, pressure=pressure,
                       stone_margins=margins, patio_margin=.01)
        local = _LocalObjective(x, **options)
        for i in (0, 3, 10, 2, 0):
            old = x.poses[i].copy()
            pose = old + x.rng.uniform(-.1, .1, 3)
            trial = local.trial(i, pose)
            assert np.array_equal(x.poses[i], old)
            x.poses[i] = pose
            expected = energy(x, **options)
            x.poses[i] = old
            assert np.isclose(trial[0], expected, atol=1e-12, rtol=1e-12)
            local.commit(i, pose, trial)
            assert np.isclose(local.current, energy(x, **options), atol=1e-12, rtol=1e-12)


def test_cached_relaxation_matches_full_energy_coordinate_descent():
    from flagstone.generate import inventory
    from flagstone.relaxation import energy
    a = inventory(seed=19, counts=(5,), areas=(.03,))
    a.active[:] = True
    a.poses[:, :2] = [[.35,.35],[.39,.35],[.5,.5],[.7,.6],[.6,.7]]
    from flagstone.phase1 import Layout
    b = Layout(a.patio, a.shapes, a.poses, a.active, a.ids, np.random.default_rng(0))
    reference = a.poses.copy()
    weights = np.array([.1,.2,.3,.4,.5])
    margins = {sid:.005 for sid in a.ids}
    kwargs = dict(gap=.01, edge=.01, reference=reference, tethers=weights,
                  stone_margins=margins, patio_margin=.002)
    current = energy(b, **kwargs)
    step = .01
    for _ in range(3):
        improved = False
        for i, sid in enumerate(b.ids):
            radius = np.sqrt(b.shapes[sid].area)
            for axis in range(3):
                original = b.poses[i, axis]
                delta = step / (radius if axis == 2 else 1)
                best, best_pos = current, original
                for sign in (-1, 1):
                    b.poses[i, axis] = original + sign*delta
                    value = energy(b, **kwargs)
                    if value < best - 1e-15:
                        best, best_pos = value, b.poses[i, axis]
                b.poses[i, axis] = best_pos
                if best < current:
                    current = best
                    improved = True
        if not validate(b, gap=.01, edge=.01, stone_margins=margins, patio_margin=.002):
            break
        if not improved:
            step *= .5
    result = relax(a, max_steps=3, step=.01, **kwargs)
    assert np.allclose(a.poses, b.poses, atol=1e-12)
    assert np.isclose(result.final_energy, energy(a, **kwargs), atol=1e-12)


def test_neighborhood_relaxation_matches_full_for_isolated_contact():
    from flagstone.phase1 import synthetic
    from flagstone.relaxation import relax_neighborhood
    a, b = synthetic(8), synthetic(8)
    a.poses[1, :2] = a.poses[0, :2] + [.15, 0]
    b.poses[:] = a.poses
    mobility = np.zeros(len(a.ids)); mobility[1] = 1
    local = relax_neighborhood(a, mobility=mobility, max_steps=30, step=.01)
    full = relax(b, mobility=mobility, max_steps=30, step=.01)
    assert not local.violations and not full.violations
    assert np.allclose(a.poses, b.poses, atol=1e-12)
    assert not validate(a)


def test_neighborhood_reports_collisions_outside_subset(monkeypatch):
    import flagstone.relaxation as module
    x = synthetic()
    x.active[:] = False
    x.active[0] = x.active[-1] = True
    mobility = np.zeros(len(x.ids)); mobility[0] = 1
    target = x.poses[-1].copy()
    def move_outside(local, *args, **kwargs):
        local.poses[0] = target
        return module.RelaxationResult(0, 0, 1, 'valid', [])
    monkeypatch.setattr(module, 'relax', move_outside)
    result = module.relax_neighborhood(x, mobility=mobility, max_steps=1, step=.001)
    assert result.reason == 'global_invalid'
    assert any(v['kind'] == 'pair_clearance' for v in result.violations)


def test_relaxation_respects_requested_validation_tolerance():
    x = synthetic(); x.active[2:] = False
    x.poses[1] = [.13+.19+.0099995, .13, 0]
    before = x.poses.copy()
    result = relax(x, tolerance=1e-6)
    assert result.reason == 'valid' and result.steps == 0
    assert np.array_equal(x.poses, before)
    strict = relax(x, tolerance=0, max_steps=20)
    assert strict.steps > 0 and not validate(x, tolerance=0)


def test_interrupted_relaxation_restores_full_state(monkeypatch):
    import pytest
    import flagstone.relaxation as relaxation
    x = synthetic(seed=7)
    x.active[2:] = False
    x.poses[1, :2] = x.poses[0, :2] + [.15, 0]
    poses, active, rng_state = x.snapshot()
    original_commit = relaxation._LocalObjective.commit
    def interrupt_after_commit(self, *args):
        original_commit(self, *args)
        self.layout.active[-1] = True
        self.layout.rng.random()
        raise KeyboardInterrupt()
    monkeypatch.setattr(relaxation._LocalObjective, 'commit', interrupt_after_commit)
    with pytest.raises(KeyboardInterrupt):
        relaxation.relax(x, max_steps=5)
    assert np.array_equal(x.poses, poses) and np.array_equal(x.active, active)
    assert x.rng.bit_generator.state == rng_state


def test_interrupted_restart_restores_poses_and_rng(monkeypatch):
    import pytest
    import flagstone.relaxation as relaxation
    x = synthetic(seed=8)
    x.active[2:] = False
    x.poses[1] = x.poses[0]
    poses, active, rng_state = x.snapshot()
    calls = 0
    def interrupted(layout, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return relaxation.RelaxationResult(1, 1, 0, 'stalled', [{'kind':'pair_clearance'}])
        layout.poses[0, 0] += 1
        raise KeyboardInterrupt()
    monkeypatch.setattr(relaxation, 'relax', interrupted)
    with pytest.raises(KeyboardInterrupt):
        relaxation.relax_with_restarts(x, restarts=1)
    assert calls == 2
    assert np.array_equal(x.poses, poses) and np.array_equal(x.active, active)
    assert x.rng.bit_generator.state == rng_state


def test_interrupted_neighborhood_global_check_restores_state(monkeypatch):
    import pytest
    import flagstone.relaxation as relaxation
    x = synthetic(seed=9)
    x.active[2:] = False
    poses, active, rng_state = x.snapshot()
    def pretend_relax(local, *args, **kwargs):
        local.poses[0, 0] += .01
        return relaxation.RelaxationResult(1, 0, 1, 'valid', [])
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(relaxation, 'relax', pretend_relax)
    monkeypatch.setattr(relaxation, 'validate', interrupted)
    with pytest.raises(KeyboardInterrupt):
        relaxation.relax_neighborhood(x, mobility=np.ones(len(x.ids)), max_steps=1)
    assert np.array_equal(x.poses, poses) and np.array_equal(x.active, active)
    assert x.rng.bit_generator.state == rng_state
