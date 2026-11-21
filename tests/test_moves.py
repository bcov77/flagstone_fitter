import numpy as np
from flagstone.moves import pulse
from flagstone.phase1 import synthetic, validate
from flagstone.search import coverage


def test_seeded_pulses_are_valid_and_preserve_frozen():
    a, b = synthetic(9), synthetic(9)
    fixed = a.poses[0].copy(); before = coverage(a)
    kw = dict(rounds=3, translation_noise=.002, angle_noise=.005, relaxation_steps=2, frozen_ids=[a.ids[0]])
    ra, rb = pulse(a, **kw), pulse(b, **kw)
    assert ra == rb and np.array_equal(a.poses, b.poses)
    assert ra.accepted > 0 and np.array_equal(a.poses[0], fixed)
    assert not validate(a) and coverage(a) == before


def test_invalid_noise_is_rolled_back_and_pressure_preserves_validity():
    x = synthetic(12)
    old = x.poses.copy()
    result = pulse(x, rounds=1, translation_noise=100, relaxation_steps=0)
    assert result.rejected == 1 and np.array_equal(x.poses, old)
    result = pulse(x, rounds=1, pressure=[1000,0], relaxation_steps=1, step=1)
    assert not result.violations and not validate(x)


def test_zero_pulse_is_noop_without_consuming_rng():
    a, b = synthetic(3), synthetic(3)
    result = pulse(a, rounds=2, relaxation_steps=1)
    assert result.unchanged == 2 and np.array_equal(a.poses,b.poses)
    assert a.rng.random() == b.rng.random()
