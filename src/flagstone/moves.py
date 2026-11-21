"""Bounded perturbation and pressure moves over a valid layout."""
from dataclasses import dataclass
import numpy as np
from .phase1 import validate
from .relaxation import relax
from .search import _frozen_indices


@dataclass
class PulseResult:
    attempted: int
    accepted: int
    rejected: int
    unchanged: int
    violations: list


def pulse(layout, *, rounds=1, translation_noise=0., angle_noise=0., pressure=None,
          relaxation_steps=20, step=.02, gap=.01, edge=.01, stone_margins=None,
          patio_margin=0, frozen_ids=None, tolerance=1e-9):
    """Apply bounded random perturbation and/or directional pressure.

    Each round begins from a valid state. Random displacements are independent
    uniform component bounds in layout units, and angles are in radians. A
    candidate that remains invalid after relaxation is rolled back; its random
    draws remain consumed. Frozen stones never move. No temperature or physical
    dynamics are implied. A zero-noise pressure pulse uses the existing soft
    pressure potential and still requires final polygon validation.
    """
    if not isinstance(rounds, int) or rounds < 0 or not isinstance(relaxation_steps, int) or relaxation_steps < 0:
        raise ValueError('rounds and relaxation_steps must be nonnegative integers')
    if not all(np.isfinite(v) and v >= 0 for v in (translation_noise, angle_noise)):
        raise ValueError('noise bounds must be finite and nonnegative')
    if not np.isfinite(step) or step <= 0:
        raise ValueError('step must be finite and positive')
    if pressure is not None:
        pressure = np.asarray(pressure, float)
        if pressure.shape != (2,) or not np.isfinite(pressure).all():
            raise ValueError('pressure must be a finite 2-vector')
    if validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin):
        raise ValueError('Initial layout must be valid')
    frozen = _frozen_indices(layout, frozen_ids)
    mobility = np.array([1. if layout.active[i] and i not in frozen else 0. for i in range(len(layout.ids))])
    accepted = rejected = unchanged = 0
    for _ in range(rounds):
        before = layout.snapshot()
        try:
            for i in np.flatnonzero(mobility):
                if translation_noise:
                    layout.poses[i, :2] += layout.rng.uniform(-translation_noise, translation_noise, 2)
                if angle_noise:
                    layout.poses[i, 2] += layout.rng.uniform(-angle_noise, angle_noise)
            if relaxation_steps:
                relax(layout, gap, edge, max_steps=relaxation_steps, step=step,
                      mobility=mobility, pressure=pressure, stone_margins=stone_margins,
                      patio_margin=patio_margin, tolerance=tolerance)
            if validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin):
                layout.poses[:] = before[0]
                layout.active[:] = before[1]
                rejected += 1
            elif np.array_equal(layout.poses, before[0]):
                unchanged += 1
            else:
                accepted += 1
        except BaseException:
            layout.restore(before)
            raise
    return PulseResult(rounds, accepted, rejected, unchanged,
                       validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin))
