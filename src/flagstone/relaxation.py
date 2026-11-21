"""Small, deliberately transparent CPU reference relaxation."""
from dataclasses import dataclass
from functools import wraps
import numpy as np
from .phase1 import transformed, validate


@dataclass
class RelaxationResult:
    initial_energy: float
    final_energy: float
    steps: int
    reason: str
    violations: list


def energy(layout, gap=.01, edge=.01, reference=None, tethers=None, pressure=None, *, stone_margins=None, patio_margin=0):
    """Area-based penalty; suitable as a local guide, not a validity test."""
    stone_margins = {} if stone_margins is None else dict(stone_margins)
    if not set(stone_margins).issubset(layout.ids) or not all(np.isfinite(v) and v >= 0 for v in stone_margins.values()):
        raise ValueError('Stone margins must be nonnegative finite values for known IDs')
    if not all(np.isfinite(v) and v >= 0 for v in (gap, edge, patio_margin)):
        raise ValueError('Clearances and margins must be finite and nonnegative')
    ids = [s for i, s in enumerate(layout.ids) if layout.active[i]]
    shapes = [transformed(layout.shapes[s], layout.poses[i]) for i, s in enumerate(layout.ids) if layout.active[i]]
    inset_cache = {}
    for sid in ids:
        distance = edge + patio_margin + stone_margins.get(sid, 0)
        if distance not in inset_cache:
            inset_cache[distance] = layout.patio.buffer(-distance)
    value = sum(g.difference(inset_cache[edge + patio_margin + stone_margins.get(sid, 0)]).area for sid, g in zip(ids, shapes))
    radii = [gap / 2 + stone_margins.get(sid, 0) for sid in ids]
    buffered = [g.buffer(r) if r else g for g, r in zip(shapes, radii)]
    bounds = [g.bounds for g in buffered]
    for i, a in enumerate(shapes):
        for j in range(i + 1, len(shapes)):
            b = shapes[j]
            # Bounds of expanded polygons are conservative for the actual contact.
            aa, bb = bounds[i], bounds[j]
            if aa[2] < bb[0] or bb[2] < aa[0] or aa[3] < bb[1] or bb[3] < aa[1]:
                continue
            value += buffered[i].intersection(buffered[j]).area
            value += a.intersection(b).area
    # Normalize by patio area, so weights remain meaningful under unit changes.
    value /= layout.patio.area
    length = np.sqrt(layout.patio.area)
    if reference is not None and tethers is not None:
        for i, sid in enumerate(layout.ids):
            if not layout.active[i]:
                continue
            displacement = (layout.poses[i, :2] - reference[i, :2]) / length
            angle = (layout.poses[i, 2] - reference[i, 2] + np.pi) % (2 * np.pi) - np.pi
            radius = np.sqrt(layout.shapes[sid].area) / length
            value += tethers[i] * (np.dot(displacement, displacement) + (radius * angle) ** 2)
    if pressure is not None:
        for i in range(len(layout.ids)):
            if layout.active[i]:
                value -= np.dot(pressure, layout.poses[i, :2] / length)
    return float(value)


class _LocalObjective:
    """Cache geometry for one-stone trial moves; poses/masks must not change externally."""
    def __init__(self, layout, gap, edge, reference, tethers, pressure, stone_margins, patio_margin):
        self.layout, self.reference, self.tethers, self.pressure = layout, reference, tethers, pressure
        self.margins = {} if stone_margins is None else dict(stone_margins)
        self.area, self.length = layout.patio.area, np.sqrt(layout.patio.area)
        self.current = energy(layout, gap, edge, reference, tethers, pressure,
                              stone_margins=self.margins, patio_margin=patio_margin)
        self.radii = [gap / 2 + self.margins.get(sid, 0) for sid in layout.ids]
        distances = [edge + patio_margin + self.margins.get(sid, 0) for sid in layout.ids]
        cache = {d: layout.patio.buffer(-d) for d in set(distances)}
        self.insets = [cache[d] for d in distances]
        self.geoms = [transformed(layout.shapes[sid], layout.poses[i]) if layout.active[i] else None
                      for i, sid in enumerate(layout.ids)]
        self.buffers = [self._buffer(i, g) if g is not None else None for i, g in enumerate(self.geoms)]
        self.bounds = [g.bounds if g is not None else None for g in self.buffers]

    def _buffer(self, i, geom):
        return geom.buffer(self.radii[i]) if self.radii[i] else geom

    def _unary(self, i, pose, geom):
        value = geom.difference(self.insets[i]).area / self.area
        delta = (pose[:2] - self.reference[i, :2]) / self.length
        angle = (pose[2] - self.reference[i, 2] + np.pi) % (2 * np.pi) - np.pi
        radius = np.sqrt(self.layout.shapes[self.layout.ids[i]].area) / self.length
        value += self.tethers[i] * (np.dot(delta, delta) + (radius * angle) ** 2)
        if self.pressure is not None:
            value -= np.dot(self.pressure, pose[:2] / self.length)
        return float(value)

    def _contacts(self, i, geom, buffered):
        bounds = buffered.bounds
        value = 0.0
        for j, other in enumerate(self.geoms):
            if j == i or other is None:
                continue
            bb = self.bounds[j]
            if bounds[2] < bb[0] or bb[2] < bounds[0] or bounds[3] < bb[1] or bb[3] < bounds[1]:
                continue
            value += buffered.intersection(self.buffers[j]).area + geom.intersection(other).area
        return value / self.area

    def trial(self, i, pose):
        if self.geoms[i] is None:
            raise ValueError('cannot move inactive stone')
        geom = transformed(self.layout.shapes[self.layout.ids[i]], pose)
        buffered = self._buffer(i, geom)
        old = self._unary(i, self.layout.poses[i], self.geoms[i]) + self._contacts(i, self.geoms[i], self.buffers[i])
        new = self._unary(i, pose, geom) + self._contacts(i, geom, buffered)
        return self.current + new - old, geom, buffered

    def commit(self, i, pose, trial):
        self.current, self.geoms[i], self.buffers[i] = trial
        self.bounds[i] = self.buffers[i].bounds
        self.layout.poses[i] = pose


def _rollback_on_exception(function):
    """Make an interrupted public relaxation leave its entire input state intact."""
    @wraps(function)
    def wrapped(layout, *args, **kwargs):
        before = layout.snapshot()
        try:
            return function(layout, *args, **kwargs)
        except BaseException:
            layout.restore(before)
            raise
    return wrapped


@_rollback_on_exception
def relax(layout, gap=.01, edge=.01, max_steps=100, step=.02, mobility=None, reference=None, tethers=None, pressure=None, *, stone_margins=None, patio_margin=0, tolerance=1e-9):
    """Coordinate descent over poses. Frozen stones have zero mobility.

    The final geometry is always checked by the authoritative validator.
    """
    if not all(np.isfinite(v) for v in (gap, edge, step)) or gap < 0 or edge < 0 or step <= 0 or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError('Clearances and step count must be nonnegative; step must be positive')
    if mobility is None:
        mobility = np.ones(len(layout.ids))
    mobility = np.asarray(mobility, dtype=float)
    if mobility.shape != (len(layout.ids),) or np.any(mobility < 0) or not np.isfinite(mobility).all():
        raise ValueError('mobility must contain one finite nonnegative value per stone')
    if reference is None:
        reference = layout.poses.copy()
    reference = np.asarray(reference, dtype=float)
    if reference.shape != layout.poses.shape or not np.isfinite(reference).all():
        raise ValueError('reference must match the pose array and be finite')
    if tethers is None:
        tethers = np.zeros(len(layout.ids))
    tethers = np.asarray(tethers, dtype=float)
    if tethers.shape != (len(layout.ids),) or not np.isfinite(tethers).all() or np.any(tethers < 0):
        raise ValueError('tethers must contain one nonnegative finite weight per stone')
    if pressure is not None:
        pressure = np.asarray(pressure, dtype=float)
        if pressure.shape != (2,) or not np.isfinite(pressure).all():
            raise ValueError('pressure must be a finite 2-vector')
    def objective():
        return energy(layout, gap, edge, reference, tethers, pressure, stone_margins=stone_margins, patio_margin=patio_margin)
    initial = current = objective()
    violations = validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin)
    if not violations and pressure is None:
        return RelaxationResult(initial, current, 0, 'valid', [])
    best_valid = layout.poses.copy() if not violations else None
    local = _LocalObjective(layout, gap, edge, reference, tethers, pressure, stone_margins, patio_margin)
    characteristic_length = np.sqrt(layout.patio.area)
    count = 0
    for count in range(1, max_steps + 1):
        improved = False
        for i, sid in enumerate(layout.ids):
            if not layout.active[i] or mobility[i] == 0:
                continue
            radius = max(np.sqrt(layout.shapes[sid].area), 1e-12 * characteristic_length)
            for axis in range(3):
                original = layout.poses[i].copy()
                delta = step * mobility[i] / (radius if axis == 2 else 1)
                best, best_pose, best_trial = current, None, None
                for sign in (-1, 1):
                    candidate_pose = original.copy()
                    candidate_pose[axis] += sign * delta
                    trial = local.trial(i, candidate_pose)
                    if trial[0] < best - 1e-15:
                        best, best_pose, best_trial = trial[0], candidate_pose, trial
                if best_pose is not None:
                    local.commit(i, best_pose, best_trial)
                    current = best
                    improved = True
        # Re-anchor to the full objective to prevent accumulated rounding drift.
        current = local.current = objective()
        if not validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin):
            return RelaxationResult(initial, current, count, 'valid', [])
        if not improved:
            step *= .5
            if step < 1e-6 * characteristic_length:
                break
    violations = validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin)
    if violations and best_valid is not None:
        layout.poses[:] = best_valid
        current = objective()
        return RelaxationResult(initial, current, count, 'restored_valid', [])
    return RelaxationResult(initial, current, count, 'stalled' if count < max_steps else 'step_limit', violations)


@dataclass
class RestartResult:
    initial_energy: float
    final_energy: float
    steps: int
    restarts_used: int
    reason: str
    violations: list


@_rollback_on_exception
def relax_with_restarts(layout, *, restarts=3, translation_noise=.05, angle_noise=.2,
                        gap=.01, edge=.01, max_steps=100, step=.02, mobility=None,
                        reference=None, tethers=None, stone_margins=None, patio_margin=0, tolerance=1e-9):
    """Try bounded perturbations after failed downhill relaxation.

    Each perturbation starts from the original poses; zero-mobility stones stay
    fixed. The best valid candidate wins, otherwise the lowest-energy candidate is
    retained. Perturbations consume the layout's RNG. This fallback is not an
    equilibrium Monte Carlo method, and may still fail on deep containment.
    """
    if not isinstance(restarts, int) or restarts < 0 or not all(np.isfinite(v) and v >= 0 for v in (translation_noise, angle_noise)):
        raise ValueError('restart count and noise bounds must be nonnegative')
    n = len(layout.ids)
    mobility = np.ones(n) if mobility is None else np.asarray(mobility, dtype=float)
    if mobility.shape != (n,) or not np.isfinite(mobility).all() or np.any(mobility < 0):
        raise ValueError('mobility must contain one finite nonnegative value per stone')
    origin = layout.poses.copy()
    reference = origin.copy() if reference is None else np.asarray(reference, dtype=float)
    first = relax(layout, gap=gap, edge=edge, max_steps=max_steps, step=step,
                  mobility=mobility, reference=reference, tethers=tethers,
                  stone_margins=stone_margins, patio_margin=patio_margin, tolerance=tolerance)
    best_poses = layout.poses.copy()
    best_energy = first.final_energy
    best_violations = first.violations
    steps = first.steps
    if not best_violations:
        return RestartResult(first.initial_energy, best_energy, steps, 0, 'valid', [])
    for attempt in range(1, restarts + 1):
        layout.poses[:] = origin
        for i in range(n):
            if layout.active[i] and mobility[i] > 0:
                layout.poses[i, :2] += layout.rng.uniform(-translation_noise, translation_noise, 2) * mobility[i]
                layout.poses[i, 2] += layout.rng.uniform(-angle_noise, angle_noise) * mobility[i]
        result = relax(layout, gap=gap, edge=edge, max_steps=max_steps, step=step,
                       mobility=mobility, reference=reference, tethers=tethers,
                  stone_margins=stone_margins, patio_margin=patio_margin, tolerance=tolerance)
        steps += result.steps
        if not result.violations:
            return RestartResult(first.initial_energy, result.final_energy, steps, attempt, 'valid', [])
        if result.final_energy < best_energy:
            best_poses, best_energy, best_violations = layout.poses.copy(), result.final_energy, result.violations
    layout.poses[:] = best_poses
    return RestartResult(first.initial_energy, best_energy, steps, restarts, 'restart_limit', best_violations)


@_rollback_on_exception
def relax_neighborhood(layout, *, mobility, gap=.01, edge=.01, max_steps=100,
                       step=.02, stone_margins=None, patio_margin=0, tolerance=1e-9):
    """Relax a nearby subset and copy its poses back to the full layout.

    Nearby fixed stones are retained as obstacles. The neighborhood is a
    heuristic, not a conservative bound for an entire multi-step trajectory.
    A final full-layout check supplies the returned violations; callers must
    roll back on failure. Energy values refer to the local subset.
    """
    from .phase1 import Layout
    mobility = np.asarray(mobility, dtype=float)
    n = len(layout.ids)
    if mobility.shape != (n,) or not np.isfinite(mobility).all() or np.any(mobility < 0):
        raise ValueError('mobility must contain one finite nonnegative value per stone')
    if not np.isfinite(step) or step <= 0:
        raise ValueError('step must be finite and positive')
    margins = {} if stone_margins is None else dict(stone_margins)
    if not set(margins).issubset(layout.ids) or not all(np.isfinite(v) and v >= 0 for v in margins.values()):
        raise ValueError('invalid stone margins')
    mobile = np.flatnonzero(layout.active & (mobility > 0))
    if not len(mobile):
        return relax(layout, gap, edge, max_steps=max_steps, step=step, mobility=mobility,
                     stone_margins=margins, patio_margin=patio_margin, tolerance=tolerance)
    radii = np.array([np.linalg.norm(np.asarray(layout.shapes[s].exterior.coords)[:, :2], axis=1).max() for s in layout.ids])
    margin_values = np.array([margins.get(s, 0) for s in layout.ids])
    active = np.flatnonzero(layout.active)
    delta = layout.poses[active, None, :2] - layout.poses[None, mobile, :2]
    distance = np.linalg.norm(delta, axis=2)
    threshold = radii[active, None] + radii[mobile][None, :] + margin_values[active, None] + margin_values[mobile][None, :] + gap + 2*step
    indices = active[np.any(distance <= threshold, axis=1)]
    ids = tuple(layout.ids[i] for i in indices)
    local = Layout(layout.patio, {sid:layout.shapes[sid] for sid in ids}, layout.poses[indices],
                   np.ones(len(indices), dtype=bool), ids, layout.rng, units=layout.units)
    local_margins = {sid:margins[sid] for sid in ids if sid in margins}
    result = relax(local, gap, edge, max_steps=max_steps, step=step, mobility=mobility[indices],
                   stone_margins=local_margins, patio_margin=patio_margin, tolerance=tolerance)
    layout.poses[indices] = local.poses
    global_violations = validate(layout, gap, edge, tolerance, stone_margins=margins, patio_margin=patio_margin)
    if global_violations:
        return RelaxationResult(result.initial_energy, result.final_energy, result.steps, 'global_invalid' if not result.violations else result.reason, global_violations)
    return RelaxationResult(result.initial_energy, result.final_energy, result.steps, result.reason, [])
