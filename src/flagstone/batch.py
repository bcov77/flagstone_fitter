"""CPU batch reference for independent candidate pose arrays.

This intentionally runs candidates serially. It defines data shapes and provides
an oracle for future vectorized or accelerator backends, not a speedup claim.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from .phase1 import Layout
from .relaxation import relax


@dataclass
class BatchResult:
    poses: np.ndarray
    active: np.ndarray
    results: list


def relax_batch(template, poses, active, *, mobility=None, gap=.01, edge=.01,
                max_steps=100, step=.02, stone_margins=None, patio_margin=0, tolerance=1e-9):
    """Relax independent [batch, stones, 3] poses and [batch, stones] masks.

    Shared shapes are never mutated and input arrays remain untouched. Each
    candidate uses the same CPU geometry and authoritative validator as relax().
    Mobility may be [stones] or [batch, stones].
    """
    poses = np.asarray(poses, dtype=float)
    active = np.asarray(active)
    n = len(template.ids)
    if poses.ndim != 3 or poses.shape[1:] != (n, 3) or not np.isfinite(poses).all():
        raise ValueError('poses must be finite with shape [batch, stones, 3]')
    batch = poses.shape[0]
    if active.dtype.kind != 'b' or active.shape != (batch, n):
        raise ValueError('active must be boolean with shape [batch, stones]')
    if mobility is not None:
        mobility = np.asarray(mobility, dtype=float)
        if mobility.shape == (n,):
            mobility = np.broadcast_to(mobility, (batch, n))
        if mobility.shape != (batch, n) or not np.isfinite(mobility).all() or np.any(mobility < 0):
            raise ValueError('mobility must be nonnegative with shape [stones] or [batch, stones]')
    result_poses = poses.copy()
    result_active = active.copy()
    results = []
    for i in range(batch):
        candidate = Layout(template.patio, template.shapes, result_poses[i], result_active[i],
                           template.ids, np.random.default_rng(0), units=template.units,
                           source_outlines=template.source_outlines)
        result = relax(candidate, gap=gap, edge=edge, max_steps=max_steps, step=step,
                       mobility=None if mobility is None else mobility[i],
                       stone_margins=stone_margins, patio_margin=patio_margin, tolerance=tolerance)
        result_poses[i] = candidate.poses
        results.append(result)
    return BatchResult(result_poses, result_active, results)


def validate_batch(template, poses, active, *, gap=.01, edge=.01, tolerance=1e-9,
                   stone_margins=None, patio_margin=0, max_batch_size=32):
    """Validate batches with NumPy broad phase and exact CPU polygon predicates.

    Returns one structured violation list per candidate. Inputs are not mutated.
    All candidates share the same geometry, clearances, and margin mapping.
    Broad-phase working arrays are limited to max_batch_size candidates at a
    time. None disables chunking. Returned violations can still grow with batch
    size and contact count.
    """
    from .broadphase import PaddedGeometry, candidate_pairs
    from .phase1 import _validate
    if not all(np.isfinite(v) and v >= 0 for v in (gap, edge, tolerance, patio_margin)):
        raise ValueError('Clearances, margins and tolerance must be finite and nonnegative')
    margins = {} if stone_margins is None else dict(stone_margins)
    if not set(margins).issubset(template.ids):
        raise ValueError('stone_margins contains an unknown stone ID')
    margin_array = np.array([margins.get(sid, 0) for sid in template.ids], dtype=float)
    if not np.isfinite(margin_array).all() or np.any(margin_array < 0):
        raise ValueError('margins must be nonnegative finite values')
    if max_batch_size is not None and (isinstance(max_batch_size, (bool, np.bool_)) or not isinstance(max_batch_size, Integral) or max_batch_size < 1):
        raise ValueError('max_batch_size must be a positive integer or None')
    poses, active = np.asarray(poses, dtype=float), np.asarray(active)
    n = len(template.ids)
    if poses.ndim != 3 or poses.shape[1:] != (n, 3) or not np.isfinite(poses).all():
        raise ValueError('poses must be finite with shape [batch, stones, 3]')
    if active.shape != (len(poses), n) or active.dtype.kind != 'b':
        raise ValueError('active must be boolean with shape [batch, stones]')
    padded = PaddedGeometry.from_layout(template)
    results = []
    if not len(poses):
        return results
    # Candidate state is copied once; immutable geometry is shared across the
    # batch, and each validation result is an independent list of violations.
    candidate = Layout(template.patio, template.shapes, poses[0], active[0], template.ids,
                       np.random.default_rng(0), units=template.units)
    chunk_size = len(poses) if max_batch_size is None else int(max_batch_size)
    for start in range(0, len(poses), chunk_size):
        stop = min(start + chunk_size, len(poses))
        pairs = candidate_pairs(padded, poses[start:stop], active[start:stop], gap=gap, margins=margin_array)
        for offset, pair in enumerate(pairs):
            i = start + offset
            candidate.poses[:] = poses[i]
            candidate.active[:] = active[i]
            results.append(_validate(candidate, gap, edge, tolerance, stone_margins=margins,
                                     patio_margin=patio_margin, candidate_pairs=pair))
    return results
