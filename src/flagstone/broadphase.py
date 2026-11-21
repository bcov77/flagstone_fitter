"""Vectorized CPU broad phase for batches of rigid polygon placements.

Bounding boxes only propose pairs. Actual contact and clearance must be checked
with the polygon validator. No GPU backend is implied by this NumPy prototype.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PaddedGeometry:
    ids: tuple
    vertices: np.ndarray  # [stones, max_vertices, 2]
    vertex_mask: np.ndarray  # [stones, max_vertices]

    @classmethod
    def from_layout(cls, layout):
        outlines = [np.asarray(layout.shapes[s].exterior.coords[:-1], dtype=float) for s in layout.ids]
        width = max((len(a) for a in outlines), default=0)
        vertices = np.zeros((len(outlines), width, 2), dtype=float)
        mask = np.zeros((len(outlines), width), dtype=bool)
        for i, points in enumerate(outlines):
            vertices[i, :len(points)] = points
            mask[i, :len(points)] = True
        vertices.flags.writeable = False
        mask.flags.writeable = False
        return cls(layout.ids, vertices, mask)


def candidate_pairs(geometry, poses, active, *, gap=0, margins=None):
    """Return per-batch index pairs whose expanded bounding boxes overlap.

    `poses` is [batch, stones, 3], `active` is a boolean [batch, stones] mask.
    `margins` can be [stones] or [batch, stones]. A small scale-dependent
    roundoff cushion makes the filter conservative near contacts. This function
    allocates O(batch*stones^2) boolean masks but no edge-pair arrays.
    """
    poses = np.asarray(poses, dtype=float)
    active = np.asarray(active)
    n = len(geometry.ids)
    if poses.ndim != 3 or poses.shape[1:] != (n, 3) or not np.isfinite(poses).all():
        raise ValueError('poses must be finite with shape [batch, stones, 3]')
    batch = poses.shape[0]
    if active.shape != (batch, n) or active.dtype.kind != 'b':
        raise ValueError('active must be boolean with shape [batch, stones]')
    if not np.isfinite(gap) or gap < 0:
        raise ValueError('gap must be finite and nonnegative')
    if margins is None:
        margins = np.zeros((batch, n))
    else:
        margins = np.asarray(margins, dtype=float)
        if margins.shape == (n,):
            margins = np.broadcast_to(margins, (batch, n))
        if margins.shape != (batch, n) or not np.isfinite(margins).all() or np.any(margins < 0):
            raise ValueError('margins must be nonnegative with shape [stones] or [batch, stones]')
    if n == 0:
        return [np.empty((0, 2), dtype=int) for _ in range(batch)]
    v = geometry.vertices
    angle = poses[:, :, 2, None]
    c, s = np.cos(angle), np.sin(angle)
    x = c * v[None, :, :, 0] - s * v[None, :, :, 1] + poses[:, :, 0, None]
    y = s * v[None, :, :, 0] + c * v[None, :, :, 1] + poses[:, :, 1, None]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('transformed vertices exceed finite numeric range')
    mask = geometry.vertex_mask[None, :, :]
    minx, maxx = np.min(np.where(mask, x, np.inf), axis=2), np.max(np.where(mask, x, -np.inf), axis=2)
    miny, maxy = np.min(np.where(mask, y, np.inf), axis=2), np.max(np.where(mask, y, -np.inf), axis=2)
    scale = max(1.0, float(np.max(np.abs(poses), initial=0)), float(np.max(np.abs(v), initial=0)), float(gap), float(np.max(margins, initial=0)))
    cushion = 64 * np.finfo(float).eps * scale
    radius = gap + margins[:, :, None] + margins[:, None, :] + cushion
    possible = ((maxx[:, :, None] + radius >= minx[:, None, :]) &
                (maxx[:, None, :] + radius >= minx[:, :, None]) &
                (maxy[:, :, None] + radius >= miny[:, None, :]) &
                (maxy[:, None, :] + radius >= miny[:, :, None]) &
                active[:, :, None] & active[:, None, :] & np.triu(np.ones((n, n), dtype=bool), 1))
    return [np.argwhere(possible[i]) for i in range(batch)]
