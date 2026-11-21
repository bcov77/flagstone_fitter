"""Private exact precheck for inserting one stone into an unchanged valid layout."""
import numpy as np
from shapely.geometry import box
from shapely.strtree import STRtree
from .phase1 import transformed


class _InsertionOracle:
    """Cache fixed geometry; rebuild whenever the active layout changes.

    This is a rejection filter only. The complete validator must still check any
    accepted proposal. The caller is responsible for supplying a valid layout
    and not mutating it for the lifetime of this instance.
    """
    def __init__(self, layout, *, gap, edge, tolerance=1e-9, stone_margins=None, patio_margin=0):
        self.layout = layout
        self.gap, self.edge, self.tolerance = gap, edge, tolerance
        self.margins = {} if stone_margins is None else dict(stone_margins)
        self.patio_margin = patio_margin
        self.inset = layout.patio.buffer(-(edge + patio_margin))
        self.indices = [i for i in range(len(layout.ids)) if layout.active[i]]
        self.ids = [layout.ids[i] for i in self.indices]
        self.geoms = [transformed(layout.shapes[sid], layout.poses[i]) for i, sid in enumerate(layout.ids) if layout.active[i]]
        self.tree = STRtree(self.geoms) if self.geoms else None
        self.max_margin = max((self.margins.get(sid, 0) for sid in self.ids), default=0)
        self._expanded_cache = {}

    def can_insert(self, index, pose):
        if self.layout.active[index]:
            raise ValueError('candidate stone must be inactive')
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError('candidate pose must contain three finite values')
        sid = self.layout.ids[index]
        geom = transformed(self.layout.shapes[sid], pose)
        if self.inset.is_empty or self.inset.geom_type == 'MultiPolygon':
            return False
        if not self.inset.covers(geom) or not self.layout.patio.covers(geom):
            return False
        required_edge = self.edge + self.patio_margin + self.margins.get(sid, 0)
        if geom.distance(self.layout.patio.boundary) < required_edge - self.tolerance:
            return False
        if self.tree is None:
            return True
        minx, miny, maxx, maxy = geom.bounds
        radius = self.gap + self.margins.get(sid, 0) + self.max_margin
        for j in self.tree.query(box(minx-radius, miny-radius, maxx+radius, maxy+radius)):
            j = int(j)
            other = self.geoms[j]
            required = self.gap + self.margins.get(sid, 0) + self.margins.get(self.ids[j], 0)
            distance = geom.distance(other)
            if distance < required - self.tolerance or (distance == 0 and geom.intersection(other).area > 0):
                return False
        return True

    def relaxation_penalty(self, index, pose):
        """Estimate normalized candidate boundary/pair violation area.

        Used only to rank or skip relaxation, never to certify validity. Polygon
        buffers approximate clearance and deep containment can be a plateau.
        """
        if self.layout.active[index]:
            raise ValueError('candidate stone must be inactive')
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError('candidate pose must contain three finite values')
        sid = self.layout.ids[index]
        geom = transformed(self.layout.shapes[sid], pose)
        margin = self.margins.get(sid, 0)
        inset = self.layout.patio.buffer(-(self.edge + self.patio_margin + margin))
        value = geom.difference(inset).area
        if self.tree is not None:
            minx, miny, maxx, maxy = geom.bounds
            radius = self.gap + margin + self.max_margin
            indices = self.tree.query(box(minx-radius, miny-radius, maxx+radius, maxy+radius))
            r = self.gap / 2 + margin
            expanded = geom.buffer(r) if r else geom
            for j in indices:
                j = int(j)
                other = self.geoms[j]
                other_r = self.gap / 2 + self.margins.get(self.ids[j], 0)
                if j not in self._expanded_cache:
                    self._expanded_cache[j] = other.buffer(other_r) if other_r else other
                other_expanded = self._expanded_cache[j]
                value += expanded.intersection(other_expanded).area + geom.intersection(other).area
        return float(value / geom.area)


    def nearby_indices(self, index, pose, *, extra=0):
        """Return placed stone indices near the candidate's required clearance."""
        if self.layout.active[index]:
            raise ValueError('candidate stone must be inactive')
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all() or not np.isfinite(extra) or extra < 0:
            raise ValueError('pose and extra distance must be finite and valid')
        if self.tree is None:
            return []
        sid = self.layout.ids[index]
        geom = transformed(self.layout.shapes[sid], pose)
        minx, miny, maxx, maxy = geom.bounds
        radius = self.gap + self.margins.get(sid, 0) + self.max_margin + extra
        candidates = self.tree.query(box(minx-radius, miny-radius, maxx+radius, maxy+radius))
        result = []
        for j in candidates:
            j = int(j)
            threshold = self.gap + self.margins.get(sid, 0) + self.margins.get(self.ids[j], 0) + extra
            if geom.distance(self.geoms[j]) <= threshold:
                result.append(self.indices[j])
        return sorted(result)
