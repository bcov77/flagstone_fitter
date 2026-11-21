"""Reproducible synthetic inventories and known feasible witnesses."""
import numpy as np
from shapely.geometry import MultiPoint
from .phase1 import Layout, local_shape, polygon, validate


def cut_witness(seed=17, rows=4, cols=4, clearance=.01):
    """Jitter a grid of shared vertices, then inset every quadrilateral."""
    if rows < 1 or cols < 1 or clearance <= 0:
        raise ValueError('rows, cols and clearance must be positive')
    rng = np.random.default_rng(seed)
    points = np.zeros((rows + 1, cols + 1, 2))
    for r in range(rows + 1):
        for c in range(cols + 1):
            points[r, c] = [c / cols, r / rows]
            if 0 < c < cols: points[r, c, 0] += rng.uniform(-.15, .15) / cols
            if 0 < r < rows: points[r, c, 1] += rng.uniform(-.15, .15) / rows
    shapes, poses, ids = {}, [], []
    for r in range(rows):
        for c in range(cols):
            piece = polygon([points[r,c], points[r,c+1], points[r+1,c+1], points[r+1,c]])
            piece = piece.buffer(-clearance, join_style='mitre')
            if piece.is_empty or piece.geom_type != 'Polygon':
                raise ValueError('clearance makes a piece vanish or split')
            sid = f'S{len(ids):03d}'; center = piece.centroid
            shapes[sid] = local_shape(piece.exterior.coords); poses.append([center.x, center.y, 0]); ids.append(sid)
    layout = Layout(polygon([(0,0),(1,0),(1,1),(0,1)]), shapes, np.array(poses), np.ones(len(ids), bool), tuple(ids), rng)
    if validate(layout, gap=clearance, edge=clearance - 1e-9):
        raise ValueError('generated witness failed clearance validation')
    return layout


def inventory(seed=17, counts=(6, 12, 24), areas=(.03, .012, .003), mixed=True, vertex_profile="basic"):
    """Generate inactive random stones; counts correspond to target area bins."""
    if len(counts) != len(areas) or any(int(n) != n or n < 0 for n in counts) or any(a <= 0 for a in areas):
        raise ValueError('counts and areas must have matching valid entries')
    if vertex_profile not in ('basic', 'detailed', 'varied'):
        raise ValueError('vertex_profile must be basic, detailed, or varied')
    rng = np.random.default_rng(seed); shapes = {}; ids = []
    from shapely.affinity import scale
    for count, area in zip(counts, areas):
        for _ in range(count):
            kind = int(rng.integers(10)) if vertex_profile == 'varied' else None
            if kind in (0, 1):
                # Retain genuinely straight-edged square and rectangle examples.
                aspect = 1. if kind == 0 else rng.uniform(1.3, 2.2)
                hull = polygon([(0,0), (aspect,0), (aspect,1), (0,1)])
            elif kind in (2, 3):
                # A synthetic state-like outline with a broad, shallow inlet.
                vertices = [(0,0),(.23,-.01),(.45,.02),(.68,0),(1,.03),
                            (1,.2),(.96,.3),(.68,.3),(.67,.45),(.71,.52),
                            (.99,.54),(1,.73),(.97,1),(.75,.98),(.55,1),
                            (.35,.98),(.1,1),(.03,.8),(.02,.6),(0,.3)]
                hull = polygon(vertices)
                if not mixed:
                    hull = hull.convex_hull
                hull = scale(hull, xfact=rng.uniform(.7, 1.4), yfact=1)
            else:
                if vertex_profile in ('detailed', 'varied'):
                    angles = np.linspace(0, 2*np.pi, 30, endpoint=False)
                    radial = rng.uniform(.96, 1.04, len(angles))
                    pts = np.column_stack((radial * np.cos(angles), radial * np.sin(angles)))
                else:
                    pts = rng.normal(size=(12, 2))
                hull = MultiPoint(pts).convex_hull
                hull = scale(hull, xfact=rng.uniform(.5, 1.5), yfact=1)
                if mixed and rng.random() < .25:
                    vertices = list(hull.exterior.coords)[:-1]
                    a, b = np.asarray(vertices[0]), np.asarray(vertices[1]); center = np.asarray(hull.centroid.coords[0])
                    vertices.insert(1, tuple(.55 * ((a+b)/2) + .45 * center))
                    hull = polygon(vertices)
            hull = scale(hull, xfact=np.sqrt(area/hull.area), yfact=np.sqrt(area/hull.area), origin='centroid')
            sid = f'S{len(ids):03d}'; ids.append(sid); shapes[sid] = local_shape(hull.exterior.coords)
    return Layout(polygon([(0,0),(1,0),(1,1),(0,1)]), shapes, np.zeros((len(ids), 3)), np.zeros(len(ids), bool), tuple(ids), rng)


def inventory_summary(layout, bins):
    areas = np.array([layout.shapes[s].area for s in layout.ids])
    counts, edges = np.histogram(areas, bins=bins)
    return {'count': len(areas), 'total_area': float(areas.sum()), 'bins': edges.tolist(), 'counts': counts.tolist()}


def flagstone_inventory(seed=17, counts=(6, 50, 100), areas=(.025, .008, .003)):
    """Synthetic angular stones inspired by the reference photograph, not traced measurements.

    Templates are deliberately varied in silhouette, then stretched and scaled;
    the area classes are experimental fractions of a unit-square patio.
    """
    if len(counts) != len(areas) or any(int(n) != n or n < 0 for n in counts) or any(not np.isfinite(a) or a <= 0 for a in areas):
        raise ValueError('counts and areas must have matching valid entries')
    templates = [
        [(0,.1),(.13,0),(.84,.04),(1,.18),(.95,.86),(.72,1),(.1,.88)],
        [(0,.2),(.24,.02),(.89,0),(1,.13),(.94,.78),(.62,1),(.08,.89)],
        [(0,.15),(.23,0),(.82,.08),(1,.48),(.75,.95),(.26,1),(.04,.68)],
        [(0,0),(.91,.04),(1,.26),(.86,1),(.15,.89)],
        [(.06,.09),(.8,0),(1,.33),(.92,.83),(.42,1),(0,.72)],
        [(0,.05),(.77,0),(1,.22),(.92,.9),(.22,1),(.04,.76)],
        [(.04,.06),(.8,0),(1,.19),(.88,.43),(.97,.79),(.55,1),(0,.84)],
        [(0,.15),(.22,0),(.95,.12),(.88,.89),(.38,1),(.11,.82)],
        [(.08,0),(.9,.08),(1,.7),(.72,.95),(.15,1),(0,.68)],
        [(0,.02),(.84,0),(1,.26),(.88,.48),(.94,.89),(.12,1),(.03,.67)],
        [(.08,.02),(.92,0),(1,.95),(.36,1),(0,.78)],
        [(0,.17),(.48,0),(.92,.12),(1,.76),(.73,1),(.12,.9)],
    ]
    from shapely.affinity import rotate, scale
    rng = np.random.default_rng(seed)
    shapes, ids = {}, []
    for count, area in zip(counts, areas):
        for _ in range(count):
            shape = polygon(templates[int(rng.integers(len(templates)))])
            shape = scale(shape, xfact=rng.uniform(.8, 1.35), yfact=rng.uniform(.8, 1.2), origin='centroid')
            shape = rotate(shape, rng.uniform(0, 360), origin='centroid')
            factor = np.sqrt(area / shape.area)
            shape = scale(shape, xfact=factor, yfact=factor, origin='centroid')
            sid = f'S{len(ids):03d}'
            ids.append(sid)
            shapes[sid] = local_shape(shape.exterior.coords)
    return Layout(polygon([(0,0),(1,0),(1,1),(0,1)]), shapes,
                  np.zeros((len(ids), 3)), np.zeros(len(ids), bool), tuple(ids), rng)


def benchmark_edge_cases(seed=17):
    """Return a concave witness and two deliberately impossible inventories.

    The four-square inventory has total stone area greater than patio area, so
    placing all of it without overlap is impossible. The oversized square has
    larger area than the patio. Both inventories start entirely inactive.
    """
    from .phase1 import synthetic
    concave = synthetic(seed)
    concave.patio = polygon([(0,0), (1,0), (1,.5), (.5,.5), (.5,1), (0,1)])
    for row in range(4):
        for col in range(4):
            if row >= 2 and col >= 2:
                concave.active[row * 4 + col] = False
    if validate(concave, gap=.01, edge=.01):
        raise ValueError('Concave benchmark witness failed validation')

    def squares(sizes):
        shapes = {}
        for i, size in enumerate(sizes):
            h = size / 2
            shapes[f'S{i:03d}'] = local_shape([(-h,-h), (h,-h), (h,h), (-h,h)])
        ids = tuple(shapes)
        return Layout(polygon([(0,0), (1,0), (1,1), (0,1)]), shapes,
                      np.zeros((len(ids), 3)), np.zeros(len(ids), bool), ids,
                      np.random.default_rng(seed))
    return {'concave_witness': concave, 'impossible_inventory': squares([.55] * 4),
            'oversized_stone': squares([1.2])}
