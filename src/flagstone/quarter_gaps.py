"""Edge/interior gap circles from the complete polygon-segment Voronoi diagram.

A gap is a local maximum of distance to the patio boundary and stone polygons.
Nonmaximal Voronoi vertices are excluded only when a common ascent direction
exists for all incident nearest features. Flat/degenerate candidates are retained
conservatively. Numerical certificates require agreement of two integer scales;
classification ambiguity cannot certify a passing layout.
"""
import hashlib
import math
import numpy as np
import shapely
import pyvoronoi
from .phase1 import load_json, validate
from .quarter import world, check_reference, write

RULE = 'local-gap-circles-edge5-interior2-v1'
TOL_IN = .002


def _diagram(patio, stones, scale):
    diagram = pyvoronoi.Pyvoronoi(scale)
    for poly in [patio, *stones]:
        coords = np.asarray(poly.exterior.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            if np.array_equal(np.rint(a*scale), np.rint(b*scale)):
                raise ValueError('A reference segment collapses at Voronoi integer scale')
            diagram.AddSegment([a.tolist(), b.tolist()])
    diagram.Construct()
    return diagram


def gap_candidates(patio, stones, *, scale=100_000_000, local_only=True):
    """Enumerate gap candidates from segment sites, not sampled boundary points."""
    if not stones:
        stones = []
    diagram = _diagram(patio, stones, scale)
    vertices = diagram.GetVertices(); cells = diagram.GetCells()
    incident = [set() for _ in vertices]
    for _, edge in diagram.EnumerateEdges():
        for index in (edge.start, edge.end):
            if index >= 0:
                incident[index].add(edge.cell)
    # A constant-radius medial segment is a plateau of locally largest circles.
    # Its endpoints can lead uphill, so include its midpoint independently.
    flat_points=[]
    for _, edge in diagram.EnumerateEdges():
        if not edge.is_primary or not edge.is_linear or edge.start<0 or edge.end<0:
            continue
        ca=cells[edge.cell];cb=cells[diagram.GetEdge(edge.twin).cell]
        if not (ca.contains_segment and cb.contains_segment):continue
        a,b=np.asarray(diagram.RetrieveScaledSegment(ca));c,d=np.asarray(diagram.RetrieveScaledSegment(cb))
        u,v=b-a,d-c
        if abs(u[0]*v[1]-u[1]*v[0])<=1e-10*np.linalg.norm(u)*np.linalg.norm(v):
            p,q=vertices[edge.start],vertices[edge.end]
            flat_points.append(([.5*(p.X+q.X),.5*(p.Y+q.Y)],{edge.cell,diagram.GetEdge(edge.twin).cell}))
    points = np.asarray([[v.X, v.Y] for v in vertices]+[p for p,_ in flat_points])
    incident += [cs for _,cs in flat_points]
    if not len(points):
        raise ValueError('No Voronoi vertices')
    pp = shapely.points(points)
    boundary_ds = shapely.distance(pp, patio.boundary)
    stone_ds = np.full(len(pp), np.inf)
    if stones:
        tree = shapely.STRtree(stones)
        pairs, ds = tree.query_nearest(pp, return_distance=True, all_matches=False)
        stone_ds[pairs[0]] = ds
    result = []
    inside = shapely.covers(patio, pp)
    for index, point in enumerate(points):
        radius = min(boundary_ds[index], stone_ds[index])
        if not inside[index] or radius <= 1e-8:
            continue
        directions = []
        for ci in incident[index]:
            cell = cells[ci]
            if cell.contains_point:
                contact = np.asarray(diagram.RetrieveScaledPoint(cell))
            else:
                a, b = np.asarray(diagram.RetrieveScaledSegment(cell))
                contact = a+np.clip(np.dot(point-a, b-a)/np.dot(b-a, b-a), 0, 1)*(b-a)
            delta = point-contact
            if np.linalg.norm(delta)>1e-12:
                directions.append(math.atan2(delta[1], delta[0]))
        angles = sorted(directions)
        max_angle = max(np.diff([*angles, angles[0]+2*math.pi])) if angles else 2*math.pi
        # Nearest-feature gradients in an open semicircle admit strict ascent.
        if local_only and max_angle > math.pi+1e-7:
            continue
        boundary_slack_in = 12*(boundary_ds[index]-radius)
        kind = 'edge' if boundary_slack_in <= TOL_IN/10 else 'interior'
        result.append(dict(center=point.tolist(), diameter_in=float(24*radius), kind=kind,
                           boundary_separation_in=float(boundary_slack_in),
                           flat_candidate=bool(abs(max_angle-math.pi)<1e-7)))
    return sorted(result, key=lambda x:x['diameter_in'], reverse=True)


def audit(patio, stones, *, tolerance_in=TOL_IN):
    """Numerically bounded dual-scale segment-Voronoi audit of local gap circles.

    The padding is a declared numerical tolerance, not a quadtree enclosing
    proof. Passing requires two complete segment diagrams to agree, original
    geometry clearance checks, and both padded maxima within their limits.
    """
    first = gap_candidates(patio, stones, scale=10_000_000)
    second = gap_candidates(patio, stones, scale=100_000_000)
    groups = {}
    stable = True
    for kind, limit in [('edge', 5), ('interior', 2)]:
        a = next((c for c in first if c['kind']==kind), None)
        b = next((c for c in second if c['kind']==kind), None)
        stable &= (a is None)==(b is None)
        if a and b:
            stable &= abs(a['diameter_in']-b['diameter_in']) <= tolerance_in/2
        maximum = max((x['diameter_in'] for x in (a,b) if x), default=0.)
        groups[kind] = dict(limit_in=limit, lower_diameter_in=max(0., maximum-tolerance_in),
                            upper_diameter_in=maximum+tolerance_in if b else 0.,
                            center=b['center'] if b else None, candidates=sum(c['kind']==kind for c in second))
    passing = bool(stable and all(g['upper_diameter_in']<=g['limit_in'] for g in groups.values()))
    return dict(rule=RULE, groups=groups, stable=bool(stable), numerical_pass=passing,
                certified_pass=passing, certification='complete segment-Voronoi numerical verification, dual integer scales, padded tolerance; not an interval-arithmetic proof',
                candidates=second, tolerance_in=tolerance_in,
                normalized_upper=max(g['upper_diameter_in']/g['limit_in'] for g in groups.values()),
                normalized_lower=max(g['lower_diameter_in']/g['limit_in'] for g in groups.values()))


def verify(path, reference, output):
    layout=load_json(path); check_reference(layout, load_json(reference))
    errors=validate(layout, gap=.5/12, edge=.5/12)
    gaps=audit(layout.patio,list(world(layout).values()))
    result=dict(rule=RULE, layout_sha256=hashlib.sha256(__import__('pathlib').Path(path).read_bytes()).hexdigest(),
                original_geometry=True, reference_matched=True, clearance_errors=errors,
                passed=not errors and gaps['certified_pass'], numerical_pass=not errors and gaps['numerical_pass'],
                visual_review='pending', gaps=gaps)
    write(output,result);return result
