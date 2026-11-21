"""Necessary area bound for excluding a given empty-circle diameter.

Read-only; no search. Uses convex hulls and analytic capsule areas so polygon
buffer discretization cannot explain a failed bound. Coordinates must be feet.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from shapely.geometry import Polygon


def inventory_bound(path, diameter_in=2):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data['units'] != 'ft' or diameter_in <= 0:
        raise ValueError('Requires feet and a positive circle diameter')
    patio = Polygon(data['patio'])
    shapes = [Polygon(s.get('source_outline', s['outline'])) for s in data['stones']]
    if not patio.is_valid or any(not s.is_valid for s in shapes):
        raise ValueError('Requires valid source polygons')
    radius = diameter_in / 24
    hulls = [s.convex_hull for s in shapes]
    # Each radius-r neighborhood of a stone is contained in that of its hull.
    # Steiner's formula is exact for convex polygons, independent of pose.
    available_upper = sum(h.area + radius*h.length + math.pi*radius**2 for h in hulls)
    # The boundary neighborhood is contained in the union of segment capsules.
    # Subtract their summed FULL areas, including portions outside the patio:
    # this deliberately underestimates the interior needing stone coverage.
    edges = len(patio.exterior.coords)-1
    boundary_upper = 2*radius*patio.length + edges*math.pi*radius**2
    required_lower = max(0, patio.area-boundary_upper)
    return dict(
        source=str(path), source_sha256=hashlib.sha256(raw).hexdigest(),
        stones=len(shapes), diameter_in=diameter_in, patio_area_sqft=patio.area,
        stone_area_sqft=sum(s.area for s in shapes),
        stone_neighborhood_union_upper_sqft=available_upper,
        required_interior_lower_sqft=required_lower,
        shortfall_lower_sqft=max(0, required_lower-available_upper),
        necessary_condition_passed=available_upper >= required_lower,
        interpretation='Failure rules out the full patio with this inventory and diameter; '
                       'passing is inconclusive. Ordinary floating-point arithmetic, '
                       'not an interval-arithmetic certificate. No prefix or clearance '
                       'restrictions are needed for this necessary condition.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='examples/real_stones_concept/layout.json')
    parser.add_argument('--diameter-in', type=float, default=2)
    args = parser.parse_args()
    print(json.dumps(inventory_bound(args.input, args.diameter_in), indent=2))
