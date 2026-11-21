"""Grid-based empty-space diagnostics; never used to certify validity."""
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
from .phase1 import transformed


def sampled_voids(layout, *, spacing, diameter):
    """Estimate empty circles and broad-gap core area on a regular grid.

    Distance is to the nearest stone OR physical patio boundary. The returned
    area counts sample cells whose center can hold a circle of the requested
    diameter; it is not total hole area or a count of distinct holes. This is a
    sampling diagnostic, not a guarantee that no larger unsampled hole exists.
    """
    import shapely
    if not np.isfinite(spacing) or spacing <= 0 or not np.isfinite(diameter) or diameter <= 0:
        raise ValueError('spacing and diameter must be finite and positive')
    minx,miny,maxx,maxy=layout.patio.bounds
    xs=np.arange(minx+spacing/2,maxx,spacing); ys=np.arange(miny+spacing/2,maxy,spacing)
    if len(xs)*len(ys)>2_000_000: raise ValueError('sampling grid exceeds two million points')
    if not len(xs) or not len(ys):
        return {'spacing':spacing,'diameter':diameter,'samples':0,'maximum_diameter':None,'largest_center':None,'broad_core_area':0.0,'broad_samples':0}
    gx,gy=np.meshgrid(xs,ys); points=shapely.points(gx.ravel(),gy.ravel())
    stones=[transformed(layout.shapes[sid],layout.poses[i]) for i,sid in enumerate(layout.ids) if layout.active[i]]
    inside=shapely.contains(layout.patio,points)
    union=shapely.union_all(stones) if stones else None
    if stones: inside &= ~shapely.covers(union,points)
    points=points[inside]
    if not len(points):
        return {'spacing':spacing,'diameter':diameter,'samples':0,'maximum_diameter':None,'largest_center':None,'broad_core_area':0.0,'broad_samples':0}
    distances=shapely.distance(points,layout.patio.boundary)
    if stones: distances=np.minimum(distances,shapely.distance(points,union))
    best=int(np.argmax(distances)); broad=int(np.count_nonzero(distances>=diameter/2))
    return {'spacing':float(spacing),'diameter':float(diameter),'samples':len(points),
            'maximum_diameter':float(2*distances[best]),
            'largest_center':[float(shapely.get_x(points[best])),float(shapely.get_y(points[best]))],
            'broad_core_area':float(broad*spacing**2),'broad_samples':broad}


def empty_space(layout, resolution=100, edge_strip_width=0, broad_distance=None):
    """Sample empty points, separating a physical boundary strip when requested.

    Areas are grid-cell estimates, not certified polygon areas. Distances measure
    only proximity to stones, not joint widths. An empty inventory has undefined
    distance-to-stone statistics.
    """
    if not isinstance(resolution, int) or resolution < 2:
        raise ValueError('resolution must be an integer >= 2')
    if not np.isfinite(edge_strip_width) or edge_strip_width < 0:
        raise ValueError('edge_strip_width must be finite and nonnegative')
    if broad_distance is not None and (not np.isfinite(broad_distance) or broad_distance < 0):
        raise ValueError('broad_distance must be finite and nonnegative')
    minx, miny, maxx, maxy = layout.patio.bounds
    dx, dy = (maxx-minx)/resolution, (maxy-miny)/resolution
    xs = np.linspace(minx, maxx, resolution, endpoint=False) + dx/2
    ys = np.linspace(miny, maxy, resolution, endpoint=False) + dy/2
    stones = [transformed(layout.shapes[sid], layout.poses[i]) for i, sid in enumerate(layout.ids) if layout.active[i]]
    union = unary_union(stones)
    distances, interior, edge = [], [], []
    for y in ys:
        for x in xs:
            p = Point(x, y)
            if layout.patio.covers(p) and not union.covers(p):
                d = p.distance(union) if stones else None
                distances.append(d)
                (edge if p.distance(layout.patio.boundary) < edge_strip_width else interior).append(d)
    def summary(samples):
        values = np.asarray([d for d in samples if d is not None])
        return {'samples': len(samples), 'estimated_area': len(samples)*dx*dy,
                'max_distance_to_stone': float(values.max()) if len(values) else None,
                'distance_quantiles': {str(q): float(np.quantile(values, q)) for q in (.5, .9, .99)} if len(values) else None,
                'broad_area': float(np.count_nonzero(values >= broad_distance)*dx*dy) if broad_distance is not None and len(values) else None}
    all_samples = summary(distances)
    return {'resolution': resolution, 'empty_samples': all_samples['samples'],
            'max_distance_to_stone': all_samples['max_distance_to_stone'],
            'distance_quantiles': all_samples['distance_quantiles'],
            'edge_strip_width': edge_strip_width, 'broad_distance': broad_distance,
            'interior': summary(interior), 'edge_strip': summary(edge)}


def layout_metrics(layout):
    """Physical area and measured clearance diagnostics for a layout.

    Clearances are nonnegative distances; outside stones have undefined edge
    clearance and their IDs are listed separately. The validator is authoritative.
    """
    stones = [(sid, transformed(layout.shapes[sid], layout.poses[i]))
              for i, sid in enumerate(layout.ids) if layout.active[i]]
    pair_distances = [a.distance(b) for i, (_, a) in enumerate(stones) for _, b in stones[i + 1:]]
    outside = [sid for sid, g in stones if not layout.patio.covers(g)]
    edge_distances = [g.distance(layout.patio.boundary) for _, g in stones if layout.patio.covers(g)]
    area = sum(g.area for _, g in stones)
    unused = [layout.shapes[sid].area for i, sid in enumerate(layout.ids) if not layout.active[i]]
    # Union/intersection remains meaningful even for an invalid overlapping layout.
    covered_area = unary_union([g for _, g in stones]).intersection(layout.patio).area if stones else 0.0
    return {'patio_area': float(layout.patio.area), 'placed_area': float(area),
            'coverage': float(area / layout.patio.area), 'placed_count': len(stones),
            'unused_count': len(layout.ids) - len(stones),
            'unused_area': float(sum(unused)),
            'smallest_unused_area': float(min(unused)) if unused else None,
            'uncovered_area': float(max(0, layout.patio.area - covered_area)),
            'minimum_pair_distance': float(min(pair_distances)) if pair_distances else None,
            'minimum_edge_distance': float(min(edge_distances)) if edge_distances and not outside else None,
            'outside_stone_ids': outside}


def sampled_neighbor_distances(layout, *, spacing, max_distance):
    """Sample directed boundary-to-nearest-other-stone distances.

    Points are evenly spaced around each stone perimeter, weighted approximately
    by boundary length. Only distances no greater than max_distance are reported;
    the excluded sample count is explicit. These are proximity proxies, not
    measurements of full joint widths or a substitute for polygon validation.
    """
    from shapely.geometry import box
    from shapely.strtree import STRtree
    if not np.isfinite(spacing) or spacing <= 0 or not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError('spacing and max_distance must be finite and positive')
    ids = [sid for i, sid in enumerate(layout.ids) if layout.active[i]]
    geoms = [transformed(layout.shapes[sid], layout.poses[i]) for i, sid in enumerate(layout.ids) if layout.active[i]]
    tree = STRtree(geoms) if geoms else None
    distances, per_stone = [], {}
    total = 0
    for i, (sid, geom) in enumerate(zip(ids, geoms)):
        boundary = geom.exterior
        count = max(1, int(np.ceil(boundary.length / spacing)))
        matched = 0
        for k in range(count):
            p = boundary.interpolate((k + .5) * boundary.length / count)
            candidates = tree.query(box(p.x-max_distance, p.y-max_distance,
                                        p.x+max_distance, p.y+max_distance))
            nearest = min((p.distance(geoms[int(j)]) for j in candidates if j != i), default=float('inf'))
            if nearest <= max_distance:
                distances.append(nearest)
                matched += 1
        total += count
        per_stone[sid] = {'samples': count, 'matched': matched,
                          'actual_spacing': float(boundary.length / count)}
    values = np.asarray(distances)
    return {'requested_spacing': spacing, 'max_distance': max_distance,
            'total_samples': total, 'matched_samples': len(distances),
            'unmatched_samples': total - len(distances), 'per_stone': per_stone,
            'minimum_distance': float(values.min()) if len(values) else None,
            'maximum_distance': float(values.max()) if len(values) else None,
            'distance_quantiles': {str(q): float(np.quantile(values, q)) for q in (.1, .5, .9, .99)} if len(values) else None}
