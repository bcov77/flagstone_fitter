"""Geometry diagnostics and conservative paired preferences; no physics or labels.

Feet internally, inches in named outputs. Samples are uniform in arc length;
the saved profiles expose where a mean would conceal a wedge or blocking bump.
Artificial pocket caps never count as matching physical edges.
"""
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point


def profile(line, spacing=.02):
    n = max(8, math.ceil(line.length / spacing))
    distances = (np.arange(n) + .5) * line.length / n
    points = shapely.line_interpolate_point(line, distances)
    return points, line.length / n, distances


def longest_run(mask, step, closed=True):
    mask = np.asarray(mask, dtype=bool)
    if not len(mask) or not mask.any():
        return 0.
    if mask.all():
        return len(mask) * step
    work = np.r_[mask, mask] if closed else mask
    best = current = 0
    for value in work:
        current = current + 1 if value else 0
        best = max(best, current)
    return min(best, len(mask)) * step


def tangent(line, positions, delta=.035):
    # Rings wrap at the seam rather than clipping the derivative there.
    a = shapely.get_coordinates(shapely.line_interpolate_point(line, (positions-delta) % line.length))
    b = shapely.get_coordinates(shapely.line_interpolate_point(line, (positions+delta) % line.length))
    v = b-a
    return v / np.maximum(np.linalg.norm(v, axis=1)[:, None], 1e-12)


def boundary_features(g, pocket, free, spacing=.02):
    """Distances to the SAME target for every piece; no candidate-defined mask."""
    pp, step, positions = profile(pocket.exterior, spacing)
    # Boundary introduced by morphological opening is a cap, not an obstacle.
    physical = np.asarray(shapely.distance(pp, free.boundary)) < .001
    distances = np.asarray(shapely.distance(pp, g)) * 12
    excess = distances[physical]
    nearest = shapely.get_point(shapely.shortest_line(pp, g.exterior), 1)
    gp = shapely.line_locate_point(g.exterior, nearest)
    alignment = np.abs(np.sum(tangent(pocket.exterior, positions) * tangent(g.exterior, gp), axis=1))
    matching = physical & (distances <= .5) & (alignment >= math.cos(math.radians(20)))
    # Mask caps out of contiguous deficit runs as well.
    unmatched = physical & ~matching
    simplified = pocket.simplify(.025, preserve_topology=True)
    xy = np.asarray(simplified.exterior.coords)[:-1]
    before = xy - np.roll(xy, 1, axis=0)
    after = np.roll(xy, -1, axis=0) - xy
    cosines = np.sum(before*after, axis=1) / np.maximum(np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1), 1e-12)
    turns = np.arccos(np.clip(cosines, -1, 1))
    cp = shapely.points(xy)
    # Both convex and concave substantial turns are diagnostics, not fitted labels.
    corner_mask = (turns >= math.radians(30)) & (np.asarray(shapely.distance(cp, free.boundary)) < .03)
    corners = np.asarray(shapely.distance(cp[corner_mask], g)) * 12
    pxy = shapely.get_coordinates(pp)
    gxy = shapely.get_coordinates(g.exterior)
    width_deficits = []
    for angle in np.arange(0, math.pi, math.pi/12):
        direction = np.array([math.cos(angle), math.sin(angle)])
        pproj, gproj = pxy@direction, gxy@direction
        # Positional support deficits distinguish one-sided gaps from centering.
        width_deficits.append(float(12*(max(0, gproj.min()-pproj.min()) + max(0, pproj.max()-gproj.max()))))
    # Morphological residual cores measure wide leftover space at two scales.
    # They do not assert a remaining inventory stone could occupy those cores.
    residual = pocket.difference(g)
    cores = {str(inches): residual.buffer(-inches/12) for inches in (.5, 1.)}
    candidate_points, _, _ = profile(g.exterior, spacing)
    candidate_dist = np.asarray(shapely.distance(candidate_points, free.boundary))*12
    return dict(
        edge_excess_q90_inches=float(np.quantile(excess, .9)) if len(excess) else None,
        edge_excess_mean_inches=float(excess.mean()) if len(excess) else None,
        matched_edge_fraction=float(matching.sum()/physical.sum()) if physical.any() else None,
        edge_angle_q75_degrees=float(np.degrees(np.quantile(np.arccos(np.clip(alignment[physical],0,1)),.75))) if physical.any() else None,
        longest_unmatched_edge_ft=longest_run(unmatched, step),
        corner_deficit_q90_inches=float(np.quantile(corners,.9)) if len(corners) else None,
        corner_count=int(len(corners)),
        support_deficit_q90_inches=float(np.quantile(width_deficits,.9)),
        support_deficits_inches=width_deficits,
        residual_core_half_inch_sqft=float(cores['0.5'].area),
        residual_core_one_inch_sqft=float(cores['1.0'].area),
        residual_core_one_inch_components=len([p for p in getattr(cores['1.0'],'geoms',[cores['1.0']]) if not p.is_empty and p.area > .0001]),
        candidate_boundary_distance_q90_inches=float(np.quantile(candidate_dist,.9)),
        boundary_profile=dict(xy_ft=pxy.tolist(), physical=physical.tolist(),
                              deficit_inches=distances.tolist(), matched=matching.tolist()),
    )


# Tolerances express an experimental indifference band, not learned human limits.
# Lower is better on every axis; no area bonus can offset an edge/corner loss.
AXES = {
    'edge_angle_q75_degrees': 3.,
    'clearance_intrusion_fraction': .0001,
    'max_clearance_intrusion_inches': .05,
    'edge_excess_q90_inches': .10,
    'longest_unmatched_edge_ft': .10,
    'corner_deficit_q90_inches': .10,
    'support_deficit_q90_inches': .10,
    'residual_core_half_inch_sqft': .01,
    'residual_core_one_inch_sqft': .01,
}


def comparison(candidate, reference, tolerance_scale=1.):
    deltas = {k: candidate[k]-reference[k] for k in AXES
              if candidate.get(k) is not None and reference.get(k) is not None}
    better = [k for k,d in deltas.items() if d < -AXES[k]*tolerance_scale]
    worse = [k for k,d in deltas.items() if d > AXES[k]*tolerance_scale]
    if not deltas:
        verdict = 'insufficient_geometry'
    elif better and not worse:
        verdict = 'candidate_dominates'
    elif worse and not better:
        verdict = 'reference_dominates'
    elif better and worse:
        verdict = 'tradeoff_review'
    else:
        verdict = 'indifferent_retain'
    return dict(verdict=verdict, deltas=deltas, better=better, worse=worse)


def screen(metrics, config):
    limits = config['screen']
    checks = {
        'physical_overlap': metrics['physical_overlap_sqft'] <= limits['physical_overlap_sqft'],
        'clearance_area': metrics['clearance_intrusion_fraction'] <= limits['clearance_intrusion_fraction'],
        'clearance_depth': metrics['max_clearance_intrusion_inches'] <= limits['clearance_depth_inches'],
        'silhouette_coverage': metrics['coverage'] >= limits['coverage'],
        'cap_overhang': metrics['outside_pocket_fraction'] <= limits['outside_pocket_fraction'],
    }
    return dict(status='passes_provisional_screen' if all(checks.values()) else 'reject', checks=checks)
