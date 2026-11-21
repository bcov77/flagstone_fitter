"""Validity-first coverage and optional sampled empty-space scores."""
from .diagnostics import empty_space
from .phase1 import validate


def score_layout(layout, *, gap=.01, edge=.01, tolerance=1e-9, stone_margins=None,
                 patio_margin=0, resolution=None, broad_distance=None, edge_strip_width=0):
    """Score a layout; invalid layouts have no usable coverage or quality score.

    Optional quality samples interior empty space on a grid. Its broad-area and
    maximum-distance values are diagnostics, not exact geometric guarantees or
    joint widths. Boundary-strip samples are reported separately by empty_space.
    """
    violations = validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin)
    if violations:
        return {'valid': False, 'violations': violations, 'coverage': None, 'secondary': None}
    coverage = sum(layout.shapes[sid].area for i, sid in enumerate(layout.ids) if layout.active[i]) / layout.patio.area
    secondary = None
    if resolution is not None:
        if broad_distance is None:
            raise ValueError('broad_distance is required when scoring empty space')
        grid = empty_space(layout, resolution=resolution, edge_strip_width=edge_strip_width,
                           broad_distance=broad_distance)
        interior = grid['interior']
        # No sampled empty points has zero sampled pocket area; an empty inventory
        # has undefined distances and is intentionally ranked worst by quality_key.
        secondary = {'resolution': resolution, 'broad_distance': broad_distance,
                     'edge_strip_width': edge_strip_width,
                     'interior_samples': interior['samples'],
                     'interior_broad_area_fraction': (interior['broad_area'] / layout.patio.area
                        if interior['broad_area'] is not None else (0.0 if interior['samples'] == 0 else None)),
                     'interior_max_distance': (interior['max_distance_to_stone']
                        if interior['max_distance_to_stone'] is not None else (0.0 if interior['samples'] == 0 else None))}
    elif broad_distance is not None:
        raise ValueError('resolution is required when scoring empty space')
    return {'valid': True, 'violations': [], 'coverage': float(coverage), 'secondary': secondary}


def quality_key(secondary):
    """Higher is better; broad area dominates maximum sampled distance."""
    if secondary is None:
        raise ValueError('secondary score is required')
    broad, maximum = secondary['interior_broad_area_fraction'], secondary['interior_max_distance']
    if broad is None or maximum is None:
        return (float('-inf'), float('-inf'))
    return (-broad, -maximum)


def compare_scores(a, b):
    """Compare exact coverage first, then equal-coverage secondary quality."""
    if a['valid'] != b['valid']:
        return 1 if a['valid'] else -1
    if not a['valid']:
        return 0
    if a['coverage'] != b['coverage']:
        return 1 if a['coverage'] > b['coverage'] else -1
    if (a['secondary'] is None) != (b['secondary'] is None):
        raise ValueError('equal-coverage scores need matching secondary settings')
    if a['secondary'] is None:
        return 0
    if any(a['secondary'][key] != b['secondary'][key] for key in ('resolution', 'broad_distance', 'edge_strip_width')):
        raise ValueError('secondary score settings must match')
    ka, kb = quality_key(a['secondary']), quality_key(b['secondary'])
    return (ka > kb) - (ka < kb)
