"""Similarity calibration for measured outlines and control points."""
from dataclasses import dataclass
import numpy as np
from shapely.geometry import MultiPoint, Point
from .phase1 import Layout, local_shape, polygon


@dataclass(frozen=True)
class SimilarityCalibration:
    """Map source points to target units using reflection, scale, rotation, translation."""
    scale: float
    rotation: tuple
    translation: tuple
    reflect_source_y: bool
    residuals: tuple
    source_controls: tuple
    target_controls: tuple

    def apply(self, points):
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
            raise ValueError('points must be finite 2D coordinates')
        arr = arr.copy()
        if self.reflect_source_y:
            arr[:, 1] *= -1
        return self.scale * (arr @ np.asarray(self.rotation)) + np.asarray(self.translation)

    def report(self):
        return {'model': '2d_similarity', 'scale': self.scale,
                'rotation_matrix_row_vectors': [list(row) for row in self.rotation],
                'translation': list(self.translation), 'reflect_source_y': self.reflect_source_y,
                'source_controls': [list(p) for p in self.source_controls],
                'target_controls': [list(p) for p in self.target_controls],
                'residuals': list(self.residuals), 'max_residual': max(self.residuals),
                'rms_residual': float(np.sqrt(np.mean(np.square(self.residuals))))}


def _control_coverage(calibration, outline):
    """Describe source-outline extrapolation beyond the controls' convex hull."""
    hull = MultiPoint(calibration.source_controls).convex_hull
    vertices = list(outline.exterior.coords[:-1])
    distances = [Point(vertex).distance(hull) for vertex in vertices]
    outside = [index for index, vertex in enumerate(vertices) if not hull.covers(Point(vertex))]
    return {'outline_vertices': len(vertices), 'outside_vertex_indices': outside,
            'outside_vertex_count': len(outside),
            'max_distance_outside_hull_source_units': float(max(distances, default=0)),
            'max_distance_outside_hull_target_units': float(max(distances, default=0) * calibration.scale)}


def fit_similarity(source, target, *, reflect_source_y=False, max_residual):
    """Fit a uniform-scale, proper-rotation map and enforce a residual limit.

    At least three non-collinear control points are required in each frame. The
    reflection flag is explicit, useful for images whose y axis points downward.
    A small residual checks the controls only; it cannot prove an image has no
    perspective or lens distortion elsewhere.
    """
    if not isinstance(reflect_source_y, bool):
        raise ValueError('reflect_source_y must be boolean')
    if not np.isfinite(max_residual) or max_residual < 0:
        raise ValueError('max_residual must be finite and nonnegative')
    x, y = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if x.ndim != 2 or x.shape[1:] != (2,) or x.shape != y.shape or len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('at least three paired finite 2D control points are required')
    original = x.copy()
    x = x.copy()
    if reflect_source_y:
        x[:, 1] *= -1
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x-xm, y-ym
    if np.linalg.matrix_rank(xc) < 2 or np.linalg.matrix_rank(yc) < 2:
        raise ValueError('control points must be non-collinear in both frames')
    u, singular, vt = np.linalg.svd(xc.T @ yc)
    correction = np.diag([1., np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    scale = float(np.sum(singular * np.diag(correction)) / np.sum(xc*xc))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('control points do not define a positive scale')
    translation = ym - scale * (xm @ rotation)
    residuals = np.linalg.norm(scale * (x @ rotation) + translation - y, axis=1)
    if max(residuals) > max_residual:
        raise ValueError(f'control point residual {max(residuals):.6g} exceeds limit {max_residual:.6g}')
    return SimilarityCalibration(scale, tuple(map(tuple, rotation)), tuple(translation), reflect_source_y,
                                 tuple(map(float, residuals)), tuple(map(tuple, original)), tuple(map(tuple, y)))


def calibrate_inventory(data, *, max_residual, seed=0):
    """Convert raw rings in one frame to a layout and a traceable calibration report."""
    if data.get('measurement_version') != 1:
        raise ValueError('Unsupported measurement_version')
    if 'holes' in data or 'patio_holes' in data:
        raise ValueError('Patio holes are unsupported')
    controls = data['control_points']
    calibration = fit_similarity([p['source'] for p in controls], [p['target'] for p in controls],
                                 reflect_source_y=data.get('reflect_source_y', False), max_residual=max_residual)
    if not isinstance(data.get('source_units'), str) or not data['source_units'].strip():
        raise ValueError('source_units must describe the input coordinate frame')
    source_patio = polygon(data['patio'])
    patio = polygon(calibration.apply(data['patio']))
    shapes, poses, active, ids, source_outlines, source_reports = {}, [], [], [], {}, []
    for stone in data['stones']:
        if 'holes' in stone:
            raise ValueError('Stone holes are unsupported')
        sid = stone['id']
        if not isinstance(sid, str) or not sid or sid in shapes:
            raise ValueError('stone IDs must be nonempty unique strings')
        if not isinstance(stone['active'], bool):
            raise ValueError('active must be boolean')
        raw = polygon(stone['outline'])
        outline = calibration.apply(np.asarray(raw.exterior.coords[:-1]))
        calibrated = polygon(outline)
        center = calibrated.centroid
        shapes[sid] = local_shape(outline)
        poses.append([center.x, center.y, 0.])
        active.append(stone['active']); ids.append(sid)
        source_outlines[sid] = outline.tolist()
        source_reports.append({'id': sid, 'outline': stone['outline'],
                               'control_coverage': _control_coverage(calibration, raw)})
    layout = Layout(patio, shapes, np.asarray(poses, float).reshape((-1, 3)), np.asarray(active, bool), tuple(ids),
                    np.random.default_rng(seed), units=data['units'], source_outlines=source_outlines)
    report = calibration.report()
    report.update({'source_units': data['source_units'], 'target_units': layout.units,
                   'allowed_max_residual': max_residual, 'source_patio': data['patio'],
                   'patio_control_coverage': _control_coverage(calibration, source_patio),
                   'source_stones': source_reports,
                   'warning': 'Residuals cover control points only; unmodeled perspective and distortion are not bounded.'})
    return layout, report


def calibrate_bundle(data, *, seed=0):
    """Calibrate a patio and independently measured stones into common units.

    Each component supplies its own control points, source units, explicit
    maximum residual, and outline. Patio target controls define patio coordinates;
    each stone's target controls define its local measurement frame. Stone poses
    use centroid x/y and counterclockwise angle in the patio frame. Active stones
    require an explicit pose; inactive stones default to zero poses.
    """
    if data.get('measurement_bundle_version') != 1:
        raise ValueError('Unsupported measurement_bundle_version')
    def component(item):
        if 'holes' in item:
            raise ValueError('Polygon holes are unsupported')
        if not isinstance(item.get('source_units'), str) or not item['source_units'].strip():
            raise ValueError('source_units must describe each input frame')
        controls = item['control_points']
        limit = item['max_residual']
        cal = fit_similarity([p['source'] for p in controls], [p['target'] for p in controls],
                             reflect_source_y=item.get('reflect_source_y', False), max_residual=limit)
        raw = polygon(item['outline'])
        result = polygon(cal.apply(np.asarray(raw.exterior.coords[:-1])))
        report = cal.report()
        report.update({'source_units': item['source_units'], 'allowed_max_residual': limit,
                       'source_outline': item['outline'], 'control_coverage': _control_coverage(cal, raw)})
        return result, report
    patio, patio_report = component(data['patio'])
    shapes, ids, poses, active, outlines, reports = {}, [], [], [], {}, []
    for stone in data['stones']:
        sid = stone['id']
        if not isinstance(sid, str) or not sid or sid in shapes:
            raise ValueError('stone IDs must be nonempty unique strings')
        if not isinstance(stone['active'], bool):
            raise ValueError('active must be boolean')
        if stone['active'] and 'pose' not in stone:
            raise ValueError('active independently calibrated stones require a patio-frame pose')
        shape, report = component(stone)
        pose = np.asarray(stone.get('pose', [0,0,0]), dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError('pose must be three finite centroid x, y, angle values')
        shapes[sid] = local_shape(shape.exterior.coords)
        outlines[sid] = list(shape.exterior.coords[:-1])
        poses.append(pose); active.append(stone['active']); ids.append(sid)
        report['id'] = sid
        report['active'] = stone['active']
        report['input_pose'] = stone.get('pose')
        reports.append(report)
    layout = Layout(patio, shapes, np.asarray(poses, float).reshape((-1,3)), np.asarray(active, bool),
                    tuple(ids), np.random.default_rng(seed), units=data['units'], source_outlines=outlines)
    return layout, {'report_version': 1, 'units': layout.units, 'patio': patio_report, 'stones': reports,
                    'warning': 'Each calibration is independent. Residuals cover control points only, not unmodeled distortion.'}


def main():
    import argparse
    import json
    from pathlib import Path
    from .phase1 import save_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='measurement JSON in one common coordinate frame')
    parser.add_argument('--max-residual', type=float, help='common-frame maximum control-point residual in target units')
    parser.add_argument('--bundle', action='store_true', help='calibrate independent patio and stone frames with per-frame limits')
    parser.add_argument('--output', required=True, type=Path, help='version 1 layout JSON')
    parser.add_argument('--report', required=True, type=Path, help='calibration and raw-measurement JSON')
    args = parser.parse_args()
    paths = [args.input.resolve(), args.output.resolve(), args.report.resolve()]
    if len(set(paths)) != 3:
        parser.error('input, output, and report paths must be distinct')
    data = json.loads(args.input.read_text())
    if args.bundle:
        if args.max_residual is not None:
            parser.error('--bundle uses a max_residual for each frame in the input')
        layout, report = calibrate_bundle(data)
    else:
        if args.max_residual is None:
            parser.error('common-frame input requires --max-residual')
        layout, report = calibrate_inventory(data, max_residual=args.max_residual)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    save_json(layout, args.output)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False))
    residual = max([report['patio']['max_residual']] + [s['max_residual'] for s in report['stones']]) if args.bundle else report['max_residual']
    print(json.dumps({'stones': len(layout.ids), 'units': layout.units,
                      'largest_control_residual': residual}, indent=2))


if __name__ == '__main__':
    main()
