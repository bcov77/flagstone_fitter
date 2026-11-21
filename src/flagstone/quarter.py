"""Final-layout quarter-patio experiments; sampled search is not certification."""
import hashlib
import json
import math
import time
from functools import wraps
from pathlib import Path

import numpy as np
import shapely
from scipy.optimize import minimize
from shapely.geometry import Point, box
from shapely.geometry.polygon import orient

from .phase1 import load_json, save_json, transformed, validate

GAP = .5 / 12
RADIUS = 1 / 12


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def prepare(source, output):
    """Declare a synthetic square, preserving every original inventory outline."""
    source = Path(source).resolve()
    raw = source.read_bytes()
    data = json.loads(raw)
    if data['units'] != 'ft':
        raise ValueError('Quarter experiment requires feet')
    original = shapely.Polygon(data['patio'])
    side = math.sqrt(original.area / 4)
    data['patio'] = [[0, 0], [side, 0], [side, side], [0, side]]
    for stone in data['stones']:
        stone.update(active=False, pose=[0, 0, 0])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'source_inventory.json').write_bytes(raw)
    write(out / 'input.json', data)
    # Validate original shape identities and source-outline consistency.
    layout = load_json(out / 'input.json')
    contract = dict(version='quarter-final-layout-v1', units='ft',
                    patio_kind='synthetic square with physical benchmark edges',
                    source=str(source), source_sha256=hashlib.sha256(raw).hexdigest(),
                    input_sha256=hashlib.sha256((out/'input.json').read_bytes()).hexdigest(),
                    full_patio_area_sqft=original.area, patio_area_sqft=layout.patio.area,
                    side_ft=side, inventory_count=len(layout.ids),
                    minimum_clearance_in=.5, maximum_empty_circle_diameter_strictly_below_in=2,
                    unused_stones_allowed=True, prefix_constraints=False, edge_relaxation=False)
    write(out / 'contract.json', contract)
    return contract


def world(layout):
    return {i: transformed(layout.shapes[layout.ids[i]], layout.poses[i])
            for i in np.flatnonzero(layout.active)}


def sample_points(patio, spacing=1/12):
    a, b, c, d = patio.bounds
    xx, yy = np.meshgrid(np.arange(a+spacing/2, c, spacing),
                         np.arange(b+spacing/2, d, spacing))
    points = shapely.points(xx.ravel(), yy.ravel())
    return points[shapely.covers(patio, points)]


def distances(points, polygons, boundary):
    result = shapely.distance(points, boundary)
    if polygons:
        tree = shapely.STRtree(list(polygons))
        pairs, ds = tree.query_nearest(points, return_distance=True, all_matches=False)
        result[pairs[0]] = np.minimum(result[pairs[0]], ds)
    return result


def measure(layout, points=None, spacing=1/12):
    if points is None:
        points = sample_points(layout.patio, spacing)
    if not len(points):
        raise ValueError('Empty sampling domain')
    ds = distances(points, list(world(layout).values()), layout.patio.boundary)
    excess_in = np.maximum(0, ds-RADIUS)*12
    worst = int(np.argmax(ds))
    return dict(sampled_max_diameter_in=float(24*ds[worst]),
                sampled_bad_center_area_sqft=float(np.count_nonzero(ds >= RADIUS)*spacing**2),
                sampled_energy=float(np.mean(excess_in**2) + .1*np.max(excess_in)**2),
                largest_center=list(points[worst].coords[0]), spacing_ft=spacing,
                coverage=sum(layout.shapes[layout.ids[i]].area for i in np.flatnonzero(layout.active))/layout.patio.area,
                placed=int(layout.active.sum()), unused=int((~layout.active).sum()),
                sampled_only=True)


def rank(metrics, objective='energy'):
    if objective == 'worst':
        return (metrics['sampled_max_diameter_in'], metrics['sampled_energy'], -metrics['coverage'])
    if objective == 'balanced':
        return (metrics['sampled_bad_center_area_sqft'], metrics['sampled_energy'],
                metrics['sampled_max_diameter_in'], -metrics['coverage'])
    if objective != 'energy':
        raise ValueError('Unknown objective')
    return (metrics['sampled_energy'], metrics['sampled_max_diameter_in'], -metrics['coverage'])


def enclosing_proxy(g):
    q = g.simplify(.003, preserve_topology=True).buffer(.004, quad_segs=4)
    return q if q.geom_type == 'Polygon' and q.covers(g) else g


def native_input(layout, seed, angle_deg=0):
    rng = np.random.default_rng(seed)
    ids = list(layout.ids)
    rng.shuffle(ids)
    angle = math.radians(angle_deg)
    from shapely.affinity import rotate, translate
    patio = rotate(layout.patio, angle, origin=(0, 0), use_radians=True)
    offset = [-patio.bounds[0], -patio.bounds[1]]
    patio = translate(patio, xoff=offset[0], yoff=offset[1])
    return dict(patio=list(patio.exterior.coords)[:-1], clearance_ft=GAP+.0001,
                bin_transform=dict(angle_rad=angle, offset_ft=offset),
                seed=seed, items=[dict(stone_id=sid, area_sqft=layout.shapes[sid].area,
                                      outline=list(orient(enclosing_proxy(layout.shapes[sid])).exterior.coords)[:-1])
                                 for sid in ids])


def apply_records(layout, records):
    """Replace the active layout using validated original inventory identities."""
    indices = {sid: i for i, sid in enumerate(layout.ids)}
    poses = layout.poses.copy()
    active = np.zeros(len(layout.ids), bool)
    for rec in records:
        sid = rec['stone_id']
        if sid not in indices or active[indices[sid]]:
            raise ValueError('Unknown or duplicate stone')
        pose = np.asarray(rec['pose'], float)
        if pose.shape != (3,) or not np.isfinite(pose).all() or rec.get('mirror', False):
            raise ValueError('Invalid rigid pose')
        poses[indices[sid]] = pose
        active[indices[sid]] = True
    layout.poses[:] = poses
    layout.active[:] = active


def neighborhood(layout, center, count=8):
    p = Point(*center)
    polygons = world(layout)
    return sorted(polygons, key=lambda i: polygons[i].distance(p))[:count]


class _Deadline(Exception):
    pass


def _rollback_on_error(function):
    @wraps(function)
    def wrapped(layout, *args, **kwargs):
        original = layout.snapshot()
        try:
            return function(layout, *args, **kwargs)
        except BaseException:
            layout.restore(original)
            raise
    return wrapped


@_rollback_on_error
def continuous_patch(layout, movable, points, *, seconds=10, maxiter=150, radius_ft=.25, objective='energy'):
    """Bounded spatial repair, including invalid intermediate poses.

    Simplified geometry guides COBYLA. Original validation controls promotion.
    Nonmovable IDs/poses are untouched, including unused inventory entries.
    """
    initial = layout.poses.copy()
    before = measure(layout, points)
    if not movable:
        return dict(accepted=False, reason='empty-neighborhood', before=before, after=before)
    if len(set(movable)) != len(movable) or any(not layout.active[i] for i in movable):
        raise ValueError('Movable set must contain distinct active indices')
    movable = list(movable)
    fixed = [g for i, g in world(layout).items() if i not in movable]
    fixed_ds = distances(points, fixed, layout.patio.boundary)
    # Search approximations are deliberately not acceptance geometry.
    shapes = [layout.shapes[layout.ids[i]].simplify(.003, preserve_topology=True) for i in movable]
    fixed_proxies = [g.simplify(.003, preserve_topology=True) for g in fixed]
    scales = np.tile([radius_ft, radius_ft, .35], len(movable))
    origin = initial[movable].ravel()
    deadline = time.monotonic()+seconds
    cache = {}
    candidates = []

    def evaluate(x):
        if time.monotonic() >= deadline:
            raise _Deadline
        if 'x' in cache and np.array_equal(cache['x'], x):
            return cache
        poses = (origin+np.asarray(x)*scales).reshape(-1, 3)
        gs = [transformed(g, pose) for g, pose in zip(shapes, poses)]
        ds = fixed_ds.copy()
        for g in gs:
            ds = np.minimum(ds, shapely.distance(points, g))
        excess = np.maximum(0, ds-RADIUS)*12
        value = float(np.mean(excess**2)+.1*np.max(excess)**2)
        if objective == 'worst':
            value = float(np.max(excess)**2+.02*np.mean(excess**2))
        elif objective == 'balanced':
            value += .5*float(np.mean(excess))
        cons = []
        for j, g in enumerate(gs):
            outside = g.difference(layout.patio).area
            cons.append(-math.sqrt(outside)-GAP if outside > 0 else g.distance(layout.patio.boundary)-GAP)
            for h in [*fixed_proxies, *gs[:j]]:
                distance = g.distance(h)
                cons.append(distance-GAP if distance > 0 else -math.sqrt(max(0, g.intersection(h).area))-GAP)
        # Keep several proxy-feasible candidates: exact outlines may reject some.
        if min(cons, default=0) >= -1e-7:
            candidates.append((value, poses.copy()))
            candidates.sort(key=lambda row: row[0])
            del candidates[8:]
        cache.clear()
        cache.update(x=np.asarray(x).copy(), value=value, constraints=np.asarray(cons)*12)
        return cache

    reason = 'optimizer-returned'
    termination = {}
    try:
        result = minimize(lambda x: evaluate(x)['value'], np.zeros(len(origin)), method='COBYLA',
                 bounds=[(-1, 1)]*len(origin),
                 constraints=[dict(type='ineq', fun=lambda x: evaluate(x)['constraints'])],
                 options=dict(maxiter=maxiter, rhobeg=.3, tol=1e-4, catol=1e-6))
        termination = dict(status=int(result.status), message=str(result.message),
                           success=bool(result.success), nfev=int(result.nfev))
    except _Deadline:
        reason = 'time-limit'
    best = before
    best_poses = initial.copy()
    try:
        for _, poses in candidates:
            layout.poses[:] = initial
            layout.poses[movable] = poses
            if validate(layout, gap=GAP, edge=GAP):
                continue
            checked = measure(layout, points)
            if rank(checked, objective) < rank(best, objective):
                best, best_poses = checked, layout.poses.copy()
    finally:
        layout.poses[:] = best_poses
    return dict(accepted=rank(best, objective) < rank(before, objective), reason=reason, termination=termination, before=before, after=best,
                movable_ids=[layout.ids[i] for i in movable], candidate_count=len(candidates))


@_rollback_on_error
def try_replacements(layout, center, points, *, seconds=10, limit=12, objective='energy'):
    """Rank the whole unused inventory; test bounded one-for-one substitutions.

    A first vertical slice: exact-clearance replacements only, before continuous
    repair. Multi-stone destroy/rebuild and invalid replacement seeds come next.
    """
    original = layout.snapshot()
    before = measure(layout, points)
    best, best_state = before, original
    deadline = time.monotonic()+seconds
    trials = 0
    for old in neighborhood(layout, center, count=2):
        old_area = layout.shapes[layout.ids[old]].area
        # Every unused ID participates in ranking; no permanent restricted pool.
        unused = sorted(np.flatnonzero(~original[1]),
                        key=lambda i: abs(math.log(layout.shapes[layout.ids[i]].area/old_area)))
        for new in unused[:limit]:
            for angle in (0, math.pi/2):
                if time.monotonic() >= deadline:
                    break
                layout.restore(original)
                layout.active[old], layout.active[new] = False, True
                layout.poses[new] = original[0][old]+[0, 0, angle]
                trials += 1
                if validate(layout, gap=GAP, edge=GAP):
                    continue
                checked = measure(layout, points)
                if rank(checked, objective) < rank(best, objective):
                    best, best_state = checked, layout.snapshot()
            if time.monotonic() >= deadline:
                break
        if time.monotonic() >= deadline:
            break
    layout.restore(best_state)
    return dict(accepted=rank(best, objective) < rank(before, objective), trials=trials, before=before, after=best,
                inventory_ranked=int((~original[1]).sum()))


def rebuild_patch(layout, center, points, *, count=3, seconds=6, proposals=12,
                  strategy='random', objective='energy'):
    """Gap-driven destroy/reinsert using the existing validated insertion engine.

    Bounding boxes restrict proposal centers only. They never act as physical
    walls or circle obstacles. All reserve IDs and removed IDs are eligible.
    A rejected patch restores every pose/active flag while randomness advances.
    Count zero is direct reserve insertion with all placed stones fixed.
    """
    from .search import greedy_insert
    original = layout.snapshot()
    before = measure(layout, points)
    removed = neighborhood(layout, center, count=count) if count else []
    polygons = world(layout)
    cx, cy = center
    focus = box(cx-.8, cy-.8, cx+.8, cy+.8)
    if removed:
        focus = shapely.union_all([focus, *[polygons[i] for i in removed]])
    a, b, c, d = focus.bounds
    result = None
    accepted = False
    after = before
    try:
        layout.active[removed] = False
        result = greedy_insert(layout, gap=GAP, edge=GAP, proposals=proposals,
                               strategy=strategy, time_limit=seconds,
                               proposal_bounds=(a-.2, b-.2, c+.2, d+.2))
        # Preserve any old pose that is still compatible with the rebuilt pocket.
        for i in removed:
            if layout.active[i]:
                continue
            pose = layout.poses[i].copy()
            layout.active[i] = True
            layout.poses[i] = original[0][i]
            if validate(layout, gap=GAP, edge=GAP):
                layout.active[i] = False
                layout.poses[i] = pose
        if not validate(layout, gap=GAP, edge=GAP):
            checked = measure(layout, points)
            if rank(checked, objective) < rank(before, objective):
                accepted, after = True, checked
    finally:
        if not accepted:
            rng = layout.snapshot()[2]
            layout.restore(original)
            layout.rng.bit_generator.state = rng
    return dict(accepted=accepted, before=before, after=after,
                removed_ids=[layout.ids[i] for i in removed],
                eligible_inventory=len(layout.ids)-int(original[1].sum())+len(removed),
                attempts=result.attempts if result else 0,
                inserted=result.accepted if result else 0)


def check_reference(layout, expected):
    if (layout.units != expected.units or layout.ids != expected.ids or
            not layout.patio.equals_exact(expected.patio, 1e-12) or
            any(not layout.shapes[s].equals_exact(expected.shapes[s], 1e-12) for s in layout.ids)):
        raise ValueError('Saved layout changed the declared patio or original inventory')


def verify_saved(path, output, reference=None):
    """Independent saved-file verification; only original geometry can pass."""
    # Reuse the existing enclosing-interval algorithm without corner history gates.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
    from five_stone_edge_start import bounded_gap
    layout = load_json(path)
    if reference is not None:
        check_reference(layout, load_json(reference))
    errors = validate(layout, gap=GAP, edge=GAP)
    gs = list(world(layout).values())
    gap = bounded_gap(layout.patio, gs, layout.patio, error=.02/12,
                      max_cells=50000, stop_on_failure=False) if gs else None
    result = dict(layout_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                  passed=not errors and gap is not None and gap['passed'],
                  clearance_errors=errors, gap=gap, original_geometry=True,
                  reference_matched=reference is not None,
                  visual_review='pending')
    write(output, result)
    return result
