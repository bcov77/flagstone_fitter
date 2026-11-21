"""Frozen multi-context placement dataset and resumable physics campaign.
Use physics_patio/.venv with PYTHONPATH=physics_patio. Never contacts the UI.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import platform
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from importlib.metadata import version
from pathlib import Path

import numpy as np
import shapely
from shapely.affinity import scale
from shapely.geometry import Polygon
from shapely.ops import unary_union

import pilot
from patio_physics.engine import PhysicsSession

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / 'campaign_v1'
DOC = None
MANIFEST = None
PROXIES = {}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def signature(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def code_hashes():
    return {str(p.relative_to(pilot.ROOT)): pilot.digest(p) for p in
            [Path(__file__).resolve(), HERE / 'pilot.py',
             pilot.ROOT / 'physics_patio/patio_physics/engine.py']}


def make_document(source, context, case):
    document = copy.deepcopy(source)
    removed = set(context['removed'])
    if case['family'] == 'replacement':
        removed.add(case['old_id'])
    document['physics'] = {'pieces': {}, 'settings': {}}
    for stone in document['stones']:
        sid = str(stone['id'])
        if sid in removed:
            stone['active'] = False
        if case['candidate'] and sid == case['candidate']['id']:
            stone['active'] = True
            stone['pose'] = case['candidate']['pose'][:]
        document['physics']['pieces'][sid] = {'anchor': stone['pose'][:], 'locked': False}
    return document


def initial_shapes(document):
    return {str(s['id']): pilot.world(s) for s in document['stones'] if s['active']}


def context_session(document):
    """Geometry-only object for independent checks; no collision decomposition."""
    d = copy.deepcopy(document)
    active = {str(s['id']): bool(s['active']) for s in d['stones']}
    for stone in d['stones']:
        stone['active'] = False
    session = PhysicsSession(d)
    for sid, piece in session.pieces.items():
        piece.active = active[sid]
    return session


def sample_grid(patio, spacing=1 / 12):
    x0, y0, x1, y1 = patio.bounds
    xx, yy = np.meshgrid(np.arange(x0 + spacing / 2, x1, spacing),
                         np.arange(y0 + spacing / 2, y1, spacing))
    return np.c_[xx.ravel(), yy.ravel()]


def distance_field(patio, shapes, coords):
    points = shapely.points(coords)
    distances = np.asarray(shapely.distance(points, patio.boundary))
    if shapes:
        idx, ds = shapely.STRtree(list(shapes)).query_nearest(
            points, return_distance=True, all_matches=False)
        distances[idx[0]] = np.minimum(distances[idx[0]], ds)
    return distances


def gap_sites(patio, shapes, count):
    coords = sample_grid(patio)
    ds = distance_field(patio, shapes, coords)
    # Greedy separated distance peaks: sampled search sites, not certified maxima.
    order = np.argsort(-ds, kind='stable')
    sites = []
    for index in order:
        point = coords[index]
        if all(np.linalg.norm(point - np.array(site['center'])) >= 1.4 for site in sites):
            sites.append({'center': point.tolist(), 'radius_ft': float(ds[index]),
                          'boundary_distance_ft': float(patio.boundary.distance(shapely.Point(point)))})
        if len(sites) == count:
            break
    if len(sites) != count:
        raise ValueError('Could not identify enough separated gap sites')
    return sites


def longest_run(mask, step):
    mask = list(map(bool, mask))
    if all(mask):
        return len(mask) * step
    best = current = 0
    for flag in mask + mask:
        current = current + 1 if flag else 0
        best = max(best, current)
    return min(best, len(mask)) * step


def boundary_profile(shape, obstacles, patio, exclusion, tree=None):
    # Arc-length sampling avoids bias from nonuniform traced vertices.
    locations = np.linspace(0, shape.length, 192, endpoint=False)
    points = shapely.line_interpolate_point(shape.exterior, locations)
    if tree is None:
        tree = shapely.STRtree(list(obstacles.geoms) if hasattr(obstacles, 'geoms') else [obstacles])
    idx, distances = tree.query_nearest(points, return_distance=True, all_matches=False)
    ds = np.asarray(shapely.distance(points, patio.boundary))
    ds[idx[0]] = np.minimum(ds[idx[0]], distances)
    ds *= 12
    near = (ds >= .4) & (ds <= 1)
    return {'intrusion_sqft': shape.intersection(exclusion).area,
            'mean_excess_inches': float(np.mean(np.clip(ds - .5, 0, 4))),
            'near_fraction': float(np.mean(near)),
            'longest_near_run_ft': longest_run(near, shape.length / 192),
            'distance_quantiles_inches': np.quantile(ds, [.1, .5, .9]).tolist()}


def feature_neighborhood(patio, others, center):
    obstacles = unary_union(others)
    exclusion = unary_union([obstacles.buffer(.5 / 12),
                             patio.buffer(5).difference(patio.buffer(-.5 / 12))])
    free = patio.buffer(-.5 / 12).difference(obstacles.buffer(.5 / 12))
    point = shapely.Point(center)
    cavity_widths = []
    for angle in np.linspace(0, math.pi, 12, endpoint=False):
        direction = np.array([math.cos(angle), math.sin(angle)]) * 20
        line = shapely.LineString([np.array(center) - direction, np.array(center) + direction])
        intersection = free.intersection(line)
        parts = list(intersection.geoms) if hasattr(intersection, 'geoms') else [intersection]
        cavity_widths.append(max((part.length for part in parts if part.distance(point) < 1e-7), default=0.))
    boundaries = [shape.exterior for shape in others] + [patio.exterior]
    return {'others': others, 'obstacles': obstacles, 'patio': patio, 'exclusion': exclusion,
            'tree': shapely.STRtree(others), 'boundaries': boundaries,
            'boundary_tree': shapely.STRtree(boundaries), 'cavity_widths': cavity_widths,
            'local_free_area': free.intersection(point.buffer(2)).area}


def features(document, case, cavity_radius, neighborhood=None):
    candidate = case['candidate']
    if candidate is None:
        return None
    stones = {str(stone['id']): stone for stone in document['stones']}
    g = pilot.world(stones[candidate['id']])
    if neighborhood is None:
        others = [pilot.world(stone) for sid, stone in stones.items() if stone['active'] and sid != candidate['id']]
        neighborhood = feature_neighborhood(Polygon(document['patio']), others, case['center'])
    others = neighborhood['others']
    obstacles, patio, exclusion = neighborhood['obstacles'], neighborhood['patio'], neighborhood['exclusion']
    profiles = {}
    for factor in [.85, .9, .95, 1.0]:
        probe = scale(g, xfact=factor, yfact=factor, origin='centroid')
        profiles[str(factor)] = boundary_profile(probe, obstacles, patio, exclusion, neighborhood['tree'])
    coords = np.asarray(g.exterior.coords)
    widths = []
    for angle in np.linspace(0, math.pi, 12, endpoint=False):
        projected = coords @ np.array([math.cos(angle), math.sin(angle)])
        widths.append(float(np.ptp(projected)))
    near_neighbors = sum(g.distance(shape) < .5 for shape in others)
    cavity_widths = neighborhood['cavity_widths']
    locations = np.linspace(0, g.length, 192, endpoint=False)
    points = shapely.line_interpolate_point(g.exterior, locations)
    boundaries = neighborhood['boundaries']
    idx, ds = neighborhood['boundary_tree'].query_nearest(points, return_distance=True, all_matches=False)
    aligned = []
    # Use an inch-wide tangent chord to suppress traced-pixel edge noise.
    for sample, other, distance in zip(idx[0], idx[1], ds):
        if distance > 1 / 12:
            continue
        boundary = boundaries[int(other)]
        at = boundary.project(points[int(sample)])
        a = boundary.interpolate((at - 1 / 24) % boundary.length)
        b = boundary.interpolate((at + 1 / 24) % boundary.length)
        ca = g.exterior.interpolate((locations[sample] - 1 / 24) % g.length)
        cb = g.exterior.interpolate((locations[sample] + 1 / 24) % g.length)
        u = np.array([b.x - a.x, b.y - a.y])
        v = np.array([cb.x - ca.x, cb.y - ca.y])
        cosine = abs(float(np.dot(u, v))) / max(1e-12, float(np.linalg.norm(u) * np.linalg.norm(v)))
        aligned.append(min(1., cosine))

    return {'area_sqft': g.area, 'perimeter_ft': g.length,
            'convexity': g.area / g.convex_hull.area,
            'widths_15deg_ft': widths, 'cavity_widths_15deg_ft': cavity_widths,
            'local_clearance_free_area_sqft': neighborhood['local_free_area'],
            'near_boundary_tangent_abs_cosine': float(np.mean(aligned)) if aligned else None,
            'near_boundary_tangent_samples': len(aligned),
            'cavity_peak_radius_ft': cavity_radius,
            'minimum_width_to_peak_diameter': min(widths) / max(.01, 2 * cavity_radius),
            'neighbor_count_within_half_ft': near_neighbors,
            'wall_distance_inches': g.distance(patio.boundary) * 12,
            'scale_profiles': profiles,
            'geometry': 'original full-size outlines; scaling is a feature probe only'}


def candidate_pool(document, target, rng, pool_size):
    stones = {str(s['id']): s for s in document['stones']}
    available = sorted([sid for sid, s in stones.items() if not s['active']],
                       key=lambda sid: Polygon(stones[sid]['outline']).area)
    # Stratified ID sample across inventory area; equal mass per area tertile.
    strata = np.array_split(available, 3)
    selected = []
    for stratum in strata:
        values = list(stratum)
        rng.shuffle(values)
        selected.extend(values[:8])
    obstacles = unary_union([pilot.world(s).simplify(.003)
                             for s in document['stones'] if s['active']])
    patio = Polygon(document['patio'])
    exclusion = unary_union([obstacles.buffer(.5 / 12),
                             patio.buffer(5).difference(patio.buffer(-.5 / 12))])
    pool = []
    while len(pool) < pool_size:
        sid = selected[len(pool) % len(selected)]
        local = Polygon(stones[sid]['outline']).simplify(.003)
        centroid = local.centroid
        angle = math.radians(rng.randrange(24) * 15)
        offset = rng.choice([0, .1, .25, .4])
        heading = rng.uniform(0, 2 * math.pi)
        cx = target['center'][0] + offset * math.cos(heading)
        cy = target['center'][1] + offset * math.sin(heading)
        c, s = math.cos(angle), math.sin(angle)
        pose = [cx - c * centroid.x + s * centroid.y,
                cy - s * centroid.x - c * centroid.y, angle]
        shape = pilot.world({'pose': pose, 'outline': list(local.exterior.coords)})
        profile = boundary_profile(shape, obstacles, patio, exclusion)
        intrusion_fraction = profile['intrusion_sqft'] / shape.area
        area_score = 8 * profile['intrusion_sqft'] - shape.area
        pool.append({'candidate': {'id': str(sid), 'pose': pose},
                     'screen_area': shape.area, 'screen_intrusion_fraction': intrusion_fraction,
                     'area_score': area_score,
                     'edge_score': area_score + .2 * shape.length * profile['mean_excess_inches']})
    return pool


def select_proposals(pool, family, rng):
    # Random proposals retain a known conditional sampling probability.
    quota = 6 if family == 'replacement' else 7
    indices = list(range(len(pool)))
    random_ids = rng.sample(indices, quota)
    used = set(random_ids)
    chosen = [(i, 'random', quota / len(pool)) for i in random_ids]
    for index in sorted(indices, key=lambda i: pool[i]['edge_score']):
        if index not in used:
            used.add(index)
            chosen.append((index, 'geometry', None))
        if sum(kind == 'geometry' for _, kind, _ in chosen) == quota:
            break
    # Keep difficult placements, including large clearance intrusion. No success filtering.
    stress = sorted((i for i in indices if i not in used),
                    key=lambda i: pool[i]['screen_intrusion_fraction'], reverse=True)
    for index in stress[:6]:
        chosen.append((index, 'stress', None))
    return chosen


def prepare(out, config_path):
    config = json.loads(Path(config_path).read_text())
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        return load_manifest(out, config)
    if config['proposals_per_target'] != 20:
        raise ValueError('Current predeclared sampling quotas require 20 proposals per target')
    source = json.loads(pilot.SOURCE.read_text())
    active = sorted([str(s['id']) for s in source['stones'] if s['active']], key=int)
    stones = {str(s['id']): s for s in source['stones']}
    patio = Polygon(source['patio'])
    rng = random.Random(config['seed'])
    contexts = []
    for i, count in enumerate(config['context_removal_counts']):
        context = {'id': f'board_{i:02d}', 'removed': sorted(rng.sample(active, count), key=int),
                   'synthetic': bool(count), 'active_count': len(active) - count,
                   'lineage_group': 'single_original_winner'}
        contexts.append(context)
    targets, cases, pools = [], [], {}
    for bi, context in enumerate(contexts):
        empty_case = {'family': 'insertion', 'candidate': None}
        board = make_document(source, context, empty_case)
        if not pilot.clearance(context_session(board))['valid']:
            raise ValueError('Source context is not clearance-valid')
        on_board = initial_shapes(board)
        sites = gap_sites(patio, on_board.values(), config['targets_per_family'])
        remaining = [sid for sid in active if sid not in context['removed']]
        # Spread replacement sites spatially; vary the first sampled stone per board.
        first = rng.choice(remaining)
        old_ids = [first]
        while len(old_ids) < config['targets_per_family']:
            chosen = max((sid for sid in remaining if sid not in old_ids),
                         key=lambda sid: min(on_board[sid].centroid.distance(on_board[old].centroid)
                                             for old in old_ids))
            old_ids.append(chosen)
        descriptions = [('insertion', j, site, None) for j, site in enumerate(sites)]
        descriptions += [('replacement', j, {'center': list(on_board[sid].centroid.coords[0]),
                                            'radius_ft': math.sqrt(on_board[sid].area / math.pi)}, sid)
                         for j, sid in enumerate(old_ids)]
        for family, index, site, old_id in descriptions:
            target = dict(id=f'{context["id"]}_{family}_{index}', context=context['id'],
                          family=family, old_id=old_id, **site)
            # Labels expose correlations. No independent-layout test split is claimed.
            target['region_group'] = f"root_region_{int(site['center'][0] // 2)}_{int(site['center'][1] // 2)}"
            target['lineage_group'] = context['lineage_group']
            targets.append(target)
            base = dict(target=target['id'], context=context['id'], family=family,
                        old_id=old_id, center=site['center'], candidate=None)
            if family == 'replacement':
                original = stones[old_id]
                baseline = dict(base, id=target['id'] + '_control', kind='control',
                                candidate={'id': old_id, 'pose': original['pose'][:]})
                cases.append(dict(base, id=target['id'] + '_empty', kind='empty_diagnostic',
                                  control=baseline['id'], features=None))
            else:
                baseline = dict(base, id=target['id'] + '_control', kind='control')
            baseline['control'] = baseline['id']
            baseline['features'] = None
            cases.append(baseline)
            gap_document = make_document(source, context, base)
            pool = candidate_pool(gap_document, target, rng, config['pool_size'])
            # An original stone is reserved for the two explicit reorientation proposals.
            if family == 'replacement':
                pool = [p for p in pool if p['candidate']['id'] != old_id]
            pools[target['id']] = pool
            selected = select_proposals(pool, family, rng)
            neighborhood = feature_neighborhood(patio, list(initial_shapes(gap_document).values()), site['center'])
            proposals = []
            for pindex, kind, probability in selected:
                proposals.append(dict(pool[pindex], kind=kind, pool_index=pindex,
                                      conditional_random_probability=probability))
            if family == 'replacement':
                for degrees in [-15, 15]:
                    pose = stones[old_id]['pose'][:]
                    pose[2] += math.radians(degrees)
                    proposals.append({'candidate': {'id': old_id, 'pose': pose},
                                      'kind': 'same_piece_reoriented', 'degrees': degrees})
            if len(proposals) != 20:
                raise ValueError('Unbalanced proposals')
            for j, proposal in enumerate(proposals):
                case = dict(base, id=target['id'] + f'_proposal_{j:02d}', control=baseline['id'], **proposal)
                candidate_doc = make_document(source, context, case)
                case['features'] = features(candidate_doc, case, site['radius_ft'], neighborhood)
                cases.append(case)
            print('Prepared', target['id'], 'pool', len(pool), 'proposals', len(proposals), flush=True)
    source_hash = pilot.digest(pilot.SOURCE)
    plan_hash = signature({'source': source_hash, 'config': config, 'cases': cases, 'code': code_hashes()})
    tasks = []
    for case in cases:
        for seed in config['seeds']:
            for protocol in config['protocols']:
                task = {'case': case['id'], 'seed': seed, 'protocol': protocol}
                task['id'] = signature({'plan': plan_hash, **task})[:24]
                tasks.append(task)
    # Coverage-balanced validation: two action regions, controls and one candidate per board.
    validation_cases = set()
    for bi, context in enumerate(contexts):
        kind = ['random', 'geometry', 'stress'][bi % 3]
        for family in ['insertion', 'replacement']:
            target_id = f'{context["id"]}_{family}_{bi % config["targets_per_family"]}'
            options = [case for case in cases if case['target'] == target_id]
            validation_cases.add(next(c['id'] for c in options if c['kind'] == 'control'))
            validation_cases.add(next(c['id'] for c in options if c['kind'] == kind))
    rng.shuffle(tasks)
    manifest = {'version': 1, 'config': config, 'source_sha256': source_hash,
                'source_path': str(pilot.SOURCE), 'code_hashes': code_hashes(), 'plan_hash': plan_hash,
                'contexts': contexts, 'targets': targets, 'cases': cases, 'tasks': tasks,
                'validation_task_ids': [t['id'] for t in tasks if t['case'] in validation_cases],
                'environment': {'python': platform.python_version(), 'shapely': version('shapely'),
                                'pymunk': version('pymunk'), 'numpy': version('numpy')},
                'limitations': ['All contexts share one source layout; no independent-layout holdout exists.',
                                'Gap sites are separated sampled distance peaks, not certified maxima.',
                                'Pool screening simplifies outlines by 0.036 inches; detailed features and validity use originals.',
                                'Random inclusion probabilities are conditional on the stratified generated pool.',
                                'This is a frozen proposal dataset, not an adaptive optimization walk.']}
    manifest['manifest_hash'] = signature(manifest)
    (out / 'source.json').write_bytes(pilot.SOURCE.read_bytes())
    pilot.write(out / 'proposal_pools.json', pools)
    pilot.write(out / 'manifest.json', manifest)
    return manifest


def load_manifest(out, config=None):
    m = json.loads((out / 'manifest.json').read_text())
    recorded = m.pop('manifest_hash')
    if signature(m) != recorded:
        raise ValueError('Manifest content hash mismatch')
    m['manifest_hash'] = recorded
    if m['code_hashes'] != code_hashes():
        raise ValueError('Simulation code changed: use a new output directory')
    if config is not None and m['config'] != config:
        raise ValueError('Configuration changed: use a new output directory')
    if pilot.digest(out / 'source.json') != m['source_sha256']:
        raise ValueError('Frozen source hash mismatch')
    return m


def initialize(out):
    global DOC, MANIFEST, PROXIES
    out = Path(out)
    DOC = json.loads((out / 'source.json').read_text())
    MANIFEST = load_manifest(out)
    MANIFEST['_cases'] = {c['id']: c for c in MANIFEST['cases']}
    MANIFEST['_contexts'] = {c['id']: c for c in MANIFEST['contexts']}
    PROXIES = {}


def run_one(out, task):
    started = time.monotonic()
    case = MANIFEST['_cases'][task['case']]
    context = MANIFEST['_contexts'][case['context']]
    config = MANIFEST['config']
    document = make_document(DOC, context, case)
    document['physics']['settings'] = config['settings'].copy()
    # Keep source geometry once; proxies are immutable and cached per worker.
    active = {str(s['id']): bool(s['active']) for s in document['stones']}
    for stone in document['stones']:
        stone['active'] = False
    session = PhysicsSession(document, seed=task['seed'])
    for sid, piece in session.pieces.items():
        piece.active = active[sid]
    session.proxy_cache = PROXIES
    session._build()
    initial = pilot.poses(session)
    before = pilot.clearance(session)
    before_quality = pilot.quality(session, case['center'])
    stages = []
    for phase, duration in enumerate(config['explore_phase_seconds']):
        gap = 1.1 if task['protocol'] == 'gap_cycle' and phase == 1 else .5
        session.edit('settings', values={'gap_inches': gap})
        session.start()
        for sid in session.drives:
            rng = random.Random(f'{task["seed"]}:{sid}:{phase}')
            session.drives[sid] = [(rng.uniform(.35, 1.35), rng.uniform(0, 2 * math.pi)) for _ in range(3)]
        session.advance(duration)
        stages.append({'phase': f'explore_{phase}', 'gap_inches': gap, 'seconds': duration,
                       'poses': pilot.poses(session), 'motion': pilot.motion(session)})
        session.stop()
    session.edit('settings', values={'gap_inches': .5, 'piece_hardness': 48, 'wall_hardness': 89,
                                    'rotation_magnitude': 0, 'translation_magnitude': 0,
                                    'rotation_force': 0, 'translation_force': 0,
                                    'angular_stiffness': 0, 'translation_stiffness': 0})
    session.start()
    trace, stable, last_quality, converged = [], 0, None, False
    interval = config['quiet_sample_seconds']
    for i in range(round(config['quiet_max_seconds'] / interval)):
        session.advance(interval)
        c, v = pilot.clearance(session), pilot.motion(session)
        low_motion = (v['max_speed_ips'] < config['max_speed_ips'] and
                      v['max_angular_speed_dps'] < config['max_angular_speed_dps'])
        q = pilot.quality(session, case['center']) if c['valid'] and low_motion else None
        steady_void = (q is not None and last_quality is not None and
                       abs(q['broad_void_1'] - last_quality['broad_void_1']) <= config['void_stability_sqft'] and
                       abs(q['far_void_1'] - last_quality['far_void_1']) <= config['void_stability_sqft'])
        stable = stable + 1 if steady_void else 0
        trace.append({'seconds': (i + 1) * interval, **c, **v, 'quality_if_low_motion': q})
        last_quality = q
        if stable >= config['stable_samples']:
            converged = True
            break
    stages.append({'phase': 'finish', 'poses': pilot.poses(session)})
    # Observe continuously: never zero velocities to manufacture convergence.
    session.advance(config['post_observation_seconds'])
    after, v, q = pilot.clearance(session), pilot.motion(session), pilot.quality(session, case['center'])
    steady_final = (last_quality is not None and
                    abs(q['broad_void_1'] - last_quality['broad_void_1']) <= config['void_stability_sqft'] and
                    abs(q['far_void_1'] - last_quality['far_void_1']) <= config['void_stability_sqft'])
    converged = (converged and after['valid'] and steady_final and
                 v['max_speed_ips'] < config['max_speed_ips'] and
                 v['max_angular_speed_dps'] < config['max_angular_speed_dps'])
    final = pilot.poses(session)
    shifts = {sid: math.hypot(final[sid][0] - p[0], final[sid][1] - p[1]) * 12 for sid, p in initial.items()}
    session.stop()
    result = {'status': 'complete', 'task': task, 'manifest_hash': MANIFEST['manifest_hash'],
              'before': before, 'before_quality': before_quality, 'initial_poses': initial,
              'stages': stages, 'trace': trace, 'after': after, 'after_quality': q,
              'final_motion': v, 'final_poses': final, 'converged': bool(converged),
              'timeout': not bool(converged), 'quiet_seconds': len(trace) * interval,
              'translation_inches_by_id': shifts, 'wall_seconds': time.monotonic() - started}
    pilot.write(Path(out) / 'results' / f'{task["id"]}.json', result)
    return {'id': task['id'], 'case': case['id'], 'valid': after['valid'],
            'converged': bool(converged), 'wall_seconds': result['wall_seconds']}


def completed_ids(out, manifest):
    ids = set()
    known = {t['id'] for t in manifest['tasks']}
    for path in (out / 'results').glob('*.json'):
        r = json.loads(path.read_text())
        tid = r['task']['id']
        if tid not in known or path.stem != tid or r.get('manifest_hash') != manifest['manifest_hash'] or r.get('status') != 'complete':
            raise ValueError(f'Invalid checkpoint identity: {path}')
        ids.add(tid)
    return ids


def run_tasks(out, manifest, remaining, selected, complete, workers, budget, limit):
    (out / 'results').mkdir(exist_ok=True)
    started = time.monotonic()
    completed = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(str(out),)) as pool:
        pending = {}
        index = 0
        while pending or index < len(remaining):
            while len(pending) < workers and index < len(remaining) and time.monotonic() - started < budget:
                task = remaining[index]
                pending[pool.submit(run_one, str(out), task)] = task['id']
                index += 1
            if not pending:
                break
            done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                tid = pending.pop(future)
                try:
                    result = future.result()
                    completed += 1
                    print(completed, '/', len(remaining), canonical(result), flush=True)
                except Exception as exc:
                    pilot.write(out / f'error_{tid}.json', {'task': tid, 'error': repr(exc)})
                    raise
    record = {'completed_new': completed, 'scheduled': len(remaining), 'selected_total': len(selected),
              'completed_before': len(complete & selected), 'limit': limit, 'workers': workers,
              'budget_seconds': budget, 'elapsed_seconds': time.monotonic() - started}
    events = out / 'runs'
    events.mkdir(exist_ok=True)
    pilot.write(events / f'{time.time_ns()}.json', record)
    print(canonical(record), flush=True)


def execute(out, manifest, selection, workers, budget, limit=None):
    if selection == 'all':
        readiness_path = out / 'readiness.json'
        if not readiness_path.exists():
            raise ValueError('Run and audit validation before starting all tasks')
        readiness = json.loads(readiness_path.read_text())
        if not readiness.get('ready') or readiness.get('manifest_hash') != manifest['manifest_hash']:
            raise ValueError('Validation readiness gate has not passed')
    chosen = set(manifest['validation_task_ids']) if selection == 'validation' else {t['id'] for t in manifest['tasks']}
    complete = completed_ids(out, manifest)
    tasks = [t for t in manifest['tasks'] if t['id'] in chosen and t['id'] not in complete]
    random.Random(manifest['config']['seed']).shuffle(tasks)
    if limit is not None:
        tasks = tasks[:limit]
    run_tasks(out, manifest, tasks, chosen, complete, workers, budget, limit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--config', type=Path, default=HERE / 'campaign_config.json')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--selection', choices=['validation', 'all'], default='validation')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-wall-seconds', type=float, default=1200)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.workers < 1 or args.max_wall_seconds <= 0 or (args.limit is not None and args.limit < 1):
        parser.error('Positive workers, budget and limit required')
    manifest = prepare(args.out, args.config)
    print(f"Frozen plan: {len(manifest['contexts'])} boards, {len(manifest['targets'])} targets, "
          f"{len(manifest['tasks'])} tasks; validation {len(manifest['validation_task_ids'])}", flush=True)
    if not args.prepare_only:
        execute(args.out, manifest, args.selection, args.workers, args.max_wall_seconds, args.limit)


if __name__ == '__main__':
    main()
