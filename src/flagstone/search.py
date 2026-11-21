"""Bounded, reproducible greedy insertion baseline."""
from dataclasses import dataclass
import time
import numpy as np
from shapely.geometry import Point, box
from shapely.ops import unary_union
from .phase1 import validate, transformed
from .relaxation import relax, relax_neighborhood
from .insertion import _InsertionOracle

STRATEGIES = ('random', 'mixed', 'corner', 'boundary', 'opening', 'contact', 'edge', 'wall')
_VALIDATED_INPUT = object()


@dataclass
class SearchResult:
    attempts: int
    accepted: int
    coverage: float
    elapsed_seconds: float
    violations: list
    relaxation_attempts: int = 0
    relaxation_skipped: int = 0
    relaxation_successes: int = 0
    relaxation_failures: int = 0
    mobile_stone_visits: int = 0


def coverage(layout):
    return sum(layout.shapes[s].area for i, s in enumerate(layout.ids) if layout.active[i]) / layout.patio.area


def _frozen_indices(layout, frozen_ids):
    frozen = set(() if frozen_ids is None else frozen_ids)
    if not frozen.issubset(layout.ids):
        raise ValueError('frozen_ids contains an unknown stone')
    indices = {i for i, sid in enumerate(layout.ids) if sid in frozen}
    if any(not layout.active[i] for i in indices):
        raise ValueError('frozen stones must already be active')
    return indices


def _contact_pose(shape, anchor, angle, clearance, rng):
    """Propose beside an anchor's box with the requested axial separation."""
    aminx, aminy, amaxx, amaxy = anchor.bounds
    bminx, bminy, bmaxx, bmaxy = transformed(shape, (0, 0, angle)).bounds
    # Scale-relative cushion avoids floating-point contact being just below the gap.
    scale = max(amaxx-aminx, amaxy-aminy, bmaxx-bminx, bmaxy-bminy, clearance, 1e-12)
    separation = clearance + scale * 1e-9
    direction = int(rng.integers(4))
    if direction == 0:
        return amaxx + separation - bminx, rng.uniform(aminy-bmaxy, amaxy-bminy)
    if direction == 1:
        return aminx - separation - bmaxx, rng.uniform(aminy-bmaxy, amaxy-bminy)
    if direction == 2:
        return rng.uniform(aminx-bmaxx, amaxx-bminx), amaxy + separation - bminy
    return rng.uniform(aminx-bmaxx, amaxx-bminx), aminy - separation - bmaxy


def _sample_edge(shape, rng):
    """Sample a nonzero edge proportionally to length and its outward normal."""
    coords = np.asarray(shape.exterior.coords, dtype=float)
    segments = np.diff(coords, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    total = lengths.sum()
    index = int(rng.choice(len(lengths), p=lengths/total))
    direction = segments[index] / lengths[index]
    normal = np.array([direction[1], -direction[0]])
    if not shape.exterior.is_ccw:
        normal = -normal
    point = coords[index] + rng.random() * segments[index]
    return point, normal


def _edge_pose(shape, anchor, clearance, rng):
    """Align sampled edges on opposite sides of a prospective joint.

    Concave outlines and other neighbors can still intersect; the validator,
    rather than this geometric proposal, decides acceptance.
    """
    a_point, a_normal = _sample_edge(anchor, rng)
    b_point, b_normal = _sample_edge(shape, rng)
    angle = np.arctan2(-a_normal[1], -a_normal[0]) - np.arctan2(b_normal[1], b_normal[0])
    c, s = np.cos(angle), np.sin(angle)
    rotated = np.array([c*b_point[0]-s*b_point[1], s*b_point[0]+c*b_point[1]])
    minx, miny, maxx, maxy = anchor.bounds
    scale = max(maxx-minx, maxy-miny, clearance, 1e-12)
    destination = a_point + (clearance + scale * 1e-9) * a_normal
    center = destination - rotated
    return float(center[0]), float(center[1]), float(angle)


def _wall_pose(shape, patio, clearance, rng):
    """Align a stone edge with a sampled patio edge on its interior side."""
    point, outward = _sample_edge(patio, rng)
    local_point, local_outward = _sample_edge(shape, rng)
    angle = np.arctan2(outward[1], outward[0]) - np.arctan2(local_outward[1], local_outward[0])
    c, s = np.cos(angle), np.sin(angle)
    rotated = np.array([c*local_point[0]-s*local_point[1], s*local_point[0]+c*local_point[1]])
    minx, miny, maxx, maxy = patio.bounds
    scale = max(maxx-minx, maxy-miny, clearance, 1e-12)
    destination = point - (clearance + scale * 1e-9) * outward
    center = destination - rotated
    return float(center[0]), float(center[1]), float(angle)


def greedy_insert(layout, gap=.01, edge=.01, proposals=50, relaxation_steps=0, strategy="random", grid_resolution=16, time_limit=None, stone_ids=None, stone_margins=None, patio_margin=0, frozen_ids=None, relaxation_step=.02, use_prefilter=True, proposal_bounds=None, local_relaxation=False, max_relaxation_penalty=None, mobility_policy="circle", tolerance=1e-9, _validation_token=None):
    """Try largest unused stones first; restore poses and flags after every failed attempt.

    Proposal randomness intentionally advances after a failure, unlike a full snapshot
    restore, so successive proposals explore different locations.
    """
    if mobility_policy not in ('circle', 'clearance') or (mobility_policy == 'clearance' and not use_prefilter):
        raise ValueError('clearance mobility requires the prefilter')
    if max_relaxation_penalty is not None:
        if not np.isfinite(max_relaxation_penalty) or max_relaxation_penalty < 0:
            raise ValueError('max_relaxation_penalty must be finite and nonnegative')
        if not use_prefilter or not relaxation_steps:
            raise ValueError('max_relaxation_penalty requires prefilter and relaxation steps')
    if not isinstance(use_prefilter, bool) or not isinstance(local_relaxation, bool):
        raise ValueError('use_prefilter and local_relaxation must be boolean')
    if not isinstance(proposals, int) or not isinstance(relaxation_steps, int) or proposals < 0 or relaxation_steps < 0:
        raise ValueError('proposal and relaxation budgets must be nonnegative integers')
    if not np.isfinite(relaxation_step) or relaxation_step <= 0:
        raise ValueError('relaxation_step must be finite and positive')
    if strategy not in STRATEGIES or not isinstance(grid_resolution, int) or grid_resolution < 2:
        raise ValueError('unknown strategy or invalid grid resolution')
    if time_limit is not None and (not np.isfinite(time_limit) or time_limit < 0):
        raise ValueError('time_limit must be finite and nonnegative')
    if _validation_token is not None and _validation_token is not _VALIDATED_INPUT:
        raise ValueError('Invalid internal validation token')
    if _validation_token is not _VALIDATED_INPUT:
        violations = validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin)
        if violations:
            raise ValueError('Initial layout must be valid')
    frozen = _frozen_indices(layout, frozen_ids)
    start = time.perf_counter(); attempts = accepted = 0
    relaxation_attempts = relaxation_skipped = relaxation_successes = relaxation_failures = 0
    mobile_stone_visits = 0
    minx, miny, maxx, maxy = layout.patio.bounds
    if proposal_bounds is not None:
        bounds = np.asarray(proposal_bounds, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError('proposal_bounds must be finite (minx, miny, maxx, maxy) with positive extent')
        minx, miny = max(minx, bounds[0]), max(miny, bounds[1])
        maxx, maxy = min(maxx, bounds[2]), min(maxy, bounds[3])
        if minx >= maxx or miny >= maxy:
            raise ValueError('proposal_bounds must overlap the patio bounding box')
    corners = list(layout.patio.exterior.coords)[:-1]
    selected = set(layout.ids) if stone_ids is None else set(stone_ids)
    if not selected.issubset(layout.ids):
        raise ValueError('stone_ids contains an unknown stone')
    order = sorted((i for i in range(len(layout.ids)) if not layout.active[i] and layout.ids[i] in selected), key=lambda i: -layout.shapes[layout.ids[i]].area)
    for i in order:
        if time_limit is not None and time.perf_counter() - start >= time_limit:
            break
        candidates = []
        if strategy not in ('random', 'contact', 'edge', 'wall'):
            stones = [transformed(layout.shapes[s], layout.poses[j]) for j, s in enumerate(layout.ids) if layout.active[j]]
            union = unary_union(stones)
            for yy in range(grid_resolution):
                for xx in range(grid_resolution):
                    px = minx + (xx + .5) * (maxx - minx) / grid_resolution
                    py = miny + (yy + .5) * (maxy - miny) / grid_resolution
                    point = Point(px, py)
                    if layout.patio.covers(point) and not union.covers(point):
                        boundary_distance = point.distance(layout.patio.boundary)
                        stone_distance = point.distance(union) if stones else float('inf')
                        if strategy == 'mixed':
                            priority = stone_distance if stones else boundary_distance
                        elif strategy == 'opening':
                            priority = min(stone_distance, boundary_distance)
                        elif strategy == 'boundary':
                            priority = -boundary_distance
                        else:
                            priority = -min((px-vx)**2 + (py-vy)**2 for vx, vy in corners)
                        candidates.append((priority, px, py))
            candidates.sort(reverse=True)
        anchors = [(sid, transformed(layout.shapes[sid], layout.poses[j])) for j, sid in enumerate(layout.ids) if layout.active[j]] if strategy in ('contact', 'edge', 'wall') else []
        if proposal_bounds is not None and anchors:
            radius = np.linalg.norm(np.asarray(layout.shapes[layout.ids[i]].exterior.coords), axis=1).max()
            zone = box(minx-radius-gap, miny-radius-gap, maxx+radius+gap, maxy+radius+gap)
            anchors = [(sid, geom) for sid, geom in anchors if geom.intersects(zone)]
        oracle = _InsertionOracle(layout, gap=gap, edge=edge, tolerance=tolerance, stone_margins=stone_margins, patio_margin=patio_margin) if use_prefilter else None
        for attempt in range(proposals):
            if time_limit is not None and time.perf_counter() - start >= time_limit:
                break
            attempts += 1
            poses, active = layout.poses.copy(), layout.active.copy()
            if strategy == 'wall' and attempt % 4 == 0:
                clearance = edge + patio_margin + (stone_margins or {}).get(layout.ids[i], 0)
                px, py, angle = _wall_pose(layout.shapes[layout.ids[i]], layout.patio, clearance, layout.rng)
            elif strategy in ('contact', 'edge', 'wall') and anchors and attempt % 2 == 0:
                if strategy == 'contact':
                    angle = layout.rng.uniform(-np.pi, np.pi)
                sid, anchor = anchors[int(layout.rng.integers(len(anchors)))]
                separation = gap + (stone_margins or {}).get(layout.ids[i], 0) + (stone_margins or {}).get(sid, 0)
                if strategy == 'contact':
                    px, py = _contact_pose(layout.shapes[layout.ids[i]], anchor, angle, separation, layout.rng)
                else:
                    px, py, angle = _edge_pose(layout.shapes[layout.ids[i]], anchor, separation, layout.rng)
            else:
                if strategy not in ('random', 'contact', 'edge', 'wall') and attempt % 2 == 0 and attempt // 2 < len(candidates):
                    _, px, py = candidates[attempt // 2]
                else:
                    px, py = layout.rng.uniform(minx, maxx), layout.rng.uniform(miny, maxy)
                angle = layout.rng.uniform(-np.pi, np.pi)
            if proposal_bounds is not None and not (minx <= px <= maxx and miny <= py <= maxy):
                continue
            prechecked_valid = oracle.can_insert(i, [px, py, angle]) if oracle is not None else False
            if oracle is not None and not prechecked_valid and not relaxation_steps:
                continue
            if (not prechecked_valid and max_relaxation_penalty is not None and
                    oracle.relaxation_penalty(i, [px, py, angle]) > max_relaxation_penalty):
                relaxation_skipped += 1
                continue
            nearby = (oracle.nearby_indices(i, [px, py, angle], extra=gap)
                      if mobility_policy == 'clearance' and relaxation_steps and not prechecked_valid else None)
            accepted_proposal = False
            relaxed = False
            try:
                layout.poses[i] = [px, py, angle]
                layout.active[i] = True
                if relaxation_steps and not prechecked_valid:
                    relaxation_attempts += 1
                    relaxed = True
                    # Mobilize only stones whose bounding circles are near the insertion.
                    if nearby is None:
                        radii = np.array([max(np.linalg.norm(np.asarray(layout.shapes[s].exterior.coords)[:, :2], axis=1)) for s in layout.ids])
                        distances = np.linalg.norm(layout.poses[:, :2] - layout.poses[i, :2], axis=1)
                        mobility = (layout.active & (distances <= radii + radii[i] + 2 * gap + 2 * max((stone_margins or {}).values(), default=0))).astype(float)
                    else:
                        mobility = np.zeros(len(layout.ids), dtype=float)
                        mobility[nearby] = 1
                    mobility[i] = 1
                    for frozen_index in frozen:
                        mobility[frozen_index] = 0
                    mobile_stone_visits += int(np.count_nonzero(mobility))
                    solver = relax_neighborhood if local_relaxation else relax
                    solver(layout, gap=gap, edge=edge, max_steps=relaxation_steps, step=relaxation_step, mobility=mobility,
                           stone_margins=stone_margins, patio_margin=patio_margin, tolerance=tolerance)
                if not validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin):
                    if relaxed:
                        relaxation_successes += 1
                    accepted += 1
                    accepted_proposal = True
                    break
                if relaxed:
                    relaxation_failures += 1
            finally:
                if not accepted_proposal:
                    layout.poses[:] = poses
                    layout.active[:] = active
    # With the private token the caller owns an already-validated state: each
    # accepted proposal above is fully validated and every rejection rolls back.
    final_violations = [] if _validation_token is _VALIDATED_INPUT else validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin)
    return SearchResult(attempts, accepted, coverage(layout), time.perf_counter() - start, final_violations, relaxation_attempts, relaxation_skipped, relaxation_successes, relaxation_failures, mobile_stone_visits)


@dataclass
class RebuildResult:
    rounds: int
    improvements: int
    attempts: int
    initial_coverage: float
    coverage: float
    best_history: list
    violations: list
    relaxation_attempts: int = 0
    relaxation_skipped: int = 0
    relaxation_successes: int = 0
    relaxation_failures: int = 0
    mobile_stone_visits: int = 0
    secondary_improvements: int = 0
    secondary_score: dict | None = None


def _patch_scope(layout, removed, patch_padding, max_new_stones):
    """Choose a patch's optional center bounds and candidate inventory."""
    region = None
    if patch_padding is not None:
        bounds = np.array([transformed(layout.shapes[layout.ids[j]], layout.poses[j]).bounds for j in removed])
        region = (float(bounds[:,0].min()-patch_padding), float(bounds[:,1].min()-patch_padding),
                  float(bounds[:,2].max()+patch_padding), float(bounds[:,3].max()+patch_padding))
    selected = None
    if max_new_stones is not None:
        unused = [sid for i, sid in enumerate(layout.ids) if not layout.active[i]]
        target = np.mean([layout.shapes[layout.ids[j]].area for j in removed])
        near = sorted(unused, key=lambda sid: abs(np.log(layout.shapes[sid].area / target)))
        first = near[:(max_new_stones+1)//2]
        first_set = set(first)
        small = sorted((sid for sid in unused if sid not in first_set), key=lambda sid: layout.shapes[sid].area)
        selected = [layout.ids[j] for j in removed] + first + small[:max_new_stones-len(first)]
    return region, selected


def rebuild_search(layout, rounds=10, patch_size=2, proposals=20, gap=.01, edge=.01,
                   strategy='mixed', relaxation_steps=0, stone_margins=None, patio_margin=0, frozen_ids=None, relaxation_step=.02, patch_padding=None, max_new_stones=None, local_relaxation=False, max_relaxation_penalty=None, mobility_policy="circle", tolerance=1e-9,
                   secondary_resolution=None, secondary_broad_distance=None, secondary_edge_strip_width=0, restore_removed=False):
    """Greedily rebuild random local patches, retaining only valid improvements.

    Active stones nearest to a random anchor are removed and the unused inventory
    is reinserted largest first. The physical inventory remains keyed by ID. The
    RNG advances on rejection to avoid retrying identical moves. This is a bounded
    local search, not simulated annealing.
    """
    if not isinstance(local_relaxation, bool) or not isinstance(restore_removed, bool):
        raise ValueError('local_relaxation and restore_removed must be boolean')
    if mobility_policy not in ('circle', 'clearance'):
        raise ValueError('unknown mobility policy')
    if max_relaxation_penalty is not None and (not np.isfinite(max_relaxation_penalty) or max_relaxation_penalty < 0 or not relaxation_steps):
        raise ValueError('max_relaxation_penalty requires nonnegative value and relaxation steps')
    if not isinstance(rounds, int) or rounds < 0 or not isinstance(patch_size, int) or patch_size < 1:
        raise ValueError('rounds must be nonnegative and patch_size positive')
    if not isinstance(proposals, int) or proposals < 0 or not isinstance(relaxation_steps, int) or relaxation_steps < 0:
        raise ValueError('proposal and relaxation budgets must be nonnegative integers')
    if not np.isfinite(relaxation_step) or relaxation_step <= 0:
        raise ValueError('relaxation_step must be finite and positive')
    if strategy not in STRATEGIES:
        raise ValueError('unknown strategy')
    if patch_padding is not None and (not np.isfinite(patch_padding) or patch_padding < 0):
        raise ValueError('patch_padding must be finite and nonnegative')
    if max_new_stones is not None and (not isinstance(max_new_stones, int) or max_new_stones < 0):
        raise ValueError('max_new_stones must be a nonnegative integer')
    if secondary_resolution is not None and (not isinstance(secondary_resolution, int) or secondary_resolution < 2):
        raise ValueError('secondary_resolution must be an integer >= 2')
    if secondary_resolution is not None and (secondary_broad_distance is None or not np.isfinite(secondary_broad_distance) or secondary_broad_distance < 0):
        raise ValueError('secondary_broad_distance must be finite and nonnegative')
    if secondary_resolution is None and secondary_broad_distance is not None:
        raise ValueError('secondary_resolution is required with secondary_broad_distance')
    if not np.isfinite(secondary_edge_strip_width) or secondary_edge_strip_width < 0:
        raise ValueError('secondary_edge_strip_width must be finite and nonnegative')
    if validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin):
        raise ValueError('Initial layout must be valid')
    from .scoring import score_layout, quality_key
    def secondary():
        return score_layout(layout, gap=gap, edge=edge, tolerance=tolerance, stone_margins=stone_margins,
                            patio_margin=patio_margin, resolution=secondary_resolution,
                            broad_distance=secondary_broad_distance, edge_strip_width=secondary_edge_strip_width)['secondary']
    best_secondary = secondary() if secondary_resolution is not None else None
    secondary_improvements = 0
    frozen = _frozen_indices(layout, frozen_ids)
    initial = best = coverage(layout)
    history = [best]
    improvements = attempts = 0
    relaxation_attempts = relaxation_skipped = relaxation_successes = relaxation_failures = 0
    mobile_stone_visits = 0
    completed = 0
    for _ in range(rounds):
        active = np.array([i for i in np.flatnonzero(layout.active) if i not in frozen], dtype=int)
        if not len(active):
            break
        anchor = int(layout.rng.choice(active))
        distances = np.linalg.norm(layout.poses[active, :2] - layout.poses[anchor, :2], axis=1)
        removed = active[np.argsort(distances, kind='stable')[:patch_size]]
        region, selected = _patch_scope(layout, removed, patch_padding, max_new_stones)
        old_poses, old_active = layout.poses.copy(), layout.active.copy()
        try:
            layout.active[removed] = False
            result = greedy_insert(layout, gap=gap, edge=edge, proposals=proposals,
                                   relaxation_steps=relaxation_steps, strategy=strategy,
                                   stone_margins=stone_margins, patio_margin=patio_margin,
                                   frozen_ids=frozen_ids, relaxation_step=relaxation_step,
                                   proposal_bounds=region, stone_ids=selected, local_relaxation=local_relaxation,
                                   max_relaxation_penalty=max_relaxation_penalty, mobility_policy=mobility_policy, tolerance=tolerance)
            attempts += result.attempts
            relaxation_attempts += result.relaxation_attempts
            relaxation_skipped += result.relaxation_skipped
            relaxation_successes += result.relaxation_successes
            relaxation_failures += result.relaxation_failures
            mobile_stone_visits += result.mobile_stone_visits
            if restore_removed:
                # An unsuccessful random reinsertion need not discard an old
                # placement if its original pose still fits the new patch.
                for j in removed:
                    if layout.active[j]:
                        continue
                    oracle = _InsertionOracle(layout, gap=gap, edge=edge, tolerance=tolerance,
                                              stone_margins=stone_margins, patio_margin=patio_margin)
                    if not oracle.can_insert(j, old_poses[j]):
                        continue
                    previous_pose = layout.poses[j].copy()
                    layout.poses[j] = old_poses[j]
                    layout.active[j] = True
                    if validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin):
                        layout.active[j] = False
                        layout.poses[j] = previous_pose
            candidate = coverage(layout)
            if not result.violations and candidate > best + 1e-12:
                best = candidate
                improvements += 1
                if secondary_resolution is not None:
                    best_secondary = secondary()
            elif not result.violations and candidate == best and secondary_resolution is not None:
                candidate_secondary = secondary()
                if quality_key(candidate_secondary) > quality_key(best_secondary):
                    best_secondary = candidate_secondary
                    secondary_improvements += 1
                else:
                    layout.poses[:] = old_poses
                    layout.active[:] = old_active
            else:
                layout.poses[:] = old_poses
                layout.active[:] = old_active
        except BaseException:
            layout.poses[:] = old_poses
            layout.active[:] = old_active
            raise
        completed += 1
        history.append(best)
    return RebuildResult(completed, improvements, attempts, initial, best, history,
                         validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin),
                         relaxation_attempts, relaxation_skipped, relaxation_successes, relaxation_failures, mobile_stone_visits,
                         secondary_improvements, best_secondary)


def batched_random_insert(layout, *, gap=.01, edge=.01, proposals=50, batch_size=8,
                          time_limit=None, stone_ids=None, stone_margins=None,
                          patio_margin=0, frozen_ids=None, tolerance=1e-9):
    """Try random pose proposals in batches, without relaxation.

    The first batch-valid candidate is rechecked by the scalar validator before
    acceptance. All poses in a batch consume RNG draws even if an early candidate
    succeeds, so trajectories depend on batch_size. `attempts` counts evaluated
    candidates, including those after the first successful candidate in a batch.
    """
    from .batch import validate_batch
    if not isinstance(proposals, int) or proposals < 0 or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('proposals must be nonnegative and batch_size positive integers')
    if time_limit is not None and (not np.isfinite(time_limit) or time_limit < 0):
        raise ValueError('time_limit must be finite and nonnegative')
    if validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin):
        raise ValueError('Initial layout must be valid')
    _frozen_indices(layout, frozen_ids)
    selected = set(layout.ids) if stone_ids is None else set(stone_ids)
    if not selected.issubset(layout.ids):
        raise ValueError('stone_ids contains an unknown stone')
    order = sorted((i for i, sid in enumerate(layout.ids) if not layout.active[i] and sid in selected),
                   key=lambda i: -layout.shapes[layout.ids[i]].area)
    start = time.perf_counter(); attempts = accepted = 0
    minx, miny, maxx, maxy = layout.patio.bounds
    for i in order:
        used = 0
        while used < proposals:
            if time_limit is not None and time.perf_counter() - start >= time_limit:
                break
            count = min(batch_size, proposals - used)
            poses = np.repeat(layout.poses[None], count, axis=0)
            active = np.repeat(layout.active[None], count, axis=0)
            for k in range(count):
                poses[k, i] = [layout.rng.uniform(minx, maxx), layout.rng.uniform(miny, maxy),
                               layout.rng.uniform(-np.pi, np.pi)]
                active[k, i] = True
            violations = validate_batch(layout, poses, active, gap=gap, edge=edge,
                                        stone_margins=stone_margins, patio_margin=patio_margin, tolerance=tolerance)
            used += count; attempts += count
            accepted_proposal = False
            old_pose, old_active = layout.poses[i].copy(), layout.active[i]
            try:
                for k, errors in enumerate(violations):
                    if errors:
                        continue
                    layout.poses[i] = poses[k, i]
                    layout.active[i] = True
                    if not validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin):
                        accepted += 1
                        accepted_proposal = True
                        break
            finally:
                if not accepted_proposal:
                    layout.poses[i] = old_pose
                    layout.active[i] = old_active
            if accepted_proposal:
                break
    return SearchResult(attempts, accepted, coverage(layout), time.perf_counter() - start,
                        validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin))
