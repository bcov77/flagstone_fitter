"""Headless simulation, editing transactions, and original-outline interchange.

Coordinates are feet and radians. Collision proxies conservatively enclose the
original outlines. Convex parts are rigidly attached to one Pymunk body per ID.
Pymunk detects contacts and integrates motion; a compliant normal-impulse law
in pre_solve provides independent ordinary/locked contact hardness controls.
"""
from __future__ import annotations

import copy
import math
import random
from collections import deque
from dataclasses import dataclass

import pymunk
from shapely import constrained_delaunay_triangles
from shapely.affinity import affine_transform
from shapely.geometry import Polygon
from shapely.ops import unary_union


# name: default, minimum, maximum, step, display unit
CONTROLS = {
    'gap_inches': (.5, .5, 2, .05, 'in'),
    'piece_hardness': (18, 0, 60, 1, 'Hz'),
    'locked_hardness': (30, 0, 90, 1, 'Hz'),
    'angular_stiffness': (2, 0, 30, .1, 's⁻²'),
    'translation_stiffness': (2, 0, 30, .1, 's⁻²'),
    'rotation_magnitude': (4, 0, 45, .5, '°'),
    'rotation_force': (12, 0, 100, 1, 'gain'),
    'rotation_frequency': (1, 0, 5, .05, '×'),
    'translation_magnitude': (.25, 0, 3, .05, 'in'),
    'translation_force': (12, 0, 100, 1, 'gain'),
    'translation_frequency': (1, 0, 5, .05, '×'),
    'speed': (1, .1, 4, .1, '×'),
    'wall_hardness': (40, 0, 90, 1, 'Hz'),
}
DT = 1 / 240
GAP = .5 / 12
PROXY_TOLERANCE = .002  # feet; 0.024 inch, conservatively enclosed


def finite_pose(pose):
    if not isinstance(pose, (list, tuple)) or len(pose) != 3:
        raise ValueError('A pose must have x, y, and angle.')
    result = [float(v) for v in pose]
    if not all(math.isfinite(v) and abs(v) < 1e6 for v in result):
        raise ValueError('Pose coordinates must be finite and reasonably bounded.')
    return result


def convex_parts(outline):
    """Enclosing simplification, then checked exact triangulation/convex merges."""
    original = Polygon(outline)
    simplified = original.simplify(PROXY_TOLERANCE, preserve_topology=True)
    proxy = simplified.buffer(PROXY_TOLERANCE, join_style='mitre')
    if not proxy.covers(original):
        proxy = simplified.buffer(2 * PROXY_TOLERANCE, join_style='mitre')
    if proxy.geom_type != 'Polygon' or proxy.interiors or not proxy.covers(original):
        proxy = original
    parts = dict(enumerate(constrained_delaunay_triangles(proxy).geoms))
    # Merge adjacent triangles when their union is convex. Keep local outlines;
    # individual part centroids must not change the stone's reference frame.
    edges = {}
    neighbors = {i: set() for i in parts}
    for i, part in parts.items():
        coords = list(part.exterior.coords)
        for a, b in zip(coords, coords[1:]):
            edge = tuple(sorted((a, b)))
            if edge in edges:
                j = edges[edge]
                neighbors[i].add(j)
                neighbors[j].add(i)
            else:
                edges[edge] = i
    queue = deque((i, j) for i, ns in neighbors.items() for j in ns if i < j)
    while queue:
        i, j = queue.popleft()
        if i not in parts or j not in parts:
            continue
        a, b = parts[i], parts[j]
        hull = a.union(b).convex_hull
        if abs(hull.area - a.area - b.area) > 1e-11:
            continue
        parts[i] = hull
        del parts[j]
        neighbors[i] = (neighbors[i] | neighbors[j]) - {i, j}
        for other in neighbors[i]:
            neighbors[other].discard(j)
            neighbors[other].add(i)
            queue.append((i, other))
    union = unary_union(list(parts.values()))
    if proxy.symmetric_difference(union).area > 1e-9:
        raise ValueError('Collision decomposition does not cover its proxy.')
    return tuple(tuple((float(x), float(y)) for x, y in p.exterior.coords[:-1])
                 for p in parts.values())


@dataclass
class Piece:
    sid: str
    outline: list
    pose: list
    active: bool
    locked: bool
    anchor: list
    area: float


class PhysicsSession:
    """Reusable backend. advance() takes simulated seconds; it never sleeps."""

    def __init__(self, document, seed=0):
        if document.get('units') != 'ft':
            raise ValueError('This sandbox currently accepts layouts in feet (ft).')
        self.document = copy.deepcopy(document)
        self.patio = Polygon(document['patio'])
        if not self.patio.is_valid or self.patio.is_empty or self.patio.interiors:
            raise ValueError('Patio must be a valid polygon without holes.')
        # This first backend explicitly supports the quarter-square boundary.
        if not self.patio.equals(self.patio.envelope):
            raise ValueError('This first physics backend requires a rectangular patio.')
        self.pieces = {}
        meta = document.get('physics', {})
        for stone in document['stones']:
            sid = str(stone['id'])
            if sid in self.pieces:
                raise ValueError('Duplicate stone ID.')
            shape = Polygon(stone['outline'])
            if not shape.is_valid or shape.is_empty or shape.area <= 0 or shape.interiors:
                raise ValueError(f'Invalid outline for stone {sid}.')
            pose = finite_pose(stone['pose'])
            extra = meta.get('pieces', {}).get(sid, {})
            self.pieces[sid] = Piece(sid, copy.deepcopy(stone['outline']), pose,
                                     bool(stone.get('active', False)),
                                     bool(extra.get('locked', False)),
                                     finite_pose(extra.get('anchor', pose)), shape.area)
        self.settings = {key: value[0] for key, value in CONTROLS.items()}
        self._validate_settings(meta.get('settings', {}))
        self.settings.update(meta.get('settings', {}))
        self.rng = random.Random(seed)
        self.history = []
        self.running = False
        self.elapsed = 0.0
        self.accumulator = 0.0
        self.run_number = 0
        self.proxy_cache = {}
        self._build()

    def _validate_settings(self, values):
        for key, value in values.items():
            if key not in CONTROLS:
                raise ValueError(f'Unknown setting: {key}')
            _, lo, hi, _, _ = CONTROLS[key]
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{key} must be between {lo} and {hi}.')

    def _build(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = .15
        self.space.iterations = 20
        self.bodies = {}
        self.drives = {}
        self.space.on_collision(None, None, pre_solve=self._contact)
        x0, y0, x1, y1 = self.patio.bounds
        # Solid outside slabs, not hollow segments: deeply misplaced stones
        # overlap the wall solid and get a meaningful inward contact normal.
        outer = 100
        r = self.settings['gap_inches'] / 24
        walls = [((x0-outer, y0-outer), (x0+r, y1+outer)),
                 ((x1-r, y0-outer), (x1+outer, y1+outer)),
                 ((x0, y0-outer), (x1, y0+r)),
                 ((x0, y1-r), (x1, y1+outer))]
        for (a, b) in walls:
            wall = pymunk.Poly(self.space.static_body,
                               [(a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1])])
            wall.collision_type = 3
            self.space.add(wall)
        for index, p in enumerate(self.pieces.values(), 1):
            if not p.active:
                continue
            if p.sid not in self.proxy_cache:
                self.proxy_cache[p.sid] = convex_parts(p.outline)
            mass = max(.1, p.area)
            moment = max(.01, pymunk.moment_for_poly(mass, Polygon(p.outline).convex_hull.exterior.coords[:-1]))
            body = (pymunk.Body(body_type=pymunk.Body.KINEMATIC) if p.locked
                    else pymunk.Body(mass, moment))
            body.position = p.pose[:2]
            body.angle = p.pose[2]
            body.piece_id = p.sid
            self.space.add(body)
            self.bodies[p.sid] = body
            for part in self.proxy_cache[p.sid]:
                shape = pymunk.Poly(body, part, radius=r)
                shape.filter = pymunk.ShapeFilter(group=index)
                shape.collision_type = 2 if p.locked else 1
                self.space.add(shape)

    @staticmethod
    def _inverse_mass(body, point, normal):
        if body.body_type != pymunk.Body.DYNAMIC:
            return 0.0
        arm = point - body.position
        return 1 / body.mass + arm.cross(normal) ** 2 / body.moment

    def _contact(self, arbiter, space, data):
        """Implicit damped spring impulse along Pymunk's contact normals.

        Native hard response is disabled for these contacts so that it cannot
        override the hardness sliders. No bounce or tangent friction is added.
        """
        arbiter.process_collision = False
        types = {s.collision_type for s in arbiter.shapes}
        key = 'wall_hardness' if 3 in types else ('locked_hardness' if 2 in types else 'piece_hardness')
        hz = self.settings[key]
        if hz == 0:
            return
        a, b = arbiter.bodies
        contact = arbiter.contact_point_set
        n = contact.normal
        for point in contact.points:
            depth = max(0, -point.distance)
            pa, pb = point.point_a, point.point_b
            inv = self._inverse_mass(a, pa, n) + self._inverse_mass(b, pb, n)
            if inv <= 0:
                continue
            effective_mass = 1 / inv
            omega = 2 * math.pi * hz
            k = effective_mass * omega * omega
            c = 2 * effective_mass * omega
            vn = (b.velocity_at_world_point(pb) - a.velocity_at_world_point(pa)).dot(n)
            impulse = max(0, (k * depth * DT - (c * DT + k * DT * DT) * vn)
                          / (1 + (c * DT + k * DT * DT) * inv))
            # Prevent explosive correction of intentionally deep insertions.
            impulse = min(impulse, effective_mass * 2)
            if a.body_type == pymunk.Body.DYNAMIC:
                a.apply_impulse_at_world_point(-n * impulse, pa)
            if b.body_type == pymunk.Body.DYNAMIC:
                b.apply_impulse_at_world_point(n * impulse, pb)

    def _sync(self):
        for sid, body in self.bodies.items():
            self.pieces[sid].pose = [body.position.x, body.position.y, body.angle]

    def snapshot(self):
        self._sync()
        return {'pieces': {sid: {'pose': p.pose[:], 'anchor': p.anchor[:],
                                'active': p.active, 'locked': p.locked}
                           for sid, p in self.pieces.items()},
                'settings': self.settings.copy(), 'rng': self.rng.getstate()}

    def _remember(self, label):
        self.history.append((label, self.snapshot()))
        # Compact poses only, never duplicate 163,000 outline vertices.
        if len(self.history) > 200:
            self.history.pop(0)

    def start(self):
        if self.running:
            return
        self._remember('Simulation run')
        self.running = True
        self.elapsed = self.accumulator = 0.0
        self.run_number += 1
        self._randomize_drives()

    def _randomize_drives(self):
        self.drives = {sid: [(self.rng.uniform(.35, 1.35), self.rng.uniform(0, 2 * math.pi))
                             for _ in range(3)] for sid in self.bodies}

    def stop(self):
        self.running = False
        self._sync()
        for body in self.bodies.values():
            body.velocity = (0, 0)
            body.angular_velocity = 0
            body.force = (0, 0)
            body.torque = 0

    def advance(self, seconds):
        if not math.isfinite(seconds) or not 0 <= seconds <= 60:
            raise ValueError('advance seconds must be between zero and 60.')
        if not self.running:
            return
        self.accumulator += seconds
        while self.accumulator + 1e-12 >= DT:
            self._forces()
            self.space.step(DT)
            self.elapsed += DT
            self.accumulator -= DT
        self._sync()

    def _forces(self):
        s = self.settings
        for sid, body in self.bodies.items():
            if body.body_type != pymunk.Body.DYNAMIC:
                continue
            p = self.pieces[sid]
            body.force = (0, 0)
            body.torque = 0
            targets = []
            for frequency, phase in self.drives[sid]:
                targets.append((frequency, phase))
            displacement = body.position - pymunk.Vec2d(*p.anchor[:2])
            desired = pymunk.Vec2d(*(s['translation_magnitude'] / 12 *
                math.sin(2 * math.pi * f * s['translation_frequency'] * self.elapsed + phase)
                for f, phase in targets[:2]))
            force = (-displacement * s['translation_stiffness'] +
                     (desired - displacement) * s['translation_force'] - body.velocity * 3)
            if force.length > 100:
                force = force.normalized() * 100
            body.force = force * body.mass
            angle_error = math.remainder(body.angle - p.anchor[2], 2 * math.pi)
            f, phase = targets[2]
            target_angle = math.radians(s['rotation_magnitude']) * math.sin(
                2 * math.pi * f * s['rotation_frequency'] * self.elapsed + phase)
            acceleration = (-s['angular_stiffness'] * angle_error +
                            s['rotation_force'] * (target_angle - angle_error) - 3 * body.angular_velocity)
            body.torque = max(-100, min(100, acceleration)) * body.moment

    def edit(self, action, sid=None, pose=None, locked=None, values=None):
        # Validate the complete transaction before stopping or writing history.
        if action not in {'pose', 'insert', 'delete', 'lock', 'settings'}:
            raise ValueError('Unknown edit action.')
        if action == 'settings':
            values = dict(values or {})
            self._validate_settings(values)
        else:
            if sid not in self.pieces:
                raise ValueError('Unknown stone ID.')
            if action in {'pose', 'insert'}:
                pose = finite_pose(pose)
            if action != 'insert' and not self.pieces[sid].active:
                raise ValueError('Select a stone on the field.')
            if action == 'insert' and self.pieces[sid].active:
                raise ValueError('This stone is already on the field.')
        was_running = self.running
        self.stop()
        self._remember({'pose': 'Move / rotate stone', 'insert': 'Insert stone',
                        'delete': 'Return stone to inventory', 'lock': 'Lock / unlock stone',
                        'settings': 'Change simulation settings'}[action])
        if action == 'settings':
            self.settings.update(values)
        else:
            p = self.pieces[sid]
            if action in {'pose', 'insert'}:
                p.pose = pose
                p.anchor = pose[:]
                p.active = True
            elif action == 'delete':
                p.active = p.locked = False
            else:
                p.locked = bool(locked) if locked is not None else not p.locked
                p.anchor = p.pose[:]
        self._build()
        if was_running:
            # Editing splits the preceding run and following run into separate
            # undo entries; the manipulation itself remains its own transaction.
            self.start()

    def undo(self):
        self.stop()
        if not self.history:
            return
        _, snapshot = self.history.pop()
        for sid, values in snapshot['pieces'].items():
            p = self.pieces[sid]
            for key, value in values.items():
                setattr(p, key, copy.deepcopy(value))
        self.settings = snapshot['settings'].copy()
        self.rng.setstate(snapshot['rng'])
        self._build()

    def state(self, geometry=False):
        self._sync()
        result = {'running': self.running, 'elapsed': self.elapsed,
                  'settings': self.settings.copy(), 'undo_count': len(self.history),
                  'undo_label': self.history[-1][0] if self.history else None,
                  'pieces': [{'id': p.sid, 'pose': p.pose[:], 'active': p.active,
                              'locked': p.locked} for p in self.pieces.values()]}
        if geometry:
            result.update(patio=list(self.document['patio']), units='ft', controls=CONTROLS)
            for item in result['pieces']:
                item['outline'] = self.pieces[item['id']].outline
        return result

    def export(self):
        """Original layout schema plus optional sandbox metadata, never proxies."""
        self._sync()
        result = copy.deepcopy(self.document)
        for stone in result['stones']:
            p = self.pieces[str(stone['id'])]
            stone['pose'] = p.pose[:]
            stone['active'] = p.active
        result['physics'] = {'version': 1, 'settings': self.settings.copy(),
                             'pieces': {sid: {'locked': p.locked, 'anchor': p.anchor[:]}
                                        for sid, p in self.pieces.items()},
                             'notice': 'Simulation output; original-clearance and gap checks required.'}
        return result

    def clearance_report(self):
        """Independent measurements on original polygons, not collision proxies."""
        self._sync()
        shapes = {}
        errors = []
        for p in self.pieces.values():
            if not p.active:
                continue
            x, y, angle = p.pose
            c, s = math.cos(angle), math.sin(angle)
            g = affine_transform(Polygon(p.outline), (c, -s, s, c, x, y))
            shapes[p.sid] = g
            if not self.patio.covers(g) or g.distance(self.patio.boundary) < GAP - 1e-8:
                errors.append({'kind': 'boundary', 'ids': [p.sid]})
        items = list(shapes.items())
        minimum = None
        for i, (sid, a) in enumerate(items):
            for tid, b in items[i+1:]:
                d = a.distance(b)
                minimum = d if minimum is None else min(minimum, d)
                if d < GAP - 1e-8:
                    errors.append({'kind': 'pair', 'ids': [sid, tid], 'distance_in': d * 12})
        return {'errors': errors, 'passes_clearance': not errors,
                'minimum_pair_in': None if minimum is None else minimum * 12,
                'coverage': sum(g.area for g in shapes.values()) / self.patio.area,
                'gap_goal_checked': False}
