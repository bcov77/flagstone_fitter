"""Phase 1 geometry, deterministic examples, state, JSON, and plotting."""
from dataclasses import dataclass, field
import copy, json
import math
from pathlib import Path
from types import MappingProxyType
import numpy as np
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree
from shapely.affinity import affine_transform

SCHEMA_VERSION = 1

def polygon(coords):
    try:
        points = np.asarray(coords, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Outline must be a single ring of finite 2D coordinates; holes are unsupported') from exc
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('Outline must be a single ring of at least three finite 2D coordinates; holes are unsupported')
    p=Polygon(points)
    if p.is_empty or not p.is_valid or p.area <= 0: raise ValueError('Expected a valid, positive-area simple polygon')
    if p.interiors: raise ValueError('Polygon holes are unsupported')
    return p

def local_shape(coords):
    p=polygon(coords); c=p.centroid
    q=Polygon(np.asarray(p.exterior.coords[:-1])- [c.x,c.y])
    if not q.exterior.is_ccw: q=Polygon(list(q.exterior.coords)[::-1])
    return q


def _is_centroid_local(shape, centroid=None):
    """Allow only floating-point residue relative to the stone's own extent."""
    centroid = shape.centroid if centroid is None else centroid
    minx, miny, maxx, maxy = shape.bounds
    return (abs(centroid.x) <= 1e-12 * (maxx-minx) and
            abs(centroid.y) <= 1e-12 * (maxy-miny))

def transformed(shape, pose):
    x,y,a=pose
    c, s = math.cos(float(a)), math.sin(float(a))
    # Match Shapely's handling of angles close to exact quarter turns.
    if abs(c) < 2.5e-16: c = 0.0
    if abs(s) < 2.5e-16: s = 0.0
    return affine_transform(shape, (c, -s, s, c, float(x), float(y)))

@dataclass
class Layout:
    patio: Polygon
    shapes: dict
    poses: np.ndarray
    active: np.ndarray
    ids: tuple
    rng: np.random.Generator
    units: str = 'normalized'
    source_outlines: dict = field(default_factory=dict)
    def __post_init__(self):
        if self.units not in ('normalized', 'm', 'cm', 'mm', 'ft', 'in'):
            raise ValueError('Unsupported units')
        self.ids = tuple(self.ids)
        if any(not isinstance(sid, str) or not sid.strip() for sid in self.ids):
            raise ValueError('Stone IDs must be nonempty strings; update IDs and all references consistently')
        if len(set(self.ids)) != len(self.ids) or set(self.ids) != set(self.shapes):
            raise ValueError('Stone IDs must be unique and match the inventory')
        self.shapes = MappingProxyType(dict(self.shapes))
        if not set(self.source_outlines).issubset(self.ids):
            raise ValueError('source outlines must refer to known stone IDs')
        self.source_outlines = MappingProxyType({sid: tuple(tuple(map(float, point)) for point in polygon(outline).exterior.coords[:-1]) for sid, outline in self.source_outlines.items()})
        self.poses = np.array(self.poses, dtype=float, copy=True)
        self.active = np.array(self.active, dtype=bool, copy=True)
        if self.poses.shape != (len(self.ids), 3) or not np.isfinite(self.poses).all():
            raise ValueError('poses must have shape (stones, 3) with finite values')
        if self.active.shape != (len(self.ids),):
            raise ValueError('active must contain one flag per stone')
        if self.patio.geom_type != 'Polygon' or self.patio.interiors or not self.patio.is_valid or self.patio.area <= 0:
            raise ValueError('patio must be a valid simple polygon without holes')
        for sid, shape in self.shapes.items():
            if shape.geom_type != 'Polygon' or shape.interiors or not shape.is_valid or shape.area <= 0:
                raise ValueError('stone shapes must be valid simple polygons without holes')
            if not _is_centroid_local(shape):
                raise ValueError(f'stone {sid} must use centroid-local coordinates; use local_shape and adjust its pose')
        # A source reference must describe the same stone after centroid centering;
        # otherwise exported orientation marks could point at the wrong outline.
        for sid, outline in self.source_outlines.items():
            # Identical already-centered rings need no second centering. Apart
            # from avoiding drift, this avoids unstable overlay of nearly
            # coincident polygons at machine precision.
            original = polygon(outline)
            if _is_centroid_local(original) and original.equals_exact(self.shapes[sid], 0):
                continue
            centered = local_shape(outline)
            shape = self.shapes[sid]
            minx, miny, maxx, maxy = shape.bounds
            length_scale = max(1., maxx-minx, maxy-miny)
            if (centered.hausdorff_distance(shape) > 1e-10 * length_scale or
                    centered.symmetric_difference(shape).area > 1e-10 * max(shape.area, centered.area)):
                raise ValueError(f'source outline does not match local shape for stone {sid}')

    def snapshot(self): return (self.poses.copy(), self.active.copy(), copy.deepcopy(self.rng.bit_generator.state))
    def restore(self,s): self.poses[:]=s[0]; self.active[:]=s[1]; self.rng.bit_generator.state=copy.deepcopy(s[2])

def _validate(layout, gap=0.01, edge=0.01, tolerance=1e-9, *, stone_margins=None, patio_margin=0, candidate_pairs=None):
    if not all(np.isfinite(v) and v >= 0 for v in (gap, edge, tolerance, patio_margin)):
        raise ValueError('Clearances, margins and tolerance must be finite and nonnegative')
    stone_margins = {} if stone_margins is None else dict(stone_margins)
    if not set(stone_margins).issubset(layout.ids):
        raise ValueError('stone_margins contains an unknown stone ID')
    if not all(np.isfinite(v) and v >= 0 for v in stone_margins.values()):
        raise ValueError('Stone margins must be finite and nonnegative')
    if layout.poses.shape != (len(layout.ids), 3) or layout.active.shape != (len(layout.ids),):
        raise ValueError('Layout state arrays have invalid shapes')
    if not np.isfinite(layout.poses[layout.active]).all():
        raise ValueError('Active stone poses must contain only finite values')
    out=[]; inset=layout.patio.buffer(-(edge + patio_margin))
    if inset.is_empty or inset.geom_type=='MultiPolygon':
        return [{'kind':'unsupported_inset','measurement':edge + patio_margin,'stone_ids':[]}]
    geoms={}
    for i,sid in enumerate(layout.ids):
        if not layout.active[i]: continue
        g=transformed(layout.shapes[sid],layout.poses[i]); geoms[sid]=g
        # The inset is approximate at curved joins. Direct distance prevents its
        # polygonal approximation from accepting insufficient edge clearance.
        contained = layout.patio.covers(g)
        d = g.distance(layout.patio.boundary) if contained else None
        required_edge = edge + patio_margin + stone_margins.get(sid, 0)
        if not inset.covers(g) or not contained or (d is not None and d < required_edge - tolerance):
            out.append({'kind':'boundary_clearance','stone_ids':[sid], 'measurement':float(d) if d is not None else None, 'outside_area':float(g.difference(layout.patio).area), 'required':required_edge})
    ids=list(geoms)
    shapes = list(geoms.values())
    if shapes:
        if candidate_pairs is None:
            tree = STRtree(shapes)
            max_margin = max((stone_margins.get(sid, 0) for sid in ids), default=0)
            pairs = []
            for i, a in enumerate(shapes):
                minx, miny, maxx, maxy = a.bounds
                radius = gap + stone_margins.get(ids[i], 0) + max_margin
                pairs.extend((i, int(j)) for j in tree.query(box(minx-radius, miny-radius, maxx+radius, maxy+radius)) if j > i)
            pairs.sort()
        else:
            active_indices = {original: rank for rank, original in enumerate(np.flatnonzero(layout.active))}
            pairs = [(active_indices[int(i)], active_indices[int(j)]) for i, j in candidate_pairs]
        for i, j in pairs:
            a, b = shapes[i], shapes[j]
            d = a.distance(b)
            required_gap = gap + stone_margins.get(ids[i], 0) + stone_margins.get(ids[j], 0)
            # A positive-area overlap has zero separation. Avoid constructing
            # an intersection polygon for the many separated candidate pairs.
            if d < required_gap-tolerance or (d == 0 and a.intersection(b).area > 0):
                out.append({'kind':'pair_clearance','stone_ids':[ids[i],ids[j]],'measurement':float(d),'required':required_gap})
    return out


def validate(layout, gap=0.01, edge=0.01, tolerance=1e-9, *, stone_margins=None, patio_margin=0):
    """Authoritative single-layout polygon validation."""
    return _validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin)

def _layout_data(layout):
    """Build the shared layout representation without an intermediate file."""
    stones = []
    for i, sid in enumerate(layout.ids):
        entry = {'id':sid, 'outline':list(layout.shapes[sid].exterior.coords)[:-1], 'pose':layout.poses[i].tolist(), 'active':bool(layout.active[i])}
        if sid in layout.source_outlines:
            entry['source_outline'] = layout.source_outlines[sid]
        stones.append(entry)
    return {'schema_version':SCHEMA_VERSION,'units':layout.units,'patio':list(layout.patio.exterior.coords)[:-1], 'stones':stones}

def save_json(layout,path):
    Path(path).write_text(json.dumps(_layout_data(layout),indent=2,allow_nan=False))

def load_json(path,seed=0):
    return _layout_from_data(json.loads(Path(path).read_text()), seed=seed)

def _layout_from_data(d, seed=0):
    if d.get('schema_version')!=SCHEMA_VERSION: raise ValueError('Unsupported schema_version')
    if 'holes' in d or 'patio_holes' in d: raise ValueError('Patio holes are unsupported')
    if d.get('units') not in ('normalized', 'm', 'cm', 'mm', 'ft', 'in'): raise ValueError('Unsupported units')
    stones=d['stones']; ids=tuple(x['id'] for x in stones)
    if any(not isinstance(sid, str) or not sid.strip() for sid in ids):
        raise ValueError('Stone IDs must be nonempty strings; update IDs and all references consistently')
    if len(set(ids))!=len(ids): raise ValueError('Stone IDs must be unique')
    shapes = {}
    poses = []
    source_outlines = {}
    for stone in stones:
        if 'holes' in stone: raise ValueError('Stone holes are unsupported')
        if not isinstance(stone['active'], bool): raise ValueError('active must be boolean')
        p = polygon(stone['outline']); c = p.centroid
        source_outlines[stone['id']] = stone.get('source_outline', stone['outline'])
        pose = np.asarray(stone['pose'], dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all(): raise ValueError('pose must contain three finite numbers')
        # Preserve the world-space polygon when an imported local outline is off-center.
        # Preserve already-centered serialized vertices exactly across checkpoints.
        # Re-centering floating-point residue on each load causes drift.
        if _is_centroid_local(p, c) and p.exterior.is_ccw:
            shape = p
        else:
            angle = pose[2]; offset = np.array([c.x*np.cos(angle)-c.y*np.sin(angle), c.x*np.sin(angle)+c.y*np.cos(angle)])
            pose = pose.copy(); pose[:2] += offset
            shape = local_shape(stone['outline'])
        poses.append(pose); shapes[stone['id']] = shape
    return Layout(polygon(d['patio']), shapes, np.asarray(poses, dtype=float).reshape((-1, 3)), np.array([x['active'] for x in stones],bool), ids, np.random.default_rng(seed), units=d['units'], source_outlines=source_outlines)

def synthetic(seed=17):
    rng=np.random.default_rng(seed); patio=polygon([(0,0),(1,0),(1,1),(0,1)])
    # Reproducible feasible grid witness, varying rectangles.
    shapes={}; poses=[]; ids=[]
    for row in range(4):
      for col in range(4):
        sid=f'S{row*4+col:02d}'; w=.19; h=.19
        shapes[sid]=local_shape([(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)])
        ids.append(sid); poses.append([.13+col*.245,.13+row*.245,0])
    return Layout(patio,shapes,np.array(poses),np.ones(16,bool),tuple(ids),rng)

def plot(layout,path='examples/layout.png',clearance=0):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,7)); x,y=layout.patio.exterior.xy; ax.plot(x,y,'k-')
    for i,sid in enumerate(layout.ids):
      if not layout.active[i]: continue
      g=transformed(layout.shapes[sid],layout.poses[i]); x,y=g.exterior.xy; ax.fill(x,y,alpha=.45); ax.text(*g.centroid.coords[0],sid,ha='center',va='center',fontsize=8)
      if clearance: x,y=g.buffer(clearance/2).exterior.xy; ax.plot(x,y,'--',lw=.6)
    ax.set_aspect('equal'); ax.autoscale_view(); ax.margins(.03); fig.savefig(path,dpi=160); plt.close(fig)
