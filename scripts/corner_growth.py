"""Corner-anchored two-dimensional experiments; original geometry decides acceptance.

Candidate-placement MILP and exact circle cuts reuse the established strip method.
Reporting rectangles constrain circle centers only. All distances are feet.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union, nearest_points

from flagstone.phase1 import load_json, save_json, transformed, validate
from five_stone_edge_start import bounded_gap

MODULE_SOURCE = Path(__file__).read_text()
SOURCE = 'examples/real_stones_concept/layout.json'
GAP = .5 / 12 + 1e-6


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def frame(patio, vertex=6):
    if not patio.exterior.is_ccw:
        raise ValueError('Corner frame requires CCW original patio')
    ring = np.asarray(patio.exterior.coords)[:-1]
    origin = ring[vertex]
    incoming = ring[vertex - 1] - origin
    outgoing = ring[(vertex + 1) % len(ring)] - origin
    cross = outgoing[0] * incoming[1] - outgoing[1] * incoming[0]
    angle = math.acos(np.clip(incoming @ outgoing / np.linalg.norm(incoming) / np.linalg.norm(outgoing), -1, 1))
    if cross <= 0 or abs(angle - math.pi / 2) > math.radians(3):
        raise ValueError('Selected corner is not convex and approximately right angled')
    theta = math.atan2(outgoing[1], outgoing[0])
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    local = Polygon((np.asarray(patio.exterior.coords) - origin) @ rotation)
    previous = incoming @ rotation
    return local, dict(vertex_index=vertex, origin=origin.tolist(), angle_radians=theta,
                       traced_interior_angle_degrees=math.degrees(angle),
                       adjacent_segment_lengths_ft=[float(np.linalg.norm(incoming)), float(np.linalg.norm(outgoing))],
                       left_wall_dx_dy=float(previous[0] / previous[1]))


def side(step):
    if step < 1:
        raise ValueError('Unsupported prefix')
    odd = step if step % 2 else step - 1
    return .65 if odd == 1 else 1.35 if odd == 3 else math.sqrt(.8 * odd)


def region(contract, step):
    rec = contract['prefix_regions'][step - 1]
    if rec['step'] != step:
        raise ValueError('Bad region numbering')
    return box(0, 0, rec['width_ft'], rec['depth_ft'])


def load_trial(directory):
    root = Path(directory)
    contract = json.loads((root / 'contract.json').read_text())
    if hashlib.sha256(Path(contract['input']).read_bytes()).hexdigest() != contract['source_sha256']:
        raise ValueError('Original source file changed')
    layout = load_json(contract['input'])
    local, spec = frame(layout.patio, contract['frame']['vertex_index'])
    if spec != contract['frame']:
        raise ValueError('Corner frame differs from original source')
    previous = None
    for n, rec in enumerate(contract['prefix_regions'], 1):
        dims = np.array([rec['width_ft'], rec['depth_ft']], dtype=float)
        if rec['step'] != n or not np.isfinite(dims).all() or min(dims) <= 0 or max(dims)/min(dims) > contract['aspect_ratio_limit'] + 1e-12:
            raise ValueError('Invalid declared two-dimensional region')
        if previous is not None and (np.any(dims < previous) or (n % 2 and not np.all(dims > previous))):
            raise ValueError('Declared odd regions must grow in both axes')
        previous = dims
    if len(contract['prefix_regions']) != 13:
        extension=contract.get('extended_from')
        if not extension or len(contract['prefix_regions']) < 13 or len(contract['prefix_regions'])%2!=1:
            raise ValueError('Missing required prefix regions')
        base_bytes=Path(extension['contract']).read_bytes()
        base=json.loads(base_bytes)
        if hashlib.sha256(base_bytes).hexdigest()!=extension['sha256']:
            raise ValueError('Extension base contract changed')
        for key in ('input','source_sha256','frame','minimum_clearance_in',
                    'maximum_empty_circle_diameter_strictly_below_in','aspect_ratio_limit','circle_obstacles'):
            if contract[key]!=base[key]:raise ValueError('Extension changed original problem definition')
        if contract['prefix_regions'][:len(base['prefix_regions'])]!=base['prefix_regions']:
            raise ValueError('Extension changed original prefix regions')
    layout.patio = local
    layout.active[:] = False
    return root, contract, layout


def plot(layout, contract, records, destination, overview=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    x, y = layout.patio.exterior.xy
    ax.plot(x, y, 'k-', label='Actual physical patio boundary')
    steps = [1, 3, 5, 7, 9, 11, 13] if overview else [len(records)]
    for step in steps:
        g = region(contract, step).intersection(layout.patio)
        x, y = g.exterior.xy
        ax.fill(x, y, color='green', alpha=.05 if overview else .12)
        width = contract['prefix_regions'][step - 1]['width_ft']
        ax.plot([width, width, 0], [0, width, width], 'r--', alpha=.35 if overview else 1)
        if overview:
            ax.text(width, width, f' {step} stones: {width:.2f}ft', fontsize=9)
    for k, rec in enumerate(records):
        g = transformed(layout.shapes[rec['stone_id']], rec['pose'])
        x, y = g.exterior.xy
        ax.fill(x, y, color=plt.get_cmap('tab20')(k), alpha=.65)
        ax.text(g.centroid.x, g.centroid.y, rec['stone_id'], ha='center')
    if records and 'center' in records[-1]:
        rec = records[-1]
        ax.add_patch(plt.Circle(rec['center'], rec['lower_diameter_in'] / 24, fill=False, color='red'))
    limit = 4 if overview else max([side(len(records)) + .4, *(transformed(layout.shapes[r['stone_id']], r['pose']).bounds[2] + .15 for r in records), *(transformed(layout.shapes[r['stone_id']], r['pose']).bounds[3] + .15 for r in records)])
    ax.set(xlim=(-.2, limit), ylim=(-.2, limit), aspect='equal', xlabel='Feet from corner along outgoing edge', ylabel='Feet inward from outgoing edge')
    ax.set_title('Predeclared corner growth regions' if overview else f'Corner prefix {len(records)}; reporting fronts are not walls')
    ax.legend(loc='upper left');fig.tight_layout();fig.savefig(destination, dpi=160);plt.close(fig)


def declare(directory):
    root = Path(directory);root.mkdir(parents=True, exist_ok=False)
    layout = load_json(SOURCE)
    local, spec = frame(layout.patio)
    contract = dict(experiment='corner_01', input=SOURCE,
        source_sha256=hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest(), frame=spec,
        rationale='Square center domains anchored at original convex corner6; .8sqft per stone at odd milestones5 onward, near the inventory median .9174sqft. Initial1/3 regions .65/1.35ft square. Both axes grow; even prefixes hold the prior region. Original physical boundary remains unchanged.',
        minimum_clearance_in=.5, maximum_empty_circle_diameter_strictly_below_in=2,
        aspect_ratio_limit=1, circle_obstacles='Actual patio boundary and original rigid stone polygons only',
        prefix_regions=[dict(step=n, width_ft=side(n), depth_ft=side(n), nominal_area_sqft=side(n)**2) for n in range(1,14)],
        method_sources=['https://fontanf.github.io/packingsolver/internals/irregular.html', 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html'])
    write(root / 'contract.json', contract)
    (root / 'source_layout.json').write_bytes(Path(SOURCE).read_bytes())
    (root / 'experiment_script.py').write_text(MODULE_SOURCE)
    layout.patio=local;layout.active[:]=False
    plot(layout, contract, [], root/'growth_schedule.png', overview=True)
    print(json.dumps(dict(frame=spec, regions=contract['prefix_regions'])), flush=True)


def wall_pose(shape, angle, slope, along=0, wall='bottom'):
    g = transformed(shape, (0, 0, angle))
    xy = np.asarray(g.exterior.coords)
    ty = GAP - g.bounds[1]
    tx = float(np.max(slope * (xy[:,1] + ty) - xy[:,0])) + GAP * math.sqrt(1+slope*slope)
    if wall == 'bottom':
        tx += along
    else:
        ty += along;tx += slope * along
    return np.array([tx, ty, angle])


def angles(shape, sid, solver, theta):
    outline = shape.simplify(.025, preserve_topology=True)
    edges = np.diff(np.asarray(outline.exterior.coords), axis=0)
    longest = np.argsort(np.linalg.norm(edges,axis=1))[-6:]
    result = [float(a) for a in np.arange(0, 2*math.pi, math.pi/18)]
    result += [-math.atan2(edges[k,1], edges[k,0]) + q*math.pi/2 for k in longest for q in range(4)]
    result += [r['angle_radians']-theta for r in solver if r['stone_id']==sid]
    return sorted(set(round(a % (2*math.pi), 7) for a in result))


def sample_points(domain, spacing=.06):
    a,b,c,d=domain.bounds
    xx,yy=np.meshgrid(np.linspace(a,c,max(2,math.ceil((c-a)/spacing)+1)),np.linspace(b,d,max(2,math.ceil((d-b)/spacing)+1)))
    pts=shapely.points(xx.ravel(),yy.ravel())
    return pts[shapely.covers(domain,pts)]


def coverage_samples(patio,domain,spacing=.16):
    """Necessary center probes include real reporting fronts and wall offsets.

    These curves are samples only, never obstacles in a circle calculation.
    """
    points=list(sample_points(domain,spacing))
    inset=patio.buffer(-(1/12+1e-6))
    curves=[(domain.boundary,.04)]
    if not inset.is_empty:curves.append((inset.boundary.intersection(domain),.08))
    for geometry,pitch in curves:
        parts=list(geometry.geoms) if hasattr(geometry,'geoms') else [geometry]
        for line in parts:
            if line.geom_type not in ('LineString','LinearRing') or line.is_empty:continue
            points.extend(line.interpolate(float(t)) for t in np.r_[np.arange(0,line.length,pitch),line.length])
    coords=np.asarray([[p.x,p.y] for p in points])
    _,indices=np.unique(np.round(coords,12),axis=0,return_index=True)
    pts=shapely.points(coords[indices])
    return pts[shapely.covers(domain,pts)]


def single_starts(directory, keep=12):
    root, contract, layout = load_trial(directory)
    out=root/'single_starts';out.mkdir(exist_ok=False)
    solver=json.loads(Path('examples/edge_packingsolver_seeds_ccw/orientations.json').read_text())['orientations']
    domain=region(contract,1).intersection(layout.patio)
    pts=sample_points(domain,.06);edge_distance=shapely.distance(pts,layout.patio.boundary)
    ranked=[]
    for sid, shape in layout.shapes.items():
        # Approximate distances guide ranking; detailed geometry certifies.
        for a in angles(shape,sid,solver,contract['frame']['angle_radians']):
            pose=wall_pose(shape,a,contract['frame']['left_wall_dx_dy'])
            g=transformed(shape,pose)
            if not layout.patio.covers(g) or g.distance(layout.patio.boundary)<.5/12:continue
            d=np.minimum(edge_distance,shapely.distance(pts,g))
            if float(d.max())>=1/12:continue
            ranked.append((float(d.max()),sid,pose.tolist()))
    ranked.sort(key=lambda row:row[0])
    winners=[];per_id={}
    for score,sid,pose in ranked:
        if per_id.get(sid,0)>=2:continue
        bound=bounded_gap(layout.patio,[transformed(layout.shapes[sid],pose)],domain,error=.005/12)
        if bound['passed']:
            rec=dict(step=1,stone_id=sid,pose=pose,**bound)
            winners.append(rec);per_id[sid]=per_id.get(sid,0)+1
        if len(winners)>=keep:break
    write(out/'candidates.json',dict(passing_starts=winners,coarse_survivors=len(ranked)))
    for k,rec in enumerate(winners):
        trial=out/f'start_{k:02d}';trial.mkdir()
        write(trial/'history.json',[rec]);plot(layout,contract,[rec],trial/'step_1.png')
    print(json.dumps(dict(passing_starts=len(winners),coarse_survivors=len(ranked),ids=[r['stone_id'] for r in winners])),flush=True)


def save_verified(root, contract, history, out):
    """Write physical placements, then independently reload them before acceptance."""
    out=Path(out);out.mkdir(exist_ok=False)
    history=[dict(step=r['step'],stone_id=r['stone_id'],pose=r['pose']) for r in history]
    original=load_json(contract['input']);original.active[:]=False
    theta=contract['frame']['angle_radians'];c,s=math.cos(theta),math.sin(theta)
    origin=np.asarray(contract['frame']['origin'])
    for rec in history:
        i=original.ids.index(rec['stone_id']);x,y,a=rec['pose']
        original.active[i]=True
        original.poses[i]=[origin[0]+c*x-s*y,origin[1]+s*x+c*y,a+theta]
    save_json(original,out/'layout.json')
    _,_,local=load_trial(root)
    for rec in history:
        i=local.ids.index(rec['stone_id']);local.active[i]=True;local.poses[i]=rec['pose']
    save_json(local,out/'layout_local.json')
    (out/'source_layout.json').write_bytes(Path(contract['input']).read_bytes())
    write(out/'history.json',history)
    write(out/'settings.json',dict(trial=str(root),contract_sha256=hashlib.sha256((root/'contract.json').read_bytes()).hexdigest(),target=len(history)))
    return verify(out)


def verify(out):
    import csv
    out=Path(out);settings=json.loads((out/'settings.json').read_text())
    root,contract,local=load_trial(settings['trial'])
    if hashlib.sha256((root/'contract.json').read_bytes()).hexdigest()!=settings['contract_sha256']:
        raise ValueError('Declared growth contract changed')
    source=load_json(contract['input']);physical=load_json(out/'layout.json')
    if physical.ids!=source.ids or not physical.patio.equals_exact(source.patio,1e-10) or any(not physical.shapes[sid].equals_exact(source.shapes[sid],1e-10) for sid in source.ids):
        raise ValueError('Saved physical geometry differs from original source')
    history=json.loads((out/'history.json').read_text())
    ids=[r['stone_id'] for r in history]
    if len(ids)!=settings['target'] or len(set(ids))!=len(ids) or set(ids)!={physical.ids[i] for i in np.flatnonzero(physical.active)}:
        raise ValueError('Saved active stones do not match the unique sequence')
    results=[];stones=[];previous=None
    theta=contract['frame']['angle_radians'];c,s=math.cos(theta),math.sin(theta)
    origin=np.asarray(contract['frame']['origin']);physical.active[:]=False
    for n,rec in enumerate(history,1):
        if rec['step']!=n:raise ValueError('Incorrect step numbering')
        i=physical.ids.index(rec['stone_id']);physical.active[i]=True
        x,y,a=physical.poses[i];q=np.array([x,y])-origin
        pose=[c*q[0]+s*q[1],-s*q[0]+c*q[1],a-theta]
        if not np.allclose(pose,rec['pose'],atol=1e-10,rtol=0):raise ValueError('Local history disagrees with physical placements')
        stones.append(transformed(source.shapes[rec['stone_id']],pose))
        domain=region(contract,n).intersection(local.patio)
        if previous is not None and not domain.buffer(1e-12).covers(previous):raise ValueError('Completed region retreated')
        if n>=7 and n%2:
            before=contract['prefix_regions'][n-3];now=contract['prefix_regions'][n-1]
            if now['width_ft']<=before['width_ft'] or now['depth_ft']<=before['depth_ft']:raise ValueError('Both axes must grow')
        previous=domain
        errors=validate(physical,gap=.5/12,edge=.5/12)
        gap=bounded_gap(local.patio,stones,domain,error=.005/12,max_cells=100000)
        results.append(dict(step=n,stone_id=rec['stone_id'],width_ft=contract['prefix_regions'][n-1]['width_ft'],depth_ft=contract['prefix_regions'][n-1]['depth_ft'],area_sqft=domain.area,minimum_edge_clearance_in=12*min(g.distance(local.patio.boundary) for g in stones),minimum_pair_clearance_in=min((12*a.distance(b) for j,a in enumerate(stones) for b in stones[j+1:]),default=None),clearance_errors=errors,**gap))
    passed=all(r['passed'] and not r['clearance_errors'] for r in results)
    from shapely.geometry import mapping,LineString
    from shapely.affinity import affine_transform
    features=[]
    for rec in results:
        n=rec['step'];w=rec['width_ft'];d=rec['depth_ft']
        geometries={'completed_region':region(contract,n).intersection(local.patio),
                    'reporting_fronts':LineString([(w,0),(w,d),(0,d)]).intersection(local.patio)}
        for kind,g in geometries.items():
            for coordinate_frame in ('local','physical'):
                geom=g if coordinate_frame=='local' else affine_transform(g,[c,-s,s,c,*origin])
                features.append(dict(type='Feature',properties=dict(step=n,kind=kind,frame=coordinate_frame,units='ft',width_ft=w,depth_ft=d,certified_area_sqft=rec['area_sqft']),geometry=mapping(geom)))
    write(out/'regions.geojson',dict(type='FeatureCollection',features=features))
    write(out/'verification.json',dict(passed=passed,results=results))
    if not passed:raise RuntimeError('Independent saved-file verification failed')
    columns=['step','stone_id','width_ft','depth_ft','area_sqft','minimum_edge_clearance_in','minimum_pair_clearance_in','lower_diameter_in','upper_diameter_in','passed']
    with (out/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    for n in range(1,len(history)+1):
        rendered=[{**rec,**result} for rec,result in zip(history[:n],results[:n])]
        plot(local,contract,rendered,out/f'step_{n}.png')
    import matplotlib.pyplot as plt
    cols=3;rows=math.ceil(len(history)/cols)
    fig,axes=plt.subplots(rows,cols,figsize=(12,4*rows),squeeze=False)
    for k,ax in enumerate(axes.ravel(),1):
        if k<=len(history):ax.imshow(plt.imread(out/f'step_{k}.png'))
        ax.axis('off')
    fig.tight_layout();fig.savefig(out/'overview.png',dpi=120);plt.close(fig)
    write(out/'contract.json',contract)
    (out/'experiment_script.py').write_text(MODULE_SOURCE)
    lines=[f'# Independently verified corner prefix: {len(history)} stones','',
        'Both reporting fronts constrain circle centers only. All radii use the actual patio boundary and original stone polygons. Unfinished space beyond the shaded square is not certified. Preliminary photographed/traced geometry remains conditional on measurement/calibration limitations in first_real_patio_test.md.','',
        '| Step | ID | Width/depth (ft) | Certified area (sqft) | Edge min (in) | Pair min (in) | Gap diameter interval (in) |',
        '|---|---|---|---|---|---|---|']
    for r in results:
        pair='—' if r['minimum_pair_clearance_in'] is None else f"{r['minimum_pair_clearance_in']:.6f}"
        lines.append(f"| {r['step']} | {r['stone_id']} | {r['width_ft']:.6f} | {r['area_sqft']:.6f} | {r['minimum_edge_clearance_in']:.6f} | {pair} | [{r['lower_diameter_in']:.6f}, {r['upper_diameter_in']:.6f}] |")
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(independent_corner_prefix_verification=True,count=len(history),directory=str(out))),flush=True)
    return results


def export_pool(directory, parent, output, seed=2711, proposals=18000, target_override=None):
    """Finite poses from existing edge alignment, wall contact and solver angles."""
    from flagstone.search import _edge_pose
    root,contract,layout=load_trial(directory)
    history=json.loads((Path(parent)/'history.json').read_text())
    n=len(history);target=n+2 if target_override is None else target_override
    if target<=n or target>len(contract['prefix_regions']):raise ValueError("Pool target must advance the saved prefix")
    domain=region(contract,target).intersection(layout.patio)
    fixed=[transformed(layout.shapes[r['stone_id']],r['pose']) for r in history]
    used={r['stone_id'] for r in history}
    fixed_simple=[g.simplify(.001,preserve_topology=True) for g in fixed]
    pts=sample_points(domain,.075)
    obstacles=unary_union([layout.patio.boundary,*fixed_simple])
    uncovered=pts[shapely.distance(pts,obstacles)>1/12]
    rng=np.random.default_rng(seed)
    solver=json.loads(Path('examples/edge_packingsolver_seeds_ccw/orientations.json').read_text())['orientations']
    angles_by_id={sid:angles(shape,sid,solver,contract['frame']['angle_radians']) for sid,shape in layout.shapes.items() if sid not in used}
    proxies={}
    for sid in angles_by_id:
        shape=layout.shapes[sid];g=shape.simplify(.002,preserve_topology=True).buffer(.003,quad_segs=4)
        proxies[sid]=g if g.covers(shape) else shape
    fixed_proxy=[]
    for r,g in zip(history,fixed):
        shape=layout.shapes[r['stone_id']];proxy=shape.simplify(.002,preserve_topology=True).buffer(.003,quad_segs=4)
        fixed_proxy.append(transformed(proxy,r['pose']) if proxy.covers(shape) else g)
    choices=list(angles_by_id);candidates=[];seen=set()
    def consider(sid,pose):
        key=(sid,*np.round(pose,5))
        if key in seen:return
        seen.add(key)
        g=transformed(layout.shapes[sid],pose)
        if g.distance(domain)>1/12 or not layout.patio.covers(g) or g.distance(layout.patio.boundary)<.5/12:return
        if any(g.distance(h)<.5/12 for h in fixed):return
        rank=float(np.sum(shapely.distance(uncovered,g)<1/12)) if len(uncovered) else 0
        if rank==0:return
        candidates.append(dict(stone_id=sid,pose=np.asarray(pose).tolist(),covered_samples=rank))
    for attempt in range(proposals):
        sid=choices[int(rng.integers(len(choices)))];shape=proxies[sid]
        a=float(rng.choice(angles_by_id[sid]))
        mode=attempt%4
        if mode==0:
            anchor=fixed_proxy[int(rng.integers(len(fixed_proxy)))]
            pose=np.array(_edge_pose(shape,anchor,GAP,rng))
            consider(sid,pose)
        elif mode in (1,2):
            # Translate along a true corner wall toward contact with fixed stones.
            pose=wall_pose(shape,a,contract['frame']['left_wall_dx_dy'],wall='bottom' if mode==1 else 'left')
            direction=np.array([1.,0.,0.]) if mode==1 else np.array([contract['frame']['left_wall_dx_dy'],1.,0.])
            def separated(t):
                g=transformed(shape,pose+direction*t)
                return all(g.distance(h)>=GAP for h in fixed_proxy)
            # Find exposed feasible contact from the far side, without crossing a neighbor.
            hi=side(target)+1
            if not separated(hi):continue
            lo=hi
            while lo>0 and separated(max(0,lo-.12)):
                lo=max(0,lo-.12)
            if lo>0:
                bad=max(0,lo-.12)
                for _ in range(14):
                    mid=(bad+lo)/2
                    if separated(mid):lo=mid
                    else:bad=mid
            consider(sid,pose+direction*lo)
        else:
            # Interior poses centered near a required uncovered point; rigid only.
            if len(uncovered):
                p=uncovered[int(rng.integers(len(uncovered)))];g=transformed(shape,(0,0,a))
                consider(sid,np.array([p.x+float(rng.uniform(-.5,.5)),p.y+float(rng.uniform(-.5,.5)),a]))
        if attempt%2000==0:print(json.dumps(dict(proposals=attempt,candidates=len(candidates))),flush=True)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    write(out/'candidate_pool.json',dict(trial=str(root),parent=str(parent),parent_history=history,target=target,seed=seed,proposals=proposals,candidates=candidates))
    (out/'experiment_script.py').write_text(MODULE_SOURCE)
    print(json.dumps(dict(pool=str(out),candidates=len(candidates),target=target)),flush=True)


def conflict_clique(polygons,a,b,gap=.5/12):
    """Lift a real spacing conflict to an adaptive half-gap occupancy clique.

    Any two polygons within gap/2 of the SAME point have separation <gap by
    the triangle inequality. Thus at most one such placement may be selected.
    """
    if polygons[a].distance(polygons[b])>=gap:raise ValueError('Conflict witness needs a true spacing violation')
    pa,pb=nearest_points(polygons[a],polygons[b]);point=Point((pa.x+pb.x)/2,(pa.y+pb.y)/2)
    cover=shapely.distance(polygons,point)<gap/2
    fallback=not(cover[a] and cover[b])
    if fallback:
        cover[:]=False;cover[[a,b]]=True
    return cover,dict(point=[point.x,point.y],covering_poses=int(cover.sum()),fallback_pair_only=bool(fallback))


def select_pair(pool,output,seconds=180,excluded_id_pairs=None,scout=False,backend="highs"):
    import time
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix,csr_matrix,vstack,hstack
    if scout and backend!='highs':raise ValueError('Radius-level scout uses HiGHS; Boolean backend applies to the strict model')
    data=json.loads((Path(pool)/'candidate_pool.json').read_text())
    root,contract,layout=load_trial(data['trial'])
    history=data['parent_history'];target=data['target'];domain=region(contract,target).intersection(layout.patio)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    (out/'experiment_script.py').write_text(MODULE_SOURCE)
    write(out/'settings.json',dict(pool=str(pool),seconds=seconds,excluded_id_pairs=excluded_id_pairs or [],scout=scout,backend=backend))
    candidates=data['candidates'];n=len(candidates)
    if n<2:raise ValueError('Not enough candidate poses')
    polygons=np.asarray([transformed(layout.shapes[r['stone_id']],r['pose']) for r in candidates],dtype=object)
    fixed=[transformed(layout.shapes[r['stone_id']],r['pose']) for r in history]
    fixed_obstacles=unary_union([layout.patio.boundary,*fixed])
    groups={sid:k for k,sid in enumerate(sorted({r['stone_id'] for r in candidates}))}
    identities=coo_matrix((np.ones(n),([groups[r['stone_id']] for r in candidates],np.arange(n))),shape=(len(groups),n)).tocsr()
    # Sampled occupancy of inscribed half-clearance buffers is a necessary
    # nonoverlap condition. Missing conflicts receive exact cuts below.
    cells={};rows=[];cols=[];spacing=.065
    for j,g in enumerate(polygons):
        inner=g.simplify(.001,preserve_topology=True).buffer(-.0011,quad_segs=4)
        if inner.is_empty or not g.covers(inner):inner=g
        inner=inner.buffer(.5/24,quad_segs=4)
        a,b,c,d=inner.bounds
        xx,yy=np.meshgrid(np.arange(math.floor(a/spacing),math.ceil(c/spacing)+1),np.arange(math.floor(b/spacing),math.ceil(d/spacing)+1))
        pts=shapely.points((xx.ravel()+.5)*spacing,(yy.ravel()+.5)*spacing)
        mask=shapely.contains(inner,pts)
        for x,y in zip(xx.ravel()[mask],yy.ravel()[mask]):
            row=cells.setdefault((int(x),int(y)),len(cells));rows.append(row);cols.append(j)
    occupancy=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(cells),n)).tocsr()
    occupancy=occupancy[np.asarray(occupancy.getnnz(axis=1)).ravel()>1]
    base=vstack([csr_matrix(np.ones((1,n))),identities,occupancy],format='csr')
    constraints=[LinearConstraint(base,np.r_[2,np.zeros(base.shape[0]-1)],np.r_[2,np.ones(base.shape[0]-1)])]
    for excluded in excluded_id_pairs or []:
        row=np.asarray([float(r['stone_id'] in excluded) for r in candidates])
        constraints.append(LinearConstraint(csr_matrix(row.reshape(1,-1)),-np.inf,1))
    cuts=[];cover_rows=[]
    def cut(center):
        cover=shapely.distance(polygons,Point(*center))<1/12
        cover_rows.append(len(constraints))
        constraints.append(LinearConstraint(csr_matrix(cover.astype(float).reshape(1,-1)),1,np.inf))
        cuts.append(dict(center=list(center),covering_candidates=int(cover.sum())))
    pts=coverage_samples(layout.patio,domain,.12 if scout else .18)
    uncovered=pts[shapely.distance(pts,fixed_obstacles)>=1/12]
    for p in uncovered:cut([p.x,p.y])
    print(json.dumps(dict(candidates=n,occupancy_rows=occupancy.shape[0],initial_gap_cuts=len(cuts))),flush=True)
    scout_sample_distances=np.asarray([shapely.distance(polygons,p) for p in uncovered]) if scout else None
    start=time.monotonic();trace=[];accepted=None
    for iteration in range(100):
        remaining=seconds-(time.monotonic()-start)
        if remaining<=0:break
        # Scout mode minimizes a global radius level over necessary sample
        # centers. This supplies repair starts; acceptance still uses the exact
        # original two-inch gate independently of the chosen proposal level.
        size=n
        active_constraints=constraints
        objective=np.zeros(n);integer=np.ones(n)
        if scout:
            fixed_dist=shapely.distance(uncovered,fixed_obstacles)
            radii=np.unique(np.r_[np.arange(1,3.001,.125)/12, max(float(fixed_dist.max())+1e-6,3.01/12)])
            m=len(radii);size=n+m
            objective=np.r_[np.zeros(n),24*radii];integer=np.ones(size)
            # Initial 1-inch coverage rows are replaced by radius-level rows.
            active_constraints=[LinearConstraint(hstack([con.A,csr_matrix((con.A.shape[0],m))],format='csr'),con.lb,con.ub) for index,con in enumerate(constraints) if index not in cover_rows]
            active_constraints.append(LinearConstraint(csr_matrix(np.r_[np.zeros(n),np.ones(m)].reshape(1,-1)),1,1))
            rr=[];cc=[];vv=[];row=0
            for k,radius in enumerate(radii):
                for point_dist,fd in zip(scout_sample_distances,fixed_dist):
                    if fd<radius:continue
                    covering=np.flatnonzero(point_dist<radius)
                    rr.extend([row]*len(covering));cc.extend(covering.tolist());vv.extend([1.]*len(covering))
                    rr.append(row);cc.append(n+k);vv.append(-1.);row+=1
            matrix=coo_matrix((vv,(rr,cc)),shape=(row,size)).tocsr()
            active_constraints.append(LinearConstraint(matrix,0,np.inf))
        result=milp(objective,integrality=integer,bounds=Bounds(0,1),constraints=active_constraints,options=dict(time_limit=min(20,remaining),presolve=False,mip_rel_gap=.05)) if scout else solve_boolean_model(size,constraints,min(20,remaining),backend=backend)
        rec=dict(iteration=iteration,status=int(result.status),message=result.message)
        if result.x is None:
            trace.append(rec);print(json.dumps(rec),flush=True);break
        chosen=np.flatnonzero(result.x[:n]>.5)
        if len(chosen)!=2:raise RuntimeError('Returned incumbent does not select exactly two candidates')
        rec.update(chosen=chosen.tolist(),stone_ids=[candidates[j]['stone_id'] for j in chosen])
        if scout:rec['sampled_minimax_diameter_objective_in']=float(result.fun)
        a,b=chosen
        if polygons[a].distance(polygons[b])<.5/12:
            cover,witness=conflict_clique(polygons,int(a),int(b))
            constraints.append(LinearConstraint(csr_matrix(cover.astype(float).reshape(1,-1)),-np.inf,1));rec['conflict']=True;rec['conflict_witnesses']=[witness]
        else:
            gap=bounded_gap(layout.patio,fixed+[polygons[j] for j in chosen],domain,error=.02/12)
            rec.update(gap)
            print(json.dumps(rec),flush=True)
            if gap['passed']:accepted=chosen.tolist()
            elif scout:
                row=np.asarray([float(r['stone_id'] in rec['stone_ids']) for r in candidates])
                constraints.append(LinearConstraint(csr_matrix(row.reshape(1,-1)),-np.inf,1))
            elif gap['lower_diameter_in']>=2:cut(gap['center'])
            else:rec['unresolved']=True
        trace.append(rec)
        write(out/'trace.json',dict(accepted=accepted,cuts=cuts,iterations=trace))
        if accepted is not None or rec.get('unresolved') or (scout and sum('chosen' in r and not r.get('conflict',False) for r in trace)>=4):break
    write(out/'trace.json',dict(accepted=accepted,cuts=cuts,iterations=trace,elapsed_seconds=time.monotonic()-start))
    if accepted is not None:
        records=list(history)
        for j in accepted:
            candidate=candidates[j]
            records.append(dict(step=len(records)+1,stone_id=candidate['stone_id'],pose=candidate['pose']))
        save_verified(root,contract,records,out/'accepted')
    print(json.dumps(dict(accepted=accepted is not None,target=target,output=str(out))),flush=True)


def augment_pool(pool,incumbent,output):
    """Retain exact verified continuation poses without fixing them in the model."""
    source=Path(pool);incumbent=Path(incumbent)
    data=json.loads((source/'candidate_pool.json').read_text())
    root,contract,layout=load_trial(data['trial'])
    verification=json.loads((incumbent/'verification.json').read_text())
    settings=json.loads((incumbent/'settings.json').read_text())
    if not verification['passed'] or settings['contract_sha256']!=hashlib.sha256((root/'contract.json').read_bytes()).hexdigest():
        raise ValueError('Incumbent is not verified under this contract')
    history=json.loads((incumbent/'history.json').read_text());parent=data['parent_history'];n=len(parent)
    if len(history)<n+2:raise ValueError('Incumbent does not contain the next odd stage')
    for old,new in zip(parent,history):
        if old['stone_id']!=new['stone_id'] or not np.allclose(old['pose'],new['pose'],atol=1e-10,rtol=0):
            raise ValueError('Incumbent does not extend the exact fixed parent poses')
    fixed=[transformed(layout.shapes[r['stone_id']],r['pose']) for r in parent]
    used={r['stone_id'] for r in parent};hint_indices=[];added=0
    for rec in history[n:data['target']]:
        sid=rec['stone_id'];pose=rec['pose'];g=transformed(layout.shapes[sid],pose)
        if sid in used or not layout.patio.covers(g) or g.distance(layout.patio.boundary)<.5/12-1e-9 or any(g.distance(h)<.5/12-1e-9 for h in fixed):
            raise ValueError('Incumbent candidate violates detailed fixed-parent clearance')
        matches=[j for j,r in enumerate(data['candidates']) if r['stone_id']==sid and np.allclose(r['pose'],pose,atol=1e-12,rtol=0)]
        if matches:j=matches[0]
        else:
            j=len(data['candidates']);added+=1
            data['candidates'].append(dict(stone_id=sid,pose=pose,source='independently_verified_incumbent'))
        if rec['step']<=n+2:hint_indices.append(j)
    if len(hint_indices)!=2:raise ValueError('Missing exactly two next-stage incumbent poses')
    data.update(derived_from_pool=str(source),included_verified_prefix=str(incumbent),first_stage_hint_indices=hint_indices)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    write(out/'candidate_pool.json',data)
    (out/'experiment_script.py').write_text(MODULE_SOURCE)
    print(json.dumps(dict(augmented_pool=str(out),candidate_count=len(data['candidates']),added_verified_poses=added,first_stage_hint_indices=hint_indices)),flush=True)


def solve_boolean_model(size,constraints,seconds,backend='highs',heuristic_effort=.05,hints=None,workers=2,state=None):
    """Two established backends for the identical 0/1 cardinality/cover model."""
    if backend=='highs':
        from scipy.optimize import milp,Bounds
        options=dict(time_limit=seconds,presolve=False,mip_rel_gap=0)
        if heuristic_effort!=.05:options['mip_heuristic_effort']=heuristic_effort
        return milp(np.zeros(size),integrality=np.ones(size),bounds=Bounds(0,1),constraints=constraints,options=options)
    if backend!='cp_sat':raise ValueError('Unsupported joint backend')
    from types import SimpleNamespace
    from ortools.sat.python import cp_model
    import time
    build_start=time.monotonic();state={} if state is None else state
    if 'model' not in state:
        state.update(model=cp_model.CpModel(),size=size,constraint_refs=[],rows_total=0)
        state['xs']=[state['model'].new_bool_var(f'pose_{j}') for j in range(size)]
        state['solver']=cp_model.CpSolver()
    old=state['constraint_refs']
    if state['size']!=size or len(constraints)<len(old) or any(a is not b for a,b in zip(old,constraints)):
        raise ValueError('Cached Boolean model requires append-only unchanged constraints')
    model=state['model'];xs=state['xs'];rows_added=0
    for con in constraints[len(old):]:
        matrix=con.A.tocsr();rows_added+=matrix.shape[0]
        lbs=np.broadcast_to(con.lb,(matrix.shape[0],));ubs=np.broadcast_to(con.ub,(matrix.shape[0],))
        for row in range(matrix.shape[0]):
            begin,end=matrix.indptr[row:row+2]
            indices=matrix.indices[begin:end];values=matrix.data[begin:end]
            if not np.all(values==1):raise ValueError('Boolean backend expects unit cardinality coefficients')
            variables=[xs[int(j)] for j in indices]
            lb,ub=lbs[row],ubs[row]
            if lb==0 and ub==1:model.add_at_most_one(variables)
            elif lb==1 and np.isposinf(ub):model.add_bool_or(variables)
            elif lb==ub and np.isfinite(lb) and lb==int(lb):model.add(sum(variables)==int(lb))
            elif np.isneginf(lb) and ub==1:model.add_at_most_one(variables)
            else:raise ValueError(f'Unsupported Boolean bounds {lb}/{ub}')
    state['constraint_refs']=list(constraints);state['rows_total']+=rows_added
    model.clear_hints()
    for index,value in (hints or {}).items():model.add_hint(xs[int(index)],int(value))
    solver=state['solver'];solver.parameters.max_time_in_seconds=seconds
    solver.parameters.num_search_workers=workers;solver.parameters.random_seed=3705
    build_seconds=time.monotonic()-build_start;solve_start=time.monotonic()
    status=solver.solve(model)
    performance=dict(build_seconds=build_seconds,solve_seconds=time.monotonic()-solve_start,new_rows=rows_added,total_rows=state['rows_total'])
    feasible=status in (cp_model.OPTIMAL,cp_model.FEASIBLE)
    mapped=0 if status==cp_model.OPTIMAL else 1 if feasible or status==cp_model.UNKNOWN else 2 if status==cp_model.INFEASIBLE else 4
    return SimpleNamespace(x=np.asarray([solver.value(x) for x in xs],dtype=float) if feasible else None,status=mapped,message=solver.status_name(status),performance=performance)


def select_joint_stages(pool,output,seconds=240,solve_limit=30,heuristic_effort=.05,backend="highs",workers=2,raster_spacing=.075,resume_selection=None):
    """Jointly assign pairs of new placements to multiple odd-prefix closures.

    Candidate-placement raster MILP with stage quotas and prefix-labeled exact
    gap cuts. Intermediate held-region prefixes are rechecked from physical files.
    """
    import time
    from scipy.optimize import milp,Bounds,LinearConstraint
    from scipy.sparse import coo_matrix,csr_matrix,vstack,hstack
    data=json.loads((Path(pool)/'candidate_pool.json').read_text())
    root,contract,layout=load_trial(data['trial'])
    history=data['parent_history'];base_count=len(history);target=data['target']
    if not math.isfinite(raster_spacing) or raster_spacing<=0:raise ValueError('Raster spacing must be positive and finite')
    if base_count%2!=1 or target<=base_count or (target-base_count)%2 or target-base_count<4:
        raise ValueError('Joint model needs at least two paired stages after an odd prefix')
    stages=list(range(base_count+2,target+1,2));stage_count=len(stages);new_count=target-base_count
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    write(out/'settings.json',dict(pool=str(pool),seconds=seconds,joint_stages=stages,solve_limit=solve_limit,heuristic_effort=heuristic_effort,backend=backend,workers=workers,raster_spacing=raster_spacing,resume_selection=str(resume_selection) if resume_selection else None,pool_sha256=hashlib.sha256((Path(pool)/'candidate_pool.json').read_bytes()).hexdigest()))
    (out/'experiment_script.py').write_text(MODULE_SOURCE)
    candidates=data['candidates'];n=len(candidates);size=stage_count*n
    polygons=np.asarray([transformed(layout.shapes[r['stone_id']],r['pose']) for r in candidates],dtype=object)
    fixed=[transformed(layout.shapes[r['stone_id']],r['pose']) for r in history]
    fixed_obstacles=unary_union([layout.patio.boundary,*fixed])
    groups={sid:k for k,sid in enumerate(sorted({r['stone_id'] for r in candidates}))}
    identities=coo_matrix((np.ones(n),([groups[r['stone_id']] for r in candidates],np.arange(n))),shape=(len(groups),n)).tocsr()
    cells={};rows=[];cols=[];spacing=raster_spacing
    for j,g in enumerate(polygons):
        inner=g.simplify(.001,preserve_topology=True).buffer(-.0011,quad_segs=4)
        if inner.is_empty or not g.covers(inner):inner=g
        inner=inner.buffer(.5/24,quad_segs=4)
        a,b,c,d=inner.bounds
        xx,yy=np.meshgrid(np.arange(math.floor(a/spacing),math.ceil(c/spacing)+1),np.arange(math.floor(b/spacing),math.ceil(d/spacing)+1))
        pts=shapely.points((xx.ravel()+.5)*spacing,(yy.ravel()+.5)*spacing)
        mask=shapely.contains(inner,pts)
        for x,y in zip(xx.ravel()[mask],yy.ravel()[mask]):
            row=cells.setdefault((int(x),int(y)),len(cells));rows.append(row);cols.append(j)
    occupancy=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(cells),n)).tocsr()
    occupancy=occupancy[np.asarray(occupancy.getnnz(axis=1)).ravel()>1]
    stage_quotas=csr_matrix(np.array([np.repeat(np.arange(stage_count)==stage,n).astype(float) for stage in range(stage_count)]))
    base=vstack([stage_quotas,hstack([identities]*stage_count),hstack([occupancy]*stage_count)],format='csr')
    constraints=[LinearConstraint(base,np.r_[np.full(stage_count,2),np.zeros(base.shape[0]-stage_count)],np.r_[np.full(stage_count,2),np.ones(base.shape[0]-stage_count)])]
    conflict_cliques=set()
    def add_conflict(a,b):
        cover,witness=conflict_clique(polygons,a,b)
        key=np.flatnonzero(cover).tobytes()
        if key not in conflict_cliques:
            conflict_cliques.add(key)
            row=np.tile(cover.astype(float),stage_count)
            constraints.append(LinearConstraint(csr_matrix(row.reshape(1,-1)),-np.inf,1))
        return witness
    cuts=[]
    def cut(stage,center):
        cover=(shapely.distance(polygons,Point(*center))<1/12).astype(float)
        coeff=np.tile(cover,stage_count);coeff[((stage-base_count)//2)*n:]=0
        constraints.append(LinearConstraint(csr_matrix(coeff.reshape(1,-1)),1,np.inf))
        cuts.append(dict(stage=stage,center=list(center),covering_candidates=int(cover.sum())))
    for stage in stages:
        pts=coverage_samples(layout.patio,region(contract,stage).intersection(layout.patio),.16)
        for p in pts[shapely.distance(pts,fixed_obstacles)>=1/12]:cut(stage,[p.x,p.y])
    print(json.dumps(dict(joint_candidates=n,variables=size,occupancy_rows=occupancy.shape[0],initial_gap_cuts=len(cuts))),flush=True)
    if resume_selection is not None:
        prior=Path(resume_selection);prior_settings=json.loads((prior/'settings.json').read_text())
        if Path(prior_settings['pool']).resolve()!=Path(pool).resolve() or prior_settings['joint_stages']!=stages:
            raise ValueError('Selection resume pool or stages differ')
        if prior_settings.get('pool_sha256') and prior_settings['pool_sha256']!=hashlib.sha256((Path(pool)/'candidate_pool.json').read_bytes()).hexdigest():
            raise ValueError('Resume pose pool bytes changed')
        prior_trace=json.loads((prior/'trace.json').read_text())
        seen_centers={(r['stage'],*r['center']) for r in cuts}
        for item in prior_trace['cuts']:
            key=(item['stage'],*item['center'])
            if key not in seen_centers:cut(item['stage'],item['center']);seen_centers.add(key)
        conflict_keys=set()
        for record in prior_trace['iterations']:
            if 'chosen' in record and record['stone_ids']!=[candidates[j]['stone_id'] for j in record['chosen']]:
                raise ValueError('Resume pose column identities changed')
            for a,b in record.get('conflicts',[]):
                key=tuple(sorted((a,b)))
                if key in conflict_keys:continue
                conflict_keys.add(key);add_conflict(a,b)
            if 'chosen' in record:
                for stage in stages:
                    take=stage-base_count
                    bound=bounded_gap(layout.patio,fixed+[polygons[j] for j in record['chosen'][:take]],region(contract,stage),error=.025/12,max_cells=30000,stop_on_failure=False)
                    if bound['lower_diameter_in']>=2:
                        key=(stage,*bound['center'])
                        if key not in seen_centers:cut(stage,bound['center']);seen_centers.add(key)
        print(json.dumps(dict(resumed_gap_rows=len(cuts),resumed_conflict_pairs=len(conflict_keys))),flush=True)
    hint_indices=data.get('first_stage_hint_indices',[])
    hints={j:int(j in hint_indices) for j in range(n)} if len(hint_indices)==2 else {}
    hints.update({phase*n+j:0 for phase in range(1,stage_count) for j in hint_indices})
    solver_state={};next_solve_limit=solve_limit
    trace=[];accepted=None;start=time.monotonic()
    for iteration in range(100):
        remaining=seconds-(time.monotonic()-start)
        if remaining<=0:break
        result=solve_boolean_model(size,constraints,min(next_solve_limit,remaining),backend=backend,heuristic_effort=heuristic_effort,hints=hints,workers=workers,state=solver_state)
        rec=dict(iteration=iteration,status=int(result.status),message=result.message)
        if hasattr(result,'performance'):rec['solver_performance']=result.performance
        if result.x is None:
            trace.append(rec);print(json.dumps(rec),flush=True)
            write(out/'trace.json',dict(accepted=accepted,cuts=cuts,iterations=trace))
            if result.status == 1:
                # Keep the same finite model and spend the declared remaining
                # budget on a longer solve; UNKNOWN is not infeasibility.
                next_solve_limit*=2
                continue
            break
        next_solve_limit=solve_limit
        assigned=np.flatnonzero(result.x>.5);chosen=(assigned%n).tolist()
        if len(assigned)!=new_count or not np.all(np.bincount(assigned//n,minlength=stage_count)==2) or len({candidates[j]['stone_id'] for j in chosen})!=new_count:
            raise RuntimeError('Joint stage incumbent violates counts/identities')
        rec.update(chosen=chosen,stage_assignments=(assigned//n).tolist(),stone_ids=[candidates[j]['stone_id'] for j in chosen])
        conflicts=[];witnesses=[]
        for k,a in enumerate(chosen):
            for b in chosen[k+1:]:
                if polygons[a].distance(polygons[b])<.5/12:
                    witnesses.append(add_conflict(a,b));conflicts.append([a,b])
        if conflicts:rec['conflict']=True;rec['conflicts']=conflicts;rec['conflict_witnesses']=witnesses
        checks=[]
        for stage in stages:
            take=stage-base_count
            active_stones=fixed+[polygons[j] for j in chosen[:take]]
            bound=bounded_gap(layout.patio,active_stones,region(contract,stage),error=.02/12)
            if not bound['passed']:
                bound=bounded_gap(layout.patio,active_stones,region(contract,stage),error=.025/12,max_cells=30000,stop_on_failure=False)
            checks.append(dict(stage=stage,**bound))
            if not bound['passed']:
                if bound['lower_diameter_in']>=2:cut(stage,bound['center'])
                else:rec['unresolved']=True
        rec.update(checks[-1]);rec['prefix_checks']=checks
        rec['gap_passed']=all(r['passed'] for r in checks)
        rec['clearance_passed']=not bool(conflicts)
        rec['passed']=rec['gap_passed'] and rec['clearance_passed']
        if rec['passed']:accepted=chosen
        print(json.dumps(rec),flush=True)
        trace.append(rec)
        write(out/'trace.json',dict(accepted=accepted,cuts=cuts,iterations=trace))
        if accepted is not None or (rec.get('unresolved') and not conflicts):break
    write(out/'trace.json',dict(accepted=accepted,cuts=cuts,iterations=trace,elapsed_seconds=time.monotonic()-start))
    if accepted is not None:
        records=list(history)
        for j in accepted:
            r=candidates[j];records.append(dict(step=len(records)+1,stone_id=r['stone_id'],pose=r['pose']))
        save_verified(root,contract,records,out/'accepted')
    print(json.dumps(dict(joint_accepted=accepted is not None,target=target,output=str(out))),flush=True)


def pair_spacing_margin(a,b,gap=.5/12+1e-7):
    """Signed search margin in inches; feasible iff exact spacing is sufficient.

    Negative square-root overlap area supplies a recovery gradient when the
    ordinary polygon distance is identically zero inside an overlap.
    """
    distance=a.distance(b)
    if distance>0:return 12*(distance-gap)
    return -12*(math.sqrt(max(0.,a.intersection(b).area))+gap)


def checked_history(layout, contract, records):
    """Validate an ordered rigid-pose job before any numerical work or writes."""
    if layout.units != 'ft':
        raise ValueError('Corner jobs require feet')
    if (contract['minimum_clearance_in'] != .5 or
            contract['maximum_empty_circle_diameter_strictly_below_in'] != 2):
        raise ValueError('Unsupported contract gates; engine implements .5/2 inches')
    if not 1 <= len(records) <= len(contract['prefix_regions']):
        raise ValueError('History must contain 1 through 13 prefixes')
    seen = set()
    cleaned = []
    for n, rec in enumerate(records, 1):
        sid = rec['stone_id']
        pose = np.asarray(rec['pose'], dtype=float)
        if rec['step'] != n or sid not in layout.shapes or sid in seen:
            raise ValueError('History needs ordered steps and unique inventory IDs')
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError('History poses must be finite rigid transforms')
        seen.add(sid)
        cleaned.append(dict(step=n, stone_id=sid, pose=pose.tolist()))
    return cleaned


def evaluate_history(directory, records, *, error_in=.005, max_cells=100000):
    """Evaluate every prefix, retaining coverage AND clearance diagnostics.

    Fully refined intervals can rank checkpoints; an optimizer objective cannot.
    This evaluates original local geometry. Promotion still requires save_verified.
    """
    import time
    started = time.monotonic()
    _, contract, layout = load_trial(directory)
    records = checked_history(layout, contract, records)
    if not math.isfinite(error_in) or error_in <= 0 or max_cells < 1:
        raise ValueError('Certification limits must be positive')
    stones = []
    results = []
    for n, rec in enumerate(records, 1):
        i = layout.ids.index(rec['stone_id'])
        layout.active[i] = True
        layout.poses[i] = rec['pose']
        stones.append(transformed(layout.shapes[rec['stone_id']], rec['pose']))
        errors = validate(layout, gap=.5/12, edge=.5/12)
        bound = bounded_gap(layout.patio, stones, region(contract, n),
                            error=error_in/12, max_cells=max_cells, stop_on_failure=False)
        gap_status = ('passed' if bound['upper_diameter_in'] < 2 else
                      'failed' if bound['lower_diameter_in'] >= 2 else 'unresolved')
        # Keep ties at junctions: several stones may define the same empty circle.
        p = Point(bound['center'])
        distances = [p.distance(g) for g in stones]
        nearby = [records[j]['stone_id'] for j, d in enumerate(distances)
                  if d <= min(distances) + .05]
        implicated = set(nearby if gap_status != 'passed' else [])
        implicated.update(sid for e in errors for sid in e['stone_ids'])
        results.append(dict(step=n, stone_id=rec['stone_id'], clearance_errors=errors,
                            gap_status=gap_status, implicated_ids=sorted(implicated), **bound))
    spacing_failed = any(r['clearance_errors'] for r in results)
    passed = all(r['gap_status'] == 'passed' for r in results) and not spacing_failed
    shortfalls = [max(0., -pair_spacing_margin(a, b, gap=.5/12))
                  for j, a in enumerate(stones) for b in stones[j+1:]]
    for g in stones:
        shortfalls.append(max(0., 12*(.5/12-g.distance(layout.patio.boundary)))
                          if layout.patio.covers(g) else
                          12*(math.sqrt(g.difference(layout.patio).area)+.5/12))
    return dict(history=records, results=results, passed=passed,
                status=('geometrically-certified' if passed else
                        'repairable-invalid' if spacing_failed else 'evaluated-near-miss'),
                worst_prefix_upper_in=max(r['upper_diameter_in'] for r in results),
                maximum_spacing_shortfall_in=max(shortfalls, default=0),
                total_spacing_shortfall_in=sum(shortfalls),
                certification_seconds=time.monotonic()-started)


def repair_history(directory, records, output, *, count=3, radius=.15,
                   angle_radius=None, centers=None, seed=3100, maxiter=700):
    """Run one bounded suffix repair and return a structured, unpromoted checkpoint.

    Inputs are poses, not a selector log. Spacing eligibility follows current
    geometry; even an overlapping complete pattern can be repaired. Translation
    bounds are feet and angle bounds are radians (legacy defaults use radius).
    The caller owns ranking, restarts, neighborhood expansion and promotion.
    """
    import copy
    from scipy.optimize import minimize
    root, contract, layout = load_trial(directory)
    records = checked_history(layout, contract, records)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('count must be a positive integer')
    angle_radius = radius if angle_radius is None else angle_radius
    if not all(math.isfinite(v) and v > 0 for v in (radius, angle_radius)):
        raise ValueError('Repair ranges must be finite and positive')
    if not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError('maxiter must be a positive integer')
    target = len(records)
    moving = min(count, target)
    fixed_count = target - moving
    centers = copy.deepcopy(centers or [])
    if any(not isinstance(c['step'], int) or not 1 <= c['step'] <= target
           or np.asarray(c['center']).shape != (2,)
           or not np.isfinite(c['center']).all() for c in centers):
        raise ValueError('Invalid prefix witness')
    for n in range(fixed_count + 1, target + 1):
        gs = [transformed(layout.shapes[r['stone_id']], r['pose']) for r in records[:n]]
        bound = bounded_gap(layout.patio, gs, region(contract, n), error=.03/12,
                            stop_on_failure=False)
        centers.append(dict(step=n, center=bound['center']))
    run_dir = Path(output)
    run_dir.mkdir(parents=True, exist_ok=False)
    scale = np.array([radius, radius, angle_radius])
    baseline=np.asarray([r['pose'] for r in records]);cache={};evaluations=0
    fixed=[transformed(layout.shapes[r['stone_id']],r['pose']) for r in records[:fixed_count]]
    domains=[region(contract,n).intersection(layout.patio) for n in range(1,target+1)]
    # Keep a fine final-domain grid and explicitly probe the earlier
    # prefix fronts, which otherwise fall between final-grid rows.
    arrays=[shapely.get_coordinates(sample_points(domains[-1],.055))]
    arrays.extend(shapely.get_coordinates(coverage_samples(layout.patio,domain,.16)) for domain in domains[fixed_count:])
    coords=np.concatenate(arrays)
    _,indices=np.unique(np.round(coords,12),axis=0,return_index=True)
    pts=shapely.points(coords[indices])
    masks=[shapely.covers(domain,pts) for domain in domains]
    fixed_field=shapely.distance(pts,unary_union([layout.patio.boundary,*fixed]))
    def evaluate(x):
        nonlocal evaluations
        if 'x' in cache and np.array_equal(cache['x'],x):return cache
        poses=baseline.copy();poses[fixed_count:]+=np.asarray(x).reshape(-1,3)*scale
        gs=fixed+[transformed(layout.shapes[r['stone_id']],pose) for r,pose in zip(records[fixed_count:],poses[fixed_count:])]
        constraints=[]
        for g in gs[fixed_count:]:
            constraints.append(12*(g.distance(layout.patio.boundary)-(.5/12+1e-7)) if layout.patio.covers(g) else -12*(math.sqrt(g.difference(layout.patio).area)+.5/12))
        constraints.extend(pair_spacing_margin(a,b) for j,a in enumerate(gs) for k,b in enumerate(gs[j+1:],j+1) if k>=fixed_count)
        distances=fixed_field.copy();objective=0
        for n,g in enumerate(gs[fixed_count:],fixed_count):
            distances=np.minimum(distances,shapely.distance(pts,g.simplify(.001,preserve_topology=True)))
            objective=max(objective,float(distances[masks[n]].max())*24)
        for item in centers:
            n=item['step'];width=contract['prefix_regions'][n-1]['width_ft'];depth=contract['prefix_regions'][n-1]['depth_ft']
            center=np.clip(np.asarray(item['center']),[0,0],[width,depth])
            def neg_radius(xy):
                p=Point(*xy)
                if not layout.patio.covers(p):return p.distance(layout.patio)
                return -min(p.distance(layout.patio.boundary),*(p.distance(g) for g in gs[:n]))
            delta=np.where(center>np.array([width,depth])-.02,-.01,.01)
            simplex=np.array([center,center+[delta[0],0],center+[0,delta[1]]])
            optimized=minimize(neg_radius,center,method='Nelder-Mead',bounds=[(0,width),(0,depth)],options=dict(xatol=1e-5,fatol=1e-7,maxiter=80,initial_simplex=simplex))
            objective=max(objective,-float(optimized.fun)*24)
        evaluations+=1
        cache.clear();cache.update(x=np.asarray(x).copy(),poses=poses,gs=gs,constraints=np.asarray(constraints),objective=objective)
        # Progress is returned as structured checkpoint metadata.
        return cache
    x0=np.zeros(3*moving) if seed is None else np.random.default_rng(seed).uniform(-.05,.05,3*moving)
    result=minimize(lambda x:evaluate(x)['objective'],x0,method='COBYLA',bounds=[(-1,1)]*(3*moving),constraints=[dict(type='ineq',fun=lambda x:evaluate(x)['constraints'])],options=dict(rhobeg=.25,tol=1e-5,catol=1e-6,maxiter=maxiter))
    final=evaluate(result.x);results=[];all_pass=True
    for n,(rec,pose) in enumerate(zip(records,final['poses']),1):
        rec['pose']=pose.tolist()
        layout.active[:]=False
        for rr,pp in zip(records[:n],final['poses'][:n]):
            i=layout.ids.index(rr['stone_id']);layout.active[i]=True;layout.poses[i]=pp
        errors=validate(layout,gap=.5/12,edge=.5/12)
        bound=bounded_gap(layout.patio,final['gs'][:n],domains[n-1],error=.01/12)
        if not bound['passed'] and not errors:
            bound=bounded_gap(layout.patio,final['gs'][:n],domains[n-1],error=.005/12,max_cells=100000,stop_on_failure=False)
        rec.update(bound);rec['clearance_errors']=errors
        results.append(dict(step=n,clearance_errors=errors,**bound))
        all_pass=all_pass and not errors and bound['passed']
        if not bound['passed'] and not any(old['step']==n and np.linalg.norm(np.asarray(old['center'])-bound['center'])<.02 for old in centers):
            centers.append(dict(step=n,center=bound['center']))
    spacing_failed = any(r['clearance_errors'] for r in results)
    payload = dict(passed=bool(all_pass), objective=float(result.fun),
                   optimizer_success=bool(result.success), optimizer_message=str(result.message),
                   results=results, centers=centers, evaluations=evaluations,
                   fixed_ids=[r['stone_id'] for r in records[:fixed_count]],
                   movable_ids=[r['stone_id'] for r in records[fixed_count:]],
                   translation_radius_ft=radius, angle_radius_radians=angle_radius,
                   seed=seed, status=('geometrically-certified' if all_pass else
                                     'repairable-invalid' if spacing_failed else 'evaluated-near-miss'))
    write(run_dir/'history.json', records)
    write(run_dir/'repair_result.json', payload)
    plot(layout, contract, [{**r, **v} for r, v in zip(records, results)], run_dir/'diagnostic.png')
    return dict(checkpoint=str(run_dir), history=records, **payload)


def repair_pairs(selection,output,count=3,radius=.15,rounds=3,resume=None,allow_spacing_failures=False):
    """Polish diverse placement groups, optionally restoring failed initial spacing.

    Earlier affected prefixes retain their declared regions and are all rechecked.
    Numerical minimax objectives never substitute for the global circle certificate.
    """
    import copy
    selection=Path(selection)
    settings=json.loads((selection/'settings.json').read_text())
    data=json.loads((Path(settings['pool'])/'candidate_pool.json').read_text())
    trace=json.loads((selection/'trace.json').read_text())
    root,contract,layout=load_trial(data['trial'])
    history=data['parent_history'];target=data['target']
    parent=[transformed(layout.shapes[r['stone_id']],r['pose']) for r in history]
    pool=data['candidates'];retained=[];seen=set()
    # Resumed geometry is authoritative, regardless of its ancestor's conflict
    # flag or the CLI flag originally used to admit invalid starts.
    for rec in ([] if resume is not None else trace['iterations']):
        if 'chosen' not in rec or (rec.get('conflict') and not allow_spacing_failures):continue
        identity=tuple(tuple(sorted(rec['stone_ids'][k:k+2])) for k in range(0,len(rec['chosen']),2))
        if identity in seen:continue
        seen.add(identity)
        gs=[transformed(layout.shapes[pool[j]['stone_id']],pool[j]['pose']) for j in rec['chosen']]
        bound=bounded_gap(layout.patio,parent+gs,region(contract,target),error=.025/12,max_cells=30000,stop_on_failure=False)
        prefix_bounds=[dict(step=target,**bound)]
        for take in range(2,len(rec['chosen']),2):
            stage=len(history)+take
            earlier=bounded_gap(layout.patio,parent+gs[:take],region(contract,stage),error=.025/12,max_cells=30000,stop_on_failure=False)
            prefix_bounds.append(dict(step=stage,**earlier))
        shortfall=sum(max(0.,-pair_spacing_margin(a,b)) for j,a in enumerate(gs) for b in gs[j+1:])
        score=max(r['upper_diameter_in'] for r in prefix_bounds)
        retained.append(dict(chosen=rec['chosen'],bound=bound,prefix_bounds=prefix_bounds,all_prefix_upper_score=score,initial_spacing_shortfall_in=shortfall,repair_priority=score+.5*shortfall,initial_spacing_failed=bool(rec.get('conflict'))))
    retained.sort(key=lambda r:r['repair_priority'])
    if resume is not None:
        resumed = checked_history(layout, contract, json.loads((Path(resume)/'history.json').read_text()))
        if len(resumed) != target:
            raise ValueError('Resume history has wrong target')
        retained = [dict(chosen=[], prefix_bounds=[], resume_geometry=True)]
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    (out/'experiment_script.py').write_text(MODULE_SOURCE)
    write(out/'rankings.json',retained)
    write(out/'settings.json',dict(selection=str(selection),moving_count=count,radius=radius,rounds=rounds,resume_source=str(resume) if resume is not None else None,feedback='fully_refined_global_failing_maxima',allow_spacing_failures=allow_spacing_failures))
    if resume is not None:
        retained=retained[:1]
    for start,retained_pair in enumerate(retained[:4]):
        records=copy.deepcopy(history)
        for j in retained_pair['chosen']:
            records.append(dict(step=len(records)+1,stone_id=pool[j]['stone_id'],pose=pool[j]['pose']))
        if resume is not None:
            records=json.loads((Path(resume)/'history.json').read_text())
            if len(records)!=target or len({r['stone_id'] for r in records})!=target:
                raise ValueError('Resume history has wrong target or duplicate IDs')
        centers=[dict(step=b['step'],center=b['center']) for b in retained_pair['prefix_bounds']]
        if resume is not None:
            centers=json.loads((Path(resume)/'repair_result.json').read_text())['centers']
        for round_index in range(rounds):
            result = repair_history(root, records, out/f'pair_{start:02d}_round_{round_index:02d}',
                                    count=count, radius=radius, centers=centers,
                                    seed=3100+start*10+round_index if round_index else None)
            records, centers = result['history'], result['centers']
            print(json.dumps(dict(pair=start,round=round_index,all_prefix_gates=result['passed'],objective=result['objective'])),flush=True)
            if result['passed']:
                save_verified(root,contract,records,out/'accepted')
                return result
    print(json.dumps(dict(accepted=False,output=str(out))),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trial',default='examples/corner_01')
    ap.add_argument('--declare',action='store_true')
    ap.add_argument('--export-pool',action='store_true')
    ap.add_argument('--pool-target',type=int)
    ap.add_argument('--parent')
    ap.add_argument('--output')
    ap.add_argument('--pool')
    ap.add_argument('--pair-backend',choices=['highs','cp_sat'],default='highs')
    ap.add_argument('--joint-pool')
    ap.add_argument('--augment-pool')
    ap.add_argument('--incumbent')
    ap.add_argument('--joint-backend',choices=['highs','cp_sat'],default='highs')
    ap.add_argument('--joint-workers',type=int,default=2)
    ap.add_argument('--raster-spacing',type=float,default=.075)
    ap.add_argument('--resume-selection')
    ap.add_argument('--joint-solve-limit',type=float,default=30)
    ap.add_argument('--joint-heuristic-effort',type=float,default=.05)
    ap.add_argument('--scout',action='store_true',help='Generate diverse repair starts using a sampled minimax-radius MILP objective; acceptance unchanged')
    ap.add_argument('--exclude-pair-ids',nargs=2,action='append',default=[])
    ap.add_argument('--verify')
    ap.add_argument('--repair')
    ap.add_argument('--repair-count',type=int,default=3)
    ap.add_argument('--repair-spacing-failures',action='store_true')
    ap.add_argument('--repair-rounds',type=int,default=3)
    ap.add_argument('--repair-radius',type=float,default=.15)
    ap.add_argument('--resume')
    ap.add_argument('--proposals',type=int,default=18000)
    ap.add_argument('--seed',type=int,default=2711)
    ap.add_argument('--single-starts',action='store_true')
    args=ap.parse_args()
    if args.declare:declare(args.trial)
    elif args.single_starts:single_starts(args.trial)
    elif args.export_pool:export_pool(args.trial,args.parent,args.output,args.seed,args.proposals,target_override=args.pool_target)
    elif args.augment_pool:augment_pool(args.augment_pool,args.incumbent,args.output)
    elif args.joint_pool:select_joint_stages(args.joint_pool,args.output,seconds=600,solve_limit=args.joint_solve_limit,heuristic_effort=args.joint_heuristic_effort,backend=args.joint_backend,workers=args.joint_workers,raster_spacing=args.raster_spacing,resume_selection=args.resume_selection)
    elif args.pool:select_pair(args.pool,args.output,excluded_id_pairs=args.exclude_pair_ids,scout=args.scout,backend=args.pair_backend)
    elif args.verify:verify(args.verify)
    elif args.repair:repair_pairs(args.repair,args.output,count=args.repair_count,radius=args.repair_radius,rounds=args.repair_rounds,resume=args.resume,allow_spacing_failures=args.repair_spacing_failures)
