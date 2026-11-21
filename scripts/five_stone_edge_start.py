"""Bounded real-outline edge experiment. All lengths are feet.

The completed region is a fixed 12-inch-deep strip, starting one foot along
one physical edge, advancing at least six inches per stone to six inches past the new stone’s left extent. Its sides/front constrain
circle CENTERS only; only the physical patio and stones constrain radii.
The distance field is 1-Lipschitz. Each intersected quadtree cell has upper
bound f(p)+max_corner_distance(p), including cells crossing the reporting front.
"""
import argparse
import heapq
import itertools
import json
import math
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from flagstone.phase1 import load_json, save_json, transformed, validate


def bounded_gap(patio, stones, region, error=0.015/12, max_cells=20000, gate_only=False, stop_on_failure=True):
    """Return enclosing diameter interval; budget exhaustion never certifies pass."""
    obstacles = unary_union([patio.boundary, *stones])
    domain = region.intersection(patio)
    if domain.is_empty or not stones:
        raise ValueError('Need a nonempty completed region and placed stones')
    queue = []
    counter = itertools.count()
    lower = 0.0
    worst = None
    def add(bounds):
        nonlocal lower, worst
        piece = box(*bounds).intersection(domain)
        if piece.is_empty:
            return
        p = piece.representative_point()
        d = p.distance(obstacles)
        if d > lower or worst is None:
            lower, worst = d, [p.x, p.y]
        a,b,c,e = bounds
        radius = max(math.hypot(x-p.x,y-p.y) for x in (a,c) for y in (b,e))
        heapq.heappush(queue, (-(d+radius), next(counter), bounds))
    add(domain.bounds)
    count = 1
    while queue and -queue[0][0]-lower > error/2 and count < max_cells:
        if gate_only and -queue[0][0]*2 < 1.95/12:
            break
        _,_,(a,b,c,d) = heapq.heappop(queue)
        x,y=(a+c)/2,(b+d)/2
        for bounds in ((a,b,x,y),(x,b,c,y),(a,y,x,d),(x,y,c,d)):
            add(bounds)
        count += 4
        # A witnessed failure needs no refinement.
        if stop_on_failure and lower*2 >= 2/12:
            break
    upper = max(lower, -queue[0][0] if queue else lower)
    if lower*2 < 2/12 <= upper*2 and error>.0001/12:
        refined=bounded_gap(patio,stones,region,error=error/10,max_cells=max(max_cells,100000),gate_only=gate_only,stop_on_failure=stop_on_failure)
        refined['refinement_passes']=refined.get('refinement_passes',0)+1
        return refined
    return dict(lower_diameter_in=24*lower, upper_diameter_in=24*upper,
                center=worst, cells=count, passed=upper*2 < 2/12)


def run(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out/'experiment_script.py').write_text(Path(__file__).read_text())
    (out/'settings.json').write_text(json.dumps(vars(args),indent=2))
    layout=load_json(args.input)
    if layout.units != 'ft':
        raise ValueError('Expected feet')
    proposal_shapes={}
    for sid,shape in layout.shapes.items():
        proxy=shape.simplify(.002,preserve_topology=True).buffer(.003,quad_segs=4)
        proposal_shapes[sid]=proxy if proxy.covers(shape) else shape
    layout.active[:]=False
    # Fixed segment on the long nearly straight physical edge; rigid rotation only.
    ring=np.asarray(layout.patio.exterior.coords)
    lengths=np.linalg.norm(np.diff(ring,axis=0),axis=1)
    k=args.edge_index; origin=ring[k]; v=ring[k+1]-origin
    angle=math.atan2(v[1],v[0])
    if not layout.patio.exterior.is_ccw:
        origin=ring[k+1]; angle+=math.pi
    c,s=math.cos(angle),math.sin(angle)
    def local(points):
        q=np.asarray(points)-origin
        return q @ np.array([[c,-s],[s,c]])
    original_patio=layout.patio
    layout.patio=Polygon(local(ring))
    start=1.0; depth=1.0; advance=.5; gap=.5/12 + 1e-6
    rng=np.random.default_rng(args.seed)
    history=[]; best_failure=None; attempts=0; highwater=0
    solver_angles={}
    if args.solver_angles:
        for item in json.loads(Path(args.solver_angles).read_text())['orientations']:
            solver_angles.setdefault(layout.ids.index(item['stone_id']),[]).append(item['angle_radians']-angle)
    def geoms():
        return [transformed(layout.shapes[sid],layout.poses[i]) for i,sid in enumerate(layout.ids) if layout.active[i]]
    def search(step):
        nonlocal attempts,best_failure,highwater
        if step==args.target+1:
            return True
        previous_front=history[-1]["front_x"] if history else start
        fixed_exact=geoms()
        fixed_proxy=[transformed(proposal_shapes[sid],layout.poses[i]) for i,sid in enumerate(layout.ids) if layout.active[i]]
        fixed_rank=[g.simplify(.001,preserve_topology=True) for g in fixed_exact]
        fixed_obstacles=unary_union([layout.patio.boundary,*fixed_rank])
        candidates=[]
        def consider(i,pose,min_x):
            nonlocal attempts
            layout.poses[i]=pose; layout.active[i]=True; attempts+=1
            if not validate(layout,gap=.5/12,edge=.5/12):
                rank_new=transformed(layout.shapes[layout.ids[i]],pose).simplify(.001,preserve_topology=True)
                if args.paired_front and step>5:
                    front=previous_front if step%2==0 else max(history[-2]['front_x']+2*advance,min_x+advance)
                else:
                    front=max(previous_front+advance,min_x+advance)
                ranking_front=previous_front+2*advance if args.paired_front and step>5 and step%2==0 else front
                region=box(start,0,front,depth)
                # Cheap lower-bound rejection on the reporting region.
                obstacles=unary_union([layout.patio.boundary,*fixed_rank,rank_new])
                xx,yy=np.meshgrid(np.linspace(start,ranking_front,max(9,int((ranking_front-start)*24))),np.linspace(0,depth,13))
                samples=shapely.points(xx.ravel(),yy.ravel())
                distances=shapely.distance(samples[shapely.covers(layout.patio,samples)],obstacles)
                boundary_points=shapely.line_interpolate_point(rank_new.exterior,np.linspace(0,rank_new.length,80,endpoint=False))
                contact_fraction=float(np.mean(shapely.distance(boundary_points,fixed_obstacles)<=.75/12))
                score=float(np.max(distances)) + .01*float(np.mean(distances)) - .01*contact_fraction
                candidates.append((score,i,pose.copy(),front))
            layout.active[i]=False
        def propose(i,a,exact=False):
            nonlocal attempts
            proposal=layout.shapes[layout.ids[i]] if exact or args.exact_proposals else proposal_shapes[layout.ids[i]]
            shape=transformed(proposal,(0,0,a))
            x0,y0,x1,y1=shape.bounds
            existing=fixed_exact if exact or args.exact_proposals else fixed_proxy
            # Explore small vertical offsets as well as boundary contact.
            ty=gap-y0+(float(rng.uniform(0,.8)) if args.paired_front and step>5 else (float(rng.uniform(0,.035)) if existing and rng.random()<.35 else 0))
            target=max((g.bounds[2] for g in existing),default=start)
            # Place along the wall, targeting the next exposed strip.
            tx=target-x0+gap if existing else start-x0+float(rng.uniform(-.5,.05))
            if existing:
                # Approach the exposed joint from the right. Stop at the first
                # clearance collision, then bisect without crossing the stones.
                def valid_offset(offset):
                    g=transformed(proposal,(offset,ty,a))
                    return all(g.distance(other)>=gap for other in existing)
                # A polygon may have several disjoint feasible intervals on
                # this horizontal slice. Placement is not constrained to a
                # physical slide from the far right: inspect pocket intervals.
                free=[]
                for min_x in np.arange(previous_front-.4,max(previous_front+2,target+.1),.04):
                    offset=float(min_x-x0)
                    if valid_offset(offset): free.append(offset)
                if not free: return
                tx=min(free,key=lambda offset:abs(offset+x0-previous_front))
                right=tx
                for shift in np.arange(.025,2.5,.025):
                    left=tx-shift
                    if not valid_offset(left):
                        for _ in range(14):
                            mid=(left+right)/2
                            if valid_offset(mid): right=mid
                            else: left=mid
                        break
                    right=left
                tx=right
            # Fit to the actual traced physical edge, which may deviate from
            # the local axis beyond the chosen segment. No artificial wall.
            def boundary_ok(y):
                g=transformed(proposal,(tx,y,a))
                return layout.patio.covers(g) and g.distance(layout.patio.boundary)>=gap
            fit_ok=True
            for _ in range(3):
                if not boundary_ok(ty):
                    lo,hi=ty,ty+.5
                    if not boundary_ok(hi):
                        fit_ok=False;break
                    for _ in range(19):
                        mid=(lo+hi)/2
                        if boundary_ok(mid): hi=mid
                        else: lo=mid
                    ty=hi
                # Moving upward along a nonhorizontal physical edge may disturb
                # contact with the previous stone. Refit that contact too.
                if existing and not valid_offset(tx):
                    lo,hi=tx,tx+.5
                    if not valid_offset(hi):
                        fit_ok=False;break
                    for _ in range(19):
                        mid=(lo+hi)/2
                        if valid_offset(mid): hi=mid
                        else: lo=mid
                    tx=hi
            if not fit_ok: return
            consider(i,np.array([tx,ty,a]),tx+x0)
        # Fixed proposal count, reproducible inventory and orientation sampling.
        for _ in range(args.proposals):
            pool=[layout.ids.index(sid) for sid in args.candidate_ids] if args.candidate_ids else [j for j,sid in enumerate(layout.ids) if sid not in args.exclude_ids]
            i=int(rng.choice(pool))
            if layout.active[i]: continue
            if rng.random()<.7:
                # Use coarse outlines only to propose long boundary matches.
                # Exact source outlines still decide all acceptance gates.
                outline=layout.shapes[layout.ids[i]].simplify(.025,preserve_topology=True)
                coords=np.asarray(outline.exterior.coords)
                edges=np.diff(coords,axis=0)
                indices=np.argsort(np.linalg.norm(edges,axis=1))[-4:]
                e=edges[int(rng.choice(indices))]
                a=-math.atan2(e[1],e[0])+(0 if outline.exterior.is_ccw else math.pi)
                a+=float(rng.uniform(-.025,.025))
            else:
                a=float(rng.uniform(0,2*math.pi))
            propose(i,a)
            if step>5 and fixed_proxy:
                # Reuse the existing edge-alignment insertion primitive for
                # upper-junction placements, rather than assuming wall contact.
                from flagstone.search import _edge_pose
                exposed=[g for g in fixed_proxy if g.bounds[2]>previous_front]
                if exposed:
                    anchor=exposed[int(rng.integers(len(exposed)))]
                    px,py,pa=_edge_pose(proposal_shapes[layout.ids[i]],anchor,gap,rng)
                    candidate=np.array([px,py,pa])
                    new=transformed(layout.shapes[layout.ids[i]],candidate)
                    if new.bounds[2]>=previous_front and new.bounds[0]<previous_front+2 and new.bounds[1]<1:
                        consider(i,candidate,new.bounds[0])
        for i,angles in solver_angles.items():
            if not layout.active[i] and layout.ids[i] not in args.exclude_ids and (not args.candidate_ids or layout.ids[i] in args.candidate_ids):
                for a in angles:
                    propose(i,a)
                    propose(i,a-.015)
                    propose(i,a+.015)
        # Refine near misses around their current contact orientations.
        refinements=[];refined_ids=set()
        for candidate in sorted(candidates,key=lambda t:t[0]):
            if candidate[1] not in refined_ids:
                refinements.append(candidate);refined_ids.add(candidate[1])
            if len(refinements)==20: break
        for _,i,pose,_ in refinements:
            for delta in (-.045,-.03,-.015,-.0075,-.003,0,.003,.0075,.015,.03,.045):
                propose(i,float(pose[2])+delta,exact=True)
        if args.export_pool:
            save_json(layout,out/'pool_parent.json')
            metadata=dict(parent_history=history,settings=vars(args),local_origin=origin.tolist(),local_angle=angle,
                start_ft=start,depth_ft=depth,candidates=[dict(score=float(score),stone_id=layout.ids[i],pose=pose.tolist()) for score,i,pose,front in candidates])
            (out/'candidate_pool.json').write_text(json.dumps(metadata,indent=2))
            print(json.dumps(dict(exported_candidates=len(candidates),parent_count=len(history))),flush=True)
            return False
        candidates.sort(key=lambda t:t[0])
        # Keep different stone choices in the beam, avoiding near-duplicate
        # orientations consuming every branch of the bounded lookahead.
        diverse=[]; seen=set()
        for candidate in candidates:
            if candidate[1] not in seen:
                diverse.append(candidate);seen.add(candidate[1])
        # Reserve half the shortlist for distinct IDs and half for alternative
        # geometries, so one-stone local refinement can test multiple angles.
        selected=diverse[:max(1,args.shortlist//2)]
        selected_ids={id(candidate) for candidate in selected}
        candidates=selected+[candidate for candidate in candidates if id(candidate) not in selected_ids]
        (out/f"shortlist_step_{step}.json").write_text(json.dumps([dict(score=float(score),stone_id=layout.ids[i],pose=pose.tolist(),front_x=front) for score,i,pose,front in candidates[:args.shortlist]],indent=2))
        for score,i,pose,front in candidates[:args.shortlist]:
            region=box(start,0,front,depth)
            layout.poses[i]=pose; layout.active[i]=True
            result=bounded_gap(layout.patio,geoms(),region,gate_only=True)
            if not result['passed']:
                # Rank failed junctions by a refined global maximum, not by
                # the loose upper bound returned after the first failure.
                result=bounded_gap(layout.patio,geoms(),region,error=.05/12,max_cells=10000,stop_on_failure=False)
            record=dict(step=step,stone_id=layout.ids[i],pose=pose.tolist(),front_x=front,**result)
            if not result['passed'] and (best_failure is None or step>best_failure['step'] or (step==best_failure['step'] and result['upper_diameter_in']<best_failure['upper_diameter_in'])):
                best_failure={**record,'prefix':list(history),'poses':layout.poses.tolist(),'active':layout.active.tolist()}
            if result['passed']:
                history.append(record)
                if len(history)>highwater:
                    highwater=len(history)
                    save_json(layout,out/'passing_prefix_local.tmp.json')
                    (out/'passing_prefix_local.tmp.json').replace(out/'passing_prefix_local.json')
                    (out/'passing_prefix.tmp.json').write_text(json.dumps(history,indent=2))
                    (out/'passing_prefix.tmp.json').replace(out/'passing_prefix.json')
                    with Path('first_real_patio_followup_plan_progress.md').open('a') as log:
                        log.write(f"\n- Search `{out}` reached {highwater} passing prefixes: {[r['stone_id'] for r in history]}; latest upper diameter {result['upper_diameter_in']:.6f} inches. Search gate only; saved-file verification pending.\n")
                print(json.dumps({'passing_prefix':[r['stone_id'] for r in history], 'front_ft':front,'gap_upper_in':result['upper_diameter_in']}),flush=True)
                if search(step+1): return True
                history.pop()
            layout.active[i]=False
        return False
    if args.prefix:
        prefix=Path(args.prefix)
        saved=load_json(prefix/'passing_prefix_local.json')
        if saved.ids != layout.ids or not saved.patio.equals_exact(layout.patio,1e-10):
            raise ValueError('Prefix inventory or patio mismatch')
        layout.poses[:]=saved.poses;layout.active[:]=saved.active
        history[:]=json.loads((prefix/'passing_prefix.json').read_text())
        history_ids=[record['stone_id'] for record in history]
        active_ids={layout.ids[i] for i in np.flatnonzero(layout.active)}
        if len(history)>args.target or len(set(history_ids))!=len(history_ids) or set(history_ids)!=active_ids:
            raise ValueError('Prefix history/active-stone mismatch or prefix exceeds target')
        for record in history:
            if not np.allclose(layout.poses[layout.ids.index(record['stone_id'])],record['pose'],rtol=0,atol=1e-10):
                raise ValueError('Prefix pose/history mismatch; recover a consistent checkpoint')
        if validate(layout,gap=.5/12,edge=.5/12):
            raise ValueError('Invalid prefix clearances')
        for n,record in enumerate(history,1):
            prefix_stones=[transformed(layout.shapes[r['stone_id']],layout.poses[layout.ids.index(r['stone_id'])]) for r in history[:n]]
            if not bounded_gap(layout.patio,prefix_stones,box(start,0,record['front_x'],depth),gate_only=True)['passed']:
                raise ValueError('Invalid prefix maximum gap')
    success=search(len(history)+1)
    if args.export_pool: return
    if not success and best_failure:
        layout.poses[:]=best_failure['poses']; layout.active[:]=best_failure['active']
    save_json(layout,out/'layout_local.json')
    if success:
        save_json(layout,out/'passing_prefix_local.json')
        (out/'passing_prefix.json').write_text(json.dumps(history,indent=2))
    # Save physical patio-frame placements too.
    local_poses=layout.poses.copy()
    for i in np.flatnonzero(layout.active):
        x,y,a=layout.poses[i]; layout.poses[i]=[origin[0]+c*x-s*y,origin[1]+s*x+c*y,a+angle]
    layout.patio=original_patio
    save_json(layout,out/'layout.json')
    layout.patio=Polygon(local(ring));layout.poses[:]=local_poses
    records=history if success else (best_failure['prefix']+[best_failure] if best_failure else [])
    for record in records:
        selected=[layout.ids.index(r['stone_id']) for r in records[:record['step']]]
        gs=[transformed(layout.shapes[layout.ids[i]],layout.poses[i]) for i in selected]
        record['minimum_edge_clearance_in']=12*min(g.distance(layout.patio.boundary) for g in gs)
        record['minimum_pair_clearance_in']=min((12*a.distance(b) for j,a in enumerate(gs) for b in gs[j+1:]),default=None)
    for record in records:
        fig,ax=plt.subplots(figsize=(9,6))
        ax.plot(*layout.patio.exterior.xy,color='black',label='Physical patio edge')
        for i in np.flatnonzero(layout.active):
            if i not in [layout.ids.index(r['stone_id']) for r in records[:record['step']]]: continue
            g=transformed(layout.shapes[layout.ids[i]],layout.poses[i]);ax.fill(*g.exterior.xy,alpha=.5)
            ax.text(g.centroid.x,g.centroid.y,layout.ids[i],ha='center')
        plotted=[transformed(layout.shapes[r['stone_id']],layout.poses[layout.ids.index(r['stone_id'])]) for r in records[:record['step']]]
        plot_bounds=unary_union(plotted).bounds
        front=record['front_x'];ax.fill([start,front,front,start],[0,0,depth,depth],color='green',alpha=.12)
        ax.plot([front,front],[0,depth],'r--',label='Advancing front (centers only)')
        p=record['center'];ax.add_patch(plt.Circle(p,record['lower_diameter_in']/24,fill=False,color='red'))
        ax.set(xlim=(min(start-.2,plot_bounds[0]-.2),max(front+.3,plot_bounds[2]+.2)),ylim=(-.2,max(depth+.2,plot_bounds[3]+.2)),aspect='equal',xlabel='Feet along physical edge',ylabel='Feet inward',title=f"Step {record['step']}: gap diameter [{record['lower_diameter_in']:.3f}, {record['upper_diameter_in']:.3f}] in")
        ax.legend();fig.tight_layout();fig.savefig(out/f"step_{record['step']}.png",dpi=160);plt.close(fig)
    report=dict(success=success,results=records,accepted_sequence=history if success else (best_failure['prefix'] if best_failure else []),earliest_failure=best_failure if not success else None,attempts=attempts,settings=vars(args),edge_segment_index=k,local_origin=origin.tolist(),local_angle=angle,completed_depth_ft=depth,start_ft=start,advance_ft=advance,convention=__doc__)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    with Path('first_real_patio_followup_plan_progress.md').open('a') as log:
        log.write(f"\n- Run `{out}` terminated: success={success}, target={args.target}, attempts={attempts}; accepted IDs={[r['stone_id'] for r in report['accepted_sequence']]}. See `report.json` for the exact first failing junction on the retained diagnostic sequence. Final saved-file verification pending if successful.\n")
    print(json.dumps(dict(success=success,attempts=attempts,failure_step=best_failure['step'] if best_failure and not success else None)))


def polish_pair_population(args,data,candidates,polygons,trace,out):
    """Polish a bounded population of exact-clearance MILP pair incumbents."""
    import copy
    parent=load_json(Path(args.pool_pair)/'pool_parent.json')
    original_active=parent.active.copy();initial_history=data['parent_history'];ranked=[];seen=set()
    parent_stones=[transformed(parent.shapes[sid],parent.poses[i]) for i,sid in enumerate(parent.ids) if parent.active[i]]
    for item in trace:
        if 'chosen' not in item or item.get('errors'):continue
        chosen=tuple(sorted(item['chosen'],key=lambda j:polygons[j].bounds[0]))
        if chosen in seen:continue
        seen.add(chosen);front=max(initial_history[-1]['front_x']+1,max(polygons[j].bounds[0] for j in chosen)+.5)
        bound=bounded_gap(parent.patio,parent_stones+[polygons[j] for j in chosen],box(data['start_ft'],0,front,data['depth_ft']),error=.02/12,max_cells=100000,stop_on_failure=False)
        ranked.append(dict(chosen=chosen,front_x=front,stone_ids=[candidates[j]['stone_id'] for j in chosen],**bound))
    ranked.sort(key=lambda item:item['upper_diameter_in'])
    (out/'ranked_pairs.json').write_text(json.dumps(ranked,indent=2))
    used_id_sets=set();attempt=0
    for item in ranked:
        identity=tuple(sorted(item['stone_ids']))
        if identity in used_id_sets:continue
        used_id_sets.add(identity)
        if attempt>=args.polish_pairs:break
        attempt+=1;trial=out/f'polish_source_{attempt}';trial.mkdir()
        history=copy.deepcopy(initial_history);parent.active[:]=original_active
        for k,j in enumerate(item['chosen'],1):
            c=candidates[j];i=parent.ids.index(c['stone_id']);parent.active[i]=True;parent.poses[i]=c['pose']
            front=initial_history[-1]['front_x'] if k==1 else item['front_x']
            stones=[transformed(parent.shapes[sid],parent.poses[i]) for i,sid in enumerate(parent.ids) if parent.active[i]]
            bound=bounded_gap(parent.patio,stones,box(data['start_ft'],0,front,data['depth_ft']))
            history.append(dict(step=len(initial_history)+k,stone_id=c['stone_id'],pose=c['pose'],front_x=front,**bound))
        if validate(parent,gap=.5/12,edge=.5/12):continue
        save_json(parent,trial/'layout_local.json')
        (trial/'report.json').write_text(json.dumps(dict(settings={**data['settings'],'paired_front':True},earliest_failure={**history[-1],'prefix':history[:-1]}),indent=2))
        (trial/'repair_centers.json').write_text(json.dumps([dict(step=len(history),center=item['center'])],indent=2))
        follow=argparse.Namespace(**vars(args));follow.local_repair=str(trial);follow.output=str(out/f'polish_{attempt}')
        follow.repair_count=2;follow.repair_radius=.18;follow.repair_rounds=3;follow.repair_seed=args.seed+attempt*101;follow.repair_kick=.15
        print(json.dumps(dict(polishing_pair=item['stone_ids'],initial_upper_in=item['upper_diameter_in'])),flush=True)
        winner=local_repair(follow)
        if winner is not None:return winner
    return None


def select_pool_pair(args):
    """Localized two-placement MILP with exact conflict and gap-cover cuts.

    Raster occupancy supplies necessary (relaxed) conflict constraints. Every
    returned pair must pass exact original-outline and adaptive gap gates.
    """
    import copy,time
    from scipy.optimize import milp,Bounds,LinearConstraint
    from scipy.sparse import coo_matrix,csr_matrix,vstack
    source=Path(args.pool_pair);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'experiment_script.py').write_text(Path(__file__).read_text())
    (out/'settings.json').write_text(json.dumps(vars(args),indent=2))
    data=json.loads((source/'candidate_pool.json').read_text());layout=load_json(source/'pool_parent.json')
    history=copy.deepcopy(data['parent_history']);original_active=layout.active.copy()
    if validate(layout,gap=.5/12,edge=.5/12):raise ValueError('Invalid fixed parent')
    candidates=data['candidates'];region=box(data['start_ft'],0,history[-1]['front_x']+1,data['depth_ft'])
    candidates=[c for c in candidates if transformed(layout.shapes[c['stone_id']],c['pose']).distance(region)<1/12]
    # Exact duplicates add no alternatives to the finite placement model.
    seen=set();unique=[]
    for c in candidates:
        key=(c['stone_id'],*c['pose'])
        if key not in seen:seen.add(key);unique.append(c)
    candidates=unique;n=len(candidates)
    if n<2:raise ValueError('Pool has fewer than two relevant placements')
    polygons=np.asarray([transformed(layout.shapes[c['stone_id']],c['pose']) for c in candidates],dtype=object)
    minxs=np.asarray([g.bounds[0] for g in polygons]);base_front=history[-1]['front_x']+1
    local_inner={}
    for sid in {c['stone_id'] for c in candidates}:
        g=layout.shapes[sid];inner=g.simplify(.001,preserve_topology=True).buffer(-.0011,quad_segs=4)
        if inner.is_empty or not g.covers(inner):inner=g
        local_inner[sid]=inner.buffer(.5/24,quad_segs=4)
    cells={};rows=[];cols=[];spacing=.04
    for j,c in enumerate(candidates):
        g=transformed(local_inner[c['stone_id']],c['pose']);a,b,e,f=g.bounds
        xs=np.arange(math.floor(a/spacing),math.ceil(e/spacing)+1);ys=np.arange(math.floor(b/spacing),math.ceil(f/spacing)+1)
        xx,yy=np.meshgrid(xs,ys);points=shapely.points((xx.ravel()+.5)*spacing,(yy.ravel()+.5)*spacing)
        mask=shapely.contains(g,points)
        for x,y in zip(xx.ravel()[mask],yy.ravel()[mask]):
            key=(int(x),int(y));row=cells.setdefault(key,len(cells));rows.append(row);cols.append(j)
    occupancy=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(cells),n)).tocsr()
    occupancy=occupancy[np.asarray(occupancy.getnnz(axis=1)).ravel()>1]
    row_keys=set();keep=[]
    for i in range(occupancy.shape[0]):
        key=occupancy.indices[occupancy.indptr[i]:occupancy.indptr[i+1]].tobytes()
        if key not in row_keys:row_keys.add(key);keep.append(i)
    occupancy=occupancy[keep]
    print(json.dumps(dict(model_candidates=n,raster_rows=occupancy.shape[0],raster_nonzeros=int(occupancy.nnz))),flush=True)
    groups={sid:k for k,sid in enumerate(sorted({c['stone_id'] for c in candidates}))}
    identities=coo_matrix((np.ones(n),([groups[c['stone_id']] for c in candidates],np.arange(n))),shape=(len(groups),n)).tocsr()
    fixed=vstack([csr_matrix(np.ones((1,n))),identities,occupancy],format='csr')
    lower=np.r_[2,np.zeros(fixed.shape[0]-1)];upper=np.r_[2,np.ones(fixed.shape[0]-1)]
    constraints=[LinearConstraint(fixed,lower,upper)];trace=[];gap_cuts=[];start=time.monotonic();accepted=False
    parent_stones=[transformed(layout.shapes[sid],layout.poses[i]) for i,sid in enumerate(layout.ids) if layout.active[i]]
    first=bounded_gap(layout.patio,parent_stones,region,error=.05/12,stop_on_failure=False)
    def add_gap_cut(center):
        point=Point(*center);cover=shapely.distance(polygons,point)<1/12
        if center[0]<=base_front:
            coeff=cover.astype(float);lb=1.0
        else:
            # Outside the mandatory base region, coverage is conditional on
            # selecting a placement whose extent advances the front this far.
            triggers=minxs+.5>=center[0];coeff=2*cover.astype(float)-triggers.astype(float);lb=0.0
        constraints.append(LinearConstraint(csr_matrix(coeff.reshape(1,-1)),lb,np.inf))
        gap_cuts.append(dict(center=list(center),covering_candidates=int(cover.sum()),conditional=center[0]>base_front))
    if first['lower_diameter_in']>=2:add_gap_cut(first['center'])
    scores=np.zeros(n)  # Feasibility only; no proof of a best ranking is needed.
    for iteration in range(80):
        remaining=120-(time.monotonic()-start)
        if remaining<=0:break
        result=milp(scores,integrality=np.ones(n),bounds=Bounds(0,1),constraints=constraints,options=dict(time_limit=min(10,remaining),mip_rel_gap=0,presolve=False))
        print(json.dumps(dict(iteration=iteration,solver_status=int(result.status),has_incumbent=result.x is not None)),flush=True)
        if result.x is None:
            trace.append(dict(iteration=iteration,status=int(result.status),message=result.message));break
        chosen=np.flatnonzero(result.x>.5)
        if len(chosen)!=2:raise RuntimeError('MILP did not return a two-placement integer incumbent')
        layout.active[:]=original_active
        for j in chosen:
            i=layout.ids.index(candidates[j]['stone_id']);layout.active[i]=True;layout.poses[i]=candidates[j]['pose']
        errors=validate(layout,gap=.5/12,edge=.5/12)
        record=dict(iteration=iteration,status=int(result.status),chosen=chosen.tolist(),stone_ids=[candidates[j]['stone_id'] for j in chosen],errors=errors)
        if errors:
            row=np.zeros(n);row[chosen]=1;constraints.append(LinearConstraint(csr_matrix(row.reshape(1,-1)),-np.inf,1));trace.append(record);continue
        front=max(base_front,float(max(minxs[chosen]))+.5)
        stones=parent_stones+[polygons[j] for j in chosen]
        gap=bounded_gap(layout.patio,stones,box(data['start_ft'],0,front,data['depth_ft']))
        record.update(front_x=front,**gap);trace.append(record)
        print(json.dumps(dict(iteration=iteration,stone_ids=record['stone_ids'],gap_lower_in=gap['lower_diameter_in'],gap_upper_in=gap['upper_diameter_in'])),flush=True)
        if gap['passed']:
            accepted=True;break
        if gap['lower_diameter_in']<2:break  # unresolved; never invent a violating cut
        add_gap_cut(gap['center'])
    (out/'pair_trace.json').write_text(json.dumps(dict(accepted=accepted,candidate_count=n,raster_spacing_ft=spacing,gap_cuts=gap_cuts,iterations=trace,elapsed_seconds=time.monotonic()-start),indent=2))
    if not accepted:
        print(json.dumps(dict(accepted=False,candidate_count=n,iterations=len(trace))),flush=True)
        with Path('first_real_patio_followup_plan_progress.md').open('a') as log:log.write(f"\n- Two-placement MILP `{out}` ended without an accepted pair in its bounded pool: {n} candidates, {len(trace)} iterations. See pair_trace.json; no global infeasibility claim.\n")
        if args.polish_pairs:
            winner=polish_pair_population(args,data,candidates,polygons,trace,out)
            if winner is not None:
                follow=argparse.Namespace(**vars(args));follow.input=data['settings']['input'];follow.edge_index=data['settings']['edge_index']
                follow.prefix=str(winner);follow.output=str(out/'accepted');follow.target=len(history)+2;follow.paired_front=True;follow.export_pool=False
                run(follow);verify_saved(out/'accepted')
                with Path('first_real_patio_followup_plan_progress.md').open('a') as log:log.write(f"\n- Hybrid finite-pair/continuous repair `{out}` independently verified milestone{follow.target}; deliverables in `{out/'accepted'}`.\n")
        return
    chosen=sorted(chosen,key=lambda j:minxs[j]);layout.active[:]=original_active
    for k,j in enumerate(chosen,1):
        c=candidates[j];i=layout.ids.index(c['stone_id']);layout.active[i]=True;layout.poses[i]=c['pose']
        step=len(data['parent_history'])+k;step_front=data['parent_history'][-1]['front_x'] if k==1 else front
        stones=[transformed(layout.shapes[sid],layout.poses[i]) for i,sid in enumerate(layout.ids) if layout.active[i]]
        bound=bounded_gap(layout.patio,stones,box(data['start_ft'],0,step_front,data['depth_ft']))
        if not bound['passed'] or validate(layout,gap=.5/12,edge=.5/12):raise RuntimeError('Pair failed independent prefix reconstruction')
        history.append(dict(step=step,stone_id=c['stone_id'],pose=c['pose'],front_x=step_front,**bound))
    save_json(layout,out/'passing_prefix_local.json');(out/'passing_prefix.json').write_text(json.dumps(history,indent=2))
    follow=argparse.Namespace(**vars(args));follow.input=data['settings']['input'];follow.edge_index=data['settings']['edge_index']
    follow.prefix=str(out);follow.output=str(out/'accepted');follow.target=len(history);follow.paired_front=True;follow.export_pool=False
    run(follow);verify_saved(out/'accepted')
    with Path('first_real_patio_followup_plan_progress.md').open('a') as log:log.write(f"\n- MILP pair `{out}` achieved and independently verified milestone{len(history)}; placements and numbered images in `{out/'accepted'}`. All prefixes and original shapes checked.\n")


def local_repair(args):
    """Joint constrained optimization; sampled objective never certifies a pass."""
    import copy
    from scipy.optimize import minimize
    source=Path(args.local_repair);out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    (out/'experiment_script.py').write_text(Path(__file__).read_text())
    (out/'settings.json').write_text(json.dumps(vars(args),indent=2))
    report=json.loads((source/'report.json').read_text())
    layout=load_json(source/'layout_local.json')
    failure=report['earliest_failure']
    records=copy.deepcopy(failure['prefix']+[failure])
    source_settings=report.get('settings',{})
    paired=source_settings.get('paired_front',False)
    moving=[layout.ids.index(rec['stone_id']) for rec in records[-args.repair_count:]]
    baseline=layout.poses.copy();ids=[layout.ids.index(rec['stone_id']) for rec in records]
    xx,yy=np.meshgrid(np.arange(1,failure['front_x']+.02,.045),np.arange(0,1.001,.045))
    points=shapely.points(xx.ravel(),yy.ravel())
    diagnostic=source/'full_failure_bound.json'
    center=json.loads(diagnostic.read_text())['center'] if diagnostic.exists() else failure['center']
    center_file=source/'repair_centers.json'
    raw_centers=json.loads(center_file.read_text()) if center_file.exists() else [center]
    raw_centers += [dict(center=rec['center'],step=rec['step']) for rec in records if 'center' in rec]
    centers=[]
    for item in raw_centers:
        item=item if isinstance(item,dict) else dict(center=item,step=len(records))
        if not any(item['step']==old['step'] and np.linalg.norm(np.asarray(item['center'])-np.asarray(old['center']))<.03 for old in centers): centers.append(item)
    points=points[shapely.covers(layout.patio,points)]
    edge_dist=shapely.distance(points,layout.patio.boundary)
    objective_proxies={}
    for i in ids:
        original=layout.shapes[layout.ids[i]]
        proxy=original.simplify(.001,preserve_topology=True).buffer(.0011,quad_segs=4)
        objective_proxies[i]=proxy if proxy.covers(original) else original
    fixed_count=len(ids)-len(moving)
    fixed_geometry=[transformed(layout.shapes[layout.ids[i]],baseline[i]) for i in ids[:fixed_count]]
    fixed_field=edge_dist.copy()
    for i in ids[:fixed_count]:
        fixed_field=np.minimum(fixed_field,shapely.distance(points,transformed(objective_proxies[i],baseline[i])))
    cache={};evaluations=0
    def evaluate(x):
        nonlocal evaluations
        if cache.get('x') is not None and np.array_equal(x,cache['x']):return cache
        layout.poses[:]=baseline
        layout.poses[moving]+=np.asarray(x).reshape(-1,3)*args.repair_radius
        gs=fixed_geometry+[transformed(layout.shapes[layout.ids[i]],layout.poses[i]) for i in ids[fixed_count:]]
        fronts=[];front=1.0
        for rec,g in zip(records,gs):
            if paired and rec['step']>5:
                front=max(front,rec['front_x']) if rec['step']%2==0 else max(front+1,rec['front_x'],g.bounds[0]+.5)
            else:
                front=max(front+.5,rec['front_x'],g.bounds[0]+.5)
            fronts.append(front)
        # A fixed minimum front prevents repair from retreating behind a defect.
        constraints=[];gap=.5/12+1e-7
        for position,g in enumerate(gs):
            if position<fixed_count:continue
            constraints.append(12*(g.distance(layout.patio.boundary)-gap) if layout.patio.covers(g) else -12*(math.sqrt(g.difference(layout.patio).area)+gap))
        constraints.extend(12*(a.distance(b)-gap) for j,a in enumerate(gs) for k,b in enumerate(gs[j+1:],start=j+1) if k>=fixed_count)
        distances=fixed_field.copy();scores=[]
        for n,(g,i) in enumerate(zip(gs[fixed_count:],ids[fixed_count:]),start=fixed_count):
            coarse=transformed(objective_proxies[i],layout.poses[i])
            distances=np.minimum(distances,shapely.distance(points,coarse))
            if n>=0:
                mask=shapely.get_x(points)<=fronts[n]
                scores.append(float(np.max(distances[mask]))*24)
        # Finite grids have plateaus where the wall is the nearest obstacle.
        # Refine the known circle center continuously to obtain a useful local
        # objective; final adaptive subdivision remains the certification gate.
        objective_shapes=gs
        objective=max(scores)
        for item in centers:
            center=item['center'];stage=min(item['step'],len(records))
            if stage<=fixed_count: continue
            def negative_radius(xy):
                p=Point(float(xy[0]),float(xy[1]))
                if not layout.patio.covers(p): return p.distance(layout.patio)
                return -min(p.distance(layout.patio.boundary),*(p.distance(g) for g in objective_shapes[:stage]))
            xlimit=fronts[stage-1]
            simplex=np.asarray([center,[min(xlimit,center[0]+.01),center[1]],[center[0],max(0,min(1,center[1]-.01 if center[1]>.98 else center[1]+.01))]])
            inner=minimize(negative_radius,np.asarray(center),method='Nelder-Mead',
                bounds=[(1,xlimit),(0,1)],
                options=dict(xatol=1e-6,fatol=1e-7,maxiter=100,initial_simplex=simplex))
            objective=max(objective,-float(inner.fun)*24)
        cache.clear();cache.update(x=np.array(x),gs=gs,fronts=fronts,constraints=np.asarray(constraints),objective=objective)
        evaluations+=1
        if evaluations%50==0:print(json.dumps(dict(evaluations=evaluations,objective=objective,min_constraint=min(constraints))),flush=True)
        return cache
    initial=np.random.default_rng(args.repair_seed).uniform(-args.repair_kick,args.repair_kick,3*len(moving))
    result=minimize(lambda x:evaluate(x)['objective'],initial,method='COBYLA',bounds=[(-1,1)]*(3*len(moving)),
        constraints=[dict(type='ineq',fun=lambda x:evaluate(x)['constraints'])],
        options=dict(rhobeg=.25,tol=1e-5,catol=1e-6,maxiter=900))
    data=evaluate(result.x)
    (out/'optimizer.json').write_text(json.dumps(dict(success=bool(result.success),message=result.message,x=result.x.tolist(),objective=float(result.fun),evaluations=evaluations,minimum_constraint=float(min(data['constraints']))),indent=2))
    passed=True
    for n,rec in enumerate(records,1):
        layout.active[:]=False
        layout.active[ids[:n]]=True
        errors=validate(layout,gap=.5/12,edge=.5/12)
        bound=bounded_gap(layout.patio,data['gs'][:n],box(1,0,data['fronts'][n-1],1))
        rec.update(bound);rec['pose']=layout.poses[ids[n-1]].tolist();rec['front_x']=data['fronts'][n-1]
        rec['clearance_errors']=errors
        passed=passed and not errors and bound['passed']
    save_json(layout,out/'passing_prefix_local.json')
    cleaned=[{k:v for k,v in rec.items() if k not in ('prefix','poses','active')} for rec in records]
    (out/'passing_prefix.json').write_text(json.dumps(cleaned,indent=2))
    (out/'repair_result.json').write_text(json.dumps(dict(passed=passed,records=cleaned),indent=2))
    if not passed:
        feedback=out/'next_source';feedback.mkdir()
        save_json(layout,feedback/'layout_local.json')
        (feedback/'report.json').write_text(json.dumps(dict(settings=source_settings,earliest_failure={**cleaned[-1],'prefix':cleaned[:-1]}),indent=2))
        new_centers=centers+[dict(center=rec['center'],step=rec['step']) for rec in cleaned if not rec['passed']]
        (feedback/'repair_centers.json').write_text(json.dumps(new_centers,indent=2))
    with Path('first_real_patio_followup_plan_progress.md').open('a') as log:
        log.write(f"\n- COBYLA joint repair `{out}` ended: all-prefix gates passed={passed}, sampled objective={result.fun:.6f}in. See optimizer and repair result JSON. Fronts were held at least at their pre-repair values. Saved-file milestone verification still required.\n")
    print(json.dumps(dict(all_prefix_gates_passed=passed,objective=float(result.fun))),flush=True)
    if not passed and args.repair_rounds>1:
        follow=argparse.Namespace(**vars(args))
        follow.local_repair=str(out/'next_source');follow.output=str(out/'continuation')
        follow.repair_rounds-=1
        follow.repair_seed+=1
        return local_repair(follow)
    return out if passed else None


def generate_solver_angles(args):
    """Use actual PackingSolver two-item subproblems as orientation proposals."""
    import packingsolver.irregular as ps
    layout=load_json(args.input)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    scale=1000
    def shape(g):
        proxy=g.simplify(.002,preserve_topology=True).buffer(.003,quad_segs=4)
        if not proxy.covers(g): proxy=g
        coords=list(proxy.exterior.coords[:-1])
        if not proxy.exterior.is_ccw: coords.reverse()
        return ps.build_shape([(float(x)*scale,float(y)*scale) for x,y in coords])
    # Historical PackingSolver poses provide broad seeds; new subproblems add
    # placements from the installed engine with at most two new items per bin.
    orientations=[dict(stone_id=sid,angle_radians=float(layout.poses[i,2]),source='historical_packingsolver') for i,sid in enumerate(layout.ids) if layout.active[i]]
    fixed=[]; fixed_ids=set()
    if args.prefix:
        prefix=Path(args.prefix)
        local_layout=load_json(prefix/'passing_prefix_local.json')
        ring=np.asarray(layout.patio.exterior.coords)
        origin=ring[args.edge_index];v=ring[args.edge_index+1]-origin
        theta=math.atan2(v[1],v[0]);c,s=math.cos(theta),math.sin(theta)
        if not layout.patio.exterior.is_ccw: raise ValueError('Expected CCW source patio for defect transform')
        for i in np.flatnonzero(local_layout.active):
            x,y,a=local_layout.poses[i]
            fixed.append(transformed(layout.shapes[layout.ids[i]],(origin[0]+c*x-s*y,origin[1]+s*x+c*y,a+theta)))
            fixed_ids.add(layout.ids[i])
    available=[i for i,sid in enumerate(layout.ids) if sid not in fixed_ids]
    groups=[available[i:i+24] for i in range(0,len(available),24)]
    certificates=[]
    for group_index,indices in enumerate(groups):
        b=ps.InstanceBuilder();b.set_objective(ps.Objective.Knapsack)
        patio=ps.build_shape([(float(x)*scale,float(y)*scale) for x,y in layout.patio.exterior.coords[:-1]])
        bin_id=b.add_bin_type(patio,copies=1,item_bin_minimum_spacing=(.5/12+.0001)*scale)
        b.set_item_item_minimum_spacing((.5/12+.0001)*scale)
        for g in fixed:
            defect=b.add_defect(bin_id,0,ps.ShapeWithHoles(shape(g)))
            b.set_item_defect_minimum_spacing(bin_id,defect,(.5/12+.0001)*scale)
        resource=b.add_bin_type_resource(bin_id,2)
        for i in indices:
            item=b.add_item_type(shape(layout.shapes[layout.ids[i]]),copies=1,profit=1,allowed_rotations=[(0,360,False)])
            b.add_resource_consumption(bin_id,resource,item,[1])
        parameters=ps.OptimizeParameters();parameters.time_limit=4;parameters.verbosity_level=0
        def retain(output):
            cert_path=out/f'group_{group_index}.json'
            output.solution.write(str(cert_path))
            cert=json.loads(cert_path.read_text())
            if not cert: return
            for bin in cert.get('bins',[]):
                if len(bin['items'])>2: raise RuntimeError('Resource count limit violated')
                for item in bin['items']:
                    if item.get('mirror'): raise RuntimeError('Unexpected mirrored seed')
                    orientations.append(dict(stone_id=layout.ids[indices[int(item['id'])]],angle_radians=math.radians(item['angle']),source=f'packingsolver_group_{group_index}'))
        parameters.new_solution_callback=retain
        result=ps.optimize(b.build(),parameters);retain(result)
    (out/'orientations.json').write_text(json.dumps(dict(orientations=orientations,method='Actual-patio two-item resource-limited PackingSolver subproblems; orientation proposals only, not accepted layouts'),indent=2))
    print(json.dumps({'orientation_proposals':len(orientations),'groups':len(groups)}),flush=True)


def verify_saved(directory):
    """Reload physical-frame output; check every prefix without proposal proxies."""
    out=Path(directory)
    report=json.loads((out/'report.json').read_text())
    layout=load_json(out/'layout.json')
    original=load_json(report['settings']['input'])
    if layout.ids!=original.ids or not layout.patio.equals_exact(original.patio,1e-10) or any(not layout.shapes[sid].equals_exact(original.shapes[sid],1e-10) for sid in layout.ids):
        raise ValueError('Saved geometry differs from the detailed source inventory/patio')
    sequence=report['accepted_sequence']
    sequence_ids=[rec['stone_id'] for rec in sequence]
    active_ids={layout.ids[i] for i in np.flatnonzero(layout.active)}
    if len(set(sequence_ids))!=len(sequence_ids) or set(sequence_ids)!=active_ids:
        raise ValueError('Saved active stones do not match the unique placement sequence')
    fronts=[rec['front_x'] for rec in sequence]
    if any(b<a for a,b in zip(fronts,fronts[1:])) or [rec['step'] for rec in sequence]!=list(range(1,len(sequence)+1)):
        raise ValueError('Retreating front or inconsistent sequence numbering')
    angle=report['local_angle']; c,s=math.cos(angle),math.sin(angle)
    origin=np.asarray(report['local_origin'])
    ring=np.asarray(layout.patio.exterior.coords)-origin
    local_patio=Polygon(ring @ np.array([[c,-s],[s,c]]))
    layout.active[:]=False
    results=[]
    for record in report['accepted_sequence']:
        i=layout.ids.index(record['stone_id']);layout.active[i]=True
        errors=validate(layout,gap=.5/12,edge=.5/12)
        stones=[]
        for j in np.flatnonzero(layout.active):
            x,y,a=layout.poses[j];q=np.array([x,y])-origin
            pose=[c*q[0]+s*q[1],-s*q[0]+c*q[1],a-angle]
            stones.append(transformed(layout.shapes[layout.ids[j]],pose))
        region=box(report['start_ft'],0,record['front_x'],report['completed_depth_ft'])
        bound=bounded_gap(local_patio,stones,region,error=.005/12,max_cells=100000)
        results.append(dict(step=record['step'],stone_id=record['stone_id'],front_ft=record['front_x'],minimum_edge_clearance_in=12*min(g.distance(local_patio.boundary) for g in stones),minimum_pair_clearance_in=min((12*a.distance(b) for j,a in enumerate(stones) for b in stones[j+1:]),default=None),clearance_errors=errors,**bound))
    passed=len(results)==report['settings'].get('target',5) and all(not r['clearance_errors'] and r['passed'] for r in results)
    (out/'verification.json').write_text(json.dumps(dict(passed=passed,results=results),indent=2))
    print(json.dumps(dict(independent_saved_prefix_validation=passed,results=results)),flush=True)
    if not passed: raise RuntimeError('Saved milestone sequence did not pass verification')
    import csv
    columns=['step','stone_id','front_ft','minimum_edge_clearance_in','minimum_pair_clearance_in','lower_diameter_in','upper_diameter_in','passed']
    with (out/'results.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    if not (out/'README.md').exists():
        lines=[f"# Verified {len(results)}-stone edge experiment",'',
            'Every saved prefix passes detailed original-outline clearances and bounded maximum-gap verification. No mirroring, cutting or reshaping.','',
            '| Step | Stone | Edge min (in) | Pair min (in) | Maximum gap bound (in) |',
            '|---|---|---|---|---|']
        for rec in results:
            pair='—' if rec['minimum_pair_clearance_in'] is None else f"{rec['minimum_pair_clearance_in']:.6f}"
            lines.append(f"| {rec['step']} | {rec['stone_id']} | {rec['minimum_edge_clearance_in']:.6f} | {pair} | [{rec['lower_diameter_in']:.6f}, {rec['upper_diameter_in']:.6f}] |")
        lines.extend(['',
            'The certified completed region is the actual patio intersected with the local twelve-inch-deep strip, starting at local x=1ft and ending at each saved nonretreating front. Circle centers lie in that region; radii are constrained only by stones and the physical patio. Circles may cross reporting edges/fronts. Unfinished space outside the strip is not certified.',
            '',f"Paired-front schedule after five stones: {report['settings'].get('paired_front',False)}. See report settings and the root progress log for the exact search/repair lineage.",
            '', 'Artifacts: `layout.json` (physical patio frame), `layout_local.json`, `report.json`, `verification.json`, `results.csv`, numbered `step_N.png` close-ups, and saved passing-prefix files for continuation. Image diameter intervals are search bounds; this table uses tighter independent bounds.',
            '', 'These remain preliminary photographed/traced geometry, not a construction-ready plan; see `first_real_patio_test.md` and documented calibration limitations. Passing this local milestone does not prove whole-patio completion.'])
        (out/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--polish-pairs',type=int,default=0,help='Continuously refine this many diverse failed finite-model pairs')
    ap.add_argument('--export-pool',action='store_true')
    ap.add_argument('--pool-pair',help='Select the next two placements jointly from an exported finite pool')
    ap.add_argument('--exclude-ids',nargs='*',default=[],help='Temporarily exclude known poor choices in a bounded branch')
    ap.add_argument('--candidate-ids',nargs='*',help='Restrict a targeted closure to these unused stone IDs')
    ap.add_argument('--paired-front',action='store_true',help='After5: hold front for even placements, advance at least one foot at odd milestones')
    ap.add_argument('--repair-seed',type=int,default=0)
    ap.add_argument('--repair-kick',type=float,default=0,help='Normalized initial pose perturbation for bounded iterated local search')
    ap.add_argument('--repair-rounds',type=int,default=1,help='Bounded exchanges adding independently detected failed gap centers')
    ap.add_argument('--repair-count',type=int,default=3)
    ap.add_argument('--repair-radius',type=float,default=.04)
    ap.add_argument('--local-repair',help='Optimize a retained failed junction jointly with COBYLA; --repair-count selects the neighborhood')
    ap.add_argument('--exact-proposals',action='store_true')
    ap.add_argument('--target',type=int,choices=[5,7,9,11,13],default=5)
    ap.add_argument('--solver-angles')
    ap.add_argument('--generate-solver-angles',action='store_true')
    ap.add_argument('--edge-index',type=int,default=6,help='Physical traced segment defining the growth direction')
    ap.add_argument('--prefix',help='Continue a saved passing prefix with a new bounded search')
    ap.add_argument('--verify',help='Reload and verify an existing experiment directory')
    ap.add_argument('--input',default='examples/real_stones_concept/layout.json')
    ap.add_argument('--output',default='examples/five_stone_edge_start')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--proposals',type=int,default=600)
    ap.add_argument('--shortlist',type=int,default=6)
    args=ap.parse_args()
    if args.pool_pair: select_pool_pair(args)
    elif args.local_repair: local_repair(args)
    elif args.generate_solver_angles: generate_solver_angles(args)
    elif args.verify: verify_saved(args.verify)
    else: run(args)
