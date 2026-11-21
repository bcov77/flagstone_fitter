"""Planning diagnostics on original outlines; no learned quality labels or physics.

Residual regions come from complete-board free space, before ROI filtering.
Morphological mouth cuts and ROI edges never become physical anchors.
"""
import math
import numpy as np
import shapely
from shapely.affinity import rotate
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from scipy import signal

from selection_geometry import profile,longest_run,tangent


def parts(g):
    return [p for p in getattr(g,'geoms',[g]) if p.geom_type=='Polygon' and not p.is_empty]


def walls(patio):
    xy=list(patio.exterior.coords)
    return {f'wall_{i}':LineString([a,b]) for i,(a,b) in enumerate(zip(xy,xy[1:]))}


def joint_profile(subject,neighbors,wall_lines,gap,config):
    points,step,positions=profile(subject.exterior,config['boundary_step_ft'])
    coords=shapely.get_coordinates(points);tv=tangent(subject.exterior,positions)
    records={};all_near=np.zeros(len(points),dtype=bool)
    for owner,geometry in {**neighbors,**wall_lines}.items():
        boundary=geometry.boundary if geometry.geom_type=='Polygon' else geometry
        ds=np.asarray(shapely.distance(points,boundary))
        near=(ds>=gap-1e-6)&(ds<=gap+config['joint_excess_inches']/12)
        # Local tangent compatibility is evaluated against actual source edges.
        nearest=shapely.get_point(shapely.shortest_line(points,boundary),1)
        if boundary.is_ring:
            pos=shapely.line_locate_point(boundary,nearest);other=tangent(boundary,pos)
        else:
            xy=shapely.get_coordinates(boundary);v=xy[-1]-xy[0]
            other=np.tile(v/max(np.linalg.norm(v),1e-12),(len(points),1))
        aligned=np.abs(np.sum(tv*other,axis=1))>=math.cos(math.radians(config['joint_angle_degrees']))
        matched=near&aligned;all_near|=matched
        if matched.any():
            records[owner]=dict(matched_length_ft=float(matched.sum()*step),
                                longest_run_ft=longest_run(matched,step))
    return dict(perimeter_ft=subject.length,matched_fraction=float(all_near.mean()),
                matched_length_ft=float(all_near.sum()*step),longest_run_ft=longest_run(all_near,step),
                by_owner=records)


def reference_stability(board,patio,gap,config):
    result={};wl=walls(patio)
    for sid,g in board.items():
        others={s:h for s,h in board.items() if s!=sid and h.distance(g)<.2}
        p=joint_profile(g,others,wl,gap,config)
        # A continuous, provisional diagnostic. It does not make a hard lock.
        result[sid]=dict(score=p['matched_fraction'],longest_joint_ft=p['longest_run_ft'])
    return result


def residual_regions(free,roi,config):
    radius=config['residual_core_inches']/12
    out=[];claimed=Polygon()
    for core in sorted(parts(free.buffer(-radius)),key=lambda p:-p.area):
        if core.area<config['min_core_sqft'] or core.intersection(roi).area<config['min_core_sqft']:
            continue
        # The full global region is recovered before local relevance filtering.
        ps=parts(core.buffer(radius).intersection(free))
        if not ps:continue
        patch=max(ps,key=lambda p:p.intersection(core).area).difference(claimed)
        claimed=unary_union([claimed,patch])
        # Clipping only limits measurement. Ownership below tests true physical
        # edges, so the new ROI boundary cannot become an anchor.
        for local in parts(patch.intersection(roi)):
            local_core=core.intersection(local)
            if local_core.area>=config['min_core_sqft']:out.append((local,local_core))
    return sorted(out,key=lambda pair:-pair[0].area)


def region_owners(patch,board,subject_id,wall_lines,gap,stability,config,buffered_edges=None):
    # Callers may share this lazy cache only across regions of the same board
    # and clearance. Shapely geometries are immutable; exact buffers are reused.
    if buffered_edges is None:buffered_edges={}
    samples=[profile(ring,config['boundary_step_ft']) for ring in [patch.exterior,*patch.interiors]]
    points=np.concatenate([p for p,_,_ in samples]);weights=np.concatenate([np.full(len(p),step) for p,step,_ in samples])
    owners={};physical=np.zeros(len(points),dtype=bool)
    for sid,g in board.items():
        if g.distance(patch)>gap+.01:continue
        if sid not in buffered_edges:buffered_edges[sid]=g.buffer(gap).boundary
        edge=buffered_edges[sid]
        hit=np.asarray(shapely.distance(points,edge))<.002
        if weights[hit].sum()>=config['owner_min_length_ft']:
            role='subject' if sid==subject_id else ('locked' if sid in config['locked_stones'] else 'neighbor')
            owners[sid]=dict(role=role,length_ft=float(weights[hit].sum()),
                             stability=stability.get(sid,{}).get('score',0.))
            physical|=hit
    for wid,line in wall_lines.items():
        # Free-space walls are offset inward by the full wall clearance.
        hit=np.abs(np.asarray(shapely.distance(points,line))-gap)<.002
        if weights[hit].sum()>=config['owner_min_length_ft']:
            owners[wid]=dict(role='wall',length_ft=float(weights[hit].sum()),stability=1.)
            physical|=hit
    return owners,float(weights[~physical].sum())


def filler_witness(patch,core,free,inventory,config):
    """Budgeted actual-shape placement search, with full-outline verification.

    Failure is 'not found in bounded search', never a proof of unfillability.
    Different region witnesses are independent and cannot be composed blindly.
    """
    allowed=free.intersection(patch.buffer(config['filler_search_pad_ft']))
    step=config['filler_raster_step_ft'];a,b,c,d=allowed.bounds
    xs=np.arange(a,c+step,step);ys=np.arange(b,d+step,step)
    if not len(xs) or not len(ys):return dict(status='no_search_region',attempted_pieces=0)
    fm=shapely.contains_xy(allowed,xs[None,:],ys[:,None]).astype(float)
    pm=shapely.contains_xy(core,xs[None,:],ys[:,None]).astype(float)
    choices=[(sid,g) for sid,g in inventory.items() if g.area<=allowed.area+1e-8]
    choices.sort(key=lambda item:abs(math.log(max(item[1].area,1e-10)/max(patch.area,.001))))
    tried=0;poses=0
    for sid,local in choices[:config['filler_max_pieces']]:
        tried+=1
        for degrees in range(0,360,config['filler_angle_step_degrees']):
            angle=math.radians(degrees);co,si=math.cos(angle),math.sin(angle)
            xy=np.asarray(local.exterior.coords)@np.array([[co,si],[-si,co]])
            g=Polygon(xy);x0,y0,x1,y1=g.bounds
            center=core.centroid
            dx,dy=center.x-g.centroid.x,center.y-g.centroid.y
            centered=shapely.affinity.affine_transform(local,[co,-si,si,co,dx,dy]);poses+=1
            fraction=centered.intersection(patch).area/patch.area
            if allowed.covers(centered) and fraction>=config['filler_min_patch_fraction'] and centered.covers(core.representative_point()):
                return dict(status='verified_filler_pose',stone=sid,pose=[dx,dy,angle],
                    patch_fraction=float(fraction),attempted_pieces=tried,checked_poses=poses,
                    scope='Independent one-region continuation; not a joint allocation or sequence')
            kx=np.arange(x0+step/2,x1,step);ky=np.arange(y0+step/2,y1,step)
            if not len(kx) or not len(ky) or len(kx)>len(xs) or len(ky)>len(ys):continue
            km=shapely.contains_xy(g,kx[None,:],ky[:,None]).astype(float)
            cost=np.where(fm>.5,0.,50.)-pm
            conv=signal.fftconvolve(cost,km[::-1,::-1],mode='valid')
            n=min(config['filler_positions_per_angle'],conv.size)
            indexes=np.argpartition(conv.ravel(),n-1)[:n]
            for index in sorted(indexes,key=lambda i:conv.ravel()[i]):
                iy,ix=np.unravel_index(index,conv.shape);dx,dy=float(xs[ix]-kx[0]),float(ys[iy]-ky[0])
                exact=shapely.affinity.affine_transform(local,[co,-si,si,co,dx,dy]);poses+=1
                if not allowed.covers(exact):continue
                fraction=exact.intersection(patch).area/patch.area
                if fraction<config['filler_min_patch_fraction'] or not exact.covers(core.representative_point()):continue
                return dict(status='verified_filler_pose',stone=sid,pose=[dx,dy,angle],
                    patch_fraction=float(fraction),attempted_pieces=tried,checked_poses=poses,
                    scope='Independent one-region continuation; not a joint allocation or sequence')
    return dict(status='no_witness_in_bounded_search',attempted_pieces=tried,checked_poses=poses,
                eligible_pieces=len(choices),piece_budget=config['filler_max_pieces'],
                search_pad_ft=config['filler_search_pad_ft'])


def alignment_options(subject,neighbor_reference,wall_lines,pocket,config):
    """Conditional single-boundary flat-face options; no neighbor-motion claim."""
    anchors={}
    for wid,line in wall_lines.items():
        if line.distance(pocket)<config['alignment_anchor_distance_ft']:
            anchors[wid]=[(line,'wall')]
    for sid,g in neighbor_reference.items():
        if g.distance(pocket)>config['alignment_anchor_distance_ft']:continue
        xy=list(g.simplify(.02,preserve_topology=True).exterior.coords)
        segments=[LineString([a,b]) for a,b in zip(xy,xy[1:])]
        segments=[line for line in segments if line.length>=config['alignment_min_edge_ft'] and line.distance(pocket)<config['alignment_anchor_distance_ft']]
        if segments:anchors[sid]=[(line,'neighbor') for line in segments]
    result={}
    center=pocket.centroid
    for owner,segments in anchors.items():
        best=dict(length_ft=0.,rotation_delta_degrees=0.,role=segments[0][1],conditional=True)
        for degrees in range(-config['alignment_angle_span_degrees'],config['alignment_angle_span_degrees']+1,5):
            g=rotate(subject,degrees,origin='centroid')
            points,step,pos=profile(g.exterior,config['boundary_step_ft'])
            xy=shapely.get_coordinates(points);tv=tangent(g.exterior,pos)
            for line,role in segments:
                ends=np.asarray(line.coords);direction=ends[-1]-ends[0];direction/=np.linalg.norm(direction)
                normal=np.array([-direction[1],direction[0]])
                if np.dot(np.array([center.x,center.y])-ends.mean(axis=0),normal)<0:normal=-normal
                proj=xy@normal
                face=(proj-proj.min())<=config['alignment_face_tolerance_inches']/12
                aligned=np.abs(tv@direction)>=math.cos(math.radians(config['joint_angle_degrees']))
                run=min(line.length,longest_run(face&aligned,step))
                if run>best['length_ft']:best.update(length_ft=float(run),rotation_delta_degrees=degrees)
        result[owner]=best
    return result


def assess(board,subject_id,reference_neighbors,patio,pocket,roi,inventory,stability,gap,config):
    subject=board.get(subject_id)
    wl=walls(patio)
    free=patio.buffer(-gap).difference(unary_union(list(board.values())).buffer(gap))
    regions=residual_regions(free,roi,config);records=[]
    indent=subject.simplify(.015,preserve_topology=True).convex_hull.difference(subject) if subject is not None else Polygon()
    totals=dict(wall_corner_risk_sqft=0.,rigid_pocket_risk_sqft=0.,wall_excess_sqft=0.,
                subject_indent_sqft=0.,stable_neighbor_exposure_sqft=0.,flexible_residual_sqft=0.,
                verified_fillable_regions=0,unsearched_regions=0)
    remaining=dict(inventory);repair_free=free;allocations=[];placed=[]
    for index,(patch,core) in enumerate(regions):
        owners,caps=region_owners(patch,board,subject_id,wl,gap,stability,config)
        roles=[v['role'] for v in owners.values()]
        flexible=[sid for sid,v in owners.items() if v['role']=='neighbor']
        wall_count=roles.count('wall');rigid=not flexible and any(r in roles for r in ('wall','locked','subject'))
        prior=next((item for item,g in placed if g.covers(core.representative_point()) and
                    g.intersection(patch).area/patch.area>=config['filler_min_patch_fraction']),None)
        if prior:
            check=dict(status='covered_by_prior_filler',stone=prior['stone'],pose=prior['pose'])
        else:
            check=filler_witness(patch,core,repair_free,remaining,config) if index<config['max_filler_regions'] else dict(status='not_searched_region_budget')
        fillable=check['status'] in ('verified_filler_pose','covered_by_prior_filler')
        if check['status']=='verified_filler_pose':
            sid=check['stone'];x,y,a=check['pose'];co,si=math.cos(a),math.sin(a)
            g=shapely.affinity.affine_transform(remaining.pop(sid),[co,-si,si,co,x,y])
            item=dict(stone=sid,pose=check['pose']);allocations.append(item);placed.append((item,g))
            repair_free=repair_free.difference(g.buffer(gap))
            check['scope']='Greedy distinct-piece, mutually clearance-checked endpoint continuation; not a motion sequence'
        totals['verified_fillable_regions']+=int(fillable)
        totals['unsearched_regions']+=int(check['status']=='not_searched_region_budget')
        # These are risk diagnostics when no concrete filler witness exists,
        # not categorical assertions that repair is impossible.
        records.append(dict(id=index,area_sqft=patch.area,core_area_sqft=core.area,
            polygon=shapely.to_wkt(patch,rounding_precision=8),owners=owners,
            artificial_mouth_length_ft=caps,direct_repair_dependencies=flexible,
            no_direct_flexible_boundary=rigid,wall_corner=wall_count>=2,
            subject_indent_sqft=float(patch.intersection(indent).area),filler=check))
    # Score actual localized remaining space, including partially filled regions.
    # Sharing a global connected component with a wall/corner does not assign
    # the whole component's area to that anchor.
    wall_band=unary_union([line.buffer(config['wall_risk_band_inches']/12) for line in wl.values()])
    corner_band=unary_union([shapely.Point(xy).buffer(config['corner_risk_radius_inches']/12) for xy in list(patio.exterior.coords)[:-1]])
    for record,(patch,_) in zip(records,regions):
        unresolved=patch.intersection(repair_free)
        radius=config['residual_core_inches']/12
        unresolved=unresolved.buffer(-radius).buffer(radius).intersection(unresolved)
        record['remaining_excess_sqft']=unresolved.area
        totals['wall_corner_risk_sqft']+=unresolved.intersection(corner_band).area
        totals['wall_excess_sqft']+=unresolved.intersection(wall_band).area
        totals['rigid_pocket_risk_sqft']+=unresolved.area if record['no_direct_flexible_boundary'] else 0.
        totals['subject_indent_sqft']+=unresolved.intersection(indent).area
        totals['flexible_residual_sqft']+=unresolved.area if record['direct_repair_dependencies'] else 0.
        remaining_region=unresolved
        for sid in sorted(reference_neighbors,key=lambda sid:-stability.get(sid,{}).get('score',0.)):
            score=stability.get(sid,{}).get('score',0.)
            if not score:continue
            g=board.get(sid,reference_neighbors[sid])
            band=g.buffer(gap+config['stable_risk_band_inches']/12).difference(g.buffer(gap))
            overlap=remaining_region.intersection(band)
            totals['stable_neighbor_exposure_sqft']+=score*overlap.area
            remaining_region=remaining_region.difference(overlap)
    joints=joint_profile(subject,{s:g for s,g in board.items() if s!=subject_id},wl,gap,config) if subject is not None else dict(by_owner={},matched_length_ft=0.,longest_run_ft=0.)
    options=alignment_options(subject,reference_neighbors,wl,pocket,config) if subject is not None else {}
    return dict(metrics=totals,regions=records,joints=joints,alignment_options=options,
                repair_filler_plan=allocations,
                scope='Complete-board free space with fixed target ROI; actual full-size endpoint geometry',
                caveat='No witness is not impossibility. Fillers form a greedy endpoint plan; alignment options are independent conditional possibilities.')


def compare(candidate,baseline,config):
    tolerance=config['risk_area_indifference_sqft']
    keys=['wall_corner_risk_sqft','rigid_pocket_risk_sqft','wall_excess_sqft','subject_indent_sqft']
    deltas={k:candidate['metrics'][k]-baseline['metrics'][k] for k in keys}
    worse=[k for k,d in deltas.items() if d>tolerance]
    better=[k for k,d in deltas.items() if d<-tolerance]
    option_losses=[];option_gains=[]
    for owner,old in baseline['alignment_options'].items():
        new=candidate['alignment_options'].get(owner,dict(length_ft=0.))
        delta=new['length_ft']-old['length_ft']
        if delta < -config['alignment_run_indifference_ft']:option_losses.append(owner)
        if delta > config['alignment_run_indifference_ft']:option_gains.append(owner)
    joint_delta=candidate['joints']['matched_length_ft']-baseline['joints']['matched_length_ft']
    old_corner=baseline.get('reference_corner_deficit_inches');new_corner=candidate.get('reference_corner_deficit_inches')
    corner_delta=new_corner-old_corner if old_corner is not None and new_corner is not None else None
    stable_delta=candidate['metrics']['stable_neighbor_exposure_sqft']-baseline['metrics']['stable_neighbor_exposure_sqft']
    if worse:status='new_residual_risk_review'
    elif corner_delta is not None and corner_delta>config['corner_reach_indifference_inches']:status='corner_reach_tradeoff_review'
    elif option_losses:status='future_alignment_tradeoff_review'
    elif better or joint_delta>config['alignment_run_indifference_ft'] or (corner_delta is not None and corner_delta<-config['corner_reach_indifference_inches']):status='geometric_planning_advantage_review'
    elif stable_delta>tolerance:status='provisional_anchor_tradeoff_review'
    else:status='unclear_retain_or_review'
    return dict(status=status,risk_deltas=deltas,new_risk_axes=worse,reduced_risk_axes=better,
                lost_alignment_options=option_losses,gained_alignment_options=option_gains,
                matched_joint_delta_ft=joint_delta,
                reference_corner_deficit_delta_inches=corner_delta,stable_neighbor_exposure_delta_sqft=stable_delta,
                interpretation='Unvalidated planning hypotheses; human review and physics remain separate')
