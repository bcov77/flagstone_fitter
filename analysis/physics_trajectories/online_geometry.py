"""V5 assessment with per-board invariant band caching for online use.

Frozen research code is untouched. Geometry formulas/order and bounded repair
semantics are retained; neighbor bands are constructed once rather than per region.
"""
import math
import shapely
from shapely.errors import GEOSException
from shapely.geometry import Polygon
from shapely.ops import unary_union
from future_gap_geometry import walls,residual_regions,region_owners,filler_witness,joint_profile,alignment_options

def planning_overlay(operation,a,b,events,context):
    """Retry failed derived planning overlays; never modify physical outlines.

    Full precision remains the first path. Retry grids are feet, at most
    0.00000012 inches. Every successful fallback is visible in its assessment.
    An unresolved numerical error propagates; it never becomes zero risk.
    """
    function=getattr(shapely,operation)
    try:
        return function(a,b)
    except GEOSException as original:
        # GEOS can assert on a mixed-dimension collection intersected with EMPTY.
        # These set identities are exact and require no coordinate rounding.
        if a.is_empty or b.is_empty:
            if operation=='intersection':result=Polygon()
            elif operation=='difference':result=a
            else:raise
            events.append(dict(operation=operation,context=context,strategy='exact_empty_set_identity',
                grid_size_ft=0.,original_error=str(original),input_valid=[a.is_valid,b.is_valid],
                scope='Exact empty-set identity; no geometry coordinates changed'))
            return result
        failures=[]
        for grid in (1e-10,1e-9,1e-8):
            try:
                result=function(a,b,grid_size=grid)
                if not result.is_valid:raise GEOSException('Invalid overlay retry output')
            except GEOSException as exc:
                failures.append(dict(grid_size_ft=grid,error=str(exc)))
                continue
            events.append(dict(operation=operation,context=context,strategy='precision_grid_retry',grid_size_ft=grid,
                max_coordinate_rounding_inches=grid*12/2,input_valid=[a.is_valid,b.is_valid],
                original_error=str(original),failed_retries=failures,
                scope='Derived planning geometry only; physical outlines and clearance checks unchanged'))
            return result
        raise GEOSException(f'{context}: exhausted bounded precision retries: {original}') from original

def assess(board,subject_id,reference_neighbors,patio,pocket,roi,inventory,stability,gap,config):
    numerical_fallbacks=[]
    def intersection(a,b,context):return planning_overlay('intersection',a,b,numerical_fallbacks,context)
    def difference(a,b,context):return planning_overlay('difference',a,b,numerical_fallbacks,context)
    subject=board.get(subject_id)
    wl=walls(patio)
    free=difference(patio.buffer(-gap),unary_union(list(board.values())).buffer(gap),'clearance_free_space')
    regions=residual_regions(free,roi,config);records=[]
    indent=difference(subject.simplify(.015,preserve_topology=True).convex_hull,subject,'subject_indent') if subject is not None else Polygon()
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
                    intersection(g,patch,'prior_filler_coverage').area/patch.area>=config['filler_min_patch_fraction']),None)
        if prior:
            check=dict(status='covered_by_prior_filler',stone=prior['stone'],pose=prior['pose'])
        else:
            check=filler_witness(patch,core,repair_free,remaining,config) if index<config['max_filler_regions'] else dict(status='not_searched_region_budget')
        fillable=check['status'] in ('verified_filler_pose','covered_by_prior_filler')
        if check['status']=='verified_filler_pose':
            sid=check['stone'];x,y,a=check['pose'];co,si=math.cos(a),math.sin(a)
            g=shapely.affinity.affine_transform(remaining.pop(sid),[co,-si,si,co,x,y])
            item=dict(stone=sid,pose=check['pose']);allocations.append(item);placed.append((item,g))
            repair_free=difference(repair_free,g.buffer(gap),'allocated_filler_exclusion')
            check['scope']='Greedy distinct-piece, mutually clearance-checked endpoint continuation; not a motion sequence'
        totals['verified_fillable_regions']+=int(fillable)
        totals['unsearched_regions']+=int(check['status']=='not_searched_region_budget')
        # These are risk diagnostics when no concrete filler witness exists,
        # not categorical assertions that repair is impossible.
        records.append(dict(id=index,area_sqft=patch.area,core_area_sqft=core.area,
            polygon=shapely.to_wkt(patch,rounding_precision=8),owners=owners,
            artificial_mouth_length_ft=caps,direct_repair_dependencies=flexible,
            no_direct_flexible_boundary=rigid,wall_corner=wall_count>=2,
            subject_indent_sqft=float(intersection(patch,indent,'region_subject_indent').area),filler=check))
    # Score actual localized remaining space, including partially filled regions.
    # Sharing a global connected component with a wall/corner does not assign
    # the whole component's area to that anchor.
    wall_band=unary_union([line.buffer(config['wall_risk_band_inches']/12) for line in wl.values()])
    corner_band=unary_union([shapely.Point(xy).buffer(config['corner_risk_radius_inches']/12) for xy in list(patio.exterior.coords)[:-1]])
    # Exact same geometry as v5, but compute invariant neighbor bands once per board.
    neighbor_order=sorted(reference_neighbors,key=lambda sid:-stability.get(sid,{}).get('score',0.))
    neighbor_bands={}
    for sid in neighbor_order:
        if stability.get(sid,{}).get('score',0.):
            g=board.get(sid,reference_neighbors[sid])
            neighbor_bands[sid]=difference(g.buffer(gap+config['stable_risk_band_inches']/12),g.buffer(gap),'stable_neighbor_band')
    for record,(patch,_) in zip(records,regions):
        unresolved=intersection(patch,repair_free,'unresolved_region')
        radius=config['residual_core_inches']/12
        unresolved=intersection(unresolved.buffer(-radius).buffer(radius),unresolved,'unresolved_excess_core')
        record['remaining_excess_sqft']=unresolved.area
        totals['wall_corner_risk_sqft']+=intersection(unresolved,corner_band,'remaining_corner_risk').area
        totals['wall_excess_sqft']+=intersection(unresolved,wall_band,'remaining_wall_risk').area
        totals['rigid_pocket_risk_sqft']+=unresolved.area if record['no_direct_flexible_boundary'] else 0.
        totals['subject_indent_sqft']+=intersection(unresolved,indent,'remaining_subject_indent').area
        totals['flexible_residual_sqft']+=unresolved.area if record['direct_repair_dependencies'] else 0.
        remaining_region=unresolved
        for sid in neighbor_order:
            score=stability.get(sid,{}).get('score',0.)
            if not score:continue
            overlap=intersection(remaining_region,neighbor_bands[sid],'remaining_stable_exposure')
            totals['stable_neighbor_exposure_sqft']+=score*overlap.area
            remaining_region=difference(remaining_region,overlap,'stable_exposure_allocation')
    joints=joint_profile(subject,{s:g for s,g in board.items() if s!=subject_id},wl,gap,config) if subject is not None else dict(by_owner={},matched_length_ft=0.,longest_run_ft=0.)
    options=alignment_options(subject,reference_neighbors,wl,pocket,config) if subject is not None else {}
    result=dict(metrics=totals,regions=records,joints=joints,alignment_options=options,
                repair_filler_plan=allocations,
                scope='Complete-board free space with fixed target ROI; actual full-size endpoint geometry',
                caveat='No witness is not impossibility. Fillers form a greedy endpoint plan; alignment options are independent conditional possibilities.')
    if numerical_fallbacks:result['numerical_fallbacks']=numerical_fallbacks
    return result

