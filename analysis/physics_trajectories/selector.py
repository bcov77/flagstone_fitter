"""Bounded online proposals and explicit stochastic choices on the CURRENT board.

Geometry prerequisites -> durable-risk tiers -> joint continuity -> area.
All thresholds and weights are experimental, not learned human-approval scores.
"""
import copy, math
import numpy as np
import shapely
from shapely.ops import unary_union
from scipy import signal, optimize
from common import GAP, world, shapes
from future_gap_geometry import compare, reference_stability, walls, joint_profile
from online_geometry import assess
from selection_geometry import boundary_features

# Deliberately independent of the frozen v5 runner and its source-board globals.
PLAN_CONFIG=dict(boundary_step_ft=.03,joint_excess_inches=.5,joint_angle_degrees=20,
 residual_core_inches=.25,min_core_sqft=.0005,owner_min_length_ft=.04,locked_stones=[],protected_stones=[],
 filler_search_pad_ft=.25,filler_raster_step_ft=.04,filler_max_pieces=3,filler_angle_step_degrees=30,
 filler_positions_per_angle=1,filler_min_patch_fraction=.35,max_filler_regions=2,
 alignment_anchor_distance_ft=.25,alignment_min_edge_ft=.25,alignment_face_tolerance_inches=.125,
 alignment_angle_span_degrees=20,risk_area_indifference_sqft=.005,alignment_run_indifference_ft=.1,
 corner_reach_indifference_inches=.1,wall_risk_band_inches=2.,corner_risk_radius_inches=3.,stable_risk_band_inches=2.)

def parts(g):return [p for p in getattr(g,'geoms',[g]) if p.geom_type=='Polygon' and not p.is_empty]
def pockets(patio,board):
    free=patio.buffer(-GAP).difference(unary_union(list(board.values())).buffer(GAP))
    out=[]
    for core in parts(free.buffer(-1.25/12)):
        if core.area<.012:continue
        ps=parts(core.buffer(1.25/12).intersection(free))
        if ps:
            p=max(ps,key=lambda p:p.intersection(core).area)
            if p.area>.08:out.append(p)
    return free,sorted(out,key=lambda p:-p.area)

def target_pool(document,local,rng,cfg,stability):
    board=shapes(document);patio=shapely.Polygon(document['patio']);targets=[]
    free,ps=pockets(patio,board)
    for i,p in enumerate(ps[:8]):targets.append(dict(id=f'insert_{i}',old=None,pocket=p,free=free,weight=2*math.sqrt(p.area)))
    locked={sid for sid,v in document.get('physics',{}).get('pieces',{}).items() if v.get('locked')}
    available=[sid for sid in board if sid not in locked]
    # Sample changeable stones more often, but keep a route to revisiting stable ones.
    weights=np.array([.15+1-stability[s]['score'] for s in available]);weights/=weights.sum() if len(weights) else 1
    count=min(len(available),max(2,cfg['target_count']))
    olds=list(rng.choice(available,count,replace=False,p=weights)) if count else []
    # Cover both a natural insertion and replacement when both exist.
    for sid in olds:
        others={s:g for s,g in board.items() if s!=sid};free,ps=pockets(patio,others)
        ps=[p for p in ps if p.intersection(board[sid]).area>board[sid].area*.35]
        if ps:
            p=max(ps,key=lambda p:p.intersection(board[sid]).area)
            targets.append(dict(id=f'replace_{sid}',old=sid,pocket=p,free=free,weight=.3+1-stability[sid]['score']))
    insertion=[t for t in targets if not t['old']];replacement=[t for t in targets if t['old']]
    selected=[]
    if insertion:
        w=np.array([t['weight'] for t in insertion]);selected.append(insertion[int(rng.choice(len(insertion),p=w/w.sum()))])
    if replacement:selected.append(replacement[0])
    remaining=[t for t in targets if all(t['id']!=s['id'] for s in selected)]
    while remaining and len(selected)<cfg['target_count']:
        w=np.array([t['weight'] for t in remaining]);selected.append(remaining.pop(int(rng.choice(len(remaining),p=w/w.sum()))))
    return selected

def coarse(local,available,pocket,free,cfg,rng,budget=None,min_fraction=.27):
    """FFT proposes positions; original outlines perform every subsequent check."""
    step=cfg['raster_step_ft'];x0,y0,x1,y1=pocket.bounds;pad=.12
    xs=np.arange(x0-pad,x1+pad+step,step);ys=np.arange(y0-pad,y1+pad+step,step)
    pm=shapely.contains_xy(pocket,xs[None,:],ys[:,None]);fm=shapely.contains_xy(free,xs[None,:],ys[:,None])
    costs=np.where(pm,-1.,1.)+np.where(fm,0.,22.)
    ids=[s for s in available if min_fraction*pocket.area<=local[s].area<=1.15*pocket.area]
    # Mostly closest area; some inventory exploration before pose scoring.
    ids=sorted(ids,key=lambda s:abs(math.log(local[s].area/(pocket.area*.85))))
    budget=budget or cfg['piece_budget'];n=min(len(ids),budget)
    head=ids[:max(1,n*2//3)];rest=ids[len(head):]
    if len(head)<n:head+=list(rng.choice(rest,n-len(head),replace=False))
    ranked=[]
    for sid in head:
        g=local[sid].simplify(.004,preserve_topology=True)
        candidates=[]
        for deg in range(0,360,cfg['angle_step_degrees']):
            a=math.radians(deg);rg=world(g,[0,0,a]);a0,b0,a1,b1=rg.bounds
            xx=np.arange(a0+step/2,a1,step);yy=np.arange(b0+step/2,b1,step)
            if not len(xx) or not len(yy) or len(xx)>len(xs) or len(yy)>len(ys):continue
            kernel=shapely.contains_xy(rg,xx[None,:],yy[:,None]).astype(float)
            conv=signal.fftconvolve(costs,kernel[::-1,::-1],mode='valid')*step**2
            iy,ix=np.unravel_index(np.argmin(conv),conv.shape)
            candidates.append((float(conv[iy,ix]),sid,[float(xs[ix]-xx[0]),float(ys[iy]-yy[0]),a]))
        ranked.extend(sorted(candidates)[:2])
    return sorted(ranked)[:cfg['coarse_per_target']]

def fit(placements,target,local,board,patio,cfg,min_coverage=None):
    gs=[world(local[p['stone']],p['pose']) for p in placements];g=unary_union(gs);p=target['pocket'];free=target['free']
    if any(not patio.buffer(-GAP).buffer(1e-8).covers(h) for h in gs):return None
    # Selected pieces are held in place together during exploration: they must not clash.
    if any(a.distance(b)<GAP-1e-8 for i,a in enumerate(gs) for b in gs[i+1:]):return None
    coverage=g.intersection(p).area/p.area;overhang=g.difference(p).area/g.area;intr=g.difference(free).area/g.area
    points=shapely.points(shapely.get_coordinates(shapely.segmentize(g.boundary,.02)))
    depth=12*float(np.max(shapely.distance(points,free)))
    affected=[sid for sid,h in board.items() if sid!=target['old'] and g.distance(h)<GAP]
    if coverage<(min_coverage if min_coverage is not None else cfg['min_pocket_coverage']) or overhang>.12 or intr>cfg['max_intrusion_fraction'] or depth>cfg['max_intrusion_inches'] or len(affected)>4:return None
    # Same opening and physical ownership mask for every alternative.
    corners=target['pocket'].simplify(.025,preserve_topology=True).exterior
    pts=shapely.points(shapely.get_coordinates(corners))
    physical=shapely.distance(pts,free.boundary)<.03
    deficit=float(np.quantile(shapely.distance(pts[physical],g),.9))*12 if np.any(physical) else None
    return dict(coverage=coverage,outside_pocket_fraction=overhang,clearance_intrusion_fraction=intr,
        max_intrusion_inches=depth,affected_neighbors=affected,area_sqft=g.area,corner_deficit_inches=deficit)

def apply(document,proposal):
    d=copy.deepcopy(document);remove=proposal['remove'];placed={p['stone']:p['pose'] for p in proposal['placements']}
    for s in d['stones']:
        sid=str(s['id'])
        if sid in remove:s['active']=False
        if sid in placed:s.update(active=True,pose=placed[sid][:])
    return d

def generate(document,cfg,rng,excluded=()):
    board=shapes(document);patio=shapely.Polygon(document['patio']);local={str(s['id']):shapely.Polygon(s['outline']) for s in document['stones']}
    pc=copy.deepcopy(PLAN_CONFIG);pc.update(max_filler_regions=cfg['planning_filler_regions'],filler_max_pieces=cfg['planning_filler_pieces'],filler_angle_step_degrees=cfg['planning_filler_angle_degrees'])
    pc['locked_stones']=[sid for sid,v in document.get('physics',{}).get('pieces',{}).items() if v.get('locked') and sid in board]
    stability=reference_stability(board,patio,GAP,pc);targets=target_pool(document,local,rng,cfg,stability);raw=[];target_log=[]
    for t in targets:
        available=[s for s in local if (s not in board or s==t['old']) and s not in excluded]
        ranked=coarse(local,available,t['pocket'],t['free'],cfg,rng)
        if t['old'] and t['old'] in available:
            original_coarse=coarse(local,[t['old']],t['pocket'],t['free'],cfg,rng)
            ranked+= [v for v in original_coarse if v not in ranked]

        singles=[]
        for i,(_,sid,pose) in enumerate(ranked):
            if i<cfg['refine_per_target']:
                def objective(q):
                    g=world(local[sid].simplify(.004,preserve_topology=True),q)
                    return -g.intersection(t['pocket']).area+g.difference(t['pocket']).area+30*g.difference(t['free']).area
                bounds=[(pose[0]-.09,pose[0]+.09),(pose[1]-.09,pose[1]+.09),(pose[2]-.10,pose[2]+.10)]
                pose=optimize.minimize(objective,pose,method='Powell',bounds=bounds,options={'maxiter':12,'xtol':.001,'ftol':.001}).x.tolist()
            pl=[dict(stone=sid,pose=pose)];f=fit(pl,t,local,board,patio,cfg,min_coverage=.27)
            if f:
                proposal=dict(target=t['id'],remove=[t['old']] if t['old'] else [],placements=pl,fit=f)
                singles.append(proposal)
                if f['coverage']>=cfg['min_pocket_coverage']:raw.append((proposal,t))
        # Two actual distinct pieces fill one opening, then undergo one shared relaxation.
        for first in singles[:cfg['double_seed_count']]:
            fg=world(local[first['placements'][0]['stone']],first['placements'][0]['pose'])
            residual=t['pocket'].difference(fg.buffer(GAP));chunks=sorted(parts(residual.buffer(-.06).buffer(.06).intersection(residual)),key=lambda p:-p.area)
            if not chunks or chunks[0].area<.08:continue
            p2=chunks[0];free2=t['free'].difference(fg.buffer(GAP))
            available2=[s for s in available if s!=first['placements'][0]['stone']]
            for _,sid,pose in coarse(local,available2,p2,free2,cfg,rng,budget=cfg['double_piece_budget'],min_fraction=.35)[:2]:
                pl=first['placements']+[dict(stone=sid,pose=pose)];f=fit(pl,t,local,board,patio,cfg)
                if f:raw.append((dict(target=t['id'],remove=first['remove'],placements=pl,fit=f),t))
        target_log.append(dict(id=t['id'],old=t['old'],pocket_wkt=shapely.to_wkt(t['pocket'],rounding_precision=7),weight=t['weight'],coarse_poses=len(ranked)))
    # Shortlist a few per action, rather than let numerous poses of one piece dominate.
    buckets={}
    for p,t in raw:
        key=(p['target'],tuple(sorted(x['stone'] for x in p['placements'])))
        buckets.setdefault(key,[]).append((p,t))
    raw=[min(v,key=lambda pt:pt[0]['fit']['clearance_intrusion_fraction']-.2*pt[0]['fit']['coverage']) for v in buckets.values()]
    raw.sort(key=lambda pt:(pt[0]['fit']['clearance_intrusion_fraction'],-pt[0]['fit']['coverage']))
    shortlist=raw[:cfg['planning_pool']]
    # Include the best searched original-pose option for each target as an explicit comparator.
    for tid in {t['id'] for p,t in shortlist}:
        refs=[pt for pt in raw if pt[1]['id']==tid and pt[1]['old'] and len(pt[0]['placements'])==1 and pt[0]['placements'][0]['stone']==pt[1]['old']]
        if refs and not any(pt[1]['id']==tid and pt[0]['placements']==refs[0][0]['placements'] for pt in shortlist):shortlist.append(refs[0])
    # Preserve at least one double in the evaluated pool when search found any.
    doubles=[pt for pt in raw if len(pt[0]['placements'])==2]
    if doubles and not any(len(p['placements'])==2 for p,t in shortlist):shortlist=shortlist[:-1]+[doubles[0]]
    baselines={};pool=[]
    for proposal,t in shortlist:
        others={s:g for s,g in board.items() if s!=t['old']};incident={s:g for s,g in board.items() if g.intersects(t['pocket'].buffer(.5))}
        # Full outlines of incident neighbors include the far side, not just the pocket.
        roi=unary_union([t['pocket'].buffer(.3),*[g.buffer(.375) for g in incident.values()]]).intersection(patio)
        if t['id'] not in baselines:
            inv={s:g for s,g in local.items() if s not in board}
            ba=assess(board,t['old'],others,patio,t['pocket'],roi,inv,stability,GAP,pc)
            if t['old']:
                original_fit=fit([dict(stone=t['old'],pose=next(s['pose'] for s in document['stones'] if str(s['id'])==t['old']))],t,local,board,patio,cfg,min_coverage=0)
                ba['reference_corner_deficit_inches']=original_fit['corner_deficit_inches'] if original_fit else None
            baselines[t['id']]=ba
        planned=shapes(apply(document,proposal));inv={s:g for s,g in local.items() if s not in planned}
        aa=assess(planned,proposal['placements'][0]['stone'],others,patio,t['pocket'],roi,inv,stability,GAP,pc)
        aa['reference_corner_deficit_inches']=proposal['fit']['corner_deficit_inches']
        # A double's indent exposure and alignment losses must include its second piece.
        profiles=[joint_profile(planned[x['stone']],{s:g for s,g in planned.items() if s!=x['stone']},walls(patio),GAP,pc) for x in proposal['placements']]
        if len(proposal['placements'])==2:
            second=proposal['placements'][1]['stone'];bb=assess(planned,second,others,patio,t['pocket'],roi,{},stability,GAP,{**pc,'max_filler_regions':0})
            aa['metrics']['subject_indent_sqft']+=bb['metrics']['subject_indent_sqft']
            # Keep the best preserved boundary option across the two pieces.
            for owner,option in bb['alignment_options'].items():
                if option['length_ft']>aa['alignment_options'].get(owner,{}).get('length_ft',0):aa['alignment_options'][owner]=option
        cmp=compare(aa,baselines[t['id']],pc)
        runs=[(owner,v['longest_run_ft']) for prof in profiles for owner,v in prof['by_owner'].items() if v['longest_run_ft']>=.25]
        hard=sum(length for owner,length in runs if owner.startswith('wall_') or owner in pc['locked_stones'])
        durable=sum(length*stability.get(owner,{}).get('score',0) for owner,length in runs if not owner.startswith('wall_'))
        # Joint lengths between the proposed pair are counted once for ranking.
        selected_ids={x['stone'] for x in proposal['placements']}
        total=sum(length*(.5 if owner in selected_ids else 1) for owner,length in runs)
        status=cmp['status'];tier=0 if status=='geometric_planning_advantage_review' else 1 if status=='unclear_retain_or_review' else 2 if status in ('future_alignment_tradeoff_review','corner_reach_tradeoff_review','provisional_anchor_tradeoff_review') else 3
        gain=sum(local[x['stone']].area for x in proposal['placements'])-sum(local[s].area for s in proposal['remove'])
        proposal.update(comparison=cmp,tier=tier,finished_connections=dict(hard_length_inches=hard*12,stable_weighted_inches=durable*12,total_length_inches=total*12,runs=[dict(owner=o,length_inches=l*12) for o,l in runs]),area_gain_sqft=gain,
            assessment=aa,baseline_assessment=baselines[t['id']],rank_key=[tier,-hard,-durable,-total,proposal['fit']['clearance_intrusion_fraction'],-gain],
            reasons=[f"{status}: compare to current original/empty opening",f"{hard*12:.1f} inches of contiguous hard-anchor joint runs; {durable*12:.1f} stability-weighted neighbor inches",f"new risk axes: {', '.join(cmp['new_risk_axes']) or 'none'}",f"lost conditional flat-edge options: {', '.join(cmp['lost_alignment_options']) or 'none'}",f"{gain:+.3f} ft² stone area; area only breaks ties after risk and connections",'Repair witnesses are bounded endpoint probes; no witness does not imply impossible repair'])
        # Do not spend relaxation attempts on exactly retaining an original pose.
        if len(proposal['placements'])==1 and proposal['remove']==[proposal['placements'][0]['stone']]:
            oldpose=next(s['pose'] for s in document['stones'] if str(s['id'])==proposal['remove'][0]);pose=proposal['placements'][0]['pose']
            if math.dist(oldpose[:2],pose[:2])<.01 and abs(math.atan2(math.sin(oldpose[2]-pose[2]),math.cos(oldpose[2]-pose[2])))<.02:continue
        pool.append(proposal)
    # Every replacement is also compared to the available searched same-stone alternatives.
    # These comparisons are evaluated before final tier/rank selection.
    for p in pool:
        refs=[q for q in pool if q['target']==p['target'] and q['remove'] and len(q['placements'])==1 and q['placements'][0]['stone']==q['remove'][0]]
        p['original_pose_comparisons']=[dict(placements=q['placements'],**compare(p['assessment'],q['assessment'],pc)) for q in refs if q is not p]
        flags=[c for c in p['original_pose_comparisons'] if c['status'] not in ('geometric_planning_advantage_review','unclear_retain_or_review')]
        if flags:
            p['tier']=max(p['tier'],2);p['rank_key'][0]=p['tier'];p['reasons'].append(f'{len(flags)} cautions against searched original-stone poses')
    pool.sort(key=lambda p:p['rank_key'])
    return pool,dict(targets=target_log,stability=stability,planning_config=pc,raw_distinct_choices=len(raw),evaluated_choices=len(pool))

def probabilities(pool,mode,cfg):
    """Explicit marginal probabilities; sample distinct opening/piece actions."""
    n=len(pool)
    if not n:return np.array([]),[]
    best_tier=min(p['tier'] for p in pool);eligible=[i for i,p in enumerate(pool) if p['tier']==best_tier]
    best=eligible[:max(1,min(len(eligible),max(3,math.ceil(n*cfg['best_fraction']))))]
    bp=np.zeros(n);bp[best]=1/len(best)
    ranked=np.exp(-np.arange(n)/max(2,n/3));ranked/=ranked.sum()
    normal=cfg['normal_best_probability']*bp+cfg['normal_ranked_probability']*ranked+(1-cfg['normal_best_probability']-cfg['normal_ranked_probability'])/n
    if mode=='golden':return bp,best
    if mode=='silver':return cfg['silver_best_probability']*bp+(1-cfg['silver_best_probability'])*normal,best
    return normal,best

def choose(pool,mode,cfg,rng):
    probs,best=probabilities(pool,mode,cfg)
    if not len(pool):return None,dict(mode=mode,reason='No pose passed the bounded initial shape/interference prerequisites')
    index=int(rng.choice(len(pool),p=probs))
    return pool[index],dict(mode=mode,index=index,probability=float(probs[index]),best_indices=best,
        best_tier=min(p['tier'] for p in pool),probabilities=probs.tolist(),candidate_count=len(pool),
        note='Probabilities conditional on this sampled target/inventory/pose pool, not on all possible placements')
