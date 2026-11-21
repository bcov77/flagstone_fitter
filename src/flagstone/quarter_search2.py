"""Independent second-campaign search policies; original validation gates promotion."""
import math
import time
import numpy as np
import shapely
from scipy.optimize import minimize
from .phase1 import transformed, validate
from .quarter import GAP, world, neighborhood, distances, _Deadline


def points_for(layout, spacing=1/12, shifted=False, witnesses=()):
    a,b,c,d=layout.patio.bounds
    offsets=(.25,.75) if shifted else (.5,)
    chunks=[]
    for offset in offsets:
        xx,yy=np.meshgrid(np.arange(a+spacing*offset,c,spacing),np.arange(b+spacing*offset,d,spacing))
        chunks.append(np.column_stack((xx.ravel(),yy.ravel())))
    if witnesses:chunks.append(np.asarray(witnesses))
    p=shapely.points(np.vstack(chunks))
    return p[shapely.covers(layout.patio,p)]


def field(ds, boundary, objective='energy'):
    # A conservative center-based search surrogate, NOT the local-circle audit.
    allowance=np.where(boundary<=ds+1e-9,2.5/12,1/12)
    ratio=ds/allowance
    excess=np.maximum(0,ratio-1)
    if objective=='worst':return float(np.max(excess)**2+.02*np.mean(excess**2))
    return float(np.mean(excess**2)+.1*np.max(excess)**2)


def measure(layout, points):
    polygons=world(layout)
    ds=distances(points,list(polygons.values()),layout.patio.boundary)
    boundary=shapely.distance(points,layout.patio.boundary)
    coverage=sum(g.area for g in polygons.values())/layout.patio.area
    return metrics_from_field(layout,points,ds,boundary,coverage)


def metrics_from_field(layout,points,ds,boundary,coverage):
    allowed=np.where(boundary<=ds+1e-9,2.5/12,1/12)
    ratios=ds/allowed
    worst=int(np.argmax(ratios))
    return dict(sampled_max_diameter_in=float(24*np.max(ds)),sampled_worst_ratio=float(np.max(ratios)),
                sampled_energy=field(ds,boundary),
                sampled_bad_center_area_sqft=float(np.mean(ratios>1)*layout.patio.area),
                largest_center=list(points[worst].coords[0]),coverage=coverage,
                placed=int(layout.active.sum()),unused=int((~layout.active).sum()),sampled_only=True,
                search_surrogate='conservative center-based ratio; final rule uses local gap circles')


def rank(metrics,objective='energy'):
    return ((metrics['sampled_worst_ratio'],metrics['sampled_energy'],-metrics['coverage']) if objective=='worst'
            else (metrics['sampled_energy'],metrics['sampled_worst_ratio'],-metrics['coverage']))


def validate_from_snapshot(layout, snapshot):
    current=layout.snapshot()
    try:
        layout.restore(snapshot);return validate(layout,gap=GAP,edge=GAP)
    finally:layout.restore(current)


def continuous(layout,movable,points,*,seconds=10,maxiter=220,radius_ft=.35,objective='energy'):
    initial=layout.snapshot();before=measure(layout,points);deadline=time.monotonic()+seconds
    if not movable:return dict(accepted=False,reason='empty-neighborhood',before=before,after=before)
    fixed=[g for i,g in world(layout).items() if i not in movable]
    fixed_ds=distances(points,fixed,layout.patio.boundary)
    boundary=shapely.distance(points,layout.patio.boundary)
    shapes=[layout.shapes[layout.ids[i]].simplify(.005,preserve_topology=True) for i in movable]
    fixed_proxy=[g.simplify(.005,preserve_topology=True) for g in fixed]
    origin=layout.poses[movable].ravel().copy();scales=np.tile([radius_ft,radius_ft,.6],len(movable))
    cache={};candidates=[]
    def evaluate(x):
        if time.monotonic()>=deadline:raise _Deadline
        if 'x' in cache and np.array_equal(cache['x'],x):return cache
        poses=(origin+x*scales).reshape(-1,3);gs=[transformed(s,p) for s,p in zip(shapes,poses)]
        ds=fixed_ds.copy()
        for g in gs:ds=np.minimum(ds,shapely.distance(points,g))
        value=field(ds,boundary,objective);cons=[]
        for j,g in enumerate(gs):
            outside=g.difference(layout.patio).area
            cons.append(-math.sqrt(outside)-GAP if outside>0 else g.distance(layout.patio.boundary)-GAP)
            for h in [*fixed_proxy,*gs[:j]]:
                distance=g.distance(h)
                cons.append(distance-GAP if distance>0 else -math.sqrt(max(0,g.intersection(h).area))-GAP)
        if min(cons,default=0)>=-1e-7:
            candidates.append((value,poses.copy()));candidates.sort(key=lambda z:z[0]);del candidates[12:]
        cache.clear();cache.update(x=x.copy(),value=value,constraints=np.asarray(cons)*12)
        return cache
    reason='unknown';termination={}
    try:
        result=minimize(lambda x:evaluate(x)['value'],np.zeros(len(origin)),method='COBYLA',
                        bounds=[(-1,1)]*len(origin),constraints=[dict(type='ineq',fun=lambda x:evaluate(x)['constraints'])],
                        options=dict(maxiter=maxiter,rhobeg=.25,tol=1e-4,catol=1e-6))
        termination=dict(status=int(result.status),message=str(result.message),success=bool(result.success),nfev=int(result.nfev))
        reason='optimizer-returned'
    except _Deadline:reason='time-limit'
    except BaseException:
        layout.restore(initial);raise
    best=before if not validate(layout,gap=GAP,edge=GAP) else None;best_state=initial
    try:
        for _,poses in candidates:
            layout.restore(initial);layout.poses[movable]=poses
            if validate(layout,gap=GAP,edge=GAP):continue
            metrics=measure(layout,points)
            if best is None or rank(metrics,objective)<rank(best,objective):best,best_state=metrics,layout.snapshot()
    finally:layout.restore(best_state)
    return dict(accepted=best is not None and (rank(best,objective)<rank(before,objective) or bool(validate_from_snapshot(layout, initial))),reason=reason,termination=termination,
                before=before,after=best or before,candidate_count=len(candidates),movable_ids=[layout.ids[i] for i in movable])


def replacement(layout,center,points,*,seconds=12,group=1,objective='energy',joint=False):
    """Shape-informed reserve proposals; joint mode repairs overlapping starts."""
    original=layout.snapshot();before=measure(layout,points);best=before;best_state=original
    deadline=time.monotonic()+seconds;attempts=0;invalid_starts=0
    removed=neighborhood(layout,center,count=group)
    if not removed:return dict(accepted=False,attempts=0,before=before,after=before)
    try:
        for _ in range(100000):
            if time.monotonic()>deadline:break
            rng=layout.snapshot()[2];layout.restore(original);layout.rng.bit_generator.state=rng
            candidates=np.flatnonzero(~original[1])
            if len(candidates)<len(removed):break
            chosen=[]
            for old in removed:
                target=layout.shapes[layout.ids[old]]
                tx0,ty0,tx1,ty1=target.bounds
                eligible=[i for i in candidates if i not in chosen]
                def similarity(i):
                    shape=layout.shapes[layout.ids[i]];a,b,c,d=shape.bounds
                    return abs(math.log(shape.area/target.area))+.25*abs(math.log(((c-a)/(d-b))/((tx1-tx0)/(ty1-ty0))))
                eligible.sort(key=similarity)
                chosen.append(int(layout.rng.choice(eligible[:min(24,len(eligible))])))
            for old,new in zip(removed,chosen):
                layout.active[old]=False;layout.active[new]=True
                layout.poses[new]=original[0][old]+layout.rng.normal(0,[.12,.12,.6])
            attempts+=1
            invalid=bool(validate(layout,gap=GAP,edge=GAP));invalid_starts+=invalid
            if joint or not invalid:
                patch=[*chosen,*[i for i in neighborhood(layout,center,5) if i not in chosen]]
                continuous(layout,patch,points,seconds=min(3,max(.01,deadline-time.monotonic())),maxiter=100,radius_ft=.4,objective=objective)
                if not validate(layout,gap=GAP,edge=GAP):
                    metrics=measure(layout,points)
                    if rank(metrics,objective)<rank(best,objective):best,best_state=metrics,layout.snapshot()
    finally:
        rng=layout.snapshot()[2];layout.restore(best_state);layout.rng.bit_generator.state=rng
    return dict(accepted=rank(best,objective)<rank(before,objective),before=before,after=best,
                attempts=attempts,invalid_starts=invalid_starts,group=group)


def repack(layout,center,points,*,seconds=15,count=4,objective='energy'):
    """Several whole-neighborhood greedy repacks, polished before comparison."""
    from .search import greedy_insert
    original=layout.snapshot();before=measure(layout,points);best=before;best_state=original
    deadline=time.monotonic()+seconds;attempts=0
    try:
        while time.monotonic()<deadline:
            rng=layout.snapshot()[2];layout.restore(original);layout.rng.bit_generator.state=rng;removed=neighborhood(layout,center,count=count)
            gs=world(layout);focus=shapely.union_all([gs[i] for i in removed]);a,b,c,d=focus.bounds
            layout.active[removed]=False
            greedy_insert(layout,gap=GAP,edge=GAP,proposals=24,strategy='contact' if attempts%2 else 'random',
                          time_limit=min(5,max(.01,deadline-time.monotonic())),proposal_bounds=(a-.25,b-.25,c+.25,d+.25))
            attempts+=1
            # Restore compatible originals to avoid leaving gratuitous missing stones.
            for i in removed:
                if layout.active[i]:continue
                layout.active[i]=True;layout.poses[i]=original[0][i]
                if validate(layout,gap=GAP,edge=GAP):layout.active[i]=False
            continuous(layout,neighborhood(layout,center,count=min(count+2,12)),points,
                       seconds=min(3,max(.01,deadline-time.monotonic())),maxiter=120,radius_ft=.4,objective=objective)
            if not validate(layout,gap=GAP,edge=GAP):
                metrics=measure(layout,points)
                if rank(metrics,objective)<rank(best,objective):best,best_state=metrics,layout.snapshot()
    finally:
        rng=layout.snapshot()[2];layout.restore(best_state);layout.rng.bit_generator.state=rng
    return dict(accepted=rank(best,objective)<rank(before,objective),before=before,after=best,attempts=attempts,removed_count=count)


def batch_move(layout,center,points,*,backend='cpu',batch_size=16384,objective='energy',scorer=None):
    from .quarter_batch import BatchScorer
    from .quarter_trial import PatchTrials
    started=time.monotonic();original=layout.snapshot()
    patch=neighborhood(layout,center,count=3)
    coordinates=shapely.get_coordinates(points[::max(1,len(points)//512)])
    scorer=scorer or BatchScorer(layout,coordinates,backend)
    key=(layout.poses.tobytes(),layout.active.tobytes(),tuple(patch),shapely.get_coordinates(points).tobytes())
    if scorer.trial_cache is not None and scorer.trial_cache[0]==key:
        _,oracle,before=scorer.trial_cache
    else:
        oracle=PatchTrials(layout,patch,points);before=oracle.measure(layout.poses)
        scorer.trial_cache=(key,oracle,before)
    best=before;best_state=original
    preparation_seconds=time.monotonic()-started
    candidates=np.broadcast_to(layout.poses,(batch_size,*layout.poses.shape)).copy()
    candidates[:,patch,:]+=layout.rng.normal(0,[.045,.045,.12],(batch_size,len(patch),3))
    candidates[0]=layout.poses
    try:
        t0=time.monotonic();scores=scorer.evaluate(candidates,layout.active);scoring_seconds=time.monotonic()-t0
        t0=time.monotonic();valid_shortlist=1
        for index in [int(i) for i in np.argsort(scores) if i!=0][:11]:
            poses=candidates[index]
            if not oracle.valid(poses):continue
            valid_shortlist+=1;metrics=oracle.measure(poses)
            if rank(metrics,objective)<rank(best,objective):
                layout.poses[:]=poses;best,best_state=metrics,layout.snapshot()
        exact_seconds=time.monotonic()-t0
        if rank(best,objective)<rank(before,objective):
            layout.restore(best_state)
            if validate(layout,gap=GAP,edge=GAP):
                best_state=original
                raise RuntimeError('Incremental batch check disagrees with full validator')
    finally:
        rng=layout.snapshot()[2];layout.restore(best_state);layout.rng.bit_generator.state=rng
    return dict(accepted=rank(best,objective)<rank(before,objective),before=before,after=best,
                preparation_seconds=preparation_seconds,scoring_seconds=scoring_seconds,exact_seconds=exact_seconds,
                valid_shortlist=valid_shortlist,gpu_wait_seconds=scorer.last_gpu_wait_seconds,backend=backend,proposed=batch_size,exact_checked=min(12,batch_size))
