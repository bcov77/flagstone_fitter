"""Matched NumPy/Metal approximate proposal scoring; never acceptance geometry."""
import time
from contextlib import contextmanager
import numpy as np
from .phase1 import transformed


def polygon_distances(poses, points, a, b, xp):
    x=points[None,:,0]-poses[:,0,None];y=points[None,:,1]-poses[:,1,None]
    ca,sa=xp.cos(poses[:,2,None]),xp.sin(poses[:,2,None])
    u,v=ca*x+sa*y,-sa*x+ca*y
    dx,dy=b[:,0]-a[:,0],b[:,1]-a[:,1]
    xx,yy=u[:,:,None]-a[:,0],v[:,:,None]-a[:,1]
    t=xp.clip((xx*dx+yy*dy)/(dx*dx+dy*dy),0,1)
    dd=xp.min((xx-t*dx)**2+(yy-t*dy)**2,axis=2)
    denominator=xp.where(xp.abs(dy)>1e-12,dy,1e-12)
    crossing=((a[:,1]>v[:,:,None])!=(b[:,1]>v[:,:,None])) & (u[:,:,None]<a[:,0]+(v[:,:,None]-a[:,1])*dx/denominator)
    inside=xp.sum(crossing.astype(xp.int32),axis=2)%2==1
    return xp.where(inside,0,xp.sqrt(xp.maximum(dd,0)))


class GPUQueueDeadline(Exception):
    pass


class BatchScorer:
    def __init__(self, layout, points, backend='cpu', chunk_size=None):
        if backend not in ('cpu', 'gpu'):
            raise ValueError('Backend must be cpu or gpu')
        self.chunk_size=chunk_size or (256 if backend=='gpu' else 32)
        self.layout = layout
        self.backend = backend
        self.xp = np
        if backend == 'gpu':
            import mlx.core as mx
            if not mx.metal.is_available():
                raise RuntimeError('Metal GPU unavailable')
            self.xp = mx
        xp = self.xp
        fn=lambda poses,points,a,b:polygon_distances(poses,points,a,b,xp)
        self.distance_fn=xp.compile(fn) if backend=='gpu' else fn
        self.contact_cache=None
        self.trial_cache=None
        self.gpu_lock_path=None
        self.queue_deadline=None
        self.last_gpu_wait_seconds=0.
        self.numpy_points=np.asarray(points,dtype=np.float32)
        self.points = xp.array(self.numpy_points)
        self.fixed_cache={}
        self.local_shapes=[]
        self.boundary = xp.array(np.asarray([layout.patio.boundary.distance(__import__('shapely').Point(p))
                                            for p in points], dtype=np.float32))
        self.segments = []
        for sid in layout.ids:
            shape=layout.shapes[sid].simplify(.015,preserve_topology=True)
            self.local_shapes.append(shape)
            coords = np.asarray(shape.exterior.coords,dtype=np.float32)
            self.segments.append((xp.array(coords[:-1]), xp.array(coords[1:])))

    def clearance_penalty(self, poses, active):
        # Linearize nearest original-geometry contacts around the incumbent.
        # This is only a ranking filter; nonlinear clearance remains exact-checked.
        from shapely.ops import nearest_points
        from .quarter import world, GAP
        layout=self.layout
        key=(layout.poses[active].tobytes(),active.tobytes())
        if self.contact_cache is None or self.contact_cache[0]!=key:
            polygons=world(layout);constraints=[]
            for i,g in polygons.items():
                near=[(None,layout.patio.boundary)]
                near += [(j,h) for j,h in polygons.items() if j<i and g.distance(h)<.3]
                for j,h in near:
                    pa,pb=nearest_points(g,h);distance=pa.distance(pb)
                    if distance<1e-10:continue
                    n=np.array([pa.x-pb.x,pa.y-pb.y])/distance
                    lever=np.array([pa.x,pa.y])-layout.poses[i,:2]
                    ga=np.array([n[0],n[1],-n[0]*lever[1]+n[1]*lever[0]])
                    gb=None
                    if j is not None:
                        lever=np.array([pb.x,pb.y])-layout.poses[j,:2]
                        gb=np.array([-n[0],-n[1],n[0]*lever[1]-n[1]*lever[0]])
                    constraints.append((i,j,distance-GAP,ga,gb))
            self.contact_cache=(key,constraints)
        delta=poses-layout.poses;penalty=np.zeros(len(poses))
        for i,j,base,ga,gb in self.contact_cache[1]:
            slack=base+delta[:,i]@ga
            if j is not None:slack+=delta[:,j]@gb
            penalty=np.maximum(penalty,np.maximum(0,-slack*12)**2)
        return 20*penalty

    @contextmanager
    def device_lock(self):
        if self.backend!='gpu' or not self.gpu_lock_path:
            yield;return
        import fcntl
        started=time.monotonic()
        with open(self.gpu_lock_path,'a') as lock:
            while True:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if self.queue_deadline is not None and time.monotonic()>=self.queue_deadline:
                        raise GPUQueueDeadline('GPU queue reached task verification reserve')
                    time.sleep(.01)
            self.last_gpu_wait_seconds=time.monotonic()-started
            yield

    def evaluate(self, poses, active, chunk=None):
        """Score candidates in chunks; caller must exact-check every shortlisted pose."""
        xp = self.xp
        chunk=chunk or self.chunk_size
        results = []
        penalties=self.clearance_penalty(poses, active)
        varying=np.flatnonzero(np.any(poses!=self.layout.poses[None,:,:],axis=(0,2)) & active)
        fixed=[i for i in np.flatnonzero(active) if i not in varying]
        key=(tuple(fixed),self.layout.poses[fixed].tobytes())
        if key not in self.fixed_cache:
            import shapely
            from .quarter import distances
            geometries=[transformed(self.local_shapes[i],self.layout.poses[i]) for i in fixed]
            pp=shapely.points(self.numpy_points);fixed_ds=np.full(len(pp),1e6,dtype=np.float32)
            if geometries:
                tree=shapely.STRtree(geometries);pairs,dd=tree.query_nearest(pp,return_distance=True,all_matches=False)
                fixed_ds[pairs[0]]=dd
            if len(self.fixed_cache)>8:self.fixed_cache.clear()
            self.fixed_cache[key]=xp.array(fixed_ds)
        fixed_ds=self.fixed_cache[key]
        with self.device_lock():
            for first in range(0, len(poses), chunk):
                pp = xp.array(np.asarray(poses[first:first+chunk], dtype=np.float32))
                distances = xp.broadcast_to(self.boundary[None, :], (len(pp), len(self.points)))
                stone_min = xp.broadcast_to(fixed_ds[None,:],distances.shape)
                for i in varying:
                    a, b = self.segments[i]
                    ds=self.distance_fn(pp[:,i,:],self.points,a,b)
                    stone_min = xp.minimum(stone_min, ds)
                distances = xp.minimum(distances, stone_min)
                allowance = xp.where(self.boundary[None, :] <= stone_min+1e-7, 2.5/12, 1/12)
                excess = xp.maximum(0, (distances-allowance)*12)
                score = xp.mean(excess**2, axis=1)+.1*xp.max(excess, axis=1)**2+xp.array(penalties[first:first+len(pp)].astype(np.float32))
                if self.backend == 'gpu':
                    xp.eval(score)
                results.append(np.asarray(score))
        values=np.concatenate(results) if results else np.empty(0)
        if not np.isfinite(values).all():raise ValueError('Nonfinite batch proposal scores')
        return values


def benchmark(layout, points, poses, repeats=3):
    """Materialize timed GPU results; includes scoring and output conversion."""
    result = {}
    scores = {}
    for backend in ('cpu', 'gpu'):
        try:
            t0 = time.perf_counter(); scorer = BatchScorer(layout, points, backend)
            warm = scorer.evaluate(poses[:2], layout.active)
            setup = time.perf_counter()-t0
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter(); values = scorer.evaluate(poses, layout.active)
                times.append(time.perf_counter()-t0)
            scores[backend] = values
            result[backend] = dict(setup_seconds=setup, median_seconds=float(np.median(times)),
                                   candidates_per_second=len(poses)/float(np.median(times)))
        except Exception as exc:
            result[backend] = dict(error=f'{type(exc).__name__}: {exc}')
    if set(scores) == {'cpu', 'gpu'}:
        result.update(speedup=result['cpu']['median_seconds']/result['gpu']['median_seconds'],
                      max_score_difference=float(np.max(np.abs(scores['cpu']-scores['gpu']))),
                      best_candidate_agrees=int(np.argmin(scores['cpu'])) == int(np.argmin(scores['gpu'])))
    return result
