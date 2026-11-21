"""Exact incremental trial evaluation for a fixed active set and movable patch.

This caches unchanged original geometry and original distance fields. It only
filters candidates; the complete original validator still checks promotion.
"""
import numpy as np
import shapely
from shapely.geometry import box
from .phase1 import transformed,validate
from .quarter import GAP,world,distances


class PatchTrials:
    def __init__(self,layout,movable,points):
        self.layout=layout;self.movable=list(movable);self.points=points
        self.poses=layout.poses.copy();self.active=layout.active.copy()
        if validate(layout,gap=GAP,edge=GAP):raise ValueError('Patch trial base must be clearance-valid')
        self.original=world(layout)
        fixed=[g for i,g in self.original.items() if i not in self.movable]
        self.fixed=fixed;self.tree=shapely.STRtree(fixed) if fixed else None
        self.fixed_ds=distances(points,fixed,layout.patio.boundary)
        self.nonzero=np.flatnonzero(self.fixed_ds>0)
        self.boundary=shapely.distance(points,layout.patio.boundary)
        self.inset=layout.patio.buffer(-GAP)
        self.coverage=sum(g.area for g in self.original.values())/layout.patio.area

    def _geometries(self,poses):
        return [transformed(self.layout.shapes[self.layout.ids[i]],poses[i]) for i in self.movable]

    def valid(self,poses):
        if poses.shape!=self.poses.shape or not np.isfinite(poses).all():return False
        fixed=np.ones(len(self.active),bool);fixed[self.movable]=False
        if not np.array_equal(poses[fixed],self.poses[fixed]):raise ValueError('Fixed poses changed')
        gs=self._geometries(poses)
        for i,g in enumerate(gs):
            if not self.inset.covers(g) or not self.layout.patio.covers(g) or g.distance(self.layout.patio.boundary)<GAP-1e-9:return False
            if self.tree is not None:
                a,b,c,d=g.bounds
                for j in self.tree.query(box(a-GAP,b-GAP,c+GAP,d+GAP)):
                    if g.distance(self.fixed[int(j)])<GAP-1e-9:return False
            for h in gs[:i]:
                if g.distance(h)<GAP-1e-9:return False
        return True

    def measure(self,poses):
        from .quarter_search2 import metrics_from_field
        ds=self.fixed_ds.copy()
        if len(self.nonzero):
            pp=self.points[self.nonzero];current=ds[self.nonzero].copy()
            for g in self._geometries(poses):current=np.minimum(current,shapely.distance(pp,g))
            ds[self.nonzero]=current
        return metrics_from_field(self.layout,self.points,ds,self.boundary,self.coverage)
