"""Classify prominent polygon sides by unoriented direction."""
import math
import numpy as np

def near_parallel_long_edge_pairs(shape):
    """Count near-parallel pairs among the four longest qualifying sides.

    A qualifying side is at least half the longest side. Triangles are exempt.
    """
    vertices=np.asarray(shape.exterior.coords[:-1]); edges=np.roll(vertices,-1,axis=0)-vertices
    if len(edges)<=3: return 0
    lengths=np.linalg.norm(edges,axis=1)
    chosen=[i for i in np.argsort(lengths)[::-1] if lengths[i]>=.5*max(lengths)][:4]
    angles=np.arctan2(edges[:,1],edges[:,0])
    return sum(abs((angles[i]-angles[j]+math.pi/2)%math.pi-math.pi/2)<=math.radians(15)
               for k,i in enumerate(chosen) for j in chosen[k+1:])
