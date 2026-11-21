from shapely.affinity import rotate
from flagstone.edge_analysis import near_parallel_long_edge_pairs
from flagstone.phase1 import polygon

def test_pair_count_rotation_invariance_and_triangle_exemption():
    square=polygon([(0,0),(2,0),(2,2),(0,2)])
    assert near_parallel_long_edge_pairs(square)==2
    assert near_parallel_long_edge_pairs(rotate(square,37))==2
    assert near_parallel_long_edge_pairs(polygon([(0,0),(3,0),(1,1)]))==0

def test_one_near_parallel_pair_and_short_sides():
    trapezoid=polygon([(0,0),(3,0),(2.7,1),(0.2,1)])
    assert near_parallel_long_edge_pairs(trapezoid)==1
    # The little bevels are below half the longest side and do not enter.
    bevel=polygon([(0,0),(3,0),(3,1),(.2,1),(0,.9)])
    assert near_parallel_long_edge_pairs(bevel)==1
