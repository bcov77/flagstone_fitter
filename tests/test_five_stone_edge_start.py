"""Deliberate pockets: front is never an obstacle."""
import importlib.util
from pathlib import Path
from shapely.geometry import box, Polygon

spec=importlib.util.spec_from_file_location('edge_start',Path(__file__).parents[1]/'scripts/five_stone_edge_start.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_closed_pocket():
    stones=[box(1,1,2,1.2),box(1,1.8,2,2),box(1,1,1.2,2),box(1.8,1,2,2)]
    assert module.bounded_gap(box(0,0,4,4),stones,box(1.2,1.2,1.8,1.8))['lower_diameter_in']>2


def test_open_wedge_front_does_not_shrink_circle():
    stones=[Polygon([(0,0),(2,0),(0,1)]),Polygon([(0,2),(2,2),(0,1.5)])]
    result=module.bounded_gap(box(-1,-1,4,4),stones,box(1,.8,1.01,1.2))
    assert result['lower_diameter_in']>2


def test_boundary_gap_and_tight_strip():
    patio=box(0,0,4,4)
    assert module.bounded_gap(patio,[box(0,.4,4,3)],box(1,0,2,.4))['lower_diameter_in']>2
    tight=module.bounded_gap(patio,[box(0,.05,4,3)],box(1,0,2,.05))
    assert tight['passed']
    assert tight['lower_diameter_in'] <= .6+1e-10 <= tight['upper_diameter_in']+1e-10
