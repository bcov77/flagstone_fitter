"""Repeated region ownership reuses exact buffers only within one fixed board."""
from pathlib import Path
import sys
from unittest.mock import patch

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'analysis/physics_placement_campaign'))
from future_gap_geometry import region_owners, walls


def test_region_cache_preserves_owners_and_avoids_repeated_buffers():
    board = {'a':box(0,0,2,2)}
    patches = [box(1.9,.5,2.1,1.5), box(.5,1.9,1.5,2.1)]
    config = dict(boundary_step_ft=.03, owner_min_length_ft=.1, locked_stones=[])
    wall_lines = walls(box(0,0,5,5))
    expected = [region_owners(p,board,'a',wall_lines,.5/12,{},config) for p in patches]
    cache = {}
    original = BaseGeometry.buffer
    with patch.object(BaseGeometry, 'buffer', autospec=True, side_effect=original) as buffered:
        actual = [region_owners(p,board,'a',wall_lines,.5/12,{},config,cache) for p in patches]
    assert actual == expected
    assert buffered.call_count == 1
