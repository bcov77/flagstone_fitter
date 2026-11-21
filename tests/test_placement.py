import numpy as np
import pytest
from shapely.geometry import Point
from flagstone.phase1 import synthetic, transformed
from flagstone.placement import placement_manifest, save_placement_manifest


def test_manifest_reference_points_and_sequence():
    x = synthetic()
    x.active[2:] = False
    x.poses[0] = [.16, .16, .4]
    x.poses[1] = [.45, .16, 0]
    data = placement_manifest(x, gap=.01, edge=.01, order=[x.ids[1], x.ids[0]], frozen_ids=[x.ids[0]])
    assert data['sequence_kind'] == 'user_supplied_order'
    assert [s['id'] for s in data['stones']] == [x.ids[1], x.ids[0]]
    mark = data['stones'][1]['orientation_mark_world']
    assert transformed(x.shapes[x.ids[0]], x.poses[0]).boundary.distance(Point(mark)) < 1e-12
    assert data['stones'][1]['frozen']
    with pytest.raises(ValueError, match='exactly once'):
        placement_manifest(x, gap=.01, edge=.01, order=[x.ids[0], x.ids[0]])


def test_invalid_manifest_does_not_overwrite(tmp_path):
    x = synthetic()
    x.poses[1] = x.poses[0]
    path = tmp_path / 'placement.json'
    path.write_text('keep')
    with pytest.raises(ValueError, match='invalid layout'):
        save_placement_manifest(x, path, gap=.01, edge=.01)
    assert path.read_text() == 'keep'


def test_manifest_outlines_and_map_are_consistent(tmp_path):
    from flagstone.placement import plot_placement
    x = synthetic()
    data = placement_manifest(x, gap=.01, edge=.01)
    first = data['stones'][0]
    assert len(first['local_outline']) == len(first['world_outline']) == len(first['edge_lengths'])
    assert np.isclose(sum(first['edge_lengths']), first['perimeter'])
    assert np.isclose(first['orientation_reference_length'], np.linalg.norm(first['orientation_mark_local']))
    path = tmp_path/'map.png'
    plot_placement(x, path, gap=.01, edge=.01, frozen_ids=[x.ids[0]])
    assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    x.poses[1] = x.poses[0]
    before = path.read_bytes()
    with pytest.raises(ValueError, match='invalid layout'):
        plot_placement(x, path, gap=.01, edge=.01)
    assert path.read_bytes() == before


def test_orientation_mark_maps_back_to_source_outline(tmp_path):
    from flagstone.phase1 import save_json, load_json, polygon
    x = synthetic()
    path = tmp_path/'layout.json'
    save_json(x, path)
    y = load_json(path)
    data = placement_manifest(y, gap=.01, edge=.01)
    first = data['stones'][0]
    assert polygon(y.source_outlines[first['id']]).boundary.distance(Point(first['orientation_mark_source'])) < 1e-12
