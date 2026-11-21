import numpy as np
from flagstone.phase1 import Layout, local_shape, polygon, synthetic, transformed, validate, save_json, load_json


def test_transform_preserves_geometry():
    s = local_shape([(0, 0), (2, 0), (1, 1)])
    t = transformed(s, (4, 3, 0.7))
    assert np.isclose(s.area, t.area)
    a, b = np.array(s.exterior.coords), np.array(t.exterior.coords)
    assert np.allclose(np.linalg.norm(a[:, None] - a[None, :], axis=2), np.linalg.norm(b[:, None] - b[None, :], axis=2))


def test_validation_and_rollback():
    x = synthetic()
    assert not validate(x)
    snap = x.snapshot()
    expected = x.rng.random()
    x.poses[1] = x.poses[0]
    x.active[2] = False
    assert any(v['kind'] == 'pair_clearance' for v in validate(x, gap=0))
    x.restore(snap)
    assert x.active[2] and x.rng.random() == expected
    assert not validate(x)


def test_concave_boundary_and_json(tmp_path):
    patio = polygon([(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)])
    shape = local_shape([(0, 0), (2, 0), (2, .3), (0, .3)])
    x = Layout(patio, {'a': shape}, np.array([[1.5, 1.5, 0.]]), np.array([True]), ('a',), np.random.default_rng(1))
    assert any(v['kind'] == 'boundary_clearance' for v in validate(x, edge=0))
    path = tmp_path / 'layout.json'
    save_json(x, path)
    y = load_json(path)
    assert transformed(y.shapes['a'], y.poses[0]).equals_exact(transformed(shape, x.poses[0]), 1e-12)


def test_crossing_containment_gap_and_boundary_measurements():
    patio = polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    a = local_shape([(-1, -.2), (1, -.2), (1, .2), (-1, .2)])
    b = local_shape([(-.2, -1), (.2, -1), (.2, 1), (-.2, 1)])
    x = Layout(patio, {'a': a, 'b': b}, [[2, 2, 0], [2, 2, 0]], [True, True], ('a', 'b'), np.random.default_rng(0))
    assert any(v['kind'] == 'pair_clearance' for v in validate(x, gap=0, edge=0))
    x.poses[1] = [2, 2, np.pi / 2]  # complete containment
    assert any(v['kind'] == 'pair_clearance' for v in validate(x, gap=0, edge=0))
    x.poses[1] = [2, 2.41, np.pi / 2]
    assert any(v['kind'] == 'pair_clearance' for v in validate(x, gap=.02, edge=0))
    assert not validate(x, gap=.005, edge=0)
    x.active[1] = False
    x.poses[0] = [1.05, 2, 0]
    v = validate(x, gap=0, edge=.1)[0]
    assert np.isclose(v['measurement'], .05) and v['outside_area'] == 0


def test_layout_owns_arrays_and_inventory_mapping():
    import pytest
    x = synthetic()
    with pytest.raises(TypeError):
        x.shapes['new'] = x.shapes[x.ids[0]]
    snapshot = x.snapshot()
    x.poses[0, 0] += 1
    assert not np.array_equal(snapshot[0], x.poses)
    x.restore(snapshot)
    x.poses[0, 0] += 1
    assert not np.array_equal(snapshot[0], x.poses)


def test_nonfinite_active_pose_is_rejected():
    import pytest
    x = synthetic()
    x.poses[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        validate(x)
    x.active[0] = False
    assert not validate(x)


def test_off_center_import_units_and_source_outline_round_trip(tmp_path):
    import json
    from shapely.affinity import rotate, translate
    original = [[10, 20], [12, 20], [12, 21], [10, 21]]
    data = {'schema_version': 1, 'units': 'ft', 'patio': [[0,0],[100,0],[100,100],[0,100]],
            'stones': [{'id': 'A', 'outline': original, 'pose': [30, 10, .7], 'active': True}]}
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(data))
    x = load_json(path)
    expected = translate(rotate(polygon(original), .7, origin=(0, 0), use_radians=True), 30, 10)
    assert transformed(x.shapes['A'], x.poses[0]).equals_exact(expected, 1e-12)
    assert x.units == 'ft' and x.source_outlines['A'] == tuple(map(tuple, original))
    save_json(x, path)
    y = load_json(path)
    assert y.units == 'ft' and y.source_outlines['A'] == x.source_outlines['A']
    assert transformed(y.shapes['A'], y.poses[0]).equals_exact(expected, 1e-12)


def test_reject_nonfinite_3d_and_nested_outlines():
    import pytest
    for coords in ([[0, 0], [1, 0], [0, float('nan')]],
                   [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                   [[[0,0],[2,0],[0,2]], [[.1,.1],[.2,.1],[.1,.2]]]):
        with pytest.raises(ValueError, match='Outline'):
            polygon(coords)


def test_reject_explicit_hole_fields(tmp_path):
    import json
    import pytest
    data = {'schema_version': 1, 'units': 'normalized', 'patio': [[0,0],[1,0],[1,1],[0,1]],
            'stones': [{'id':'a','outline':[[0,0],[.1,0],[0,.1]],'pose':[.5,.5,0],'active':True,'holes': []}]}
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='holes'):
        load_json(path)


def test_spatial_filter_matches_exhaustive_pairs():
    from flagstone.generate import inventory
    for seed in range(5):
        x = inventory(seed=seed, counts=(25,), areas=(.008,))
        x.active[:] = True
        x.poses[:, :2] = x.rng.uniform(.1, .9, (25, 2))
        x.poses[:, 2] = x.rng.uniform(-np.pi, np.pi, 25)
        x.poses[1] = x.poses[0]  # include complete containment / overlap
        placed = [transformed(x.shapes[sid], x.poses[i]) for i, sid in enumerate(x.ids)]
        for gap in (0, .005, .1):
            expected = {(x.ids[i], x.ids[j]) for i, a in enumerate(placed) for j, b in enumerate(placed)
                        if j > i and (a.intersection(b).area > 0 or a.distance(b) < gap - 1e-9)}
            found = {tuple(v['stone_ids']) for v in validate(x, gap=gap, edge=0) if v['kind'] == 'pair_clearance'}
            assert found == expected


def test_source_outline_must_match_local_stone(tmp_path):
    import json
    import pytest
    x = synthetic()
    path = tmp_path / 'layout.json'
    save_json(x, path)
    data = json.loads(path.read_text())
    data['stones'][0]['source_outline'] = [[0,0], [1,0], [1,1], [0,1]]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='source outline does not match'):
        load_json(path)


def test_combined_transform_matches_rotate_then_translate():
    from shapely.affinity import rotate, translate
    shape = polygon([(-.3, -.2), (.2, -.25), (.4, .05), (.05, .1), (-.1, .35)])
    rng = np.random.default_rng(104)
    poses = [[0, 0, 0], [1, -2, np.pi / 2], [-3, 5, np.pi]]
    poses.extend(rng.uniform([-10, -10, -20], [10, 10, 20], size=(100, 3)))
    for x, y, angle in poses:
        old = translate(rotate(shape, angle, origin=(0, 0), use_radians=True), x, y)
        assert transformed(shape, (x, y, angle)).equals_exact(old, 2e-14)


def test_layout_rejects_ids_that_cannot_round_trip_as_json_keys(tmp_path):
    import json
    import pytest
    shape = local_shape([[0,0],[.1,0],[0,.1]])
    patio = polygon([[0,0],[1,0],[1,1],[0,1]])
    for sid in (1, True, '', '  '):
        with pytest.raises(ValueError, match='nonempty strings'):
            Layout(patio, {sid:shape}, [[.2,.2,0]], [True], (sid,), np.random.default_rng(0))
    path = tmp_path / 'invalid.json'
    for sid in (1, True, '', '  ', None, ['not', 'hashable']):
        data = {'schema_version':1, 'units':'normalized', 'patio':[[0,0],[1,0],[1,1],[0,1]],
                'stones':[{'id':sid, 'outline':[[0,0],[.1,0],[0,.1]], 'pose':[.2,.2,0], 'active':True}]}
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match='nonempty strings'):
            load_json(path)


def test_direct_layout_rejects_off_center_stone():
    import pytest
    patio = polygon([(0,0),(1,0),(1,1),(0,1)])
    off_center = polygon([(0,0),(.2,0),(.2,.1),(0,.1)])
    with pytest.raises(ValueError, match='centroid-local'):
        Layout(patio, {'A':off_center}, [[.4,.4,0]], [True], ('A',), np.random.default_rng(0))
    slender = polygon([(-5e11,.49),(5e11,.49),(5e11,.51),(-5e11,.51)])
    with pytest.raises(ValueError, match='centroid-local'):
        Layout(patio, {'A':slender}, [[.4,.4,0]], [True], ('A',), np.random.default_rng(0))


def test_tiny_off_center_import_preserves_world_shape_and_centroid(tmp_path):
    import json
    from shapely.affinity import rotate, translate
    outline = [[1e-16,0],[3e-16,0],[3e-16,1e-16],[1e-16,1e-16]]
    pose = [1e-15,1e-15,.3]
    raw = {'schema_version':1, 'units':'normalized',
           'patio':[[0,0],[1e-14,0],[1e-14,1e-14],[0,1e-14]],
           'stones':[{'id':'A','outline':outline,'pose':pose,'active':True}]}
    path = tmp_path / 'tiny.json'; path.write_text(json.dumps(raw))
    expected = translate(rotate(polygon(outline), pose[2], origin=(0,0), use_radians=True), pose[0], pose[1])
    x = load_json(path)
    assert transformed(x.shapes['A'], x.poses[0]).equals_exact(expected, 1e-29)
    assert np.allclose(x.poses[0,:2], [expected.centroid.x,expected.centroid.y], rtol=0, atol=1e-29)
    original_pose = x.poses.copy()
    for _ in range(5):
        save_json(x, path); x = load_json(path)
    assert np.array_equal(x.poses, original_pose)
    assert transformed(x.shapes['A'], x.poses[0]).equals_exact(expected, 1e-29)
