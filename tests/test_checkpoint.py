import numpy as np
from flagstone.checkpoint import save_checkpoint, load_checkpoint
from flagstone.generate import inventory


def test_checkpoint_restores_state_and_rng(tmp_path):
    a = inventory(seed=3, counts=(2,), areas=(.01,))
    a.active[0] = True
    a.poses[0] = [0.4, 0.5, 0.7]
    path = tmp_path / 'checkpoint.json'
    save_checkpoint(a, path, {'gap': .01})
    b, metadata = load_checkpoint(path)
    assert metadata['gap'] == .01
    assert np.allclose(a.poses, b.poses)
    assert np.array_equal(a.active, b.active)
    assert a.rng.random() == b.rng.random()


def test_integrity_detects_altered_state_and_geometry(tmp_path):
    import json
    import pytest
    x = inventory(seed=4, counts=(2,), areas=(.01,))
    path = tmp_path / 'checkpoint.json'
    save_checkpoint(x, path, {'run_id':'example', 'gap':.01})
    original = json.loads(path.read_text())
    assert original['checkpoint_version'] == 2
    assert original['run_id'] == 'example' and original['software']['numpy']
    for modify in ('pose', 'geometry', 'rng', 'metadata'):
        data = json.loads(json.dumps(original))
        if modify == 'pose': data['stones'][0]['pose'][0] += 1
        elif modify == 'geometry': data['stones'][0]['outline'][0][0] += .1
        elif modify == 'rng': data['rng_state']['state']['state'] += 1
        else: data['metadata']['gap'] = .2
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match='integrity'):
            load_checkpoint(path)


def test_legacy_v1_checkpoint_remains_readable(tmp_path):
    import json
    from flagstone.phase1 import save_json
    x = inventory(seed=9, counts=(2,), areas=(.01,))
    path = tmp_path / 'legacy.json'
    save_json(x, path)
    data = json.loads(path.read_text())
    data.update(checkpoint_version=1, rng_state=x.rng.bit_generator.state, metadata={'gap':.01})
    path.write_text(json.dumps(data))
    restored, metadata = load_checkpoint(path)
    assert metadata['gap'] == .01 and np.allclose(restored.poses, x.poses)
    assert restored.rng.random() == x.rng.random()


def test_failed_write_keeps_previous_checkpoint(tmp_path, monkeypatch):
    import pytest
    import flagstone.checkpoint as checkpoint
    x = inventory(seed=1, counts=(1,), areas=(.01,))
    path = tmp_path / 'checkpoint.json'
    save_checkpoint(x, path)
    original = path.read_bytes()
    def fail(_):
        raise OSError('simulated failure')
    monkeypatch.setattr(checkpoint, '_digest', fail)
    with pytest.raises(OSError):
        save_checkpoint(x, path)
    assert path.read_bytes() == original


def test_geometry_hash_is_stable_across_pose_changes_and_resume(tmp_path):
    import json
    x = inventory(seed=2, counts=(2,), areas=(.01,))
    path = tmp_path / 'a.json'
    save_checkpoint(x, path)
    a = json.loads(path.read_text())
    x.poses[0] = [.2, .3, .4]
    x.active[0] = True
    save_checkpoint(x, path)
    b = json.loads(path.read_text())
    y, _ = load_checkpoint(path)
    save_checkpoint(y, path)
    c = json.loads(path.read_text())
    assert a['geometry_sha256'] == b['geometry_sha256'] == c['geometry_sha256']
    assert a['integrity']['digest'] != b['integrity']['digest']


def test_checkpoint_layout_matches_normal_json_export(tmp_path):
    import json
    from flagstone.phase1 import load_json, save_json
    exported = tmp_path / 'layout.json'
    checkpoint = tmp_path / 'checkpoint.json'
    raw = {'schema_version': 1, 'units': 'ft', 'patio': [[0,0],[3,0],[3,3],[0,3]],
           'stones': [{'id':'A', 'outline': [[1,1],[2,1],[2,2],[1,2]],
                       'pose':[.2,.3,.4], 'active':True}]}
    exported.write_text(json.dumps(raw))
    layout = load_json(exported)
    save_json(layout, exported)
    save_checkpoint(layout, checkpoint)
    plain = json.loads(exported.read_text())
    saved = json.loads(checkpoint.read_text())
    for key in ('schema_version', 'units', 'patio', 'stones'):
        assert saved[key] == plain[key]
    restored, _ = load_checkpoint(checkpoint)
    assert restored.units == layout.units and restored.source_outlines == layout.source_outlines


def test_standard_numpy_generators_round_trip(tmp_path):
    for bit_generator in (np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937,
                          np.random.Philox, np.random.SFC64):
        layout = inventory(seed=8, counts=(2,), areas=(.01,))
        layout.rng = np.random.Generator(bit_generator(42))
        layout.rng.random(13)
        path = tmp_path / f'{bit_generator.__name__}.json'
        save_checkpoint(layout, path)
        restored, _ = load_checkpoint(path)
        assert type(restored.rng.bit_generator) is bit_generator
        assert np.array_equal(restored.rng.random(30), layout.rng.random(30))


def test_string_ids_remain_distinct_in_margins_and_checkpoint(tmp_path):
    from flagstone.phase1 import Layout, polygon, local_shape
    shape = local_shape([[0,0],[.1,0],[0,.1]])
    layout = Layout(polygon([[0,0],[1,0],[1,1],[0,1]]), {'1':shape, '01':shape},
                    [[.2,.2,0],[.7,.7,0]], [True,True], ('1','01'), np.random.default_rng(1))
    path = tmp_path / 'checkpoint.json'
    save_checkpoint(layout, path, {'stone_margins': {'1': .01, '01': .02}})
    restored, metadata = load_checkpoint(path)
    assert restored.ids == ('1', '01')
    assert metadata['stone_margins'] == {'1': .01, '01': .02}


def test_geometry_cache_reuses_only_unchanged_immutable_geometry(tmp_path, monkeypatch):
    import json
    import flagstone.checkpoint as checkpoint
    from flagstone.phase1 import local_shape
    x = inventory(seed=3, counts=(2,), areas=(.01,))
    cache = checkpoint._GeometryDigestCache(x)
    original = checkpoint._geometry_digest
    calls = []
    def counted(data):
        calls.append(1)
        return original(data)
    monkeypatch.setattr(checkpoint, '_geometry_digest', counted)
    path = tmp_path / 'checkpoint.json'
    save_checkpoint(x, path, _geometry_cache=cache)
    first = json.loads(path.read_text())['geometry_sha256']
    x.poses[0] = [.2,.3,.4]
    save_checkpoint(x, path, _geometry_cache=cache)
    assert len(calls) == 1 and json.loads(path.read_text())['geometry_sha256'] == first
    x.shapes = dict(x.shapes)
    x.shapes[x.ids[0]] = local_shape([[0,0],[.2,0],[0,.2]])
    save_checkpoint(x, path, _geometry_cache=cache)
    second = json.loads(path.read_text())['geometry_sha256']
    assert len(calls) == 2 and second != first
    x.shapes[x.ids[0]] = local_shape([[0,0],[.3,0],[0,.2]])
    save_checkpoint(x, path, _geometry_cache=cache)
    third = json.loads(path.read_text())['geometry_sha256']
    assert len(calls) == 3 and third != second
    assert load_checkpoint(path)[0].shapes[x.ids[0]].equals_exact(x.shapes[x.ids[0]], 0)
