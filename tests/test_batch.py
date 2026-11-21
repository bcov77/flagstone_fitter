import numpy as np
import pytest
from flagstone.batch import relax_batch
from flagstone.phase1 import Layout, synthetic
from flagstone.relaxation import relax


def test_batch_matches_independent_cpu_candidates():
    x = synthetic()
    poses = np.stack([x.poses, x.poses])
    poses[0, 1, :2] = poses[0, 0, :2] + [.15, 0]
    masks = np.zeros((2, len(x.ids)), dtype=bool)
    masks[:, :2] = True
    masks[1, 1] = False
    before = poses.copy()
    mobility = np.zeros((2, len(x.ids))); mobility[:, 1] = 1
    batch = relax_batch(x, poses, masks, mobility=mobility, max_steps=20)
    assert np.array_equal(poses, before)
    for i in range(2):
        candidate = Layout(x.patio, x.shapes, poses[i], masks[i], x.ids, np.random.default_rng(0))
        expected = relax(candidate, mobility=mobility[i], max_steps=20)
        assert np.array_equal(batch.poses[i], candidate.poses)
        assert batch.results[i] == expected
    assert not np.shares_memory(batch.poses, poses)


def test_batch_rejects_bad_shapes_and_masks():
    x = synthetic()
    with pytest.raises(ValueError):
        relax_batch(x, x.poses, x.active)
    with pytest.raises(ValueError):
        relax_batch(x, x.poses[None], x.active[None].astype(int))


def test_batch_validation_matches_scalar_with_concavity_and_margins():
    from flagstone.batch import validate_batch
    from flagstone.generate import benchmark_edge_cases
    from flagstone.phase1 import validate
    template = benchmark_edge_cases()['concave_witness']
    rng = np.random.default_rng(123)
    poses = np.repeat(template.poses[None], 6, axis=0)
    poses[:, :, :2] += rng.uniform(-.15, .15, poses[:, :, :2].shape)
    poses[:, :, 2] = rng.uniform(-1, 1, poses[:, :, 2].shape)
    poses[0, 1] = poses[0, 0]
    active = rng.random((6, len(template.ids))) > .3
    active[0, :2] = True
    margins = {sid: float(rng.uniform(0, .02)) for sid in template.ids}
    before = poses.copy(); flags = active.copy()
    for gap in (0, .01, .1):
        batch = validate_batch(template, poses, active, gap=gap, edge=.01, stone_margins=margins, patio_margin=.005)
        for b in range(len(poses)):
            x = Layout(template.patio, template.shapes, poses[b], active[b], template.ids, np.random.default_rng(0))
            expected = validate(x, gap=gap, edge=.01, stone_margins=margins, patio_margin=.005)
            assert batch[b] == expected
    assert np.array_equal(poses, before) and np.array_equal(active, flags)


def test_batch_validation_handles_unsupported_inset_and_empty_batch():
    from flagstone.batch import validate_batch
    x = synthetic()
    poses = x.poses[None]
    active = x.active[None]
    assert validate_batch(x, poses, active, edge=1)[0][0]['kind'] == 'unsupported_inset'
    assert validate_batch(x, poses[:0], active[:0]) == []


def test_empty_batch_still_checks_clearance_arguments():
    from flagstone.batch import validate_batch
    x = synthetic()
    with pytest.raises(ValueError, match='nonnegative'):
        validate_batch(x, x.poses[None][:0], x.active[None][:0], edge=-1)


def test_batch_validation_uses_explicit_tolerance():
    from flagstone.batch import validate_batch
    x = synthetic(); x.active[2:] = False
    x.poses[1] = [.13+.19+.0099995, .13, 0]
    assert validate_batch(x, x.poses[None], x.active[None], tolerance=0)[0]
    assert not validate_batch(x, x.poses[None], x.active[None], tolerance=1e-6)[0]


def test_chunked_validation_matches_unchunked_and_scalar():
    from flagstone.batch import validate_batch
    from flagstone.generate import benchmark_edge_cases
    from flagstone.phase1 import validate
    template = benchmark_edge_cases()['concave_witness']
    rng = np.random.default_rng(127)
    poses = np.repeat(template.poses[None], 37, axis=0)
    poses[:,:,:2] += rng.uniform(-.12, .12, poses[:,:,:2].shape)
    poses[:,:,2] = rng.uniform(-np.pi, np.pi, poses[:,:,2].shape)
    active = rng.random((37,len(template.ids))) > .4
    poses[0, 0] = poses[0, 1]; active[0, :2] = True
    poses[1, -1, :2] = 1e9; active[1, -1] = False
    margins = {sid: .001*(i % 3) for i,sid in enumerate(template.ids)}
    before, flags = poses.copy(), active.copy()
    options = dict(gap=.015, edge=.01, tolerance=1e-9, stone_margins=margins, patio_margin=.002)
    expected = validate_batch(template, poses, active, max_batch_size=None, **options)
    for size in (1, 7, 32, 100):
        assert validate_batch(template, poses, active, max_batch_size=size, **options) == expected
    for i in range(len(poses)):
        layout = Layout(template.patio, template.shapes, poses[i], active[i], template.ids, np.random.default_rng(0))
        assert expected[i] == validate(layout, **options)
    assert np.array_equal(poses, before) and np.array_equal(active, flags)


def test_batch_chunk_size_and_empty_batch_margins_are_validated():
    from flagstone.batch import validate_batch
    x = synthetic()
    for value in (0, -1, 1.5, True, np.bool_(True)):
        with pytest.raises(ValueError, match='max_batch_size'):
            validate_batch(x, x.poses[None], x.active[None], max_batch_size=value)
    assert validate_batch(x, x.poses[None], x.active[None], max_batch_size=np.int64(1)) == [[]]
    with pytest.raises(ValueError, match='margins'):
        validate_batch(x, x.poses[None][:0], x.active[None][:0], stone_margins={x.ids[0]:-1})
