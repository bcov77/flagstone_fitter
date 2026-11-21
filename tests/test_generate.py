from flagstone.generate import cut_witness, inventory, inventory_summary
from flagstone.phase1 import validate


def test_generated_witness_and_inventory():
    witness = cut_witness()
    assert not validate(witness, gap=.01, edge=.009999999)
    x = inventory(counts=(2, 3), areas=(.03, .003))
    assert inventory_summary(x, [0, .01, .04])['counts'] == [3, 2]
    assert not x.active.any()


def test_edge_case_families():
    from flagstone.generate import benchmark_edge_cases
    cases = benchmark_edge_cases()
    witness = cases['concave_witness']
    assert not validate(witness) and witness.active.sum() == 12
    impossible = cases['impossible_inventory']
    assert sum(g.area for g in impossible.shapes.values()) > impossible.patio.area
    oversized = cases['oversized_stone']
    assert next(iter(oversized.shapes.values())).area > oversized.patio.area
    assert not impossible.active.any() and not oversized.active.any()


def test_detailed_inventory_has_more_vertices_and_is_reproducible():
    import numpy as np
    a = inventory(seed=8, counts=(10,), areas=(.01,), vertex_profile='detailed')
    b = inventory(seed=8, counts=(10,), areas=(.01,), vertex_profile='detailed')
    counts = [len(g.exterior.coords)-1 for g in a.shapes.values()]
    assert min(counts) >= 15 and np.mean(counts) > 20
    assert all(a.shapes[s].equals_exact(b.shapes[s], 0) for s in a.ids)
    assert all(np.isclose(g.area, .01) for g in a.shapes.values())


def test_large_witness_fits_and_round_trips(tmp_path):
    from flagstone.phase1 import load_json, save_json
    from flagstone.search import coverage
    witness = cut_witness(seed=42, rows=12, cols=12, clearance=.006)
    assert len(witness.ids) == 144 and witness.active.all()
    assert not validate(witness, gap=.01, edge=.005)
    path = tmp_path / 'witness.json'
    save_json(witness, path)
    restored = load_json(path)
    assert not validate(restored, gap=.01, edge=.005)
    assert abs(coverage(witness) - coverage(restored)) < 1e-12


def test_witness_benchmark_zero_budget(tmp_path):
    import json
    import subprocess
    import sys
    subprocess.run([sys.executable, 'scripts/benchmark_witness.py', '--size', '3',
                    '--seconds', '0', '--output', str(tmp_path)], check=True, capture_output=True)
    report = json.loads((tmp_path / 'metrics.json').read_text())
    assert report['witness']['placed'] == 9 and report['witness']['violations'] == 0
    assert report['result']['placed'] == 0 and report['result']['violations'] == 0
    assert (tmp_path / 'checkpoint.json').exists()


def test_witness_rebuild_zero_budget(tmp_path):
    import json
    import subprocess
    import sys
    initial, rebuilt = tmp_path / 'initial', tmp_path / 'rebuilt'
    subprocess.run([sys.executable, 'scripts/benchmark_witness.py', '--size', '3',
                    '--seconds', '0', '--output', str(initial)], check=True, capture_output=True)
    subprocess.run([sys.executable, 'scripts/benchmark_witness_rebuild.py', '--input', str(initial / 'checkpoint.json'),
                    '--seconds', '0', '--output', str(rebuilt)], check=True, capture_output=True)
    report = json.loads((rebuilt / 'metrics.json').read_text())
    assert report['result']['rounds'] == 0 and report['result']['violations'] == 0
    assert (rebuilt / 'checkpoint.json').exists()


def test_varied_inventory_has_rectangles_broad_notches_and_detailed_outlines():
    import numpy as np
    a = inventory(seed=41, counts=(6,50,100), areas=(.025,.008,.003), vertex_profile='varied')
    b = inventory(seed=41, counts=(6,50,100), areas=(.025,.008,.003), vertex_profile='varied')
    vertices = [len(s.exterior.coords)-1 for s in a.shapes.values()]
    notches = [1-s.area/s.convex_hull.area for s in a.shapes.values()]
    assert sum(n == 4 for n in vertices) >= 10
    assert sum(n >= 20 for n in vertices) >= 50
    assert sum(d > .04 for d in notches) >= 10
    assert sum(d < 1e-9 for d in notches) > len(notches)/2
    assert all(a.shapes[s].equals_exact(b.shapes[s], 0) for s in a.ids)
    assert np.allclose([a.shapes[s].area for s in a.ids], [.025]*6 + [.008]*50 + [.003]*100)
    convex = inventory(seed=41, counts=(40,), areas=(.01,), vertex_profile='varied', mixed=False)
    assert all(abs(s.area-s.convex_hull.area) < 1e-12 for s in convex.shapes.values())
def test_revised_inventory_round_trip(tmp_path):
    """A near-zero centroid must not turn an identical source ring into a mismatch."""
    from flagstone.phase1 import Layout, local_shape, polygon, save_json, load_json
    import numpy as np
    vertices = [[-0.5026359476728275, -0.30458802861435025], [0.4576290616010002, -0.37396406938523824], [0.555137312411258, -0.22623730220776433], [0.4197427855454589, 0.33759796534020325], [-0.3548194905938785, 0.3856498312088339], [-0.5269171325228109, 0.19460338755040563]]
    stone = polygon(vertices)
    x = Layout(polygon([(0,0),(2,0),(2,2),(0,2)]), {'S':stone}, np.zeros((1,3)), np.zeros(1,bool), ('S',), np.random.default_rng(0))
    path = tmp_path / 'layout.json'; save_json(x, path)
    restored = load_json(path)
    assert restored.shapes['S'].equals_exact(stone, 0)
