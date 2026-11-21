import numpy as np
import pytest
from flagstone.calibration import fit_similarity, calibrate_inventory
from flagstone.phase1 import transformed


def test_similarity_recovers_rotation_scale_and_reflection():
    source = np.array([[0,0], [100,0], [0,80], [70,40]], float)
    angle = .4
    r = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    reflected = source * [1, -1]
    target = .03 * (reflected @ r) + [4, 5]
    cal = fit_similarity(source, target, reflect_source_y=True, max_residual=1e-10)
    assert np.allclose(cal.apply(source), target)
    assert np.isclose(cal.scale, .03)
    assert np.linalg.det(np.asarray(cal.rotation)) > 0
    with pytest.raises(ValueError, match='residual'):
        fit_similarity(source, target, reflect_source_y=False, max_residual=.01)


def test_calibration_rejects_degenerate_and_residual_overrun():
    with pytest.raises(ValueError, match='non-collinear'):
        fit_similarity([[0,0],[1,0],[2,0]], [[0,0],[1,0],[2,0]], max_residual=.01)
    with pytest.raises(ValueError, match='residual'):
        fit_similarity([[0,0],[1,0],[0,1],[1,1]], [[0,0],[1,0],[0,1],[2,1]], max_residual=.001)


def test_inventory_calibration_preserves_rings_and_original_data():
    data = {'measurement_version': 1, 'source_units': 'pixels', 'units': 'ft',
            'control_points': [{'source':[0,0], 'target':[2,3]}, {'source':[100,0], 'target':[12,3]}, {'source':[0,100], 'target':[2,13]}],
            'patio': [[0,0],[100,0],[100,100],[0,100]],
            'stones':[{'id':'A','outline':[[10,10],[30,10],[30,20],[10,20]],'active':False}]}
    layout, report = calibrate_inventory(data, max_residual=1e-10)
    assert layout.units == 'ft' and not layout.active[0]
    assert np.isclose(layout.shapes['A'].area, 2.)
    assert np.allclose(layout.poses[0, :2], [4, 4.5])
    assert transformed(layout.shapes['A'], layout.poses[0]).equals_exact(__import__('shapely').geometry.Polygon([[3,4],[5,4],[5,5],[3,5]]), 1e-12)
    assert report['source_stones'][0]['outline'] == data['stones'][0]['outline']
    coverage = report['patio_control_coverage']
    assert coverage['outside_vertex_indices'] == [2]
    assert np.isclose(coverage['max_distance_outside_hull_source_units'], 50 * np.sqrt(2))
    assert np.isclose(coverage['max_distance_outside_hull_target_units'], 5 * np.sqrt(2))
    assert report['source_stones'][0]['control_coverage']['outside_vertex_count'] == 0


def test_calibration_command_produces_traceable_files(tmp_path):
    import json
    import subprocess
    import sys
    from flagstone.phase1 import load_json
    data = {'measurement_version':1, 'source_units':'pixels', 'units':'ft',
            'control_points':[{'source':[0,0],'target':[0,0]}, {'source':[100,0],'target':[10,0]}, {'source':[0,100],'target':[0,10]}],
            'patio':[[0,0],[100,0],[100,100],[0,100]],
            'stones':[{'id':'A','outline':[[10,10],[30,10],[30,20],[10,20]],'active':False}]}
    source = tmp_path / 'measurements.json'; source.write_text(json.dumps(data)); original = source.read_bytes()
    output, report = tmp_path / 'layout.json', tmp_path / 'report.json'
    subprocess.run([sys.executable, '-m', 'flagstone.calibration', str(source), '--max-residual', '.001', '--output', str(output), '--report', str(report)], check=True, capture_output=True, text=True)
    assert source.read_bytes() == original and load_json(output).units == 'ft'
    assert json.loads(report.read_text())['source_stones'][0]['outline'] == data['stones'][0]['outline']
    invalid = subprocess.run([sys.executable, '-m', 'flagstone.calibration', str(source), '--max-residual', '.001', '--output', str(source), '--report', str(report)], capture_output=True, text=True)
    assert invalid.returncode != 0 and source.read_bytes() == original


def test_independent_frames_and_placed_pose():
    from flagstone.calibration import calibrate_bundle
    def component(scale, outline):
        return {'source_units':'pixels', 'max_residual':1e-9, 'outline':outline,
                'control_points':[{'source':[0,0],'target':[0,0]},
                                  {'source':[100,0],'target':[100*scale,0]},
                                  {'source':[0,100],'target':[0,100*scale]}]}
    data={'measurement_bundle_version':1,'units':'ft',
          'patio':component(.1, [[0,0],[100,0],[100,100],[0,100]]),
          'stones':[{'id':'A','active':True,'pose':[4,5,.5], **component(.02, [[0,0],[100,0],[100,50],[0,50]])},
                    {'id':'B','active':False, **component(.04, [[0,0],[50,0],[50,25],[0,25]])}]}
    layout, report = calibrate_bundle(data)
    assert np.isclose(layout.patio.area,100)
    assert np.isclose(layout.shapes['A'].area,2) and np.isclose(layout.shapes['B'].area,2)
    assert np.allclose(layout.poses[0], [4,5,.5]) and np.allclose(layout.poses[1], [0,0,0])
    assert layout.active.tolist() == [True,False]
    assert report['stones'][0]['source_outline'] == data['stones'][0]['outline']
    assert report['stones'][0]['control_coverage']['outside_vertex_count'] == 1
    del data['stones'][0]['pose']
    with pytest.raises(ValueError, match='pose'):
        calibrate_bundle(data)


def test_independent_bundle_cli_and_failed_fit_preserve_source(tmp_path):
    import json
    import subprocess
    import sys
    source = tmp_path / 'bundle.json'
    controls = [{'source':[0,0], 'target':[0,0]}, {'source':[10,0], 'target':[1,0]}, {'source':[0,10], 'target':[0,1]}]
    base = {'source_units':'pixels', 'max_residual':.001, 'control_points':controls,
            'outline':[[0,0],[10,0],[10,10],[0,10]]}
    data = {'measurement_bundle_version':1, 'units':'ft', 'patio':base,
            'stones':[{'id':'s', 'active':False, **base}]}
    source.write_text(json.dumps(data)); before = source.read_bytes()
    out, report = tmp_path/'layout.json', tmp_path/'report.json'
    cmd = [sys.executable, '-m', 'flagstone.calibration', str(source), '--bundle', '--output', str(out), '--report', str(report)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert source.read_bytes() == before
    assert json.loads(report.read_text())['stones'][0]['source_outline'] == base['outline']
    data['stones'][0]['control_points'] = controls + [{'source':[10,10], 'target':[2,1]}]
    source.write_text(json.dumps(data))
    out.unlink(); report.unlink()
    failed = subprocess.run(cmd, capture_output=True, text=True)
    assert failed.returncode != 0 and 'residual' in failed.stderr
    assert not out.exists() and not report.exists()
