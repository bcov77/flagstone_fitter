import json
import subprocess
import sys
from flagstone.generate import inventory
from flagstone.runner import start_insertion


def test_resume_keeps_saved_clearances_and_margins(tmp_path):
    x = inventory(seed=2, counts=(3,), areas=(.01,))
    margins = {sid: .005 for sid in x.ids}
    path = tmp_path / 'insertion.json'
    start_insertion(x, path, gap=.03, edge=.04, proposals=10, stone_margins=margins, patio_margin=.01)
    output = tmp_path / 'output'
    subprocess.run([sys.executable, '-m', 'flagstone', '--resume', str(path), '--durable-rebuild', '--rebuild-rounds', '1', '--output', str(output)], check=True, capture_output=True, text=True)
    rebuild = json.loads((output / 'rebuild_checkpoint.json').read_text())['metadata']['config']
    assert rebuild['gap'] == .03 and rebuild['edge'] == .04
    assert rebuild['stone_margins'] == margins and rebuild['patio_margin'] == .01
    assert not json.loads((output / 'metrics.json').read_text())['violations']


def test_import_requires_clearances_and_retains_units_and_frozen(tmp_path):
    from flagstone.phase1 import save_json
    x = inventory(seed=2, counts=(3,), areas=(.01,))
    x.units = 'ft'
    x.active[0] = True
    x.poses[0] = [.5, .5, .3]
    path = tmp_path / 'input.json'
    save_json(x, path)
    original = path.read_bytes()
    output = tmp_path / 'out'
    base = [sys.executable, '-m', 'flagstone', '--input', str(path), '--output', str(output)]
    failure = subprocess.run(base, capture_output=True, text=True)
    assert failure.returncode != 0 and 'explicit --gap and --edge' in failure.stderr
    margins = tmp_path / 'margins.json'
    margins.write_text(json.dumps({sid: .005 for sid in x.ids}))
    subprocess.run(base + ['--gap', '.02', '--edge', '.03', '--patio-margin', '.01', '--stone-margins', str(margins), '--freeze', x.ids[0], '--proposals', '10'], check=True, capture_output=True, text=True)
    manifest = json.loads((output / 'placement.json').read_text())
    assert manifest['units'] == 'ft'
    assert manifest['clearance']['pair'] == .02 and manifest['clearance']['edge'] == .03
    assert manifest['clearance']['patio_margin'] == .01
    assert next(s for s in manifest['stones'] if s['id'] == x.ids[0])['frozen']
    assert path.read_bytes() == original


def test_import_refuses_to_overwrite_input(tmp_path):
    from flagstone.phase1 import save_json
    path = tmp_path / 'layout.json'
    save_json(inventory(counts=(1,), areas=(.01,)), path)
    original = path.read_bytes()
    result = subprocess.run([sys.executable, '-m', 'flagstone', '--input', str(path), '--gap', '.01', '--edge', '.01', '--output', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0 and 'would be overwritten' in result.stderr
    assert path.read_bytes() == original


def test_cli_exploration_resume_exports_best(tmp_path):
    out = tmp_path / 'first'
    subprocess.run([sys.executable, '-m', 'flagstone', '--case', 'search', '--proposals', '2', '--explore-rounds', '2', '--patch-padding', '.08', '--patch-candidates', '4', '--nudge-probability', '.5', '--nudge-translation', '.01', '--nudge-angle', '.02', '--output', str(out)], check=True, capture_output=True, text=True)
    checkpoint = out / 'exploration_checkpoint.json'
    final = tmp_path / 'final'
    subprocess.run([sys.executable, '-m', 'flagstone', '--resume', str(checkpoint), '--explore-rounds', '4', '--output', str(final)], check=True, capture_output=True, text=True)
    metrics = json.loads((final / 'metrics.json').read_text())
    assert metrics['exploration']['rounds'] == 4
    assert metrics['exploration']['config']['patch_padding'] == .08
    assert metrics['exploration']['config']['max_new_stones'] == 4
    assert metrics['exploration']['config']['nudge_probability'] == .5
    assert abs(metrics['layout']['coverage'] - metrics['exploration']['best_history'][-1]) < 1e-12
    assert not metrics['violations']


def test_cli_batched_durable_insertion(tmp_path):
    output = tmp_path / 'out'
    subprocess.run([sys.executable, '-m', 'flagstone', '--case', 'search', '--strategy', 'random', '--batch-size', '4', '--proposals', '4', '--durable-insertion', '--output', str(output)], check=True, capture_output=True, text=True)
    state = json.loads((output / 'insertion_checkpoint.json').read_text())['metadata']
    assert state['config']['batch_size'] == 4
    assert not json.loads((output / 'metrics.json').read_text())['violations']


def test_cli_relaxation_and_pulse_export_valid_layout(tmp_path):
    output = tmp_path / 'pulse'
    subprocess.run([sys.executable, '-m', 'flagstone', '--case', 'search', '--proposals', '3',
                    '--relaxation-steps', '2', '--relaxation-step', '.01', '--pulse-rounds', '2',
                    '--translation-noise', '.001', '--angle-noise', '.002', '--pulse-steps', '2',
                    '--durable-insertion', '--output', str(output)], check=True, capture_output=True, text=True)
    metrics = json.loads((output / 'metrics.json').read_text())
    config = json.loads((output / 'insertion_checkpoint.json').read_text())['metadata']['config']
    assert metrics['pulse']['attempted'] == 2 and not metrics['violations']
    assert config['relaxation_steps'] == 2 and config['relaxation_step'] == .01


def test_physical_relaxation_requires_explicit_step(tmp_path):
    from flagstone.phase1 import save_json
    x = inventory(seed=1, counts=(1,), areas=(.01,)); x.units = 'ft'
    source = tmp_path / 'input.json'; save_json(x, source)
    cmd = [sys.executable, '-m', 'flagstone', '--input', str(source), '--gap', '.01', '--edge', '.01',
           '--relaxation-steps', '1', '--proposals', '2', '--output', str(tmp_path / 'out')]
    failed = subprocess.run(cmd, capture_output=True, text=True)
    assert failed.returncode != 0 and 'explicit --relaxation-step' in failed.stderr
    subprocess.run(cmd + ['--relaxation-step', '.05'], check=True, capture_output=True, text=True)


def test_cli_output_checkpoint_uses_durable_run_identity(tmp_path):
    output = tmp_path / 'out'
    subprocess.run([sys.executable, '-m', 'flagstone', '--case', 'search', '--proposals', '2', '--durable-insertion', '--output', str(output)], check=True, capture_output=True, text=True)
    insertion = json.loads((output / 'insertion_checkpoint.json').read_text())
    final = json.loads((output / 'checkpoint.json').read_text())
    assert final['run_id'] == insertion['run_id']


def test_uncertainty_model_survives_cli_resume_and_rebuild(tmp_path):
    from flagstone.phase1 import save_json
    x = inventory(seed=10, counts=(4,), areas=(.01,)); x.units = 'ft'
    source = tmp_path/'input.json'; save_json(x, source)
    model = {'uncertainty_version':1, 'units':'ft', 'patio':.01,
             'stones':{sid:{'shape':.001,'translation':.002,'rotation':.01} for sid in x.ids}}
    bounds = tmp_path/'uncertainty.json'; bounds.write_text(json.dumps(model))
    output = tmp_path/'first'
    subprocess.run([sys.executable, '-m', 'flagstone', '--input', str(source), '--gap', '.01', '--edge', '.01',
                    '--uncertainty-model', str(bounds), '--durable-insertion', '--proposals', '4', '--output', str(output)],
                   check=True, capture_output=True, text=True)
    state = json.loads((output/'insertion_checkpoint.json').read_text())['metadata']
    assert state['uncertainty_model'] == model
    assert all(v > .003 for v in state['config']['stone_margins'].values())
    resumed = tmp_path/'resumed'
    subprocess.run([sys.executable, '-m', 'flagstone', '--resume', str(output/'insertion_checkpoint.json'),
                    '--durable-rebuild', '--rebuild-rounds', '1', '--proposals', '4', '--output', str(resumed)],
                   check=True, capture_output=True, text=True)
    rebuild = json.loads((resumed/'rebuild_checkpoint.json').read_text())['metadata']
    manifest = json.loads((resumed/'placement.json').read_text())
    assert rebuild['uncertainty_model'] == model and manifest['uncertainty_model'] == model
    assert rebuild['config']['stone_margins'] == state['config']['stone_margins']
    assert not json.loads((resumed/'metrics.json').read_text())['violations']


def test_uncertainty_model_rejects_incomplete_inventory(tmp_path):
    from flagstone.phase1 import save_json
    x = inventory(counts=(2,), areas=(.01,))
    source = tmp_path/'input.json'; save_json(x, source)
    model = {'uncertainty_version':1, 'units':'normalized', 'patio':0,
             'stones':{x.ids[0]:{'shape':0,'translation':0,'rotation':0}}}
    bounds = tmp_path/'uncertainty.json'; bounds.write_text(json.dumps(model))
    result = subprocess.run([sys.executable, '-m', 'flagstone', '--input', str(source), '--gap', '.01', '--edge', '.01',
                             '--uncertainty-model', str(bounds), '--output', str(tmp_path/'out')], capture_output=True, text=True)
    assert result.returncode != 0 and 'every stone ID' in result.stderr


def test_cli_local_rebuild_settings_survive_resume(tmp_path):
    out = tmp_path/'first'
    subprocess.run([sys.executable, '-m', 'flagstone', '--case', 'search', '--proposals', '2',
                    '--strategy', 'edge', '--durable-rebuild', '--rebuild-rounds', '1',
                    '--patch-padding', '.1', '--patch-candidates', '4', '--restore-removed', '--output', str(out)],
                   check=True, capture_output=True, text=True)
    path = out/'rebuild_checkpoint.json'
    resumed = tmp_path/'resumed'
    subprocess.run([sys.executable, '-m', 'flagstone', '--resume', str(path), '--rebuild-rounds', '2',
                    '--output', str(resumed)], check=True, capture_output=True, text=True)
    state = json.loads(path.read_text())['metadata']
    assert state['rounds'] == 2
    assert state['config']['patch_padding'] == .1 and state['config']['max_new_stones'] == 4
    assert state['config']['restore_removed'] is True
    assert not json.loads((resumed/'metrics.json').read_text())['violations']


def test_cli_neighborhood_relaxation_persists_setting(tmp_path):
    out = tmp_path/'local'
    subprocess.run([sys.executable, '-m', 'flagstone', '--case', 'search', '--proposals', '2',
                    '--strategy', 'edge', '--relaxation-steps', '2', '--local-relaxation',
                    '--durable-insertion', '--output', str(out)], check=True, capture_output=True, text=True)
    state = json.loads((out/'insertion_checkpoint.json').read_text())['metadata']
    assert state['config']['local_relaxation'] is True
    assert not json.loads((out/'metrics.json').read_text())['violations']


def test_cli_selective_relaxation_records_settings_and_counts(tmp_path):
    out = tmp_path/'out'
    subprocess.run([sys.executable, '-m', 'flagstone', '--case', 'search', '--strategy', 'edge',
                    '--proposals', '4', '--relaxation-steps', '2', '--local-relaxation',
                    '--max-relaxation-penalty', '.2', '--durable-insertion', '--output', str(out)],
                   check=True, capture_output=True, text=True)
    state = json.loads((out/'insertion_checkpoint.json').read_text())['metadata']
    assert state['config']['max_relaxation_penalty'] == .2
    assert state['relaxation_attempts'] == state['relaxation_successes'] + state['relaxation_failures']
    assert not json.loads((out/'metrics.json').read_text())['violations']


def test_cli_mobility_policy_is_saved(tmp_path):
    out=tmp_path/'run'
    subprocess.run([sys.executable, '-m', 'flagstone', '--case', 'search', '--strategy', 'edge',
                    '--proposals', '2', '--relaxation-steps', '2', '--local-relaxation',
                    '--mobility-policy', 'clearance', '--durable-insertion', '--output', str(out)],
                   check=True, capture_output=True, text=True)
    state=json.loads((out/'insertion_checkpoint.json').read_text())['metadata']
    assert state['config']['mobility_policy'] == 'clearance'
    assert state['mobile_stone_visits'] >= state['relaxation_attempts']
    assert not json.loads((out/'metrics.json').read_text())['violations']


def test_iterative_cli_resumes_saved_schedule_and_strategy(tmp_path):
    import json
    out = tmp_path/'first'
    subprocess.run([sys.executable, '-m', 'flagstone', '--case', 'search', '--strategy', 'edge',
                    '--proposals', '3', '--passes', '3', '--max-seconds', '0', '--output', str(out)],
                   check=True, capture_output=True, text=True)
    path = out/'iterative_checkpoint.json'
    start = json.loads(path.read_text())['metadata']
    assert start['stones_tried'] == 0 and start['config']['strategy'] == 'edge'
    final = tmp_path/'final'
    subprocess.run([sys.executable, '-m', 'flagstone', '--resume', str(path), '--passes', '3',
                    '--output', str(final)], check=True, capture_output=True, text=True)
    state = json.loads(path.read_text())['metadata']
    summary = json.loads((final/'checkpoint.json').read_text())['metadata']
    assert state['stones_tried'] > 0 and state['config']['strategy'] == 'edge'
    assert summary['strategy'] == 'edge' and not json.loads((final/'metrics.json').read_text())['violations']
    bad = subprocess.run([sys.executable, '-m', 'flagstone', '--resume', str(path), '--passes', '3',
                          '--strategy', 'random', '--output', str(final)], capture_output=True, text=True)
    assert bad.returncode != 0 and 'saved proposal strategy' in bad.stderr


def test_tolerance_is_saved_exported_and_enforced_on_resume(tmp_path):
    import numpy as np
    from flagstone.phase1 import Layout, synthetic, save_json
    x = synthetic(); ids = x.ids[:2]
    y = Layout(x.patio, {sid:x.shapes[sid] for sid in ids}, [[.13,.13,0],[.13+.19+.0099995,.13,0]],
               [True,True], ids, np.random.default_rng(2))
    source=tmp_path/'input.json'; save_json(y, source)
    out=tmp_path/'out'
    subprocess.run([sys.executable, '-m', 'flagstone', '--input', str(source), '--gap', '.01', '--edge', '.01',
                    '--tolerance', '1e-6', '--passes', '1', '--output', str(out)], check=True, capture_output=True, text=True)
    state=json.loads((out/'iterative_checkpoint.json').read_text())['metadata']
    manifest=json.loads((out/'placement.json').read_text())
    assert state['config']['tolerance'] == 1e-6 and manifest['clearance']['tolerance'] == 1e-6
    assert not json.loads((out/'metrics.json').read_text())['violations']
    bad=subprocess.run([sys.executable, '-m', 'flagstone', '--resume', str(out/'iterative_checkpoint.json'),
                        '--passes', '1', '--tolerance', '0', '--output', str(tmp_path/'other')], capture_output=True, text=True)
    assert bad.returncode != 0 and 'tolerance saved' in bad.stderr


def test_secondary_rebuild_cli_saves_settings_and_score(tmp_path):
    out=tmp_path/'out'
    subprocess.run([sys.executable,'-m','flagstone','--case','search','--strategy','edge','--proposals','3',
                    '--durable-rebuild','--rebuild-rounds','1','--secondary-resolution','12',
                    '--secondary-broad-distance','.1','--secondary-edge-strip','.03','--output',str(out)],
                   check=True,capture_output=True,text=True)
    state=json.loads((out/'rebuild_checkpoint.json').read_text())['metadata']
    assert state['config']['secondary_resolution']==12 and state['config']['secondary_broad_distance']==.1
    assert state['secondary_score']['resolution']==12
    assert not json.loads((out/'metrics.json').read_text())['violations']
