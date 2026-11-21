"""Allocation, transaction recovery, stopping, timeout and reporting regressions."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('full_trajectories', ROOT/'scripts/full_trajectories.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


def document():
    return dict(units='ft', patio=[[0,0],[3,0],[3,3],[0,3]], stones=[
        dict(id=1, outline=[[0,0],[1,0],[1,1],[0,1]], pose=[.5,.5,0], active=True)], physics={})


def metrics(area=50):
    return dict(area_percent=area, clearance=dict(valid=True),
                gaps=dict(stable=True, largest_interior_inches=3, largest_edge_inches=4), layout_pass=False)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        f._IDLE_STARTED, f._OVERHEAD_SECONDS, f._SOURCE_PATH = None, 0., None
        self.cfg = f.read(ROOT/'analysis/aggressive_trajectories/config.json')
        self.design = dict(lineages=2, replicates=2, classical_minutes=[0., 1., 4.],
            physics_minutes=[.01, .02], extension_minutes=.04, extension_fraction=.25,
            seed=123, search_timeout=1, physics_timeout=1, native_seconds=1,
            native_python='unused', patch_seconds=1, classical_timeout=1, audit_timeout=1)
        self.manifest = dict(design=self.design, physics_config=self.cfg, extension_keys=[])

    def tearDown(self):
        self.tmp.cleanup()

    def setup_parent(self):
        parent = self.out/'parents/p000'
        parent.mkdir(parents=True)
        f.write(parent/'handoff_00.json', dict(document=document(), metrics=metrics(), parent=0,
            horizon_index=0, nominal_minutes=0, actual_refinement_seconds=0,
            cumulative_worker_seconds=2, seed=1))
        folder = self.out/'physics/p000_h00_r00'
        folder.mkdir(parents=True)
        task = dict(kind='physics', key=folder.name, parent=0, horizon=0, replicate=0, seed=345)
        return folder, task

    def fake_selection(self):
        return dict(chosen=dict(remove=[], placements=[dict(stone='1',pose=[.5,.5,0])], family='ordinary'),
                    candidates=[], selection={}, outcome='proposal')

    def test_balanced_tasks_and_matched_seeds(self):
        parents, jobs = f.tasks_for(self.manifest)
        self.assertEqual(len(parents), 2)
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({t['key'] for t in jobs}), 12)
        same = [t for t in jobs if t['parent']==0 and t['replicate']==0]
        self.assertEqual(len({t['seed'] for t in same}), 1)
        self.assertNotEqual(jobs, sorted(jobs, key=lambda t:t['key']))

    def test_invalid_horizons(self):
        for text in ('2,1', '1,1', '-1,2', 'nan', 'inf', ''):
            with self.assertRaises((ValueError, f.argparse.ArgumentTypeError)):
                f.minutes(text)

    def test_empty_search_costs_time_not_physical_attempt(self):
        folder, task = self.setup_parent()
        def invoke(*args):
            return dict(status='ok', worker_seconds=2.), dict(value=dict(chosen=None, outcome='no_candidate'))
        with patch.object(f, 'invoke', side_effect=invoke):
            result = f.physics_job(self.out, folder, task, self.manifest)
        self.assertEqual(result['attempts'], 0)
        self.assertEqual(result['samples'], 2)
        cp = f.read(folder/'checkpoint.json')
        self.assertEqual(cp['empty_searches'], 1)
        self.assertEqual(cp['worker_seconds'], 2.)
        self.assertEqual(f.read(folder/'sample_00.json')['actual_worker_seconds'], 2.)

    def test_search_timeout_is_bounded_and_reported(self):
        folder, task = self.setup_parent()
        with patch.object(f, 'invoke', return_value=(dict(status='timeout', worker_seconds=2.), None)):
            result = f.physics_job(self.out, folder, task, self.manifest)
        self.assertEqual(result['attempts'], 0)
        cp = f.read(folder/'checkpoint.json')
        self.assertEqual(cp['search_timeouts'], 1)
        self.assertEqual(cp['search_round'], 1)

    def test_physics_rejection_rolls_back_and_counts(self):
        folder, task = self.setup_parent()
        before = document()
        def invoke(_folder, index, request, timeout):
            if request['kind']=='select':
                value = self.fake_selection()
            else:
                value = dict(accepted=False, document=before, attempted_metrics=metrics(70))
            return dict(status='ok', worker_seconds=.8), dict(value=value)
        with patch.object(f, 'invoke', side_effect=invoke):
            result = f.physics_job(self.out, folder, task, self.manifest)
        self.assertEqual(result['attempts'], 1)
        self.assertEqual(result['accepted'], 0)
        self.assertEqual(f.read(folder/'checkpoint.json')['document'], before)
        record = f.read(folder/'attempt_00001.json.gz')
        self.assertEqual(record['start'], record['end'])
        self.assertEqual(record['end_metrics'], record['start_metrics'])

    def test_physics_timeout_does_not_commit_partial_board(self):
        folder, task = self.setup_parent()
        def invoke(_folder, index, request, timeout):
            if request['kind']=='select':
                return dict(status='ok', worker_seconds=.8), dict(value=self.fake_selection())
            return dict(status='timeout', worker_seconds=.8), None
        with patch.object(f, 'invoke', side_effect=invoke):
            result = f.physics_job(self.out, folder, task, self.manifest)
        self.assertEqual(result['attempts'], 1)
        self.assertEqual(f.read(folder/'checkpoint.json')['physics_timeouts'], 1)
        self.assertTrue(f.read(folder/'attempt_00001.json.gz')['timed_out'])

    def test_stop_resume_keeps_pending_proposal_and_seed(self):
        folder, task = self.setup_parent()
        calls = []
        def first(_folder, index, request, timeout):
            calls.append(copy.deepcopy(request))
            (self.out/'STOP').touch()
            return dict(status='ok', worker_seconds=.8), dict(value=self.fake_selection())
        with patch.object(f, 'invoke', side_effect=first):
            self.assertEqual(f.physics_job(self.out, folder, task, self.manifest)['status'], 'checkpointed')
        cp = f.read(folder/'checkpoint.json')
        self.assertIsNotNone(cp['pending'])
        self.assertEqual(cp['attempts'], 0)
        (self.out/'STOP').unlink()
        def second(_folder, index, request, timeout):
            calls.append(copy.deepcopy(request))
            return dict(status='ok', worker_seconds=.8), dict(value=dict(
                accepted=True, document=document(), attempted_metrics=metrics(60)))
        with patch.object(f, 'invoke', side_effect=second):
            result = f.physics_job(self.out, folder, task, self.manifest)
        self.assertEqual([v['kind'] for v in calls], ['select','physics'])
        self.assertEqual(calls[1]['seed'], f.seed_for(task['seed'], 'physics', 1))
        self.assertEqual(result['best_metrics']['area_percent'], 60)

    def test_random_extension_has_its_own_horizon(self):
        folder, task = self.setup_parent()
        self.manifest['extension_keys'] = [task['key']]
        with patch.object(f, 'invoke', return_value=(dict(status='timeout', worker_seconds=2.), None)):
            result = f.physics_job(self.out, folder, task, self.manifest)
        self.assertEqual(result['samples'], 3)
        self.assertEqual(result['worker_seconds'], 4.)

    def test_cached_receipt_replay_and_tamper_detection(self):
        folder = self.out/'job'
        req = dict(kind='audit', document=document())
        op = folder/'operations/op_000000'
        f.write(op/'request.json.gz', req)
        f.write(op/'result.json.gz', dict(status='ok', value=dict(metrics=metrics()), cpu_seconds=1))
        rec = dict(status='ok', request_sha256=f.digest(op/'request.json.gz'),
                   result_sha256=f.digest(op/'result.json.gz'), worker_seconds=2)
        f.write(op/'receipt.json', rec)
        with patch.object(f.subprocess, 'Popen', side_effect=AssertionError('Must reuse committed work')):
            self.assertEqual(f.invoke(folder, 0, req, 1)[0], rec)
        f.write(op/'result.json.gz', dict(status='ok', value='tampered'))
        with self.assertRaisesRegex(RuntimeError, 'result changed'):
            f.invoke(folder, 0, req, 1)

    def test_completed_child_survives_missing_receipt(self):
        folder = self.out/'job'
        op = folder/'operations/op_000000'
        req = dict(kind='audit', document=document())
        f.write(op/'request.json.gz', req)
        f.write(op/'running.json', dict(pid=99999999, started_unix=100., observed_worker_seconds=2.))
        f.write(op/'result.json.gz', dict(status='ok', value={}, cpu_seconds=1, finished_unix=104.))
        with patch.object(f.subprocess, 'Popen', side_effect=AssertionError('Must preserve completed result')):
            rec, _ = f.invoke(folder, 0, req, 1)
        self.assertGreaterEqual(rec['worker_seconds'], 4.)
        self.assertEqual(rec['status'], 'ok')

    def test_real_subprocess_timeout_kills_process_group(self):
        sleeper = self.out/'sleeper.py'
        sleeper.write_text('import time\ntime.sleep(30)\n')
        with patch.object(f, 'SCRIPT', sleeper):
            start = time.monotonic()
            rec, result = f.invoke(self.out/'timeout', 0, dict(kind='select'), .2)
        self.assertLess(time.monotonic()-start, 3)
        self.assertEqual(rec['status'], 'timeout')
        self.assertIsNone(result)

    def test_replay_request_must_match(self):
        folder = self.out/'job'
        f.write(folder/'operations/op_000000/request.json.gz', dict(kind='audit', value=1))
        with self.assertRaisesRegex(RuntimeError, 'request mismatch'):
            f.invoke(folder, 0, dict(kind='audit', value=2), 1)

    def test_zero_attempt_verification_does_not_use_sibling_counter(self):
        # An empty trajectory still has a valid initial/champion board.
        folder, task = self.setup_parent()
        handoff = self.out/'parents/p000/handoff_00.json'
        parent = f.read(handoff)
        parent['metrics'] = metrics(100/9)
        f.write(handoff, parent)
        with patch.object(f, 'invoke', return_value=(dict(status='timeout', worker_seconds=2.), None)):
            f.physics_job(self.out, folder, task, self.manifest)
        with patch.object(f, 'verify_pins', return_value=self.manifest):
            f.write(self.out/'source.json', document())
            result = f.verify(self.out)
        self.assertEqual(result['attempts'], 0)
        self.assertEqual(result['handoffs'], 1)

    def test_pose_only_operation_requests_preserve_original_inventory(self):
        source = self.out/'source.json'
        f.write(source, document())
        f._SOURCE_PATH = source
        captured = []
        def fake(folder, index, request, timeout):
            captured.append(request)
            return {}, {}
        with patch.object(f, '_invoke', side_effect=fake):
            f.invoke(self.out/'job', 0, dict(kind='audit', document=document()), 1)
        self.assertNotIn('document', captured[0])
        self.assertIn('frame', captured[0])
        from common import restore, state
        reconstructed = restore(f.read(captured[0]['source']), captured[0]['frame'])
        self.assertEqual(state(reconstructed), state(document()))
        self.assertEqual(reconstructed['stones'][0]['outline'], document()['stones'][0]['outline'])

    def test_gap_champion_can_differ_from_coverage_champion(self):
        better_gap = metrics(49)
        better_gap['gaps']['largest_interior_inches'] = 2.5
        self.assertLess(f.gap_key(better_gap), f.gap_key(metrics(60)))

    def test_report_costs_and_outline_metadata_cache(self):
        folder, task = self.setup_parent()
        f.write(folder/'task.json', task)
        f.write(self.out/'manifest.json', self.manifest)
        with patch.object(f, 'invoke', return_value=(dict(status='timeout', worker_seconds=2.), None)):
            f.physics_job(self.out, folder, task, self.manifest)
        f.write(folder/'operations/op_000000/receipt.json', dict(worker_seconds=2., cpu_seconds=None,
            interrupted_accounting_lower_bound=False))
        f.write(folder/'sessions/test.json', dict(orchestration_worker_seconds=.5))
        first = f.report(self.out)
        self.assertEqual(first['costs']['worker_seconds'], 2.5)
        self.assertEqual(first['costs']['timeout_cpu_unknown_count'], 1)
        seen = []
        real_read = f.read
        def tracked(path):
            seen.append(str(path))
            return real_read(path)
        with patch.object(f, 'read', side_effect=tracked):
            second = f.report(self.out)
        self.assertEqual(first, second)
        self.assertFalse(any(path.endswith(('initial.json','handoff_00.json')) for path in seen))
        self.assertTrue((self.out/'samples.csv').exists())

    def test_compute_pin_changes_refuse_resume(self):
        source = self.out/'source.json'
        f.write(source, document())
        f.write(self.out/'manifest.json', dict(source_sha256=f.digest(source), code={'old.py':'old'},
            versions={}, python=f.sys.version, design={}, native_python_sha256='unused'))
        with patch.object(f, 'pins', return_value={'new.py':'new'}):
            with self.assertRaisesRegex(RuntimeError, 'Compute changed'):
                f.verify_pins(self.out)


if __name__ == '__main__':
    unittest.main()
