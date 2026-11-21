import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('allocation', ROOT/'scripts/allocation_campaigns.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_policies_spend_different_amounts_on_siblings(self):
        for policy, parents, children in [('broad', 1, 1), ('multiple', 1, 3), ('adaptive', 2, 6)]:
            s = a.initial_state(policy)
            a.fill_queue(s, 123)
            self.assertEqual(sum(t['kind']=='classical' for t in s['queue']), parents)
            self.assertEqual(sum(t['kind']=='physics' for t in s['queue']), children)
            self.assertEqual(len({t['key'] for t in s['queue']}), parents+children)

    def test_adaptive_only_preparation_and_pinned_resume(self):
        def fake_prepare(folder, design, source):
            return dict(design=design)
        with patch.object(a.f, 'prepare', side_effect=fake_prepare) as prepare:
            campaigns = a.prepare(self.out, 60, 10, 20261121, 72, ['adaptive'])
            self.assertEqual(len(campaigns), 10)
            self.assertEqual({p.name for p, _ in campaigns},
                             {f'adaptive_{i:02d}' for i in range(10)})
            self.assertEqual(prepare.call_count, 10)
            seeds = [seed for _, seed in campaigns]
            self.assertEqual(len(set(seeds)), 10)
            self.assertEqual(campaigns, a.prepare(self.out, 60, 10, 20261121, 72, ['adaptive']))
            with self.assertRaisesRegex(RuntimeError, 'Settings/driver changed'):
                a.prepare(self.out, 60, 10, 20261121, 72, ['broad'])
        self.assertEqual(a.f.read(self.out/'allocation_manifest.json')['policies'], ['adaptive'])

    def test_default_preparation_retains_three_policy_comparison(self):
        with patch.object(a.f, 'prepare', side_effect=lambda folder, design, source: dict(design=design)):
            campaigns = a.prepare(self.out, 14, 4, 20261120)
        self.assertEqual(len(campaigns), 12)
        self.assertEqual(a.f.read(self.out/'allocation_manifest.json')['policies'], list(a.POLICIES))

    def test_duplicate_policies_rejected_before_preparing(self):
        with self.assertRaisesRegex(RuntimeError, 'Invalid policy selection'):
            a.prepare(self.out, 60, 10, 1, 72, ['adaptive', 'adaptive'])
        self.assertFalse((self.out/'allocation_manifest.json').exists())

    def test_charge_parent_once_and_include_failed_work(self):
        for i, seconds in enumerate([40, 300, 20]):
            a.f.write(self.out/f'parents/p000/operations/op_{i:06d}/receipt.json',
                      dict(worker_seconds=seconds, cpu_seconds=None))
        a.f.write(self.out/'parents/p000/sessions/s.json', dict(orchestration_worker_seconds=5))
        c = a.costs(self.out)
        self.assertEqual(c['worker_seconds'], 365)
        self.assertEqual(c['unknown_cpu_operations'], 3)
        s = a.initial_state('multiple')
        self.assertIsNone(a.next_action(self.out, s, 123, 360))
        self.assertTrue(s['complete'])

    def test_final_continuation_uses_remaining_budget_and_exact_prefix(self):
        s = a.initial_state('adaptive')
        t = a.task('physics', 0)
        t.update(stage='extension', target_minutes=90.)
        s['queue'] = [t]
        a.f.write(self.out/'physics/p000_h00_r00/progress.json', dict(worker_seconds=510.))
        with patch.object(a, 'costs', return_value=dict(worker_seconds=3500)):
            action = a.next_action(self.out, s, 123, 3600)
        self.assertEqual(action['horizons'], [2., 8., 610/60])
        # A resume must reuse the dispatched target even if costs have changed.
        with patch.object(a, 'costs', side_effect=AssertionError('Must not reallocate pending work')):
            self.assertEqual(a.next_action(self.out, s, 123, 3600), action)

    def test_promotion_is_reproducible_and_preserves_exploration(self):
        candidates = [dict(key=str(i), coverage=70+i, task=a.task('physics', i)) for i in range(6)]
        for i in range(100):
            d = a.promote(candidates, 123, i)
            self.assertEqual(d, a.promote(candidates, 123, i))
            self.assertEqual(d['chosen']['key']=='5', not d['exploration'])
        self.assertTrue(any(a.promote(candidates, 123, i)['exploration'] for i in range(100)))

    def test_only_complete_probe_batches_can_promote(self):
        s = a.initial_state('adaptive')
        a.fill_queue(s, 123)
        while s['queue']:
            t = copy.deepcopy(s['queue'][0])
            t['action_id'] = s['next_action']
            s['pending'] = t
            a.commit_action(s, dict(action=t, result=dict(status='complete', best_metrics=dict(area_percent=75))))
        self.assertEqual(len(s['candidates']), 6)
        a.fill_queue(s, 123)
        self.assertEqual(len(s['queue']), 1)
        self.assertEqual(s['queue'][0]['stage'], 'extension')
        self.assertEqual(s['queue'][0]['target_minutes'], 90)
        self.assertEqual(len(s['decisions']), 1)

    def test_stop_does_not_commit_action(self):
        s = a.initial_state('multiple')
        action = a.next_action(self.out, s, 123, 3600)
        before = copy.deepcopy(s)
        a.commit_action(s, dict(action=action, result=dict(status='checkpointed')))
        self.assertEqual(s, before)

    def test_extension_archives_opening_and_resumes_same_directory(self):
        folder = self.out/'physics/p000_h00_r00'
        a.f.write(self.out/'manifest.json', dict(design={}, extension_keys=[]))
        a.f.write(folder/'summary.json', dict(status='complete', samples=2))
        action = dict(a.task('physics', 0), action_id=1, horizons=[2., 8., 90.], stage='extension')
        result = dict(status='complete', samples=3)
        with patch.object(a.f, 'physics_job', return_value=result) as job:
            first = a.run_action(str(self.out), action)
            second = a.run_action(str(self.out), action)
        self.assertEqual(first, second)
        self.assertEqual(job.call_count, 1)
        self.assertEqual(job.call_args.args[1], folder)
        self.assertEqual(job.call_args.args[3]['design']['physics_minutes'], [2., 8., 90.])
        self.assertEqual(a.f.read(folder/'stages/before_action_00001.json')['samples'], 2)

    def test_partial_campaign_cannot_receive_complete_marker(self):
        a.f.write(self.out/'allocation_state.json', a.initial_state('broad'))
        with self.assertRaisesRegex(RuntimeError, 'all campaign budgets'):
            a.verify_all(self.out, [(self.out, 123)])
        self.assertFalse((self.out/'COMPLETE.json').exists())

    def test_audit_deadline_leaves_completion_pending(self):
        s = a.initial_state('broad')
        s['complete'] = True
        a.f.write(self.out/'allocation_state.json', s)
        with patch.object(a.subprocess, 'Popen', side_effect=AssertionError('Deadline has expired')):
            a.verify_all(self.out, [(self.out, 123)], deadline=0.)
        self.assertFalse((self.out/'COMPLETE.json').exists())

    def test_safe_stop_terminates_only_current_audit(self):
        s = a.initial_state('broad')
        s['complete'] = True
        a.f.write(self.out/'allocation_state.json', s)
        (self.out/'STOP').touch()
        from unittest.mock import Mock
        process = Mock(pid=12345)
        process.poll.return_value = None
        with patch.object(a.subprocess, 'Popen', return_value=process), patch.object(a.os, 'killpg') as kill:
            a.verify_all(self.out, [(self.out, 123)])
        kill.assert_called_once_with(12345, a.signal.SIGTERM)
        self.assertFalse((self.out/'COMPLETE.json').exists())

    def test_final_report_reflects_verified_completion(self):
        s = a.initial_state('broad')
        s['complete'] = True
        a.f.write(self.out/'campaign/allocation_state.json', s)
        a.f.write(self.out/'campaign/allocation_verification.json', dict(result='passed'))
        a.f.write(self.out/'allocation_manifest.json', dict(hours_per_campaign=1))
        a.verify_all(self.out, [(self.out/'campaign', 123)])
        self.assertTrue(a.f.read(self.out/'report.json')['verification_complete'])


if __name__ == '__main__':
    unittest.main()
