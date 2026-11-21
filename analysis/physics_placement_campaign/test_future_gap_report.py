import unittest
from future_gap_report import completed_connections


class FinishedConnections(unittest.TestCase):
    def fixture(self,runs):
        return dict(assessment=dict(joints=dict(by_owner={k:dict(longest_run_ft=v) for k,v in runs.items()})),
                    provisional_stability={'neighbor':dict(score=.5)})

    def test_scattered_matches_do_not_count_as_finished_connections(self):
        scattered=completed_connections(self.fixture({'a':.05,'b':.08,'c':.1}),dict(locked_stones=[]))
        continuous=completed_connections(self.fixture({'neighbor':.6}),dict(locked_stones=[]))
        self.assertEqual(scattered['finished_owner_count'],0)
        self.assertEqual(continuous['finished_owner_count'],1)
        self.assertAlmostEqual(continuous['provisional_finished_length_ft'],.3)

    def test_walls_and_locks_are_separate_from_soft_stability(self):
        r=completed_connections(self.fixture({'wall_0':.8,'locked':.5,'neighbor':.6}),dict(locked_stones=['locked']))
        self.assertAlmostEqual(r['hard_finished_length_ft'],1.3)
        self.assertAlmostEqual(r['provisional_finished_length_ft'],.3)


if __name__=='__main__':unittest.main()
