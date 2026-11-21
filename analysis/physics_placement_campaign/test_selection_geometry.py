"""Behavioral geometry/preference tests; review labels are not unit-test targets."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from shapely.geometry import box, Polygon
from shapely.affinity import rotate

import selection_campaign as campaign
from selection_geometry import AXES, boundary_features, comparison, longest_run, screen


class SelectionGeometry(unittest.TestCase):
    def test_equal_area_wrong_shape_and_orientation(self):
        p = Polygon([(1,1),(4,1),(4,2),(2,2),(2,4),(1,4)])
        good = boundary_features(p,p,p)
        bad = boundary_features(rotate(p,180,origin=(2.5,2.5)),p,p)
        self.assertAlmostEqual(good['edge_excess_q90_inches'],0)
        self.assertGreater(bad['edge_excess_q90_inches'],good['edge_excess_q90_inches'])
        self.assertGreater(bad['corner_deficit_q90_inches'],good['corner_deficit_q90_inches'])

    def test_caps_do_not_supply_matching_edges(self):
        p = box(1,1,3,3)
        # Right edge is an artificial cut inside real free space.
        free = box(1,1,5,3)
        m = boundary_features(p,p,free)
        pp = np.array(m['boundary_profile']['xy_ft'])
        artificial = (pp[:,0] > 2.999) & (pp[:,1] > 1.01) & (pp[:,1] < 2.99)
        self.assertTrue(artificial.any())
        self.assertFalse(np.array(m['boundary_profile']['physical'])[artificial].any())
        self.assertFalse(np.array(m['boundary_profile']['matched'])[artificial].any())

    def test_contiguous_runs_wrap_at_ring_seam(self):
        self.assertAlmostEqual(longest_run([1,1,0,0,1],.2),.6)
        self.assertAlmostEqual(longest_run([1,1,1],.2),.6)

    def test_bad_edge_cannot_be_bought_with_area_or_smaller_void(self):
        old = {k:1. for k in AXES}
        new = dict(old,edge_angle_q75_degrees=20.,residual_core_one_inch_sqft=.01,area_sqft=100.)
        c = comparison(new,old)
        self.assertEqual(c['verdict'],'tradeoff_review')
        self.assertIn('edge_angle_q75_degrees',c['worse'])

    def test_indifference_retains_original(self):
        old = {k:1. for k in AXES}
        self.assertEqual(comparison(old,old)['verdict'],'indifferent_retain')
        new = {k:old[k]-2*AXES[k] for k in AXES}
        self.assertEqual(comparison(new,old)['verdict'],'candidate_dominates')

    def test_tiny_area_deep_bump_fails_gate(self):
        m = dict(physical_overlap_sqft=0,clearance_intrusion_fraction=.001,
                 max_clearance_intrusion_inches=1.,coverage=.8,outside_pocket_fraction=0.)
        self.assertEqual(screen(m,campaign.DEFAULT_CONFIG)['status'],'reject')

    def test_preparation_availability_and_resume_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            manifest = campaign.prepare(out,copy.deepcopy(campaign.DEFAULT_CONFIG),['board_00_replace_50'])
            tasks = manifest['tasks']
            self.assertIn('50',{t['stone'] for t in tasks})
            self.assertFalse(({t['stone'] for t in tasks}-{'50'}) & set(campaign.b.ACTIVE))
            self.assertEqual(campaign.load_manifest(out)['tasks'],tasks)
            path = out/'manifest.json'
            manifest['dependencies']['selection_geometry.py'] = 'modified'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError,'hashes changed'):
                campaign.load_manifest(out)


    def test_underfilled_repositioned_original_remains_competitor(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            manifest = campaign.prepare(out,copy.deepcopy(campaign.DEFAULT_CONFIG),['board_00_replace_50'])
            def candidate(stone,origin,value,passing):
                return dict(stone=stone,origin=origin,pose=[0.,0.,0.],
                            metrics={k:value for k in AXES},
                            fit_gate=dict(status='passes_provisional_screen' if passing else 'reject',
                                          checks=dict(physical_overlap=True,clearance_area=True,
                                                      clearance_depth=True,silhouette_coverage=passing)))
            for task in manifest['tasks']:
                if task['stone']=='50':
                    cs=[candidate('50','retain_original',3.,False),candidate('50','boundary_tail',.1,False)]
                elif task['stone']=='116':
                    cs=[candidate('116','boundary_tail',1.,True)]
                else:
                    continue
                campaign.atomic_json(out/'results'/f"{task['id']}.json",dict(candidates=cs))
            campaign.report(out)
            r=json.loads((out/'report.json').read_text())[0]
            self.assertEqual(len(r['original_options']),2)
            self.assertEqual(r['frontier'][0]['benefit_gate'],'retain_or_review_tradeoff')
            self.assertEqual(r['action'],'incomplete_search_do_not_select')


if __name__=='__main__':
    unittest.main()
