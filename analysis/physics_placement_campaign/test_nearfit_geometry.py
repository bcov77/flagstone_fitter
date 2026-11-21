"""Controlled near-fit and local slack tests; all dimensions are feet."""
import copy
from pathlib import Path
import tempfile
import unittest
import json
from unittest.mock import patch
from shapely.ops import unary_union

from shapely.geometry import box
from shapely.affinity import translate

import selection_campaign as c
from nearfit_geometry import shrink_probe,translation_witness,nearfit_checks
from selection_geometry import screen


class NearFitGeometry(unittest.TestCase):
    def setUp(self):
        self.config=copy.deepcopy(c.DEFAULT_CONFIG)
        self.near=self.config['nearfit']
        self.patio=box(0,0,5,5)
        self.candidate=box(1,1,2.01,2)
        self.neighbors={'right':box(2.04,1,3.04,2)}

    def test_shrink_is_centered_diagnostic_full_size_stays_unchanged(self):
        full=box(2,2,4.8,4.8)
        original=full.wkb
        probe=shrink_probe(full,.97)
        self.assertEqual(full.wkb,original)
        self.assertAlmostEqual(probe.area/full.area,.97**2)
        self.assertLess(probe.centroid.distance(full.centroid),1e-12)
        obstacles=c.b.PATIO.difference(probe.buffer(c.b.GAP))
        fm=c.b.measures(full,probe,probe,obstacles)
        pm=c.b.measures(probe,probe,probe,obstacles)
        self.assertEqual(screen(fm,self.config)['status'],'reject')
        checks=nearfit_checks(fm,screen(pm,self.config),True,self.near)
        self.assertTrue(all(checks.values()))
        self.assertGreater(fm['physical_overlap_sqft'],0)

    def test_slack_produces_full_size_clearance_witness(self):
        w=translation_witness(self.candidate,self.neighbors,self.patio,c.b.GAP,self.near)
        self.assertEqual(w['status'],'bounded_translation_witness')
        dx,dy=w['translations_ft']['right']
        moved=translate(self.neighbors['right'],dx,dy)
        self.assertGreaterEqual(moved.distance(self.candidate),c.b.GAP-1e-8)
        self.assertLessEqual(w['max_neighbor_translation_inches'],self.near['max_translation_inches'])

    def test_no_slack_does_not_push_neighbor_into_another_stone(self):
        neighbors={**self.neighbors,'blocked':box(3.04+c.b.GAP,1,4.04+c.b.GAP,2)}
        w=translation_witness(self.candidate,neighbors,self.patio,c.b.GAP,self.near)
        self.assertEqual(w['status'],'no_bounded_translation_witness')
        self.assertEqual(w['translations_ft'],{})

    def test_fixed_wall_and_locked_neighbor_cannot_be_displaced(self):
        wall=translation_witness(box(-.01,1,1,2),{},self.patio,c.b.GAP,self.near)
        self.assertEqual(wall['reason'],'candidate_crosses_fixed_wall_clearance')
        self.near['locked_neighbors']=['right']
        locked=translation_witness(self.candidate,self.neighbors,self.patio,c.b.GAP,self.near)
        self.assertEqual(locked['reason'],'affected_neighbor_locked')

    def test_full_size_deep_overlap_cannot_hide_behind_good_probe(self):
        m=dict(area_sqft=1.,physical_overlap_sqft=.5,clearance_intrusion_fraction=.5,
               max_clearance_intrusion_inches=3.)
        checks=nearfit_checks(m,dict(status='passes_provisional_screen'),True,self.near)
        self.assertFalse(checks['full_physical_overlap'])
        self.assertFalse(checks['full_clearance_depth'])

    def test_movement_budget_is_a_limit_even_when_more_space_exists(self):
        near=dict(self.near,max_translation_inches=.1)
        w=translation_witness(self.candidate,self.neighbors,self.patio,c.b.GAP,near)
        self.assertEqual(w['status'],'no_bounded_translation_witness')

    def test_larger_budget_admits_a_fit_requiring_more_than_three_quarter_inch(self):
        candidate=box(1,1,2.09,2)
        small=dict(self.near,max_translation_inches=.75)
        old=translation_witness(candidate,self.neighbors,self.patio,c.b.GAP,small)
        new=translation_witness(candidate,self.neighbors,self.patio,c.b.GAP,self.near)
        self.assertEqual(old['status'],'no_bounded_translation_witness')
        self.assertEqual(new['status'],'bounded_translation_witness')
        self.assertGreater(new['max_neighbor_translation_inches'],.75)
        self.assertLessEqual(new['max_neighbor_translation_inches'],1.5)

    def test_eighty_percent_probe_can_pass_expanded_full_size_screen(self):
        full=box(1,1,2,2);probe=shrink_probe(full,.80)
        obstacles=c.b.PATIO.difference(probe.buffer(c.b.GAP))
        fm=c.b.measures(full,probe,probe,obstacles)
        pm=c.b.measures(probe,probe,probe,obstacles)
        checks=nearfit_checks(fm,screen(pm,self.config),True,self.near)
        self.assertTrue(all(checks.values()))
        self.assertGreater(fm['clearance_intrusion_fraction'],.16)
        self.assertAlmostEqual(probe.area/full.area,.64)

    def test_eighty_percent_probe_restores_full_size_by_moving_four_neighbors(self):
        full=box(1,1,2,2);gap=c.b.GAP
        neighbors=dict(left=box(.2,1.1,1.1-gap,1.9),right=box(1.9+gap,1.1,2.8,1.9),
                       bottom=box(1.1,.2,1.9,1.1-gap),top=box(1.1,1.9+gap,1.9,2.8))
        old=translation_witness(full,neighbors,self.patio,gap,dict(self.near,max_translation_inches=.75))
        new=translation_witness(full,neighbors,self.patio,gap,self.near)
        self.assertEqual(old['status'],'no_bounded_translation_witness')
        self.assertEqual(new['status'],'bounded_translation_witness')
        self.assertEqual(len(new['translations_ft']),4)
        self.assertGreater(new['max_neighbor_translation_inches'],1.)
        self.assertLessEqual(new['max_neighbor_translation_inches'],1.5)

    def test_prefilter_includes_pieces_only_probe_size_can_admit(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'full',Path(d)/'near'
            a.mkdir();b.mkdir()
            full=copy.deepcopy(self.config);full['nearfit']['scales']=[]
            first=c.prepare(a,full,['board_00_replace_50'])
            second=c.prepare(b,self.config,['board_00_replace_50'])
            ids={t['stone'] for t in first['tasks']}
            extras={t['stone'] for t in second['tasks']}-ids
            self.assertTrue(extras)
            p=c.environment(second['targets'][0])[0]
            for sid in extras:
                ratio=c.b.LOCAL[sid].area/p.area
                self.assertGreater(ratio,full['max_area_ratio'])
                self.assertTrue(any(self.config['min_area_ratio']<=ratio*s*s<=self.config['max_area_ratio'] for s in self.near['scales']))

    def test_full_size_nearfit_survives_report_without_passing_fixed_neighbor_gate(self):
        target=dict(id='synthetic_slack',old=None,removed=[],family='insertion')
        pocket=shrink_probe(self.candidate,.95)
        obstacles=unary_union(list(self.neighbors.values()))
        free=self.patio.buffer(-c.b.GAP).difference(obstacles.buffer(c.b.GAP))
        target.update(pocket=pocket.wkt,free=free.wkt)
        # Synthetic world geometry exercises actual exact metrics, screens,
        # neighbor search and reporting, without depending on source-board labels.
        with patch.object(c.b,'PATIO',self.patio),patch.object(c.b,'ACTIVE',['right']), \
             patch.object(c.b,'ORIGINAL',self.neighbors), \
             patch.object(c.b,'STONES',{'right':dict(pose=[2.04,1.,0.])}), \
             patch.object(c.b,'world',lambda *args:self.candidate):
            candidate=c.evaluate_nearfit('candidate',[1.,1.,0.],.95,target,pocket,free,obstacles,self.config)
        self.assertEqual(candidate['fit_gate']['status'],'reject')
        self.assertEqual(candidate['nearfit_gate']['status'],'passes_nearfit_review_screen')
        self.assertAlmostEqual(candidate['metrics']['area_sqft'],self.candidate.area)
        self.assertAlmostEqual(candidate['shrink_probe']['metrics']['area_sqft'],self.candidate.area*.95**2)
        self.assertEqual(candidate['space_plan']['status'],'bounded_translation_witness')
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'results').mkdir()
            task=dict(id='case',target=target['id'],stone='candidate')
            c.atomic_json(out/'manifest.json',dict(config=self.config,dependencies=c.dependencies(),targets=[target],tasks=[task]))
            c.atomic_json(out/'results/case.json',dict(candidates=[candidate]))
            c.report(out)
            result=json.loads((out/'report.json').read_text())[0]
            self.assertEqual(result['passing_poses'],0)
            self.assertEqual(result['nearfit_passing_poses'],1)
            self.assertEqual(result['nearfit_frontier'][0]['benefit_gate'],'conditional_value_requires_review')


if __name__=='__main__':
    unittest.main()
