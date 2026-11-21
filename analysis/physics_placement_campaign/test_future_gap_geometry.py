"""Controlled planning contrasts; human-reviewed examples are not test labels."""
import copy
import unittest
import shapely
from shapely.geometry import box,Polygon
from shapely.ops import unary_union
from shapely.affinity import affine_transform

from future_gap_campaign import DEFAULT_CONFIG,respects_locks
import build_prefit_review as b
from future_gap_geometry import walls,joint_profile,residual_regions,region_owners,filler_witness,alignment_options,assess,compare


class FutureGaps(unittest.TestCase):
    def setUp(self):
        self.c=copy.deepcopy(DEFAULT_CONFIG);self.gap=.5/12;self.patio=box(0,0,5,5)

    def test_fixed_corner_and_flexible_neighbors_have_different_owners(self):
        lo,hi=self.gap,.35
        p=box(lo,lo,hi,hi)
        subject=unary_union([box(hi+self.gap,0,2,2),box(0,hi+self.gap,2,2)])
        owners,_=region_owners(p,{'subject':subject},'subject',walls(self.patio),self.gap,{},self.c)
        self.assertEqual(sum(v['role']=='wall' for v in owners.values()),2)
        moved=shapely.affinity.translate(p,2,2)
        board={'subject':shapely.affinity.translate(subject,2,2),
               'left':box(1,2,2,3),'bottom':box(2,1,3,2)}
        owners,_=region_owners(moved,board,'subject',walls(box(0,0,10,10)),self.gap,{},self.c)
        self.assertEqual(sum(v['role']=='wall' for v in owners.values()),0)
        self.assertTrue(any(v['role']=='neighbor' for v in owners.values()))

    def test_artificial_region_edges_are_not_physical_anchors(self):
        owners,caps=region_owners(box(2,2,3,3),{},None,walls(self.patio),self.gap,{},self.c)
        self.assertEqual(owners,{})
        self.assertGreater(caps,3.9)

    def test_normal_half_inch_joint_is_not_a_residual_hole(self):
        patio=box(0,0,2,2)
        occupied=unary_union([box(0,0,.9,2),box(.9+self.gap,0,2,2)])
        free=patio.buffer(-self.gap).difference(occupied.buffer(self.gap))
        self.assertEqual(residual_regions(free,patio,self.c),[])

    def test_actual_filler_witness_preserves_scale_and_clear_region(self):
        p=box(1,1,2,2);stone=box(-.3,-.3,.3,.3)
        w=filler_witness(p,p.buffer(-.1),p,{'spare':stone},self.c)
        self.assertEqual(w['status'],'verified_filler_pose')
        x,y,a=w['pose'];import math
        g=affine_transform(stone,[math.cos(a),-math.sin(a),math.sin(a),math.cos(a),x,y])
        self.assertTrue(p.covers(g));self.assertAlmostEqual(g.area,stone.area)

    def test_no_inventory_fit_is_labeled_bounded_failure(self):
        p=box(1,1,2,2)
        w=filler_witness(p,p.buffer(-.1),p,{'too_big':box(-1,-1,1,1)},self.c)
        self.assertEqual(w['status'],'no_witness_in_bounded_search')
        self.assertNotIn('unfillable',w)

    def test_flat_wall_alignment_option_beats_a_point(self):
        pocket=box(self.gap,1,2.1,3.1);wl=walls(self.patio)
        wid=min(wl,key=lambda k:wl[k].distance(pocket))
        flat=alignment_options(box(1,1,2,3),{},wl,pocket,self.c)
        point=alignment_options(Polygon([(1,2),(2,1),(2,3)]),{},wl,pocket,self.c)
        self.assertGreater(flat[wid]['length_ft'],point[wid]['length_ft']+.5)
        self.assertTrue(flat[wid]['conditional'])

    def test_finishing_a_side_registers_a_contiguous_joint(self):
        subject=box(1,1,2,2);neighbor=box(2+self.gap,1,3,2)
        p=joint_profile(subject,{'neighbor':neighbor},{},self.gap,self.c)
        self.assertGreater(p['by_owner']['neighbor']['longest_run_ft'],.8)

    def test_indent_records_a_residual_inside_the_subject_hull(self):
        subject=box(1,1,3,3).difference(box(2.6,1.5,3.1,2.5))
        board={'subject':subject,'right':box(3+self.gap,1,4,3)}
        config=dict(self.c,max_filler_regions=0)
        result=assess(board,'subject',{'right':board['right']},self.patio,box(1,1,3,3),
                      box(.8,.8,4.2,3.2),{}, {},self.gap,config)
        self.assertGreater(result['metrics']['subject_indent_sqft'],.01)
        self.assertGreater(result['metrics']['unsearched_regions'],0)

    def test_area_cannot_override_new_rigid_corner_risk(self):
        base=dict(metrics={k:0. for k in ('wall_corner_risk_sqft','rigid_pocket_risk_sqft','wall_excess_sqft','subject_indent_sqft','stable_neighbor_exposure_sqft')},
                  alignment_options={},joints=dict(matched_length_ft=1.))
        candidate=copy.deepcopy(base)
        candidate['metrics'].update(wall_corner_risk_sqft=.1,area_gain_sqft=100.)
        candidate['joints']['matched_length_ft']=5.
        self.assertEqual(compare(candidate,base,self.c)['status'],'new_residual_risk_review')

    def test_local_move_does_not_inherit_remote_wall_corners(self):
        patio=box(0,0,8,8);subject=box(3,3,4,4);roi=box(2.7,2.7,4.3,4.3)
        result=assess({'subject':subject},'subject',{},patio,roi,roi,{}, {},self.gap,dict(self.c,max_filler_regions=0))
        self.assertEqual(result['metrics']['wall_corner_risk_sqft'],0.)
        self.assertEqual(result['metrics']['wall_excess_sqft'],0.)

    def test_one_spare_cannot_be_allocated_to_two_holes(self):
        patio=box(0,0,8,8);holes=unary_union([box(1,1,2,2),box(4,1,5,2)])
        subject=patio.difference(holes);inventory={'one_spare':box(-.35,-.35,.35,.35)}
        result=assess({'subject':subject},'subject',{},patio,holes,holes.buffer(.2),inventory,{},self.gap,self.c)
        self.assertEqual(len(result['repair_filler_plan']),1)
        self.assertEqual(result['metrics']['verified_fillable_regions'],1)
        self.assertGreater(result['metrics']['rigid_pocket_risk_sqft'],0.)

    def test_gap_opened_behind_a_moved_neighbor_is_measured(self):
        original=box(3.9,1,4.9,3);shifted=shapely.affinity.translate(original,-.125,0)
        subject=box(2,1,3,3);roi=box(3.6,.8,5,3.2);config=dict(self.c,max_filler_regions=0)
        reference={'neighbor':original}
        before=assess({'subject':subject,'neighbor':original},'subject',reference,self.patio,roi,roi,{}, {},self.gap,config)
        after=assess({'subject':subject,'neighbor':shifted},'subject',reference,self.patio,roi,roi,{}, {},self.gap,config)
        self.assertGreater(after['metrics']['wall_excess_sqft'],before['metrics']['wall_excess_sqft']+.05)

    def test_new_lock_policy_rejects_old_pool_moves_and_replacements(self):
        config=dict(self.c,locked_stones=['62','120']);target=dict(old='62')
        self.assertFalse(respects_locks(dict(stone='91',pose=[0,0,0]),target,config))
        original=list(b.STONES['62']['pose'])
        self.assertTrue(respects_locks(dict(stone='62',pose=original),target,config))
        shifted=original.copy();shifted[0]+=.01
        self.assertFalse(respects_locks(dict(stone='62',pose=shifted),target,config))
        self.assertFalse(respects_locks(dict(stone='91',pose=[0,0,0],space_plan=dict(translations_ft={'120':[.01,0]})),dict(old=None),config))


if __name__=='__main__':unittest.main()
