"""Small synthetic checks for pre-fit measurements; no physics or UI."""
import unittest
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
import build_prefit_review as b

class PrefitGeometry(unittest.TestCase):
 def measure(self,g,p,free=None):
  free=p if free is None else free
  return b.measures(g,p,free,b.PATIO.difference(free))
 def test_exact_match(self):
  p=box(1,1,3,3);m=self.measure(p,p)
  self.assertAlmostEqual(m['coverage'],1)
  self.assertAlmostEqual(m['clearance_intrusion_sqft'],0)
 def test_low_overlap_does_not_establish_fit(self):
  p=box(1,1,3,3);g=Polygon([(1,1),(3,1),(1,3)])
  m=self.measure(g,p)
  self.assertEqual(m['physical_overlap_sqft'],0)
  self.assertAlmostEqual(m['coverage'],.5)
  self.assertAlmostEqual(m['unfilled_pocket_sqft'],2)
 def test_narrow_piece_leaves_directional_gap(self):
  p=box(1,1,3,3);m=self.measure(box(1,1,1.5,3),p)
  self.assertAlmostEqual(m['coverage'],.25)
  self.assertEqual(m['clearance_intrusion_sqft'],0)
 def test_small_bump_records_depth(self):
  p=box(1,1,3,3)
  g=unary_union([p,box(2.45,2.98,2.55,3.2)])
  m=self.measure(g,p)
  self.assertLess(m['clearance_intrusion_fraction'],.01)
  self.assertAlmostEqual(m['max_clearance_intrusion_inches'],2.4,places=6)
 def test_clearance_intrusion_is_separate_from_physical_overlap(self):
  p=box(1,1,3,3);free=p
  # Move the actual obstruction outward by the half-inch clearance.
  obstacles=b.PATIO.difference(p.buffer(b.GAP))
  g=box(1,1,3+.02,3);m=b.measures(g,p,free,obstacles)
  self.assertAlmostEqual(m['physical_overlap_sqft'],0)
  self.assertGreater(m['clearance_intrusion_sqft'],0)
 def test_wrong_orientation_changes_silhouette(self):
  p=Polygon([(1,1),(3,1),(3,2),(2,2),(2,3),(1,3)])
  from shapely.affinity import rotate
  good=self.measure(p,p);bad=self.measure(rotate(p,90,origin=(2,2)),p)
  self.assertGreater(good['coverage'],bad['coverage'])
  self.assertGreater(bad['clearance_intrusion_sqft'],0)

if __name__=='__main__':unittest.main()
