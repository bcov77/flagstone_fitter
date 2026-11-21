"""Independent geometry and experiment-control checks; no simulation side effects."""
import copy,json,unittest
from pathlib import Path
from pilot import DEFAULT,prepare,clearance,DOC
from patio_physics.engine import PhysicsSession
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.manifest=prepare(DEFAULT);cls.doc=json.loads((DEFAULT/'source.json').read_text())
 def session(self,changes=None):
  d=copy.deepcopy(self.doc)
  if changes:changes(d)
  active={str(s['id']):s['active'] for s in d['stones']}
  for s in d['stones']:s['active']=False
  session=PhysicsSession(d)
  for sid,p in session.pieces.items():p.active=active[sid]
  return session
 def test_valid_baseline_and_easy_insertion(self):
  for removed in [[],['50','61'],['50','60','61']]:
   session=self.session(lambda d:[s.update(active=False) for s in d['stones'] if str(s['id']) in removed])
   self.assertTrue(clearance(session)['valid'])
   self.assertTrue(session.clearance_report()['passes_clearance'])
 def test_checker_detects_pair_and_wall_damage(self):
  session=self.session();session.pieces['60'].pose=session.pieces['61'].pose[:]
  self.assertGreater(clearance(session)['pair_violations'],0)
  session.pieces['60'].pose=[-2,-2,0]
  ours=clearance(session);engine=session.clearance_report()
  self.assertGreater(ours['wall_violations'],0)
  self.assertEqual(ours['pair_violations'],sum(e['kind']=='pair' for e in engine['errors']))
  self.assertEqual(ours['wall_violations'],sum(e['kind']=='boundary' for e in engine['errors']))
 def test_matched_controls_and_balance(self):
  m=self.manifest;self.assertEqual(len(m['tasks']),96);self.assertEqual(len({t['id'] for t in m['tasks']}),96)
  names={c['name'] for c in m['cases']}
  for case in m['cases']:self.assertIn(case['control'],names)
  for name in names:
   subset=[t for t in m['tasks'] if t['case']['name']==name]
   self.assertEqual(len(subset),12)
  original=next(c for c in m['cases'] if c['name']=='replace_original')
  originalpose=next(s['pose'] for s in self.doc['stones'] if str(s['id'])=='61')
  self.assertEqual(original['candidate']['pose'],originalpose)
if __name__=='__main__':unittest.main()
