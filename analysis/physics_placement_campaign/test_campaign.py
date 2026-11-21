"""Geometry, candidate controls, isolation and checkpoint-integrity tests."""
import copy,json,tempfile,unittest,subprocess,sys,fcntl,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import campaign as c
import pilot
from shapely.geometry import box

class CampaignChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=json.loads(pilot.SOURCE.read_text())
    def test_subset_and_original_replacement_preserve_source(self):
        untouched=copy.deepcopy(self.source)
        context={'removed':['50','60']}
        old=next(s for s in self.source['stones'] if str(s['id'])=='61')
        case={'family':'replacement','old_id':'61','candidate':{'id':'61','pose':old['pose']}}
        document=c.make_document(self.source,context,case)
        self.assertEqual(self.source,untouched)
        self.assertTrue(pilot.clearance(c.context_session(document))['valid'])
        self.assertEqual(document['physics']['pieces']['61']['anchor'],old['pose'])
        ids={str(s['id']) for s in document['stones'] if s['active']}
        self.assertNotIn('50',ids);self.assertNotIn('60',ids);self.assertIn('61',ids)
    def test_circular_matching_runs(self):
        self.assertEqual(c.longest_run([1,0,0,1],.25),.5)
        self.assertEqual(c.longest_run([1]*4,.25),1)
        self.assertEqual(c.longest_run([0]*4,.25),0)
    def test_gap_sites_separated_and_not_in_stones(self):
        patio=box(0,0,8,8);stone=box(3,3,5,5)
        sites=c.gap_sites(patio,[stone],3)
        self.assertEqual(len(sites),3)
        for site in sites:
            self.assertGreater(site['radius_ft'],0)
            self.assertTrue(patio.contains(c.shapely.Point(site['center'])))
            self.assertFalse(stone.contains(c.shapely.Point(site['center'])))
    def test_sampling_retains_random_and_stress(self):
        pool=[{'edge_score':i,'screen_intrusion_fraction':i/100} for i in range(96)]
        choices=c.select_proposals(pool,'insertion',c.random.Random(1))
        self.assertEqual(len(choices),20)
        self.assertEqual(len({i for i,_,_ in choices}),20)
        self.assertEqual(sum(kind=='stress' for _,kind,_ in choices),6)
        self.assertEqual(sum(kind=='random' for _,kind,_ in choices),7)
        self.assertTrue(any(i>85 for i,kind,_ in choices if kind=='stress'))
    def test_checkpoint_identity_rejected_and_readiness_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'results').mkdir()
            manifest={'tasks':[{'id':'abc'}],'manifest_hash':'frozen'}
            pilot.write(out/'results/abc.json',{'status':'complete','manifest_hash':'wrong','task':{'id':'abc'}})
            with self.assertRaises(ValueError):c.completed_ids(out,manifest)
            with self.assertRaises(ValueError):c.execute(out,manifest,'all',1,1)
            pilot.write(out/'results/abc.json',{'status':'complete','manifest_hash':'frozen','task':{'id':'abc'}})
            self.assertEqual(c.completed_ids(out,manifest),{'abc'})
    def test_cached_neighborhood_matches_direct_features(self):
        document={'units':'ft','patio':[[0,0],[8,0],[8,8],[0,8]],'stones':[
            {'id':'a','outline':[[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]],'pose':[2,3,0],'active':True},
            {'id':'b','outline':[[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]],'pose':[3.05,3,0],'active':True}]}
        case={'candidate':{'id':'b','pose':[3.05,3,0]},'center':[3.05,3]}
        neighbors=c.feature_neighborhood(box(0,0,8,8),[pilot.world(document['stones'][0])],case['center'])
        cached=c.features(document,case,.5,neighbors)
        direct=c.features(document,case,.5)
        self.assertEqual(cached,direct)
        self.assertGreater(cached['scale_profiles']['1.0']['longest_near_run_ft'],0)

    def test_launcher_excludes_duplicate_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with (out/'.campaign.lock').open('a+') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                result=subprocess.run([sys.executable,str(Path(c.__file__).with_name('launch_campaign.py')),
                                       '--out',str(out),'--prepare-only'],capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,2)
                self.assertIn('already has a controller',result.stderr)
                self.assertFalse((out/'manifest.json').exists())

    def test_budget_stops_new_dispatch_and_drains_only_workers(self):
        def pool_factory(max_workers,**kwargs):return ThreadPoolExecutor(max_workers=max_workers)
        def fake_trial(out,task):
            time.sleep(.03)
            return {'id':task['id']}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);tasks=[{'id':str(i)} for i in range(10)]
            with patch.object(c,'ProcessPoolExecutor',side_effect=pool_factory),patch.object(c,'run_one',side_effect=fake_trial):
                c.run_tasks(out,{},tasks,{str(i) for i in range(10)},set(),2,.01,None)
            record=json.loads(next((out/'runs').glob('*.json')).read_text())
            self.assertGreater(record['completed_new'],0)
            self.assertLessEqual(record['completed_new'],2)
            self.assertLess(record['completed_new'],record['scheduled'])

    def test_manifest_tampering_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);m={'code_hashes':c.code_hashes(),'config':{},'source_sha256':'x'}
            m['manifest_hash']=c.signature(m);m['config']['changed']=True
            pilot.write(out/'manifest.json',m)
            with self.assertRaisesRegex(ValueError,'Manifest content'):c.load_manifest(out)

if __name__=='__main__':unittest.main()
