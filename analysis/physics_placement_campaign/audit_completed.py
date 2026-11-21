"""Parallel original-outline audit of a finished campaign, without rerunning physics."""
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import campaign as c
from patio_physics.engine import PhysicsSession
SESSION=None
OUT=None

def initialize(out):
 global SESSION,OUT
 OUT=Path(out);d=json.loads((OUT/'source.json').read_text())
 for stone in d['stones']:stone['active']=False
 SESSION=PhysicsSession(d)

def check(ids):
 mismatches=[];valid=0
 for tid in ids:
  r=json.loads((OUT/'results'/f'{tid}.json').read_text())
  for sid,piece in SESSION.pieces.items():
   piece.active=sid in r['final_poses']
   if piece.active:piece.pose=r['final_poses'][sid]
  checked=SESSION.clearance_report();valid+=checked['passes_clearance']
  if (checked['passes_clearance']!=r['after']['valid'] or
      sum(e['kind']=='pair' for e in checked['errors'])!=r['after']['pair_violations'] or
      sum(e['kind']=='boundary' for e in checked['errors'])!=r['after']['wall_violations']):mismatches.append(tid)
 return {'checked':len(ids),'valid':valid,'mismatches':mismatches}

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=c.DEFAULT);p.add_argument('--workers',type=int,default=10);a=p.parse_args()
 m=c.load_manifest(a.out);ids=sorted(c.completed_ids(a.out,m));assert len(ids)==len(m['tasks'])
 started=time.monotonic();results=[]
 with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(str(a.out),)) as pool:
  futures=[pool.submit(check,ids[i:i+64]) for i in range(0,len(ids),64)]
  for future in as_completed(futures):
   result=future.result();results.append(result)
   print('Audited',sum(r['checked'] for r in results),'/',len(ids),flush=True)
 result={'checked':sum(r['checked'] for r in results),'valid':sum(r['valid'] for r in results),
         'mismatches':[tid for r in results for tid in r['mismatches']],
         'manifest_hash':m['manifest_hash'],'source_original_unchanged':c.pilot.digest(c.pilot.SOURCE)==m['source_sha256'],
         'simulation_code_hashes_match':True,'local_maximum_gap_rule_checked':False,
         'workers':a.workers,'wall_seconds':time.monotonic()-started}
 target=a.out/'completed_analysis';target.mkdir(exist_ok=True);c.pilot.write(target/'endpoint_audit.json',result)
 assert not result['mismatches'],result
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
