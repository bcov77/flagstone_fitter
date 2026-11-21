"""Recheck every saved endpoint using the production original-outline checker."""
import copy,json,hashlib
from pathlib import Path
from patio_physics.engine import PhysicsSession
OUT=Path(__file__).with_name('pilot_v1');m=json.loads((OUT/'manifest.json').read_text());d=json.loads((OUT/'source.json').read_text())
for s in d['stones']:s['active']=False
session=PhysicsSession(d);count=0
for path in sorted((OUT/'results').glob('*.json')):
 r=json.loads(path.read_text());active=r['final_poses']
 for sid,p in session.pieces.items():
  p.active=sid in active
  if p.active:p.pose=active[sid]
 engine=session.clearance_report()
 assert engine['passes_clearance']==r['after']['valid'],path.name
 assert sum(x['kind']=='pair' for x in engine['errors'])==r['after']['pair_violations'],path.name
 assert sum(x['kind']=='boundary' for x in engine['errors'])==r['after']['wall_violations'],path.name
 count+=1
assert count==len(m['tasks'])
assert hashlib.sha256(Path(m['source']).read_bytes()).hexdigest()==m['source_sha256']
engine_path=Path(__file__).resolve().parents[2]/'physics_patio/patio_physics/engine.py'
assert hashlib.sha256(engine_path.read_bytes()).hexdigest()==m['engine_sha256']
(OUT/'audit.json').write_text(json.dumps({'endpoints_checked':count,'engine_checker_agrees':True,'source_unchanged':True,'engine_unchanged':True,'local_maximum_gap_rule_checked':False},indent=2))
print('Independent engine-checker audit passed for',count,'endpoints; source and engine hashes unchanged.')
