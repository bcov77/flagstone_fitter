"""Read-only verification of pinned inputs, transactions and retained champions."""
from core import *
from store import preflight,load

def verify(out):
 out=Path(out);m=preflight(out);_,nodes,jobs,results=load(out);source=read(out/'source.json');cfg=m['config'];physics=attempts=accepted=empty=0
 for number,r in results.items():
  meta=jobs[number];parent=nodes[meta['parent']]
  if not r.get('physics_evaluated'):
   assert r['status'] in ['no_candidate','geometry_error'];empty+=1;continue
  attempts+=1;assert r['start']==parent['frame'];p=r['chosen'];doc=restore(source,r['start'])
  if p.get('operator')!='local_pulse':assert eligible(doc,p,cfg['physics'])
  else:
   assert all(not doc['physics']['pieces'].get(s,{}).get('locked') for s in p['movable_ids'])
   for sid in r['start']['poses']:
    if sid not in p['movable_ids']:assert all(abs(a-b)<1e-7 for a,b in zip(r['start']['poses'][sid],r['attempted_end']['poses'][sid]))
  if not r['accepted']:assert r['end']==r['start']
  else:
   accepted+=1;assert r['settled'] and r['clearance']['valid'] and r['end_metrics']['clearance']['valid']
   for t in r['phases'][-1]['trace'][-cfg['physics']['stable_samples']:]:
    assert t['clearance']['valid'] and t['motion']['max_speed_ips']<cfg['physics']['max_speed_ips'] and t['motion']['max_angular_speed_dps']<cfg['physics']['max_angular_speed_dps']
   assert len(r['phases'][-1]['trace'])>=cfg['physics']['stable_samples']
   assert r['end_metrics']['area_percent']>=parent['metrics']['area_percent']-1e-6
   before=set(r['start']['poses']);after=set(r['end']['poses'])
   expected=before if p.get('operator')=='local_pulse' else (before-set(p['remove']))|{x['stone'] for x in p['placements']}
   assert after==expected
   for sid,prop in doc['physics']['pieces'].items():
    if prop.get('locked') and sid in before:assert all(abs(a-b)<1e-7 for a,b in zip(r['start']['poses'][sid],r['end']['poses'][sid]))
  saved=read(out/'jobs'/f'job_{number:08d}'/'selected.json.gz');assert saved['chosen']==p and saved['physics_seed']==r['physics_seed']
 for folder in (out/'roots').glob('run_*'):
  for label in ['best_coverage','best_interior_gap','best_wall_fit']:
   path=folder/(label+'.json')
   if path.exists():assert clearance(read(path))['valid']
 value=dict(jobs=len(jobs),committed=len(results),pending=len(jobs)-len(results),physics_attempts=attempts,accepted=accepted,empty_searches=empty,source_code_config_and_library_pins=True,eligibility_inventory_rollbacks_locks_and_pre_stop_settling=True,exported_champions_full_outline_clearance=True,scope='Saved gap audits not independently recomputed. Run after a checkpointed stop for a stable snapshot.')
 write(out/'verification.json',value);return value
