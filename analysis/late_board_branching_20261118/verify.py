"""Check committed trial invariants, acceptance observations and campaign pins."""
from experiment import HERE,CASES,read,write,manifest,sha,BATCH,restore,eligible,config,state,canonical,clearance

def main():
 assert read(HERE/'manifest.json')==manifest(),'Input or compute hashes changed'
 cfg=config();count=accepted=0;cases=[]
 for mf in ['mobility_manifest.json','repack_manifest.json','fine_insert_manifest.json']:
  if (HERE/mf).exists():
   for name,digest in read(HERE/mf)['files'].items():assert sha(HERE/name)==digest
 for n in CASES:
  folder=HERE/f'run_{n:06d}';source=read(folder/'source.json');saved={}
  assert source==read(BATCH/f'runs/run_{n:06d}/final.json'),'Snapshot differs from campaign root'
  for p in folder.glob('*.json.gz'):
   if 'pool' in p.name or 'portfolio' in p.name or p.name=='pareto_archive.json.gz':continue
   r=read(p)
   if 'accepted' in r and 'id' in r:saved[r['id']]=r
  assert read(folder/'summary.json')['complete']
  for si in range(2):assert f'local_agitation_s{si}' in saved
  for stem,portfolio in [('repack','repack_portfolio.json.gz'),('fine_insert','fine_insert_portfolio.json.gz')]:
   pool=read(folder/portfolio)['candidates']
   if stem=='repack':
    if pool:
     for si in range(2):assert f'repack_s{si}' in saved
   else:
    for ci in range(min(2,len(pool))):
     for si in range(2):assert f'fine_insert_c{ci}_s{si}' in saved
  for r in saved.values():
   start=restore(source,r['start']);end=restore(source,r['end']);p=r['action'];count+=1
   if p:assert eligible(start,p,cfg),(n,r['id'],'eligibility')
   if r.get('operator')=='local_agitation':
    for sid in r['start']['poses']:
     if sid not in r['movable_ids']:
      assert all(abs(a-b)<1e-7 for a,b in zip(r['start']['poses'][sid],r['attempted_end']['poses'][sid]))
   if r['parent'] is not None:assert r['start']==saved[r['parent']]['end']
   else:assert r['start']==state(source)
   if not r['accepted']:assert r['start']==r['end'],(n,r['id'],'rollback')
   else:
    accepted+=1
    assert r['settled'] and r['clearance']['valid'] and r['end_metrics']['clearance']['valid']
    trace=r['phases'][-1]['trace']
    assert len(trace)>=cfg['stable_samples']
    for t in trace[-cfg['stable_samples']:]:
     assert t['clearance']['valid']
     assert t['motion']['max_speed_ips']<cfg['max_speed_ips']
     assert t['motion']['max_angular_speed_dps']<cfg['max_angular_speed_dps']
    assert r['final_motion']['max_speed_ips']<cfg['max_speed_ips']
    assert r['final_motion']['max_angular_speed_dps']<cfg['max_angular_speed_dps']
    before=set(r['start']['poses']);after=set(r['end']['poses'])
    if p:assert after==(before-set(p['remove']))|{x['stone'] for x in p['placements']}
    else:assert before==after
    for sid,meta in start['physics']['pieces'].items():
     if meta.get('locked') and sid in before:
      assert all(abs(a-b)<1e-7 for a,b in zip(r['start']['poses'][sid],r['end']['poses'][sid]))
   if r.get('local_end') and not r['accepted']:assert r['local_end']==r['local_before']
  for label in ['best_coverage','best_interior_gap']:
   if (folder/(label+'.json')).exists():
    check=clearance(read(folder/(label+'.json')));assert check['valid'],(n,label,'actual full-outline clearance')
  cases.append(dict(run=n,trials=len(saved),complete=read(folder/'summary.json')['complete']))
 result=dict(trials=count,accepted=accepted,cases=cases,source_and_dependency_hashes_unchanged=True,eligibility_inventory_rollbacks_locks_and_settling_observations_pass=True,best_layouts_rechecked_full_outline_clearance=True,scope='Saved global gap audits were not independently recomputed; exact full-outline clearance rechecked for coverage and interior-gap champions.')
 write(HERE/'verification.json',result);print(result)
if __name__=='__main__':main()
