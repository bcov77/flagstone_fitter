"""Supplement: physics-only local agitation followed by a blind insertion search.
No deliberate poses are edited. Four nearest unlocked stones may move; all other
active stones are temporarily held. Original locks and settings are restored.
"""
import copy,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from experiment import *

def insertion_search(doc,cfg,seed):
 board=shapes(doc);patio=shapely.Polygon(doc['patio']);local={str(s['id']):shapely.Polygon(s['outline']) for s in doc['stones']};free,holes=pockets(patio,board);rng=np.random.default_rng(seed);pool=[]
 from shapely.affinity import scale
 for ti,pocket in enumerate(holes[:12]):
  for factor in [1.,cfg['probe_scale']]:
   if factor<1 and not cfg['scaled_min_pocket_sqft']<=pocket.area<=cfg['scaled_max_pocket_sqft']:continue
   probe_local={s:scale(g,factor,factor,origin='centroid') for s,g in local.items()}
   available=[s for s in local if s not in board]
   ranked=coarse(probe_local,available,pocket,free,{**cfg,'piece_budget':137,'coarse_per_target':200},rng)
   for _,sid,pose in ranked:
    full=world(local[sid],pose)
    if not patio.buffer(-GAP).buffer(1e-8).covers(full):continue
    if factor==1:
     target=dict(old=None,pocket=pocket,free=free)
     f=fit([dict(stone=sid,pose=pose)],target,local,board,patio,cfg)
     if f is None:continue
     protocol='held_release';family='ordinary';rank=[f['clearance_intrusion_fraction'],-f['coverage'],-local[sid].area]
    else:
     probe=world(probe_local[sid],pose);coverage=probe.intersection(pocket).area/pocket.area;intr=probe.difference(free).area/probe.area;overhang=probe.difference(pocket).area/probe.area
     if coverage<cfg['probe_min_coverage'] or intr>cfg['probe_max_intrusion_fraction'] or overhang>cfg['probe_max_overhang_fraction']:continue
     f=dict(coverage=coverage,clearance_intrusion_fraction=intr,outside_pocket_fraction=overhang);protocol='free_pulses';family='scaled';rank=[intr,-coverage,-local[sid].area]
    p=dict(family=family,physics_protocol=protocol,target=f'mobility_insert_{ti}',remove=[],placements=[dict(stone=sid,pose=pose)],fit=f,rank_key=rank)
    if factor<1:p['probe']=dict(scale=factor,coverage=coverage,intrusion_fraction=intr,overhang=overhang)
    if eligible(doc,p,cfg):pool.append(p)
 unique={canonical(p['placements']):p for p in pool}
 return sorted(unique.values(),key=lambda p:p['rank_key']),dict(pockets=[dict(area_sqft=p.area,center=list(p.centroid.coords[0])) for p in holes],smallest_unused_sqft=min((g.area for s,g in local.items() if s not in board),default=None))

def agitate(doc,cfg,seed):
 before=copy.deepcopy(doc);board=shapes(doc);m=read(HERE/f'run_{CURRENT_RUN:06d}/root_metrics.json')
 point=shapely.Point(m['gaps']['groups']['interior']['center'])
 oldmeta=copy.deepcopy(doc['physics']);locked={sid for sid,v in oldmeta['pieces'].items() if v.get('locked')}
 movable=sorted((s for s in board if s not in locked),key=lambda s:board[s].distance(point))[:4]
 roi=unary_union([board[s].buffer(.25) for s in movable]).intersection(shapely.Polygon(doc['patio']))
 meta=before['physics']
 for s in before['stones']:
  sid=str(s['id']);meta['pieces'][sid]['anchor']=s['pose'][:]
  if s['active']:meta['pieces'][sid]['locked']=sid not in movable
 meta['settings'].update(gap_inches=.5,piece_hardness=60,wall_hardness=80,angular_stiffness=2,translation_stiffness=2,**{k:cfg[k] for k in ['rotation_magnitude','rotation_force','rotation_frequency','translation_magnitude','translation_force','translation_frequency']})
 session=session_for(before,seed);pulse=[]
 for duration in cfg['pulse_seconds']:
  session.start();session.advance(duration);pulse.append(dict(seconds=duration,motion=motion(session)));session.stop()
 settled,trace=quiet(session,cfg);c=clearance(session.export());repair=not c['valid'];phases=[dict(name='local_free_pulses',pulses=pulse),dict(name='direct',settled=settled,trace=trace)]
 if repair:
  session.settings.update(gap_inches=cfg['repair_gap_inches'],rotation_magnitude=4,translation_magnitude=.25,rotation_force=12,translation_force=12,angular_stiffness=2,translation_stiffness=2)
  session._build();session._randomize_drives();session.running=True;session.advance(cfg['repair_inflate_seconds'])
  session.settings['gap_inches']=.5;session._build();session._randomize_drives();session.running=True;session.advance(cfg['repair_deflate_seconds']);settled,trace=quiet(session,cfg)
  phases.append(dict(name='gap_cycle',settled=settled,trace=trace))
 velocity=motion(session);end=session.export();c=clearance(end);session.stop();end['physics']=oldmeta
 for s in end['stones']:end['physics']['pieces'][str(s['id'])]['anchor']=s['pose'][:]
 return end,dict(accepted=bool(settled and c['valid']),settled=settled,clearance=c,final_motion=velocity,phases=phases,reason='valid_and_observed_settled' if settled and c['valid'] else 'clearance_failure_rolled_back' if not c['valid'] else 'unsettled_timeout_rolled_back',repair_attempted=repair,movable_ids=movable,focus=list(point.coords[0])),roi

def run(n):
 global CURRENT_RUN
 CURRENT_RUN=n
 cfg=config();folder=HERE/f'run_{n:06d}';doc=read(folder/'source.json');root=read(folder/'root_metrics.json')
 # Baseline geometry search uses the identical broader insertion-only routine.
 pool,diag=insertion_search(doc,cfg,20261118+n)
 write(folder/'mobility_baseline_search.json.gz',dict(candidates=pool,diagnostics=diag))
 for si,seed in enumerate(SEEDS):
  ident=f'local_agitation_s{si}';path=folder/(ident+'.json.gz')
  if path.exists():r=read(path)
  else:
   start=time.monotonic();end,result,roi=agitate(doc,cfg,seed);attempted=evaluate(end);retained=end if result['accepted'] else doc
   r=dict(id=ident,parent=None,seed=seed,action=None,operator='local_agitation',start=state(doc),attempted_end=state(end),end=state(retained),attempted_metrics=attempted,end_metrics=attempted if result['accepted'] else root,seconds=time.monotonic()-start,roi_wkt=roi.wkt,local_before=local_metrics(doc,roi),local_attempted=local_metrics(end,roi),local_end=local_metrics(retained,roi),**result)
   write(path,r)
  print(f'run {n} {ident}: accepted={r["accepted"]} interior={r["end_metrics"]["gaps"]["largest_interior_inches"]:.3f}',flush=True)
  if not r['accepted']:continue
  current=restore(doc,r['end']);pool,diag=insertion_search(current,cfg,seed+100)
  roi=shapely.from_wkt(r['roi_wkt'])
  local=[p for p in pool if world(shapely.Polygon(next(s['outline'] for s in doc['stones'] if str(s['id'])==p['placements'][0]['stone'])),p['placements'][0]['pose']).intersection(roi).area>.05]
  write(folder/f'mobility_search_s{si}.json.gz',dict(candidates=pool,local_candidates=local,diagnostics=diag))
  print(f'run {n} {ident}: {len(local)} local insertion hypotheses',flush=True)
  for ci,p in enumerate(local[:2]):trial(folder,f'follow_{ident}_{ci}',current,p,cfg,seed+1000+ci,parent=ident,roi=roi)
 return n

if __name__=='__main__':
 import multiprocessing
 paths=[Path(__file__)];m=dict(files={p.name:sha(p) for p in paths},physics_seeds=SEEDS,scope=__doc__)
 dest=HERE/'mobility_manifest.json'
 if dest.exists():assert read(dest)==m
 else:write(dest,m)
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as ex:
  for f in as_completed([ex.submit(run,n) for n in CASES]):print(f'Local mobility completed run {f.result()}',flush=True)
