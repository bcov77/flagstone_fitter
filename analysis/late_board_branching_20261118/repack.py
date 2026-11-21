"""Supplement: two-for-two local repacking around an interior gap.
Only unused inventory pieces are proposed; removed full-size area is preserved.
Pair geometry uses the ordinary fit gates, with original neighbors outside the
removed pair and no mutual selected-piece overlap. Two seeds per top pair.
"""
from experiment import *
from local_mobility import insertion_search

def proposals(doc,cfg,seed):
 rng=np.random.default_rng(seed);board=shapes(doc);patio=shapely.Polygon(doc['patio']);local={str(s['id']):shapely.Polygon(s['outline']) for s in doc['stones']};locked={s for s,v in doc['physics']['pieces'].items() if v.get('locked')}
 from aggressive_geometry import joint_summary
 joints=joint_summary(board,patio)
 m=evaluate(doc);point=shapely.Point(m['gaps']['groups']['interior']['center'])
 ids=sorted((s for s in board if s not in locked),key=lambda s:(joints['by_id'][s]['wall']>=6,board[s].distance(point)))[:8]
 pairs=[]
 for i,a in enumerate(ids):
  for b in ids[i+1:]:
   if board[a].distance(board[b])>.25:continue
   pairs.append((board[a].distance(point)+board[b].distance(point),a,b))
 pairs.sort();result=[];diag=[]
 search_cfg={**cfg,'piece_budget':96,'coarse_per_target':200}
 for _,a,b in pairs[:3]:
  others={s:g for s,g in board.items() if s not in [a,b]};free,holes=pockets(patio,others)
  relevant=[h for h in holes if h.intersection(board[a]).area>.25*board[a].area and h.intersection(board[b]).area>.25*board[b].area]
  if not relevant:diag.append(dict(remove=[a,b],reason='No combined detected pocket'));continue
  pocket=max(relevant,key=lambda h:h.intersection(board[a]).area+h.intersection(board[b]).area)
  target=dict(old=None,pocket=pocket,free=free);available=[s for s in local if s not in board];count=0
  ranked=coarse(local,available,pocket,free,search_cfg,rng,min_fraction=.20)
  for index,(_,sid,pose) in enumerate(ranked[:10]):
   if index<3:
    def objective(v):
     g=world(local[sid].simplify(.004,preserve_topology=True),v)
     return -g.intersection(pocket).area+g.difference(pocket).area+30*g.difference(free).area
    pose=minimize(objective,pose,method='Powell',bounds=[(pose[0]-.09,pose[0]+.09),(pose[1]-.09,pose[1]+.09),(pose[2]-.10,pose[2]+.10)],options={'maxiter':10,'xtol':.001,'ftol':.001}).x.tolist()
   first=[dict(stone=sid,pose=pose)]
   if fit(first,target,local,others,patio,cfg,min_coverage=.20) is None:continue
   fg=world(local[sid],pose);residual=pocket.difference(fg.buffer(GAP));chunks=sorted(parts(residual.buffer(-.04).buffer(.04).intersection(residual)),key=lambda h:-h.area)
   if not chunks:continue
   free2=free.difference(fg.buffer(GAP))
   for _,second,q in coarse(local,[s for s in available if s!=sid],chunks[0],free2,search_cfg,rng,min_fraction=.20)[:4]:
    placements=first+[dict(stone=second,pose=q)];f=fit(placements,target,local,others,patio,cfg)
    if f is None:continue
    gain=local[sid].area+local[second].area-local[a].area-local[b].area
    p=dict(family='compound',physics_protocol='held_release',target=f'repack_{a}_{b}',remove=[a,b],placements=placements,fit=f,area_gain_sqft=gain,rank_key=[f['clearance_intrusion_fraction'],-f['coverage'],-gain],research_operator='two_for_two_repack')
    if eligible(doc,p,cfg):result.append(p);count+=1
  diag.append(dict(remove=[a,b],pocket_sqft=pocket.area,removed_sqft=local[a].area+local[b].area,first_poses=len(ranked),eligible_pairs=count))
 return sorted(result,key=lambda p:p['rank_key']),diag

def run(n):
 folder=HERE/f'run_{n:06d}';cfg=config();doc=read(folder/'source.json');path=folder/'repack_portfolio.json.gz'
 if path.exists():pool=read(path)['candidates']
 else:
  start=time.monotonic();pool,diag=proposals(doc,cfg,20261118+n);write(path,dict(candidates=pool,diagnostics=diag,seconds=time.monotonic()-start))
 print(f'run {n} repack: {len(pool)} eligible pair hypotheses',flush=True)
 for si,seed in enumerate(SEEDS):
  if not pool:break
  p=pool[0];roi=roi_for(doc,p);r=trial(folder,f'repack_s{si}',doc,p,cfg,seed,roi=roi)
  if r['accepted']:
   current=restore(doc,r['end']);choices,diag=insertion_search(current,cfg,seed+200)
   local=[p for p in choices if roi_for(current,p).intersection(roi).area>.05]
   write(folder/f'repack_follow_search_s{si}.json.gz',dict(candidates=local,diagnostics=diag))
   for ci,p in enumerate(local[:2]):trial(folder,f'follow_repack_s{si}_{ci}',current,p,cfg,seed+2000+ci,parent=r['id'],roi=roi)
 return n

if __name__=='__main__':
 import multiprocessing
 m=dict(files={Path(__file__).name:sha(Path(__file__))},seeds=SEEDS,scope=__doc__);dest=HERE/'repack_manifest.json'
 if dest.exists():assert read(dest)==m
 else:write(dest,m)
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as ex:
  for f in as_completed([ex.submit(run,n) for n in CASES]):print(f'Repack completed run {f.result()}',flush=True)
