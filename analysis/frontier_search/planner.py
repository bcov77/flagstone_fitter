"""Bounded multi-pose FFT proposals; full original outlines enforce final gates."""
from core import *
from scipy import signal
from shapely.affinity import scale


def search_settings(cfg,phase):
 rung=min(phase//4,len(cfg['piece_budgets'])-1)
 return dict(raster_step_ft=cfg['grid_steps_ft'][phase%len(cfg['grid_steps_ft'])],angle_step_degrees=cfg['angle_steps_degrees'][(phase//len(cfg['grid_steps_ft']))%len(cfg['angle_steps_degrees'])],piece_budget=cfg['piece_budgets'][rung],target_budget=cfg['target_budgets'][rung],poses_per_angle=cfg['poses_per_angle'],poses_per_piece=cfg['poses_per_piece'],coarse_limit=48+48*rung,beam=cfg['geometry_beam'],rung=rung)


def coarse_poses(local,available,pocket,free,spec,rng,stopped,min_fraction=.20):
 step=spec['raster_step_ft'];x0,y0,x1,y1=pocket.bounds;pad=.12
 phase=rng.random(2)*step;angle_offset=float(rng.choice([0.,.5]))*spec['angle_step_degrees']
 xs=np.arange(x0-pad+phase[0],x1+pad+step,step);ys=np.arange(y0-pad+phase[1],y1+pad+step,step)
 pm=shapely.contains_xy(pocket,xs[None,:],ys[:,None]);fm=shapely.contains_xy(free,xs[None,:],ys[:,None]);costs=np.where(pm,-1.,1.)+np.where(fm,0.,22.)
 ids=[s for s in available if min_fraction*pocket.area<=local[s].area<=1.15*pocket.area]
 ids=sorted(ids,key=lambda s:abs(math.log(local[s].area/(pocket.area*.85))))
 budget=spec['piece_budget'];head=ids[:max(1,budget*2//3)];rest=ids[len(head):]
 if len(head)<min(len(ids),budget):head+=list(rng.choice(rest,min(len(ids),budget)-len(head),replace=False))
 result=[]
 for sid in head:
  checked(stopped);g=local[sid].simplify(.004,preserve_topology=True);candidates=[]
  for deg in np.arange(angle_offset,360+angle_offset,spec['angle_step_degrees']):
   a=math.radians(float(deg)%360);rg=world(g,[0,0,a]);a0,b0,a1,b1=rg.bounds
   xx=np.arange(a0+step/2,a1,step);yy=np.arange(b0+step/2,b1,step)
   if not len(xx) or not len(yy) or len(xx)>len(xs) or len(yy)>len(ys):continue
   kernel=shapely.contains_xy(rg,xx[None,:],yy[:,None]).astype(float)
   costs_pose=signal.fftconvolve(costs,kernel[::-1,::-1],mode='valid')*step**2
   for _ in range(spec['poses_per_angle']):
    iy,ix=np.unravel_index(np.argmin(costs_pose),costs_pose.shape);cost=float(costs_pose[iy,ix])
    if not math.isfinite(cost):break
    candidates.append((cost,sid,[float(xs[ix]-xx[0]),float(ys[iy]-yy[0]),a]))
    radius=max(1,round(.04/step));costs_pose[max(0,iy-radius):iy+radius+1,max(0,ix-radius):ix+radius+1]=np.inf
  # Keep several angles/positions for each inventory piece, not just one best pose.
  result.extend(sorted(candidates)[:spec['poses_per_piece']])
 return sorted(result)[:spec['coarse_limit']]


def insertion_targets(doc,board,patio,spec,metrics):
 free,holes=pockets(patio,board);targets=[]
 for i,hole in enumerate(holes):
  targets.append(dict(id=f'insert_{i}',remove=[],pocket=hole,free=free))
  grown=parts(hole.buffer(.08 if spec['rung']==0 else .16).intersection(free))
  if grown:
   patch=max(grown,key=lambda p:p.intersection(hole).area)
   if patch.area>hole.area*1.05:targets.append(dict(id=f'insert_patch_{i}',remove=[],pocket=patch,free=free))
 # Local safe-space patches at saved gap witnesses also reach narrow connections
 # discarded by pocket erosion. This changes target discovery, never fit gates.
 for kind in ['interior','edge']:
  group=metrics.get('gaps',{}).get('groups',{}).get(kind,{})
  if group.get('center'):
   point=shapely.Point(group['center']);patches=parts(free.intersection(point.buffer(.5+.15*spec['rung'])))
   if patches:
    patch=max(patches,key=lambda p:p.area)
    if patch.area>.08:targets.append(dict(id=f'witness_{kind}',remove=[],pocket=patch,free=free))
 # Interleave witnesses with ordinary pockets, without starving the natural holes.
 witnesses=[t for t in targets if t['id'].startswith('witness')];normal=[t for t in targets if not t['id'].startswith('witness')]
 return (normal[:2]+witnesses+normal[2:])[:spec['target_budget']]


def replacement_targets(doc,board,patio,spec,rng,metrics,size):
 locked={s for s,v in doc.get('physics',{}).get('pieces',{}).items() if v.get('locked')};joints=joint_summary(board,patio)
 center=metrics.get('gaps',{}).get('groups',{}).get('interior',{}).get('center',list(patio.centroid.coords[0]));point=shapely.Point(center)
 ids=[s for s in board if s not in locked]
 ordered=sorted(ids,key=lambda s:(joints['by_id'][s]['wall']>=6,board[s].distance(point),joints['by_id'][s]['matched_fraction']))
 # Sample extra neighborhoods so a fixed worst gap cannot monopolize the search.
 shuffled=list(rng.permutation(ids));ordered=list(dict.fromkeys(ordered[:max(4,spec['target_budget']//2)]+shuffled))
 clusters=[]
 if size==1:clusters=[[s] for s in ordered[:spec['target_budget']]]
 else:
  for a in ordered:
   near=sorted((b for b in ids if b!=a and board[b].distance(board[a])<.4),key=lambda b:board[b].distance(board[a])+board[b].distance(point)*.15)
   for b in near[:2]:
    chosen=[a,b]
    if size==3:
     union=unary_union([board[a],board[b]])
     third=sorted((c for c in ids if c not in chosen and board[c].distance(union)<.4),key=lambda c:board[c].distance(union)+board[c].distance(point)*.15)
     if not third:continue
     chosen.append(third[0])
    chosen=sorted(chosen)
    if chosen not in clusters:clusters.append(chosen)
   if len(clusters)>=spec['target_budget']:break
 targets=[]
 for removed in clusters[:spec['target_budget']]:
  others={s:g for s,g in board.items() if s not in removed};free,holes=pockets(patio,others)
  relevant=[p for p in holes if all(p.intersection(board[s]).area>.25*board[s].area for s in removed)]
  if not relevant:
   region=unary_union([board[s] for s in removed]).buffer(.25);relevant=parts(region.intersection(free))
   relevant=[p for p in relevant if all(p.intersection(board[s]).area>.25*board[s].area for s in removed)]
  if not relevant:continue
  pocket=max(relevant,key=lambda p:sum(p.intersection(board[s]).area for s in removed))
  targets.append(dict(id='replace_'+'_'.join(removed),remove=removed,pocket=pocket,free=free))
 return targets


def proposal(doc,t,placements,local,others,patio,physics,factor=1.):
 removed=t['remove'];gs=[world(local[x['stone']],x['pose']) for x in placements]
 if factor==1:
  f=fit(placements,dict(old=None,pocket=t['pocket'],free=t['free']),local,others,patio,physics)
  if f is None:return None
  protocol='held_release';probe=None
 else:
  if len(placements)!=1:return None
  if not physics['scaled_min_pocket_sqft']<=t['pocket'].area<=physics['scaled_max_pocket_sqft']:return None
  if removed:
   original=shapes(doc);original_free=patio.buffer(-GAP).difference(unary_union(list(original.values())).buffer(GAP))
   if t['pocket'].intersection(original_free).area<.08:return None
  if not patio.buffer(-GAP).buffer(1e-8).covers(gs[0]):return None
  p=placements[0];probe_poly=world(scale(local[p['stone']],factor,factor,origin='centroid'),p['pose']);coverage=probe_poly.intersection(t['pocket']).area/t['pocket'].area;intr=probe_poly.difference(t['free']).area/probe_poly.area;overhang=probe_poly.difference(t['pocket']).area/probe_poly.area
  if coverage<physics['probe_min_coverage'] or intr>physics['probe_max_intrusion_fraction'] or overhang>physics['probe_max_overhang_fraction']:return None
  probe=dict(scale=factor,coverage=coverage,intrusion_fraction=intr,overhang=overhang);f=dict(coverage=coverage,clearance_intrusion_fraction=intr,outside_pocket_fraction=overhang);protocol='free_pulses'
 gain=sum(local[x['stone']].area for x in placements)-sum(local[s].area for s in removed)
 p=dict(operator='placement',family='compound' if len(removed)>1 or len(placements)>1 else 'scaled' if factor<1 else 'ordinary',physics_protocol=protocol,target=t['id'],remove=removed,placements=placements,fit=f,area_gain_sqft=gain,rank_key=[f['clearance_intrusion_fraction'],-f['coverage'],-gain])
 if probe:p['probe']=probe
 return p if eligible(doc,p,physics) else None


def generate(doc,metrics,cfg,operator,phase,rng,stopped):
 spec=search_settings(cfg,phase);physics=cfg['physics'];board=shapes(doc);patio=shapely.Polygon(doc['patio']);local={str(s['id']):shapely.Polygon(s['outline']) for s in doc['stones']};pool=[];logs=[]
 if operator=='local_pulse':
  center=metrics.get('gaps',{}).get('groups',{}).get('interior',{}).get('center',list(patio.centroid.coords[0]));point=shapely.Point(center);locked={s for s,v in doc['physics']['pieces'].items() if v.get('locked')};count=[4,6,8,12][phase%4];ids=sorted((s for s in board if s not in locked),key=lambda s:board[s].distance(point))[:count]
  return [dict(operator='local_pulse',movable_ids=ids,focus=center,rank_key=[0])],dict(search=spec,operator=operator)
 if operator in ['insert','pair_fill']:targets=insertion_targets(doc,board,patio,spec,metrics)
 else:targets=replacement_targets(doc,board,patio,spec,rng,metrics,2 if operator=='pair_repack' else 3 if operator=='triple_repack' else 1)
 for t in targets:
  checked(stopped);removed=t['remove'];others={s:g for s,g in board.items() if s not in removed};available=[s for s in local if s not in board or s in removed]
  before=len(pool)
  if operator in ['insert','replace','reorient']:
   if operator=='reorient':available=removed
   elif operator=='replace':available=[s for s in available if s not in removed]
   for factor in [1.,physics['probe_scale']] if operator!='reorient' else [1.]:
    if factor<1 and not physics['scaled_min_pocket_sqft']<=t['pocket'].area<=physics['scaled_max_pocket_sqft']:continue
    probes={s:scale(g,factor,factor,origin='centroid') for s,g in local.items()} if factor<1 else local
    ranked=coarse_poses(probes,available,t['pocket'],t['free'],spec,rng,stopped)
    for _,sid,pose in ranked:
     checked(stopped);p=proposal(doc,t,[dict(stone=sid,pose=pose)],local,others,patio,physics,factor)
     if p:pool.append(p)
  else:
   # Joint full-size geometry beam: actual selected pieces never overlap each other.
   count=3 if operator=='triple_repack' else 2;beam=[([],t['pocket'],t['free'])]
   for depth in range(count):
    children=[]
    for placements,residual,free in beam:
     checked(stopped)
     if operator=='neighbor_refill' and depth==0:ids=removed
     else:ids=[s for s in available if s not in board and s not in {x['stone'] for x in placements}]
     chunks=sorted(parts(residual.buffer(-.03).buffer(.03).intersection(residual)),key=lambda p:-p.area)
     for chunk in chunks[:2]:
      ranked=coarse_poses(local,ids,chunk,free,spec,rng,stopped,min_fraction=.15)
      for _,sid,pose in ranked[:12]:
       chosen=placements+[dict(stone=sid,pose=pose)]
       if operator=='neighbor_refill' and depth==0 and not eligible(doc,dict(remove=removed,placements=chosen),physics):continue
       seed_fit=fit(chosen,dict(old=None,pocket=t['pocket'],free=t['free']),local,others,patio,physics,min_coverage=.15)
       if seed_fit is None:continue
       full=unary_union([world(local[x['stone']],x['pose']) for x in chosen]);rest=t['pocket'].difference(full.buffer(GAP));free2=t['free'].difference(full.buffer(GAP))
       children.append(([-seed_fit['coverage'],seed_fit['clearance_intrusion_fraction']],chosen,rest,free2))
    children.sort(key=lambda x:x[0]);beam=[(p,r,f) for _,p,r,f in children[:spec['beam']]]
    if not beam:break
   for chosen,_,_ in beam:
    if len(chosen)!=count:continue
    p=proposal(doc,t,chosen,local,others,patio,physics)
    if p:pool.append(p)
  logs.append(dict(target=t['id'],remove=removed,pocket_sqft=t['pocket'].area,eligible=len(pool)-before))
 unique={action_key(p):p for p in sorted(pool,key=lambda p:p['rank_key'],reverse=True)}
 return sorted(unique.values(),key=lambda p:p['rank_key']),dict(search=spec,operator=operator,targets=logs,scope='Broader multi-pose proposal search; original full-size final fit, eligibility and physics gates unchanged')
