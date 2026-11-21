"""One immutable search/selection/physics transaction; no shared frontier writes."""
from core import *
from planner import generate

def perform(out,number,parent_pid,select_only=False):
 out=Path(out);folder=out/'jobs'/f'job_{number:08d}';meta=read(folder/'job.json');cfg=read(out/'manifest.json')['config'];parent=read(out/'nodes'/f'{meta["parent"]}.json.gz');source=read(out/'source.json');doc=restore(source,parent['frame']);start=time.monotonic()
 def stopped():return (out/'STOP').exists() or os.getppid()!=parent_pid
 def progress(phase):write(folder/'progress.json',dict(phase=phase,number=number,root=meta['root'],updated_unix=time.time()))
 if (folder/'result.json.gz').exists():return 'complete'
 try:
  checked(stopped);selected_path=folder/'selected.json.gz'
  if not selected_path.exists():
   progress('searching');rng=np.random.default_rng(meta['seed'])
   try:pool,diagnostics=generate(doc,parent['metrics'],cfg,meta['operator'],meta['phase'],rng,stopped)
   except shapely.errors.GEOSException as exc:
    import traceback
    write(folder/'result.json.gz',dict(status='geometry_error',physics_evaluated=False,accepted=False,diagnostics=dict(error=str(exc),traceback=traceback.format_exc()),seconds=time.monotonic()-start));progress('geometry_error');return 'complete'
   # Exclusions are snapshotted at allocation, so completion order cannot change
   # a pending job's proposal choice on resume.
   excluded=set(meta.get('excluded',[]));pool=[p for p in pool if action_key(p) not in excluded]
   if not pool:
    write(folder/'result.json.gz',dict(status='no_candidate',physics_evaluated=False,accepted=False,diagnostics=diagnostics,seconds=time.monotonic()-start));progress('no_candidate');return 'complete'
   weights=np.array([.7/min(3,len(pool)) if i<3 else .3/(len(pool)-3) for i in range(len(pool))]) if len(pool)>3 else np.ones(len(pool))/len(pool)
   weights/=weights.sum();index=int(rng.choice(len(pool),p=weights));selected=dict(chosen=pool[index],probability=float(weights[index]),index=index,probabilities=weights.tolist(),rng_state=rng.bit_generator.state,physics_seed=seed_for(meta['seed'],'physics'),diagnostics=diagnostics)
   write(folder/'proposals.json.gz',pool);write(selected_path,selected)
  if select_only:progress('selected_checkpoint');return 'checkpointed'
  checked(stopped);selected=read(selected_path);p=selected['chosen'];progress('physics')
  if p.get('operator')=='local_pulse':r=local_pulse(doc,p,cfg['physics'],selected['physics_seed'])
  else:
   if not eligible(doc,p,cfg['physics']):raise ValueError('Selected proposal no longer eligible')
   r=relax(doc,p,cfg['physics'],selected['physics_seed'])
  attempted=r.pop('attempted_document');progress('auditing_endpoint');metrics=evaluate(attempted)
  if r['accepted'] and not metrics['clearance']['valid']:raise RuntimeError('Independent endpoint clearance disagrees with physical acceptance')
  retained=attempted if r['accepted'] else doc
  usable=bool(metrics.get('gaps',{}).get('stable') and metrics.get('quality') and all(math.isfinite(v) for v in axes(metrics)))
  result=dict(status='physics_complete',physics_evaluated=True,chosen=p,start=parent['frame'],proposed_start=r.pop('proposed_start'),attempted_end=state(attempted),attempted_metrics=metrics,end=state(retained),end_metrics=metrics if r['accepted'] else parent['metrics'],metrics_usable=usable,seconds=time.monotonic()-start,**r)
  write(folder/'result.json.gz',result);progress('committed');return 'complete'
 except SearchInterrupted:
  progress('checkpointed');return 'checkpointed'
 except BaseException:
  import traceback
  write(folder/'error.json',dict(traceback=traceback.format_exc(),updated_unix=time.time()));raise
