"""Foreground, checkpointed, multi-root frontier search. User launches the long run."""
import argparse,fcntl,multiprocessing,signal,traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from core import *
from store import prepare,preflight,load,allocate,summary
from worker import perform


def init_worker():
 for sig in [signal.SIGINT,signal.SIGTERM,getattr(signal,'SIGHUP',signal.SIGTERM)]:signal.signal(sig,signal.SIG_IGN)


def exclusions(out,meta,jobs,cfg):
 counts={}
 for j in jobs.values():
  if j['number']==meta['number'] or j['parent_geometry']!=meta['parent_geometry']:continue
  folder=Path(out)/'jobs'/f'job_{j["number"]:08d}';selected=folder/'selected.json.gz'
  if selected.exists():
   p=read(selected)['chosen'];key=action_key(p);counts[key]=counts.get(key,0)+1
 return sorted(k for k,v in counts.items() if v>=cfg['repeats_per_action'])


def run(out,cfg,args):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 with (out/'.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('This output already has a search supervisor')
  prepare(out,cfg);m,nodes,jobs,results=load(out);cfg=m['config']
  if args.prepare:
   value=summary(out,nodes,jobs,results,cfg)
   from review import render
   render(out,nodes,jobs,results,value,export=True)
   print(f'Prepared {out}\n{len(m["seed_nodes"])} checked starting nodes across {len(cfg["roots"])} roots. No workers launched.\nReview: {out}/index.html',flush=True);return
  (out/'STOP').unlink(missing_ok=True);stopping=False;failed=False;deadline=time.monotonic()+args.hours*3600 if args.hours else None;last_print=0
  def stop(*_):
   nonlocal stopping
   if not stopping:print('\nStopping: search passes checkpoint; current physics/audit transactions finish. Please wait.',flush=True)
   stopping=True;(out/'STOP').touch()
  previous={sig:signal.signal(sig,stop) for sig in set([signal.SIGINT,signal.SIGTERM,getattr(signal,'SIGHUP',signal.SIGTERM)])}
  workers=args.workers;per_root=max(1,math.ceil(workers/len(cfg['roots'])));pending=sorted(set(jobs)-set(results));futures={};session_jobs=0;executed_before=sum(bool(r.get('physics_evaluated')) for r in results.values())
  print(f'Frontier search: {out}\n{workers} workers; roots {cfg["roots"]}; up to {per_root} active searches/root. No fixed placement limit.\nCtrl-C checkpoints. Viewer: {out}/index.html',flush=True)
  try:
   with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker) as executor:
    while True:
     if (out/'STOP').exists() or (deadline is not None and time.monotonic()>=deadline):stop()
     physics_done=sum(bool(r.get('physics_evaluated')) for r in results.values())
     while not stopping and len(futures)<workers:
      if args.max_evaluations is not None and physics_done+len(futures)>=args.max_evaluations:break
      active={r:sum(jobs[n]['root']==r for n in futures.values()) for r in cfg['roots']}
      candidates=[n for n in pending if active[jobs[n]['root']]<per_root]
      if candidates:
       number=candidates[0];pending.remove(number)
      else:
       if args.max_jobs is not None and len(jobs)>=args.max_jobs:break
       roots=[r for r in cfg['roots'] if active[r]<per_root]
       if not roots:break
       root=min(roots,key=lambda r:sum(j['root']==r for j in jobs.values()))
       meta=allocate(out,root,nodes,jobs,cfg,args.operator,exclusion_provider=lambda meta,previous:exclusions(out,meta,previous,cfg));number=meta['number']
      futures[executor.submit(perform,str(out),number,os.getpid(),args.select_only)]=number;session_jobs+=1
     if not futures:break
     done,_=wait(futures,timeout=1,return_when=FIRST_COMPLETED)
     changed=False
     for future in done:
      number=futures.pop(future)
      try:
       status=future.result()
       if status=='checkpointed':stop();continue
       r=read(out/'jobs'/f'job_{number:08d}'/'result.json.gz');results[number]=r;meta=jobs[number]
       if r.get('accepted') and r.get('metrics_usable'):
        n=dict(id=f'n{number:08d}',root=meta['root'],kind='search',depth=meta['depth'],sequence=number,parent=meta['parent'],frame=r['end'],metrics=r['end_metrics'],job=number);nodes[n['id']]=n;write(out/'nodes'/f'{n["id"]}.json.gz',n)
       print(f'Job {number} · root {meta["root"]} · {meta["operator"]}: '+('no proposal (no placement attempt)' if not r.get('physics_evaluated') else f'{"accepted" if r["accepted"] else "rollback"} · {r["end_metrics"]["area_percent"]:.3f}% · {r["reason"]}'),flush=True);changed=True
      except BaseException:
       failed=True;stop();print(f'Job {number} failed:\n{traceback.format_exc()}',flush=True)
     if changed or time.monotonic()-last_print>60:
      value=summary(out,nodes,jobs,results,cfg)
      try:
       from review import render
       render(out,nodes,jobs,results,value,export=changed)
      except Exception:write(out/'review_error.json',dict(traceback=traceback.format_exc()))
      print(' | '.join(f'Root {r["root"]}: {r["best"]:.3f}% · {r["physics_attempts"]} physics / {r["empty_searches"]} empty' for r in value['roots']),flush=True);last_print=time.monotonic()
   value=summary(out,nodes,jobs,results,cfg)
   from review import render
   render(out,nodes,jobs,results,value,export=True)
   if failed:raise RuntimeError('A worker failed; all workers have stopped. Inspect jobs/job_*/error.json before resuming.')
   print(f'All workers exited. {value["completed_jobs"]} committed search jobs; {len(value["pending_jobs"])} pending. Same command resumes.\nViewer: {out}/index.html',flush=True)
  finally:
   # ProcessPoolExecutor above waits for worker transactions before releasing the lock.
   for sig,handler in previous.items():signal.signal(sig,handler)


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=DEFAULT_OUT);ap.add_argument('--config',type=Path,default=HERE/'config.json');ap.add_argument('--workers',type=int,default=8);ap.add_argument('--hours',type=float);ap.add_argument('--max-jobs',type=int);ap.add_argument('--max-evaluations',type=int);ap.add_argument('--prepare',action='store_true');ap.add_argument('--status',action='store_true');ap.add_argument('--stop',action='store_true');ap.add_argument('--roots',type=int,nargs='+');ap.add_argument('--no-warm-start',action='store_true');ap.add_argument('--operator',choices=['insert','replace','reorient','pair_fill','neighbor_refill','pair_repack','triple_repack','local_pulse']);ap.add_argument('--select-only',action='store_true',help='Checkpoint selected proposals without running physics (validation)');ap.add_argument('--verify',action='store_true');args=ap.parse_args();out=args.out.resolve()
 if args.stop:
  if not (out/'manifest.json').exists():ap.error('Output is not prepared')
  (out/'STOP').touch();print('Stop requested. Search stages checkpoint; active physics/audit transactions finish.');return
 if args.status:
  if not (out/'manifest.json').exists():print('Not prepared; no workers launched.');return
  value=read(out/'summary.json') if (out/'summary.json').exists() else dict(state='initializing',roots=[])
  with (out/'.lock').open('a') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB);value['running']=False
   except BlockingIOError:value['running']=True
  value['progress']=[read(p) for p in (out/'jobs').glob('job_*/progress.json') if not (p.parent/'result.json.gz').exists()]
  print(json.dumps(value,indent=2));return
 if args.verify:
  from verify import verify
  print(json.dumps(verify(out),indent=2));return
 if args.workers<1 or any(v is not None and v<=0 for v in [args.hours,args.max_jobs,args.max_evaluations]):ap.error('Positive counts/durations required')
 if out==ROOT or any(out==p or p in out.parents or out in p.parents for p in [BATCH,PILOT,HERE]):ap.error('Use a separate output directory')
 cfg=config(args.config)
 if args.roots:cfg['roots']=args.roots
 if args.no_warm_start:cfg['warm_start']=False
 if not cfg['roots'] or set(cfg['roots'])-{62,105,70,76} or len(set(cfg['roots']))!=len(cfg['roots']):ap.error('Distinct researched roots required')
 run(out,cfg,args)

if __name__=='__main__':main()
