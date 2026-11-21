"""Prepared source snapshots, immutable nodes/jobs, and derived frontier summaries."""
from core import *

def prepare(out,cfg):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);manifest_path=out/'manifest.json'
 if manifest_path.exists():preflight(out,cfg);return read(manifest_path)
 marker=out/'preparing.json'
 initial_paths=[BATCH/'source.json']+[BATCH/f'runs/run_{n:06d}/{name}' for n in cfg['roots'] for name in ['final.json','summary.json']]
 if cfg['warm_start']:initial_paths += [PILOT/f'run_{n:06d}/pareto_archive.json.gz' for n in cfg['roots']]
 plan=dict(config=cfg,dependencies=dependencies(),inputs={str(p):sha(p) for p in initial_paths})
 if marker.exists():
  if read(marker)!=plan:raise RuntimeError('Interrupted preparation inputs/code changed; use a new output')
 elif any(p.name not in ['.lock'] for p in out.iterdir()):raise ValueError('Use an empty output directory or an existing prepared search')
 else:write(marker,plan)
 base=read(BATCH/'source.json');raw=(BATCH/'source.json').read_bytes();(out/'source.json').write_bytes(raw);inputs={str(BATCH/'source.json'):sha(BATCH/'source.json')};seeds=[]
 geometry=canonical([(str(s['id']),s['outline']) for s in base['stones']])
 for number in cfg['roots']:
  source_path=BATCH/f'runs/run_{number:06d}/final.json';summary_path=source_path.parent/'summary.json';doc=read(source_path);metrics=read(summary_path)['end_metrics']
  if canonical([(str(s['id']),s['outline']) for s in doc['stones']])!=geometry or doc['patio']!=base['patio']:raise ValueError('Root geometry differs from shared inventory')
  inputs[str(source_path)]=sha(source_path);inputs[str(summary_path)]=sha(summary_path)
  items=[dict(id=f'r{number:06d}_original',root=number,kind='original',depth=0,sequence=0,parent=None,frame=state(doc),metrics=metrics,provenance=str(source_path))]
  if cfg['warm_start']:
   archive=PILOT/f'run_{number:06d}/pareto_archive.json.gz';inputs[str(archive)]=sha(archive)
   for i,e in enumerate(read(archive)):
    m=e['metrics'];a=axes(m);b=axes(metrics)
    meaningful=(m['area_percent']>metrics['area_percent']+.01 or any(a[k]<b[k]-tol for k,tol in [(1,.1),(2,.1),(3,.02),(4,3)]))
    if meaningful:items.append(dict(id=f'r{number:06d}_seed_{i:03d}',root=number,kind='pilot_seed',depth=0,sequence=0,parent=None,frame=e['frame'],metrics=m,provenance=dict(archive=str(archive),trial=e['id'])))
  seen=set()
  for n in items:
   key=canonical(n['frame'])
   if key in seen:continue
   seen.add(key);layout=restore(base,n['frame']);check=clearance(layout)
   if not check['valid']:raise ValueError(f'Invalid seed {n["id"]}: {check}')
   actual=sum(shapely.Polygon(s['outline']).area for s in layout['stones'] if s['active'])*100/shapely.Polygon(base['patio']).area
   if abs(actual-n['metrics']['area_percent'])>1e-6:raise ValueError('Seed coverage differs from original full-size outlines')
   write(out/'nodes'/f'{n["id"]}.json.gz',n);seeds.append(n['id'])
 (out/'assets.js').write_text('window.PATIO_SOURCE='+json.dumps(dict(patio=base['patio'],stones=[dict(id=s['id'],outline=s['outline']) for s in base['stones']]),separators=(',',':')).replace('</','<\\/')+';\n')
 m=dict(version=1,config=cfg,inputs=inputs,source_sha256=sha(out/'source.json'),dependencies=dependencies(),versions=versions(),seed_nodes=seeds,policy=dict(attempt_count='Only a selected action evaluated by physics counts as a placement attempt; empty/interrupted searches are separate.',frontier='Bounded multiobjective frontier plus geometrically distinct exploratory setups; all raw endpoints and retained nodes remain on disk.',eligibility='Full original outlines, locks, no net downsizing, >=45 degree deliberate same-piece edits.',settling='Whole-board half-inch clearance and two consecutive pre-stop low-motion observations.',seed_metrics='Existing campaign/pilot gap audits reused; every initial seed rechecked for full-outline clearance and area. New endpoints receive new complete gap audits.',resume='Immutable job identity, parent, operator, phase and RNG seed; selected proposals survive stop/resume. Parallel scheduling is timing-dependent, not claimed globally identical across restarts.'))
 write(manifest_path,m);(out/'STOP').touch();return m

def preflight(out,cfg=None):
 out=Path(out);m=read(out/'manifest.json')
 if cfg is not None and cfg!=m['config']:raise RuntimeError('Configuration changed; resume with the prepared config or use a new versioned output')
 if dependencies()!=m['dependencies'] or versions()!=m['versions']:raise RuntimeError('Compute code or libraries changed; preserve this output and prepare a new version')
 if sha(out/'source.json')!=m['source_sha256']:raise RuntimeError('Source snapshot changed')
 for path,digest in m['inputs'].items():
  if sha(path)!=digest:raise RuntimeError(f'Pinned input changed: {path}')
 continuation=m.get('continuation')
 if continuation:
  for name,digest in continuation['inherited_file_hashes'].items():
   if sha(out/name)!=digest:raise RuntimeError(f'Inherited checkpoint changed: {name}')
  if sha(out/'lineage/parent_manifest.json')!=continuation['parent_manifest_sha256']:raise RuntimeError('Parent manifest archive changed')
  for name,digest in continuation['parent_compute_hashes'].items():
   if sha(out/'lineage/parent_code'/name)!=digest:raise RuntimeError(f'Parent compute archive changed: {name}')
 return m

def load(out):
 out=Path(out);m=read(out/'manifest.json');nodes={n:read(out/'nodes'/f'{n}.json.gz') for n in m['seed_nodes']};jobs={};results={}
 for p in sorted((out/'jobs').glob('job_*/job.json')):
  meta=read(p);jobs[meta['number']]=meta;result_path=p.parent/'result.json.gz'
  if result_path.exists():
   r=read(result_path);results[meta['number']]=r
   if r.get('accepted') and r.get('metrics_usable'):
    n=dict(id=f'n{meta["number"]:08d}',root=meta['root'],kind='search',depth=meta['depth'],sequence=meta['number'],parent=meta['parent'],frame=r['end'],metrics=r['end_metrics'],job=meta['number'])
    nodes[n['id']]=n
    dest=out/'nodes'/f'{n["id"]}.json.gz'
    if not dest.exists():write(dest,n)
 return m,nodes,jobs,results

def parent_for(root,nodes,jobs,cfg,rng):
 available=[n for n in nodes.values() if n['root']==root and n['depth']<cfg['max_depth']];front=frontier(available,cfg['frontier_limit']);visits={n['id']:sum(j['parent']==n['id'] for j in jobs.values()) for n in front}
 pareto=[n for n in front if not any(dominates(o,n) for o in front)]
 pool=pareto if pareto and rng.random()<.65 else front
 weights=np.array([1/math.sqrt(1+visits[n['id']]) for n in pool]);return pool[int(rng.choice(len(pool),p=weights/weights.sum()))]

def allocate(out,root,nodes,jobs,cfg,operator_override=None,exclusion_provider=None):
 number=max(jobs,default=0)+1;rng=np.random.default_rng(seed_for(cfg['seed'],f'schedule-{number}'));parent=parent_for(root,nodes,jobs,cfg,rng)
 root_jobs=[j for j in jobs.values() if j['root']==root];cycle=cfg['operator_cycle'];operator=operator_override or cycle[len(root_jobs)%len(cycle)];phase=sum(j['operator']==operator for j in root_jobs)
 meta=dict(number=number,root=root,parent=parent['id'],depth=parent['depth']+1,operator=operator,phase=phase,seed=seed_for(cfg['seed'],number),parent_geometry=geometry_key(parent['frame']))
 meta['excluded']=exclusion_provider(meta,jobs) if exclusion_provider else []
 folder=Path(out)/'jobs'/f'job_{number:08d}';write(folder/'job.json',meta);jobs[number]=meta;return meta

def summary(out,nodes,jobs,results,cfg):
 rows=[]
 for root in cfg['roots']:
  ns=[n for n in nodes.values() if n['root']==root];original=next(n for n in ns if n['kind']=='original');best=max(ns,key=lambda n:n['metrics']['area_percent'])
  if best['metrics']['area_percent']<=original['metrics']['area_percent']+1e-7:best=original
  gap=min(ns,key=lambda n:axes(n['metrics'])[1:3]);r=[results[j['number']] for j in jobs.values() if j['root']==root and j['number'] in results]
  rows.append(dict(root=root,original=original['metrics']['area_percent'],best=best['metrics']['area_percent'],best_node=best['id'],gap_node=gap['id'],physics_attempts=sum(bool(v.get('physics_evaluated')) for v in r),accepted=sum(bool(v.get('accepted')) for v in r),empty_searches=sum(v['status'] in ['no_candidate','geometry_error'] for v in r),numerical_search_errors=sum(v['status']=='geometry_error' for v in r),nodes=len(ns),frontier=[n['id'] for n in frontier(ns,cfg['frontier_limit'])]))
 value=dict(roots=rows,scheduled_jobs=len(jobs),completed_jobs=len(results),pending_jobs=sorted(set(jobs)-set(results)),updated_unix=time.time());write(Path(out)/'summary.json',value);return value
