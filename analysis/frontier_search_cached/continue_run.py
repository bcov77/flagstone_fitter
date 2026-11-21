"""Create a separately pinned cache-only continuation of a checkpointed search."""
import argparse
import fcntl
import shutil
from core import *
from store import load,summary,preflight


def prepare_continuation(parent,destination):
 parent=Path(parent).resolve();destination=Path(destination).resolve()
 if parent==destination or parent in destination.parents or destination in parent.parents:
  raise ValueError('Parent and continuation must be separate directories')
 if destination.exists():raise ValueError('Use a new continuation output directory')
 staging=destination.with_name(destination.name+'.preparing')
 if staging.exists():raise ValueError(f'Interrupted staging output exists: {staging}')
 with (parent/'.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('Checkpoint and stop the parent supervisor first')
  old=read(parent/'manifest.json')
  if not (parent/'STOP').exists():raise RuntimeError('Parent must have a checkpointed STOP marker')
  if versions()!=old['versions']:raise RuntimeError('Parent library versions changed')
  if sha(parent/'source.json')!=old['source_sha256']:raise RuntimeError('Parent source changed')
  for name,digest in old['dependencies'].items():
   if sha(ROOT/name)!=digest:raise RuntimeError(f'Parent compute changed: {name}')
  for name,digest in old['inputs'].items():
   if sha(name)!=digest:raise RuntimeError(f'Parent input changed: {name}')
  protected=[parent/'source.json',*sorted((parent/'nodes').glob('*.json.gz')),
             *sorted((parent/'jobs').glob('job_*/job.json')),
             *sorted((parent/'jobs').glob('job_*/selected.json.gz')),
             *sorted((parent/'jobs').glob('job_*/result.json.gz'))]
  # Progress/error files are mutable. A proposal pool without a committed
  # selection can also be regenerated after interruption.
  protected += [p for p in sorted((parent/'jobs').glob('job_*/proposals.json.gz'))
                if (p.parent/'selected.json.gz').exists()]
  hashes={str(p.relative_to(parent)):sha(p) for p in protected}
  parent_manifest_sha=sha(parent/'manifest.json')
  shutil.copytree(parent,staging,ignore=shutil.ignore_patterns('.lock'))
  lineage=staging/'lineage';lineage.mkdir(exist_ok=True)
  shutil.copy2(parent/'manifest.json',lineage/'parent_manifest.json')
  for name,digest in old['dependencies'].items():
   target=lineage/'parent_code'/name;target.parent.mkdir(parents=True,exist_ok=True)
   shutil.copy2(ROOT/name,target)
   if sha(target)!=digest:raise RuntimeError(f'Parent code archive mismatch: {name}')
  current=copy.deepcopy(old)
  current.update(dependencies=dependencies(),versions=versions(),compute_revision='geometry_target_cache_v2',
   continuation=dict(parent=str(parent),parent_manifest_sha256=parent_manifest_sha,
    inherited_file_hashes=hashes,parent_compute_hashes=old['dependencies'],
    description='Byte-preserved nodes, jobs, selected proposals, physics seeds and results. '
                'Only target-invariant scaled-replacement free-space computation is cached; '
                'ranking, RNG, schedule and physical gates are unchanged.'))
  write(staging/'manifest.json',current);(staging/'STOP').touch()
  for name,digest in hashes.items():
   if sha(staging/name)!=digest:raise RuntimeError(f'Inherited bytes differ: {name}')
  preflight(staging)
  _,nodes,jobs,results=load(staging)
  write(staging/'inherited_prefix_verification.json',dict(parent=str(parent),
   inherited_files=len(hashes),jobs=len(jobs),committed=len(results),
   pending=len(jobs)-len(results),nodes=len(nodes),byte_identical=True,
   selected_proposals_and_seeds_preserved=True))
  staging.rename(destination)
  value=summary(destination,nodes,jobs,results,current['config'])
  from review import render
  render(destination,nodes,jobs,results,value,export=True)
  return read(destination/'inherited_prefix_verification.json')


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--parent',type=Path,default=ROOT/'examples/frontier_search_20261118')
 parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
 args=parser.parse_args()
 print(json.dumps(prepare_continuation(args.parent,args.out),indent=2))
