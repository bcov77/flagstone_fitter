"""Prepare an explicit revision with immutable committed prefixes from a stopped run.

No trajectories are launched. Existing manifests and attempt files are never
rewritten to disguise a code change. Each inherited record is identified by hash.
"""
import argparse,copy,fcntl,json,os,shutil,time
from pathlib import Path
from common import ROOT,write,read,sha,attempt_files,state,restore
from run import dependencies,versions,DEFAULT_OUT


def prepare_continuation(parent,out,archive=None,revision='geos_empty_overlay_fix_r2'):
    parent=Path(parent).resolve();out=Path(out).resolve()
    if out.exists():raise ValueError('Continuation destination already exists; use its normal run command to resume')
    if parent==out or parent in out.parents or out in parent.parents:raise ValueError('Use separate sibling output directories')
    with (parent/'.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Stop the parent runner before making a continuation')
        old=read(parent/'manifest.json')
        if sha(parent/'source.json')!=old['source_sha256']:raise RuntimeError('Parent source snapshot changed')
        if old['versions']!=versions():raise RuntimeError('Library versions differ; this continuation is for the recorded code repair only')
        source=read(parent/'source.json');stage=out.with_name(out.name+f'.preparing_{os.getpid()}')
        if stage.exists():raise RuntimeError('Staging directory already exists')
        inherited={};counts={};old_errors={}
        # Validate the entire immutable chain before copying anything.
        for folder in sorted((parent/'runs').glob('run_*')):
            previous=state(source);files=attempt_files(folder)
            for i,path in enumerate(files,1):
                r=read(path)
                if r['number']!=i or r['run']!=int(folder.name.split('_')[1]) or r['start']!=previous:raise RuntimeError(f'Invalid parent prefix: {path}')
                previous=r['end']
                inherited[str(path.relative_to(parent))]=sha(path)
            counts[folder.name]=len(files)
        try:
            stage.mkdir(parents=True);shutil.copy2(parent/'source.json',stage/'source.json')
            lineage=stage/'lineage';lineage.mkdir()
            shutil.copy2(parent/'manifest.json',lineage/'parent_manifest.json')
            # Preserve the archived exact parent code, when available.
            archive=Path(archive) if archive is not None else ROOT/'analysis/physics_trajectories/revisions/pre_geos_fix_20261116'
            if (parent/'lineage').exists():shutil.copytree(parent/'lineage',lineage/'parent_lineage')
            code_preserved={}
            for name,expected in old['dependencies'].items():
                src=archive/name
                if not src.exists() or sha(src)!=expected:raise RuntimeError(f'Missing verified parent code archive: {name}')
                dest=lineage/'parent_code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);code_preserved[name]=expected
            for folder in sorted((parent/'runs').glob('run_*')):
                dest=stage/'runs'/folder.name;dest.mkdir(parents=True)
                for path in attempt_files(folder):shutil.copy2(path,dest/path.name)
                for name in ('run.json','summary.json','final.json','start_end.png','trajectory.png','metrics.png'):
                    if (folder/name).exists():shutil.copy2(folder/name,dest/name)
                # Prior errors and in-flight trials are history, not new-run failures.
                for name in ('error.json','pending_attempt.json.gz','render_error.json'):
                    if (folder/name).exists():
                        (dest/'parent_history').mkdir(exist_ok=True);shutil.copy2(folder/name,dest/'parent_history'/name)
                        old_errors[str((folder/name).relative_to(parent))]=sha(folder/name)
                files=attempt_files(folder)
                if files:
                    last=read(files[-1]);write(dest/'progress.json',dict(run=last['run'],mode=last['mode'],attempt=len(files),
                        completed=len(files),phase='checkpointed_from_parent',updated_unix=time.time(),metrics=last['end_metrics']))
            current=copy.deepcopy(old);current.update(version=2,compute_revision=revision,created_unix=time.time(),
                dependencies=dependencies(),versions=versions(),continuation=dict(parent=str(parent),parent_manifest_sha256=sha(parent/'manifest.json'),
                inherited_attempt_hashes=inherited,inherited_counts=counts,total_inherited_attempts=sum(counts.values()),
                parent_error_hashes=old_errors,parent_code_archive='lineage/parent_code',
                interpretation=f'Inherited attempts retain their recorded compute provenance; only subsequent attempts use revision {revision}.'))
            current.setdefault('policy',{})['numerical_geometry']='Exact empty-set identities after GEOS failure; otherwise bounded, logged planning-only precision retries. Unrecoverable selection geometry is an explicit unsuccessful attempt with unchanged board, not a low-risk proposal or a physics trial.'
            write(stage/'manifest.json',current)
            for name,expected in inherited.items():
                if sha(stage/name)!=expected:raise RuntimeError(f'Copied prefix differs: {name}')
            stage.rename(out)
        except BaseException:
            # Staging is diagnostic evidence; never delete or overwrite the parent.
            raise
    return current

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--from',dest='parent',type=Path,required=True);ap.add_argument('--out',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--archive',type=Path,help='Verified archive of the exact parent compute files')
    ap.add_argument('--revision',default='geos_empty_overlay_fix_r2')
    args=ap.parse_args();m=prepare_continuation(args.parent,args.out,args.archive,args.revision)
    print(f'Prepared {args.out}: {m["continuation"]["total_inherited_attempts"]} immutable attempts across {len(m["continuation"]["inherited_counts"])} runs. No physics launched.')
