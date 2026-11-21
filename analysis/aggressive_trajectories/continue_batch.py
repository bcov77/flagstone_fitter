"""Prepare a stopped two-policy continuation; preserve exact committed prefixes."""
import argparse
import copy
import fcntl
import importlib.util
import shutil
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'physics_trajectories')]
from common import read,write,sha,state,attempt_files
from proposal_search import ATTEMPT_POLICY


def runner(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prepare(parent,out,archive):
    parent=Path(parent).resolve();out=Path(out).resolve();archive=Path(archive).resolve()
    if out.exists():raise ValueError('Destination already exists')
    if parent==out or parent in out.parents or out in parent.parents:raise ValueError('Use sibling output directories')
    if not (parent/'STOP').exists():raise RuntimeError('Stop the parent batch first')
    locks=[];manifests={};inherited={}
    for track in ['aggressive','baseline']:
        src=parent/track;lock=(src/'.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        m=read(src/'manifest.json');manifests[track]=m
        if sha(src/'source.json')!=m['source_sha256']:raise RuntimeError('Parent source changed')
        for name,digest in m['dependencies'].items():
            if sha(archive/name)!=digest:raise RuntimeError(f'Parent code archive differs: {name}')
        source=read(src/'source.json');hashes={}
        for folder in sorted((src/'runs').glob('run_*')):
            previous=state(source)
            for index,path in enumerate(attempt_files(folder),1):
                r=read(path)
                if r['number']!=index or r['run']!=int(folder.name.split('_')[1]) or r['start']!=previous:raise RuntimeError(f'Broken parent chain: {path}')
                previous=r['end'];hashes[str(path.relative_to(src))]=sha(path)
        inherited[track]=hashes
    stage=out.with_name(out.name+'.preparing')
    stage.mkdir()
    for track in ['aggressive','baseline']:
        src=parent/track;dest=stage/track;dest.mkdir();old=manifests[track]
        module=runner(HERE/'run.py' if track=='aggressive' else HERE.parent/'physics_trajectories/run.py',f'{track}_continuation')
        if old['versions']!=module.versions():raise RuntimeError('Parent library versions differ')
        shutil.copy2(src/'source.json',dest/'source.json')
        shutil.copytree(src/'runs',dest/'runs')
        lineage=dest/'lineage';lineage.mkdir()
        shutil.copy2(src/'manifest.json',lineage/'parent_manifest.json')
        if (src/'lineage').exists():shutil.copytree(src/'lineage',lineage/'parent_lineage')
        for name,digest in old['dependencies'].items():
            target=lineage/'parent_code'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(archive/name,target)
        for folder in (dest/'runs').glob('run_*'):
            for name in ['pending_attempt.json.gz','error.json','render_error.json']:
                if (folder/name).exists():
                    history=folder/'parent_history';history.mkdir(exist_ok=True)
                    (folder/name).rename(history/name)
        current=copy.deepcopy(old)
        current.update(version=2,compute_revision='proposal_retry_v2',created_unix=time.time(),dependencies=module.dependencies(),versions=module.versions(),
            continuation=dict(parent=str(src),parent_manifest_sha256=sha(src/'manifest.json'),inherited_attempt_hashes=inherited[track],
                total_inherited_attempts=len(inherited[track]),parent_code_archive='lineage/parent_code',
                interpretation='Inherited records are byte-identical and keep their original attempt semantics; new attempts require a physics proposal.'))
        current['policy']['attempt_count']=ATTEMPT_POLICY
        write(dest/'manifest.json',current)
        for name,digest in inherited[track].items():
            if sha(dest/name)!=digest:raise RuntimeError(f'Copied prefix differs: {name}')
        write(dest/'inherited_prefix_verification.json',dict(attempts=len(inherited[track]),byte_identical=True,chains_valid=True,parent=str(src)))
        print(f'{track}: preserved {len(inherited[track])} committed attempts',flush=True)
    # Leave STOP in the prepared batch; normal launch clears it under output locks.
    (stage/'STOP').touch();stage.rename(out)
    from launch import index
    index(out)
    for lock in locks:lock.close()
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from',dest='parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args();print(f'Prepared {prepare(a.parent,a.out,a.archive)}. No physics launched.')
