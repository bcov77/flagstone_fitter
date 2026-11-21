"""Shared IO and original-outline geometry for the online trajectory experiment."""
from pathlib import Path
import copy, gzip, hashlib, json, math, os, sys
import numpy as np
import shapely
from shapely.affinity import affine_transform

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'physics_patio'), str(ROOT/'src'), str(ROOT/'analysis/physics_placement_campaign')]
GAP = .5/12

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    if path.name.endswith('.gz'):payload=gzip.compress(payload,compresslevel=3,mtime=0)
    with tmp.open('wb') as f:
        f.write(payload);f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def read(path):
    path=Path(path);raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.name.endswith('.gz') else raw)

def attempt_path(folder,number):
    compressed=Path(folder)/f'attempt_{number:03d}.json.gz'
    plain=Path(folder)/f'attempt_{number:03d}.json'
    if compressed.exists() and plain.exists():raise RuntimeError('Duplicate compressed/plain attempt')
    return plain if plain.exists() else compressed

def attempt_files(folder):
    files=[p for p in Path(folder).glob('attempt_*.json*') if p.name.endswith(('.json','.json.gz'))]
    if len({p.name.split('.')[0] for p in files})!=len(files):raise RuntimeError('Duplicate attempt files')
    return sorted(files)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def seed_for(seed, run, attempt=0):
    return int(hashlib.sha256(f'{seed}:{run}:{attempt}'.encode()).hexdigest()[:15],16)
def world(local, pose):
    x,y,a=pose;c,s=math.cos(a),math.sin(a)
    return affine_transform(local,[c,-s,s,c,x,y])
def shapes(document):
    return {str(s['id']):world(shapely.Polygon(s['outline']),s['pose']) for s in document['stones'] if s['active']}
def state(document):
    return dict(poses={str(s['id']):s['pose'] for s in document['stones'] if s['active']},physics=document.get('physics',{}))
def restore(source, frame):
    d=copy.deepcopy(source)
    for s in d['stones']:
        sid=str(s['id']);s['active']=sid in frame['poses']
        if s['active']: s['pose']=frame['poses'][sid][:]
    d['physics']=copy.deepcopy(frame['physics']);return d

def clearance(document):
    board=shapes(document);patio=shapely.Polygon(document['patio']);errors=[];min_pair=None;min_wall=None;overlap=0.
    items=list(board.items())
    for i,(sid,g) in enumerate(items):
        wd=g.distance(patio.boundary);min_wall=wd if min_wall is None else min(min_wall,wd)
        if not patio.covers(g) or wd<GAP-1e-8:errors.append(dict(kind='wall',ids=[sid],distance_inches=12*wd))
        overlap+=g.difference(patio).area
        for tid,h in items[i+1:]:
            dist=g.distance(h);min_pair=dist if min_pair is None else min(min_pair,dist)
            if dist<GAP-1e-8:errors.append(dict(kind='pair',ids=[sid,tid],distance_inches=12*dist))
            if dist==0:overlap+=g.intersection(h).area
    return dict(valid=not errors,errors=errors,overlap_or_outside_sqft=overlap,
        min_pair_inches=None if min_pair is None else 12*min_pair,min_wall_inches=None if min_wall is None else 12*min_wall)

def measure(document):
    from flagstone.quarter_gaps import audit
    board=shapes(document);patio=shapely.Polygon(document['patio']);gs=list(board.values())
    covered=shapely.union_all(gs).intersection(patio).area
    result=dict(clearance=clearance(document),active_count=len(gs),area_sqft=covered,area_percent=100*covered/patio.area)
    try:
        g=audit(patio,gs)
        result['gaps']=dict(method=g['certification'],stable=g['stable'],numerical_gap_pass=g['numerical_pass'],
            largest_interior_inches=g['groups']['interior']['upper_diameter_in'],largest_edge_inches=g['groups']['edge']['upper_diameter_in'],
            groups=g['groups'],largest_circles=g['candidates'][:20])
    except (ValueError,RuntimeError,OverflowError) as exc:
        # Never silently replace a failed segment audit with a sampled certificate.
        result['gaps']=dict(error=repr(exc),stable=False,numerical_gap_pass=False)
    result['layout_pass']=result['clearance']['valid'] and result['gaps']['numerical_gap_pass']
    return result
