"""Resumable v5 durable-joint/residual-repair evaluation of saved v4 poses.

Preserves v4 inputs and all human review numbering. Foreground, ten workers,
completion percentage and rolling ETA. No physics or changes to the board.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import math
from pathlib import Path
import time

import shapely
from shapely.ops import unary_union
import selection_campaign as v4
from future_gap_geometry import assess,compare,reference_stability

HERE=Path(__file__).resolve().parent
DEFAULT_OUT=HERE/'selection_v5_future_gaps'
DEFAULT_CONFIG=dict(boundary_step_ft=.025,joint_excess_inches=.5,joint_angle_degrees=20,
    residual_core_inches=.25,min_core_sqft=.0005,owner_min_length_ft=.04,
    locked_stones=[],protected_stones=[],
    filler_search_pad_ft=.25,filler_raster_step_ft=.035,filler_max_pieces=12,
    filler_angle_step_degrees=15,filler_positions_per_angle=2,filler_min_patch_fraction=.35,
    max_filler_regions=6,alignment_anchor_distance_ft=.25,alignment_min_edge_ft=.25,
    alignment_face_tolerance_inches=.125,alignment_angle_span_degrees=20,
    risk_area_indifference_sqft=.005,alignment_run_indifference_ft=.10,corner_reach_indifference_inches=.10,
    wall_risk_band_inches=2.,corner_risk_radius_inches=3.,stable_risk_band_inches=2.)
CACHE={}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependencies():
    return {str(p.relative_to(HERE)):sha(p) for p in [Path(__file__),HERE/'future_gap_geometry.py',
        HERE/'selection_campaign.py',HERE/'selection_geometry.py',HERE/'build_prefit_review.py',
        HERE/'campaign_v1/source.json']}


def eligible(c,target):
    if c['fit_gate']['status']=='passes_provisional_screen':return True
    if c.get('nearfit_gate',{}).get('status')=='passes_nearfit_review_screen':return True
    return c['stone']==target['old'] and all(c['fit_gate']['checks'][k] for k in ('physical_overlap','clearance_area','clearance_depth'))


def respects_locks(c,target,config):
    locked=set(config['locked_stones'])
    if any(sid in locked and math.hypot(*xy)>1e-10 for sid,xy in c.get('space_plan',{}).get('translations_ft',{}).items()):return False
    old=target['old']
    if old in locked:
        if c['stone']!=old:return False
        before=v4.b.STONES[old]['pose'];after=c['pose']
        return math.hypot(after[0]-before[0],after[1]-before[1])<1e-10 and abs(math.atan2(math.sin(after[2]-before[2]),math.cos(after[2]-before[2])))<1e-10
    return True


def prepare(out,source,config,target_ids=None,review_only=False):
    if (out/'manifest.json').exists():raise RuntimeError('Already prepared; run to resume, or choose a fresh directory')
    m=v4.load_manifest(source);targets={t['id']:t for t in m['targets']}
    reviewed=json.loads((source/'review_18_v1/placements.json').read_text()) if (source/'review_18_v1/placements.json').exists() else []
    if review_only:
        if not reviewed:raise ValueError('No reviewed panel available')
        wanted={r['target']['id'] for r in reviewed};targets={k:v for k,v in targets.items() if k in wanted}
    if target_ids:
        if set(target_ids)-set(targets):raise ValueError('Unknown target')
        targets={k:v for k,v in targets.items() if k in target_ids}
    (out/'inputs').mkdir(exist_ok=True);(out/'results').mkdir(exist_ok=True)
    tasks=[];seen=set();moved={tid:set() for tid in targets}
    for t in targets.values():
        old=t['old'];c=None
        if old:
            p,free,obstacles=v4.environment(t)
            c=v4.evaluate(old,v4.b.STONES[old]['pose'],p,free,obstacles,v4.DEFAULT_CONFIG,'retain_original')
        item=dict(id=f"baseline__{t['id']}",target=t['id'],role='baseline',candidate=v4.compact(c) if c else None)
        path=out/'inputs'/f"{item['id']}.json";v4.atomic_json(path,item)
        tasks.append(dict(id=item['id'],target=t['id'],role='baseline',input_sha256=sha(path)))
        if c:seen.add((t['id'],c['stone'],*c['pose'],()))
    pose_reviews={}
    for r in reviewed:
        pose_reviews[(r['target']['id'],r['candidate']['stone'],tuple(r['candidate']['pose']))]=r['number']
    source_hashes={}
    for task in m['tasks']:
        if task['target'] not in targets:continue
        path=source/'results'/f"{task['id']}.json"
        if not path.exists():raise RuntimeError(f'Incomplete v4 input: {path.name}')
        source_hashes[path.name]=sha(path)
        for index,c in enumerate(json.loads(path.read_text())['candidates']):
            if not eligible(c,targets[task['target']]):continue
            if not respects_locks(c,targets[task['target']],config):continue
            if review_only and (task['target'],c['stone'],tuple(c['pose'])) not in pose_reviews:continue
            translations=c.get('space_plan',{}).get('translations_ft',{})
            move_key=tuple((s,*xy) for s,xy in sorted(translations.items()))
            key=(task['target'],c['stone'],*c['pose'],move_key)
            if key in seen:continue
            seen.add(key)
            identity=hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
            item=dict(id=identity,target=task['target'],role='candidate',candidate=v4.compact(c),
                      source_task=task['id'],source_candidate_index=index,
                      review_number=pose_reviews.get((task['target'],c['stone'],tuple(c['pose']))))
            ipath=out/'inputs'/f'{identity}.json';v4.atomic_json(ipath,item)
            tasks.append(dict(id=identity,target=task['target'],role='candidate',input_sha256=sha(ipath)))
            moved[task['target']].update(translations)
    # A common, candidate-independent ROI includes far sides of every neighbor
    # moved by any candidate at this target, plus potential incident neighbors.
    enriched=[]
    for t in targets.values():
        p=shapely.from_wkt(t['pocket']);active=set(v4.b.ACTIVE)-set(t['removed'])-({t['old']} if t['old'] else set())
        incident={s for s in active if v4.b.ORIGINAL[s].intersects(p.buffer(.5))}|moved[t['id']]
        roi=unary_union([p.buffer(.25),*[v4.b.ORIGINAL[s].buffer(.375) for s in sorted(incident)]]).intersection(v4.b.PATIO)
        enriched.append({**t,'roi':shapely.to_wkt(roi,rounding_precision=8)})
    feedback=source/'review_18_v1/human_feedback.jsonl'
    manifest=dict(version=5,created_unix=time.time(),source=str(source),source_manifest_sha256=sha(source/'manifest.json'),
        source_result_hashes=source_hashes,feedback_sha256=sha(feedback) if feedback.exists() else None,
        config=config,dependencies=dependencies(),targets=enriched,tasks=tasks,
        reviewed_development_cases_only=review_only,
        scope='Rescore saved eligible poses and original controls; bounded actual-inventory repair probes. No new placement search or physics.')
    v4.atomic_json(out/'manifest.json',manifest)
    print(f'Prepared {len(tasks)} assessments across {len(targets)} targets, including {len(targets)} original/empty controls.',flush=True)


def load(out):
    m=json.loads((out/'manifest.json').read_text())
    if m['dependencies']!=dependencies():raise RuntimeError('Code/input dependencies changed; use a fresh output directory')
    for task in m['tasks']:
        if sha(out/'inputs'/f"{task['id']}.json")!=task['input_sha256']:
            raise RuntimeError(f"Changed input {task['id']}")
    return m


def board_for(target,candidate):
    active=set(v4.b.ACTIVE)-set(target['removed'])
    if candidate is not None and target['old']:active.discard(target['old'])
    board={s:v4.b.ORIGINAL[s] for s in sorted(active)}
    if candidate is not None:
        for sid,pose in candidate.get('space_plan',{}).get('neighbor_poses',{}).items():
            if sid not in board:raise ValueError('Invalid moved-neighbor identity')
            board[sid]=v4.b.world(sid,pose)
        if candidate['stone'] in board:raise ValueError('Duplicate physical piece')
        board[candidate['stone']]=v4.b.world(candidate['stone'],candidate['pose'])
    return board


def work(out,task,target,config):
    started=time.time();item=json.loads((out/'inputs'/f"{task['id']}.json").read_text());c=item['candidate']
    board=board_for(target,c);subject=c['stone'] if c else None
    context_key=tuple(sorted(target['removed']))
    if context_key not in CACHE:
        reference={s:v4.b.ORIGINAL[s] for s in sorted(set(v4.b.ACTIVE)-set(target['removed']))}
        CACHE[context_key]=(reference,reference_stability(reference,v4.b.PATIO,v4.b.GAP,config))
    reference,stability=CACHE[context_key]
    stability={s:dict(v) for s,v in stability.items()}
    for sid in config['protected_stones']:
        if sid in stability:stability[sid]['score']=1.
    neighbors={s:g for s,g in reference.items() if s!=target['old']}
    inventory={s:g for s,g in v4.b.LOCAL.items() if s not in board and s not in target['removed']}
    assessed=assess(board,subject,neighbors,v4.b.PATIO,shapely.from_wkt(target['pocket']),
        shapely.from_wkt(target['roi']),inventory,stability,v4.b.GAP,config)
    assessed['reference_corner_deficit_inches']=c['metrics'].get('corner_deficit_q90_inches') if c else None
    # Explicitly separate an initial pose with clearance intrusion from the
    # independently checked near-fit endpoint witnesses inherited from v4.
    minimum=None
    if subject:
        g=board[subject]
        minimum=12*min([g.distance(h) for sid,h in board.items() if sid!=subject]+[g.distance(v4.b.PATIO.boundary)])
    result=dict(id=task['id'],target=target['id'],role=task['role'],candidate=c,review_number=item.get('review_number'),
        assessment=assessed,subject_min_clearance_inches=minimum,seconds=time.time()-started,
        physics_steps=0,provisional_stability=stability)
    v4.atomic_json(out/'results'/f"{task['id']}.json",result)
    return task['id'],result['seconds']


def run(out,workers,max_tasks):
    m=load(out);targets={t['id']:t for t in m['targets']};total=len(m['tasks'])
    done=sum((out/'results'/f"{t['id']}.json").exists() for t in m['tasks'])
    todo=[t for t in m['tasks'] if not (out/'results'/f"{t['id']}.json").exists()]
    if max_tasks is not None:todo=todo[:max_tasks]
    print(f'{done}/{total} ({100*done/total:.1f}%) complete; {len(todo)} queued, {workers} workers. No physics.',flush=True)
    started=time.time();finished=0;errors=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(work,out,t,targets[t['target']],m['config']):t for t in todo}
        for future in as_completed(futures):
            task=futures[future]
            try:
                _,seconds=future.result();finished+=1;done+=1
                elapsed=time.time()-started;eta=(total-done)*elapsed/finished
                print(f'{done}/{total} ({100*done/total:.1f}%) | ETA ~{eta/60:.1f} min | {task["target"]} | {seconds:.1f}s',flush=True)
            except Exception as exc:
                errors.append(dict(task=task['id'],error=repr(exc)));print(errors[-1],flush=True)
    v4.atomic_json(out/'last_run_errors.json',errors)
    if errors:raise RuntimeError(f'{len(errors)} failed assessments; rerun to retry')


def report(out):
    m=load(out);by_target={};missing=[]
    for t in m['tasks']:
        path=out/'results'/f"{t['id']}.json"
        if not path.exists():missing.append(t['id']);continue
        by_target.setdefault(t['target'],[]).append(json.loads(path.read_text()))
    reports=[]
    for target,items in by_target.items():
        baseline=next((r for r in items if r['role']=='baseline'),None)
        if not baseline:continue
        comparisons=[]
        for r in items:
            if r['role']=='baseline':continue
            decision=compare(r['assessment'],baseline['assessment'],m['config'])
            c=r['candidate'];comparisons.append(dict(id=r['id'],stone=c['stone'],pose=c['pose'],
                review_number=r['review_number'],comparison=decision,requires_clearance_refinement=r['subject_min_clearance_inches']<.5-1e-6,
                min_clearance_inches=r['subject_min_clearance_inches'],metrics=r['assessment']['metrics'],
                joint_length_ft=r['assessment']['joints']['matched_length_ft']))
        reports.append(dict(target=target,baseline=baseline['id'],assessments=comparisons))
    complete=not missing
    v4.atomic_json(out/'report.json',dict(complete=complete,missing=missing,targets=reports,physics_steps=0))
    counts={}
    for r in reports:
        for c in r['assessments']:counts[c['comparison']['status']]=counts.get(c['comparison']['status'],0)+1
    text=['# Future-gap planning report','',f'Complete: {complete}. {len(m["tasks"])-len(missing)}/{len(m["tasks"])} assessments saved.',
        '', 'All dispositions are unvalidated review hypotheses. No physics; no-witness results are not impossibility proofs.', '',
        *[f'- {k}: {v}' for k,v in sorted(counts.items())]]
    (out/'report.md').write_text('\n'.join(text)+'\n')
    print('Report:',counts,'complete:',complete,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['prepare','run','report'])
    ap.add_argument('--out',type=Path,default=DEFAULT_OUT);ap.add_argument('--source',type=Path,default=HERE/'selection_v4_aggressive')
    ap.add_argument('--workers',type=int,default=10);ap.add_argument('--max-tasks',type=int);ap.add_argument('--target',action='append')
    ap.add_argument('--config',type=Path)
    ap.add_argument('--review-only',action='store_true',help='Prepare only the 18 reviewed poses and their original/empty controls')
    args=ap.parse_args();out=args.out.resolve();source=args.source.resolve()
    if args.workers<1 or (args.max_tasks is not None and args.max_tasks<0):ap.error('Invalid worker/task count')
    protected=[source,v4.b.SRC.resolve(),v4.b.DEST.resolve()]
    if out==HERE or any(out==p or p in out.parents or out in p.parents for p in protected):ap.error('Use an independent output directory outside frozen inputs')
    with v4.lock(out):
        if args.command=='prepare':prepare(out,source,json.loads(args.config.read_text()) if args.config else DEFAULT_CONFIG,args.target,args.review_only)
        elif args.command=='run':run(out,args.workers,args.max_tasks)
        else:report(out)


if __name__=='__main__':main()
