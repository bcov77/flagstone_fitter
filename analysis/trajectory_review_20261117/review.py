"""Read-only retrospective of every committed attempt; never modifies batch inputs."""
import csv
import gzip
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import shapely
from shapely.affinity import affine_transform

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'analysis/physics_placement_campaign'))
from future_gap_geometry import joint_profile, walls

OUT = Path(__file__).parent
BATCH = ROOT / 'examples/physics_trajectories_20261117_r3'
CFG = dict(boundary_step_ft=.03, joint_excess_inches=.5, joint_angle_degrees=20)


def transform(g, p):
    c, s = math.cos(p[2]), math.sin(p[2])
    return affine_transform(g, [c, -s, s, c, p[0], p[1]])


def angle(a, b):
    return abs(math.degrees(math.atan2(math.sin(a-b), math.cos(a-b))))


def small_metrics(m):
    return dict(coverage=m['area_percent'], interior=m['gaps'].get('largest_interior_inches'),
                edge=m['gaps'].get('largest_edge_inches'))


def joints(board, subjects, wl):
    runs = []
    for sid in subjects:
        if sid not in board:
            continue
        g = board[sid]
        neighbors = {s: h for s, h in board.items() if s != sid and h.distance(g) <= 1.05/12}
        nearwalls = {s: h for s, h in wl.items() if h.distance(g) <= 1.05/12}
        if not neighbors and not nearwalls:
            continue
        p = joint_profile(g, neighbors, nearwalls, .5/12, CFG)
        for owner, v in p['by_owner'].items():
            length = v['longest_run_ft'] * 12
            if length >= 3:
                runs.append(dict(stone=sid, owner=owner, inches=length,
                                 weight=.5 if owner in subjects else 1))
    return dict(total=sum(r['inches']*r['weight'] for r in runs),
                wall=sum(r['inches'] for r in runs if r['owner'].startswith('wall_')), runs=runs)


def main():
    started = time.monotonic()
    source = json.loads((BATCH/'source.json').read_text())
    local = {str(s['id']): shapely.Polygon(s['outline']) for s in source['stones']}
    # Joint diagnostics only: 0.024-inch simplification; gap witness tracking uses full outlines.
    simplified = {s: g.simplify(.002, preserve_topology=True) for s, g in local.items()}
    patio = shapely.Polygon(source['patio']); wl = walls(patio)
    paths = sorted((BATCH/'runs').glob('run_*/attempt_*.json.gz'))
    snapshot = [(str(p.relative_to(BATCH)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    (OUT/'snapshot.json').write_text(json.dumps(dict(batch=str(BATCH), files=snapshot), indent=2))
    folders = sorted(set(p.parent for p in paths)); rows=[]; run_rows=[]
    for fi, folder in enumerate(folders):
        records = [json.loads(gzip.decompress(p.read_bytes())) for p in paths if p.parent == folder]
        records.sort(key=lambda r:r['number'])
        assert all(r['start']==records[i-1]['end'] for i,r in enumerate(records) if i)
        states = [records[0]['start']] + [r['end'] for r in records]
        exact = [{s:transform(local[s],p) for s,p in st['poses'].items()} for st in states]
        boards = [{s:g.simplify(.002, preserve_topology=True) for s,g in b.items()} for b in exact]
        trees = [shapely.STRtree([*b.values(), patio.boundary]) for b in exact]
        def diameter(index, centers):
            if not len(centers):return np.array([])
            points=shapely.points(np.array(centers));_,ds=trees[index].query_nearest(points,all_matches=False,return_distance=True)
            return ds*24
        final_metrics=small_metrics(records[-1]['end_metrics'])
        for i,r in enumerate(records):
            p=r['chosen']; before=small_metrics(r['start_metrics']); after=small_metrics(r['end_metrics'])
            row=dict(run=r['run'], mode=r['mode'], attempt=r['number'],
                     complete=(folder/'summary.json').exists(), accepted=r['accepted'],
                     reason=r['reason'], before=before, after=after, final=final_metrics,
                     remaining_attempts=len(records)-i-1, holes=[], type='no_candidate',
                     placed=[],removed=[],small_adjustment=False)
            if not p:
                row.update(verdict='no placement', notes=['No candidate passed the bounded search.'])
                rows.append(row);continue
            ids=[x['stone'] for x in p['placements']];removed=p['remove']
            same=len(ids)==1 and removed==ids
            row.update(type='reorientation' if same else 'double replacement' if removed and len(ids)==2 else 'double insertion' if len(ids)==2 else 'replacement' if removed else 'insertion',placed=ids,removed=removed,
                       tier=p['tier'], planning_status=p['comparison']['status'],
                       risk_axes=p['comparison']['new_risk_axes'],lost_alignment=p['comparison']['lost_alignment_options'],
                       planned_hard=p['finished_connections']['hard_length_inches'],
                       planned_joints=p['finished_connections']['total_length_inches'],
                       fit=p['fit'],planning_metrics=p['assessment']['metrics'],
                       physics_rotation=max(angle(x['pose'][2],r['attempted_end']['poses'][x['stone']][2]) for x in p['placements']),
                       physics_translation=max(math.dist(x['pose'][:2],r['attempted_end']['poses'][x['stone']][:2])*12 for x in p['placements']))
            if same:
                original=r['start']['poses'][ids[0]];proposed=p['placements'][0]['pose']
                row.update(rotation=angle(original[2],proposed[2]), translation=math.dist(original[:2],proposed[:2])*12)
                row['small_adjustment']=row['rotation']<45-1e-8
            initial=joints(boards[i],removed,wl)
            planned_board={s:transform(simplified[s],pose) for s,pose in r['proposed_start']['poses'].items()}
            planned=joints(planned_board,ids,wl)
            post=joints(boards[i+1],ids,wl) if r['accepted'] else joints({s:transform(simplified[s],pose) for s,pose in r['attempted_end']['poses'].items()},ids,wl)
            row.update(joints_before=initial,joints_planned=planned,joints_after=post,
                       delta_coverage=after['coverage']-before['coverage'],
                       delta_interior=after['interior']-before['interior'],
                       delta_edge=after['edge']-before['edge'],
                       delta_joints=post['total']-initial['total'],delta_wall=post['wall']-initial['wall'])
            if r['accepted']:
                candidates=r['end_metrics']['gaps'].get('largest_circles',[])
                centers=[c['center'] for c in candidates];pre=diameter(i,centers)
                novel=[]
                for c,old in zip(candidates,pre):
                    point_before=float(old)
                    # Avoid calling an existing cavity "new" just because its maximum moved.
                    nearby=[h['diameter_in'] for h in r['start_metrics']['gaps'].get('largest_circles',[])
                            if math.dist(c['center'],h['center'])*12 <= max(1,min(c['diameter_in'],h['diameter_in'])/2)]
                    old=max([point_before,*nearby])
                    if c['diameter_in'] <= (2 if c['kind']=='interior' else 5) or c['diameter_in']-old < .5:continue
                    if any(math.dist(c['center'],h['center'])*12 < max(1,min(c['diameter_in'],h['post'])/2) for h in novel):continue
                    novel.append(dict(center=c['center'],kind=c['kind'],before=float(old),point_before=point_before,post=c['diameter_in']))
                # All distinct novel saved witnesses, not just the globally largest gap.
                future=[diameter(j,[h['center'] for h in novel]) for j in range(i+1,len(states))]
                for hi,h in enumerate(novel):
                    history=[float(v[hi]) for v in future];threshold=h['before']+.25
                    assert abs(history[0]-h['post']) < .02, 'Gap witness does not match saved pose'
                    for k in range(1,len(history)):
                        later=records[i+k]['end_metrics']['gaps'].get('largest_circles',[])
                        nearby=[c['diameter_in'] for c in later if math.dist(c['center'],h['center'])*12 <= max(1,min(c['diameter_in'],h['post'])/2)]
                        history[k]=max([history[k],*nearby])
                    h.update(final=history[-1],history=history,
                             repaired_at_end=history[-1]<=threshold,
                             ever_repaired=any(v<=threshold for v in history[1:]),
                             sustained_repair_attempt=None)
                    for k in range(1,len(history)):
                        if all(v<=threshold for v in history[k:]):
                            h['sustained_repair_attempt']=r['number']+k;break
                    row['holes'].append(h)
            persistent=[h for h in row['holes'] if not h['repaired_at_end']]
            row['persistent_holes']=len(persistent)
            positives=[];negatives=[]
            if row['delta_coverage']>=.15:positives.append('adds at least 0.15 percentage points of coverage')
            if row['delta_joints']>=3:positives.append('adds at least 3 inches of contiguous matched joints')
            if row['delta_interior']<=-.5 or row['delta_edge']<=-.5:positives.append('reduces a largest gap by at least 0.5 inch')
            if persistent:negatives.append('newly enlarged hole witness remains enlarged at the last observed state')
            if row['delta_wall']<=-3:negatives.append('loses at least 3 inches of wall joint')
            if row['delta_joints']<=-6 and row['delta_wall']<3:negatives.append('loses at least 6 inches of contiguous matched joints')
            if row['delta_coverage']<=-.15 and row['delta_joints']<3:negatives.append('loses coverage without compensating joint gain')
            if (row['delta_interior']>=.5 or row['delta_edge']>=.5) and not row['holes'] and (final_metrics['interior']>before['interior']+.25 or final_metrics['edge']>before['edge']+.25):
                negatives.append('largest gap worsens substantially and final board remains worse')
            row['positives']=positives;row['negatives']=negatives
            row['verdict']='bad candidate' if negatives else 'good candidate' if positives else 'neutral'
            if not r['accepted']:row['verdict']='rejected physics'
            # Observed retained-piece fate, not a counterfactual effect of this action.
            row['placed_present_final']=[s for s in ids if s in states[-1]['poses']]
            row['removed_later']=[s for s in ids if any(s not in st['poses'] for st in states[i+2:])]
            rows.append(row)
        run_rows.append(dict(run=records[0]['run'],mode=records[0]['mode'],attempts=len(records),complete=(folder/'summary.json').exists(),
                             final=final_metrics,best_coverage=max(r['end_metrics']['area_percent'] for r in records)))
        if fi%5==0:print(f'{fi+1}/{len(folders)} runs reviewed · {len(rows)} attempts · {time.monotonic()-started:.0f}s',flush=True)
    report=dict(batch=str(BATCH),runs=run_rows,attempts=rows,
                caveats=['Retrospective assistant screening labels, not user verdicts or validated quality labels.',
                         'Joint measurements use 0.024-inch outline simplification and 0.36-inch boundary samples; source acceptance and gap maxima remain original-outline audits.',
                         'A matched joint is a continuous run >=3 inches, at 0.5–1 inch separation with tangents within 20 degrees; this measures geometry, not permanence.',
                         'Hole tracking covers distinct newly enlarged saved top-20 gap witnesses: >=0.5-inch diameter increase and above the 2/5-inch limits.',
                         'Existing and later saved gap circles are matched within half the smaller diameter (at least 1 inch) to reduce false new/repair claims from moving maxima. Fixed-center clearance is also checked. This still does not prove entire-region repair; unsaved smaller holes are not exhaustively tracked.',
                         'Partial runs have shorter follow-up. Later improvements are observed continuations, not proof a bad move was necessary.',
                         'All trajectories share one starting board; modes and action types are adaptively selected, not randomized controlled comparisons.'])
    (OUT/'review.json').write_text(json.dumps(report,separators=(',',':')))
    fields=['run','mode','attempt','complete','type','verdict','small_adjustment','rotation','translation','delta_coverage','delta_interior','delta_edge','delta_joints','delta_wall','persistent_holes','physics_rotation','physics_translation']
    with (OUT/'placements.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:r.get(k) for k in fields})
    summary=dict(runs=len(run_rows),complete_runs=sum(r['complete'] for r in run_rows),attempts=len(rows),
                 types=dict(Counter(r['type'] for r in rows)),verdicts=dict(Counter(r['verdict'] for r in rows)),
                 small_adjustments=sum(r['small_adjustment'] for r in rows))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
