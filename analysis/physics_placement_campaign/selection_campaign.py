"""Resumable, geometry-only candidate selection experiment.

Each target/piece is a checkpoint. Search every eligible piece rather than
refining only the top few pieces by area. Never writes historical reviews.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import html
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy import optimize, signal

import build_prefit_review as b
from selection_geometry import AXES, boundary_features, comparison, profile, screen
from nearfit_geometry import shrink_probe, nearfit_checks, translation_witness

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE/'selection_v4_aggressive'
DEFAULT_CONFIG = dict(
    angle_step_degrees=5, raster_step_ft=.035, poses_per_piece=3,
    refine_iterations=40, boundary_spacing_ft=.02,
    min_area_ratio=.50, max_area_ratio=1.10,
    screen=dict(physical_overlap_sqft=1e-6, clearance_intrusion_fraction=.005,
                clearance_depth_inches=.4, coverage=.65, outside_pocket_fraction=.05),
    comparison_tolerance_scales=[.5, 1., 2.],
    nearfit=dict(scales=[.98,.95,.92,.80], physical_overlap_fraction=.36,
                 clearance_intrusion_fraction=.40, clearance_depth_inches=2.25,
                 max_translation_inches=1.5, translation_step_inches=.125,
                 directions=24, max_neighbors=4, beam_width=32, locked_neighbors=[]),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependencies():
    return {str(p.relative_to(HERE)): digest(p) for p in [
        Path(__file__), HERE/'selection_geometry.py', HERE/'nearfit_geometry.py', HERE/'build_prefit_review.py',
        b.SRC/'source.json', b.SRC/'manifest.json', b.DEST/'targets.json',
        *sorted((b.DEST/'refined_results').glob('*.json'))]}


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


@contextmanager
def lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another process is using {out}') from None
        yield


def prepare(out, config, target_ids=None):
    if (out/'manifest.json').exists():
        raise RuntimeError('Already prepared; resume with run, or use a new output directory.')
    near=config.get('nearfit')
    if near is not None:
        if any(not 0<s<1 for s in near['scales']) or len(set(near['scales']))!=len(near['scales']):
            raise ValueError('Near-fit scales must be distinct and in (0, 1)')
        if (near['max_translation_inches']<=0 or near['translation_step_inches']<=0 or
            near['directions']<4 or near['beam_width']<1 or near['max_neighbors']<1):
            raise ValueError('Invalid bounded-movement search configuration')
    # Frozen candidate-independent masks. Opening sensitivity is a separate
    # experiment; all choices for one target see exactly the same geometry.
    targets = json.loads((b.DEST/'targets.json').read_text())
    if target_ids:
        requested = set(target_ids)
        missing = requested-{t['id'] for t in targets}
        if missing:
            raise ValueError(f'Unknown targets: {sorted(missing)}')
        targets = [t for t in targets if t['id'] in requested]
    tasks = []
    for t in targets:
        active = set(b.ACTIVE)-set(t['removed'])-({t['old']} if t['old'] else set())
        pocket = shapely.from_wkt(t['pocket'])
        for sid in sorted(b.STONES):
            if sid in active or (t['family']=='insertion' and sid in t['removed']):
                continue
            ratio = b.LOCAL[sid].area/pocket.area
            # Always search the old stone, even when outside size prefilters.
            scales=config.get('nearfit',{}).get('scales',[])
            # Scale-aware prefilter: a large piece must not be discarded before
            # its smaller probe can be tested. The manifest still uses full IDs.
            possible=any(config['min_area_ratio']<=ratio*s*s<=config['max_area_ratio'] for s in [1.,*scales])
            if sid != t['old'] and not possible:
                continue
            tasks.append(dict(id=f"{t['id']}__{sid}", target=t['id'], stone=sid))
    manifest = dict(version=4, created_unix=time.time(), config=config,
                    dependencies=dependencies(), targets=targets, tasks=tasks,
                    physics_steps=0, purpose='Unvalidated selection hypotheses; human review required')
    atomic_json(out/'manifest.json', manifest)
    (out/'results').mkdir(exist_ok=True)
    print(f'Prepared {len(targets)} openings, {len(tasks)} target/piece searches', flush=True)
    return manifest


def load_manifest(out):
    manifest = json.loads((out/'manifest.json').read_text())
    if manifest['dependencies'] != dependencies():
        raise RuntimeError('Code/source hashes changed. Use a new output directory; mixed runs are forbidden.')
    return manifest


def environment(target):
    p, free = shapely.from_wkt(target['pocket']), shapely.from_wkt(target['free'])
    active = set(b.ACTIVE)-set(target['removed'])-({target['old']} if target['old'] else set())
    obstacles = unary_union([b.ORIGINAL[s] for s in sorted(active)])
    return p, free, obstacles


def evaluate(sid, pose, p, free, obstacles, config, origin):
    g = b.world(sid, pose)
    metrics = {**b.measures(g,p,free,obstacles),
               **boundary_features(g,p,free,config['boundary_spacing_ft'])}
    return dict(stone=sid, pose=list(map(float,pose)), origin=origin,
                metrics=metrics, fit_gate=screen(metrics,config),
                benefit_gate='not_compared', settled_gate='not_run', physics_steps=0)


def coarse_poses(sid, p, free, config, factor=1.):
    step = config['raster_step_ft']
    x0,y0,x1,y1 = p.bounds
    xs = np.arange(x0-.25,x1+.25+step,step)
    ys = np.arange(y0-.25,y1+.25+step,step)
    pm, fm = b.mask_at(p,xs,ys), b.mask_at(free,xs,ys)
    cost = np.where(pm>.5,-1.,.8)+np.where(fm<.5,24.,0.)
    coords = np.asarray(shrink_probe(b.SIMPLE[sid],factor).exterior.coords)
    trials = []
    for degrees in range(0,360,config['angle_step_degrees']):
        angle = math.radians(degrees);c,s = math.cos(angle),math.sin(angle)
        g = Polygon(coords@np.array([[c,s],[-s,c]]))
        a,bb,cc,d = g.bounds
        kx,ky = np.arange(a+step/2,cc,step),np.arange(bb+step/2,d,step)
        if not len(kx) or not len(ky) or len(kx)>len(xs) or len(ky)>len(ys):
            continue
        km = b.mask_at(g,kx,ky)
        conv = signal.fftconvolve(cost,km[::-1,::-1],mode='valid')*step**2
        iy,ix = np.unravel_index(np.argmin(conv),conv.shape)
        trials.append((float(conv[iy,ix]),[float(xs[ix]-kx[0]),float(ys[iy]-ky[0]),angle]))
    # Diverse angular basins, rather than neighboring 5-degree versions only.
    selected = []
    for _,pose in sorted(trials):
        if all(abs(math.atan2(math.sin(pose[2]-q[2]),math.cos(pose[2]-q[2]))) >= math.radians(20) for q in selected):
            selected.append(pose)
        if len(selected) >= config['poses_per_piece']:
            break
    return selected


def evaluate_nearfit(sid,pose,factor,target,p,free,obstacles,config):
    # Full-size geometry is always the actual proposal and primary metrics.
    candidate=evaluate(sid,pose,p,free,obstacles,config,f'nearfit_probe_{factor:g}')
    full=b.world(sid,pose)
    probe=shrink_probe(full,factor)
    pm={**b.measures(probe,p,free,obstacles),
        **boundary_features(probe,p,free,config['boundary_spacing_ft'])}
    pg=screen(pm,config)
    candidate['shrink_probe']=dict(linear_scale=factor,diagnostic_only=True,
                                  metrics=pm,fit_gate=pg)
    nc=config['nearfit']
    checks=nearfit_checks(candidate['metrics'],pg,b.PATIO.buffer(-b.GAP).buffer(1e-8).covers(full),nc)
    checks['needs_space']=not all(candidate['fit_gate']['checks'][k] for k in ('physical_overlap','clearance_area','clearance_depth'))
    candidate['nearfit_gate']=dict(status='reject',checks=checks)
    if candidate['fit_gate']['status']=='passes_provisional_screen':
        candidate['nearfit_gate']['status']='full_size_screen_passes'
    elif all(checks.values()):
        active=set(b.ACTIVE)-set(target['removed'])-({target['old']} if target['old'] else set())
        neighbors={s:b.ORIGINAL[s] for s in sorted(active)}
        witness=translation_witness(full,neighbors,b.PATIO,b.GAP,nc)
        witness['neighbor_poses']={s:[b.STONES[s]['pose'][0]+dx,b.STONES[s]['pose'][1]+dy,b.STONES[s]['pose'][2]]
                                   for s,(dx,dy) in witness['translations_ft'].items()}
        candidate['space_plan']=witness
        checks['bounded_neighbor_slack']=witness['status'] in ('bounded_translation_witness','no_movement_needed')
        candidate['nearfit_gate']['status']='passes_nearfit_review_screen' if all(checks.values()) else 'reject'
    candidate['benefit_gate']='conditional_value_requires_review'
    return candidate


def solve_task(task, target, config, out):
    start = time.time()
    p,free,obstacles = environment(target)
    sid = task['stone']
    starts = coarse_poses(sid,p,free,config)
    if sid == target['old']:
        starts.insert(0,list(b.STONES[sid]['pose']))
    # Earlier poses are useful extra starts, never a substitute for all-piece search.
    old_path = b.DEST/'refined_results'/f"{target['id']}.json"
    if old_path.exists():
        starts += [c['pose'] for c in json.loads(old_path.read_text())['candidates'] if c['stone']==sid]
    candidates = []
    if sid == target['old']:
        candidates.append(evaluate(sid,b.STONES[sid]['pose'],p,free,obstacles,config,'retain_original'))
    for q0 in starts:
        # Two complementary pose objectives: coverage and edge/corner tails.
        # These are search heuristics only, never the final preference rule.
        for mode in ('silhouette','boundary_tail'):
            pp,_,_ = profile(p.exterior,.07)
            def objective(q):
                g = b.world(sid,q,False)
                intrusion = g.difference(free).area
                value = -g.intersection(p).area+1.2*g.difference(p).area+40*intrusion
                if mode == 'boundary_tail':
                    ds = np.asarray(shapely.distance(pp,g))
                    value += 2*float(np.mean(ds**2)) + float(np.quantile(ds,.9))**2
                return value
            bounds = [(q0[0]-.25,q0[0]+.25),(q0[1]-.25,q0[1]+.25),(q0[2]-.18,q0[2]+.18)]
            opt = optimize.minimize(objective,q0,method='Powell',bounds=bounds,
                                    options=dict(maxiter=config['refine_iterations'],xtol=.00015,ftol=.00005))
            candidate = evaluate(sid,opt.x,p,free,obstacles,config,mode)
            candidate['optimizer_success'] = bool(opt.success)
            candidates.append(candidate)
    # Shrunk silhouettes discover additional basins. They never replace the
    # full-size outline in saved poses, physical metrics, or movement checks.
    for factor in config.get('nearfit',{}).get('scales',[]):
        for q0 in coarse_poses(sid,p,free,config,factor):
            pp,_,_=profile(p.exterior,.07)
            wall=b.PATIO.buffer(-b.GAP)
            def probe_objective(q):
                full=b.world(sid,q,False)
                g=shrink_probe(full,factor)
                ds=np.asarray(shapely.distance(pp,g))
                return (-g.intersection(p).area+1.2*g.difference(p).area+
                        40*g.difference(free).area+80*full.difference(wall).area+
                        2*float(np.mean(ds**2))+float(np.quantile(ds,.9))**2)
            bounds=[(q0[0]-.25,q0[0]+.25),(q0[1]-.25,q0[1]+.25),(q0[2]-.18,q0[2]+.18)]
            opt=optimize.minimize(probe_objective,q0,method='Powell',bounds=bounds,
                                  options=dict(maxiter=config['refine_iterations'],xtol=.00015,ftol=.00005))
            candidate=evaluate_nearfit(sid,opt.x,factor,target,p,free,obstacles,config)
            candidate['optimizer_success']=bool(opt.success)
            candidates.append(candidate)
            # Area penalties can leave tiny blocking bumps in otherwise good
            # probes. A separate depth pass searches for a genuinely fitting
            # reduced silhouette while keeping the FULL stone inside the wall.
            inset=p.simplify(.002,preserve_topology=True).buffer(-.003)
            if inset.is_empty:
                continue
            def feasibility(q):
                full=b.world(sid,q,False)
                probe=shrink_probe(full,factor)
                points=shapely.points(shapely.get_coordinates(shapely.segmentize(probe.exterior,.035)))
                ds=np.asarray(shapely.distance(points,inset))
                fp=shapely.points(shapely.get_coordinates(shapely.segmentize(full.exterior,.035)))
                wd=np.asarray(shapely.distance(fp,wall))
                return float(ds.max()+5*np.mean(ds**2)+5*wd.max()+20*np.mean(wd**2))
            q1=opt.x
            fit=optimize.minimize(feasibility,q1,method='Powell',
                bounds=[(q1[0]-.2,q1[0]+.2),(q1[1]-.2,q1[1]+.2),(q1[2]-.15,q1[2]+.15)],
                options=dict(maxiter=config['refine_iterations'],xtol=.0001,ftol=.00001))
            if np.linalg.norm(fit.x-q1)>.001:
                candidate=evaluate_nearfit(sid,fit.x,factor,target,p,free,obstacles,config)
                candidate['origin']+='_depth_refined'
                candidate['optimizer_success']=bool(fit.success)
                candidates.append(candidate)
    result = dict(task=task, candidates=candidates, seconds=time.time()-start, physics_steps=0)
    atomic_json(out/'results'/f"{task['id']}.json",result)
    return task['id'],len(candidates),round(result['seconds'],2)


def run(out,workers,max_tasks):
    manifest = load_manifest(out)
    todo = [t for t in manifest['tasks'] if not (out/'results'/f"{t['id']}.json").exists()]
    if max_tasks is not None:
        todo = todo[:max_tasks]
    targets = {t['id']:t for t in manifest['targets']}
    print(f'Searching {len(todo)} target/piece tasks, {workers} workers; no physics',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(solve_task,t,targets[t['target']],manifest['config'],out):t for t in todo}
        errors = []
        for future in as_completed(futures):
            try:
                print(future.result(),flush=True)
            except Exception as exc:
                errors.append(dict(task=futures[future],error=repr(exc)))
                print(errors[-1],flush=True)
        atomic_json(out/'last_run_errors.json',errors)
    if errors:
        raise RuntimeError(f'{len(errors)} tasks failed; successful tasks checkpointed, retry run to resume')


def compact(candidate):
    c = {**candidate,'metrics':dict(candidate['metrics'])}
    c['metrics'].pop('boundary_profile',None)
    if 'shrink_probe' in candidate:
        c['shrink_probe']={**candidate['shrink_probe'],'metrics':dict(candidate['shrink_probe']['metrics'])}
        c['shrink_probe']['metrics'].pop('boundary_profile',None)
    return c


def nearfit_axes(candidate):
    # Shape is measured at probe size; interference and movement remain full size.
    return {**candidate['shrink_probe']['metrics'],
            'clearance_intrusion_fraction':candidate['metrics']['clearance_intrusion_fraction'],
            'max_clearance_intrusion_inches':candidate['metrics']['max_clearance_intrusion_inches'],
            'shrink_fraction':1-candidate['shrink_probe']['linear_scale'],
            'movement_inches':candidate['space_plan']['max_neighbor_translation_inches']}


def nearfit_dominates(a,b):
    axes=[*AXES,'shrink_fraction','movement_inches']
    a,b=nearfit_axes(a),nearfit_axes(b)
    keys=[k for k in axes if a.get(k) is not None and b.get(k) is not None]
    return bool(keys) and all(a[k]<=b[k] for k in keys) and any(a[k]<b[k] for k in keys)


def report(out):
    manifest = load_manifest(out)
    reports = []
    for target in manifest['targets']:
        tasks = [t for t in manifest['tasks'] if t['target']==target['id']]
        complete = all((out/'results'/f"{t['id']}.json").exists() for t in tasks)
        candidates = [c for t in tasks if (out/'results'/f"{t['id']}.json").exists()
                      for c in json.loads((out/'results'/f"{t['id']}.json").read_text())['candidates']]
        eligible = [c for c in candidates if c['fit_gate']['status']=='passes_provisional_screen']
        baseline = next((c for c in candidates if c['origin']=='retain_original'),None)
        # The original remains a legitimate competitor even when its opening
        # coverage is below the new-piece screen. Otherwise undersized originals
        # would be excluded from their own comparison by construction.
        old_options = [c for c in candidates if c['stone']==target['old'] and
                       all(c['fit_gate']['checks'][k] for k in ('physical_overlap','clearance_area','clearance_depth'))]
        references = old_options + ([baseline] if baseline and baseline not in old_options else [])
        for c in eligible:
            if target['old']:
                comparisons = [comparison(c['metrics'],ref['metrics'],scale)
                               for ref in references for scale in manifest['config']['comparison_tolerance_scales']]
                c['benefit_gate'] = ('robust_geometric_advantage' if comparisons and all(
                    x['verdict']=='candidate_dominates' for x in comparisons) else 'retain_or_review_tradeoff')
                c['comparisons_to_original_options'] = comparisons
            else:
                c['benefit_gate'] = 'insertion_value_requires_review'
        # Exact Pareto dominance (zero tolerance) is transitive; tolerance-based
        # dominance is used only in the paired comparison, avoiding frontier cycles.
        frontier = [c for c in eligible if not any(
            comparison(other['metrics'],c['metrics'],0)['verdict']=='candidate_dominates'
            for other in eligible if other is not c)]
        nearfits=[c for c in candidates if c.get('nearfit_gate',{}).get('status')=='passes_nearfit_review_screen']
        for c in nearfits:
            c['benefit_gate']='conditional_value_requires_review'
            # Keep the honest full-size comparison to the unchanged original;
            # a good shrink probe cannot erase its interference disadvantage.
            c['comparisons_to_original_options']=[comparison(c['metrics'],ref['metrics']) for ref in references]
        near_frontier=[c for c in nearfits if not any(nearfit_dominates(other,c) for other in nearfits if other is not c)]
        # Preserve the best passing pose at each probe scale even when it is
        # dominated overall, so the requested aggressive probes remain reviewable.
        probe_representatives={str(factor):compact(min(
            [c for c in nearfits if c['shrink_probe']['linear_scale']==factor],
            key=lambda c:(c['shrink_probe']['metrics']['edge_excess_q90_inches'],
                          c['space_plan']['max_neighbor_translation_inches'],c['stone'])))
            for factor in sorted({c['shrink_probe']['linear_scale'] for c in nearfits})}
        reports.append(dict(target=target['id'], complete=complete, candidates=len(candidates),
                            passing_poses=len(eligible), baseline=compact(baseline) if baseline else None,
                            original_options=[compact(c) for c in old_options],
                            frontier=[compact(c) for c in frontier],
                            nearfit_passing_poses=len(nearfits),nearfit_frontier=[compact(c) for c in near_frontier],
                            nearfit_probe_representatives=probe_representatives,
                            action='human_review_required' if complete else 'incomplete_search_do_not_select'))
    atomic_json(out/'report.json',reports)
    lines = ['# Geometry selection experiment','',
             'No physics or automatic board changes. Frontiers are unvalidated review candidates.', '',
             '| Opening | Complete | Poses | Fixed-neighbor passing | Frontier | Near-fit passing | Near-fit frontier |', '|---|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {r['target']} | {r['complete']} | {r['candidates']} | {r['passing_poses']} | {len(r['frontier'])} | {r['nearfit_passing_poses']} | {len(r['nearfit_frontier'])} |" for r in reports]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Reported {len(reports)} targets; {sum(r["complete"] for r in reports)} complete',flush=True)


def review(out,limit):
    """Deterministic metric extremes per opening; no visual/manual curation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from render_prefit_review import draw,lines
    from shapely.affinity import translate
    manifest = load_manifest(out)
    targets = {t['id']:t for t in manifest['targets']}
    reports = json.loads((out/'report.json').read_text())
    dest = out/'review'
    dest.mkdir(exist_ok=True)
    if (dest/'labels.json').exists():
        raise RuntimeError('Review is frozen; use a fresh run directory for a different panel.')
    records = []
    # Fixed opening order, then metric extremes; all alternatives remain in report.
    for r in sorted(reports,key=lambda r:r['target']):
        if not r['complete'] or not (r['frontier'] or r.get('nearfit_frontier')):
            continue
        choices = []
        for key in ['edge_angle_q75_degrees','edge_excess_q90_inches',
                    'corner_deficit_q90_inches','residual_core_one_inch_sqft']:
            valid = [c for c in r['frontier'] if c['metrics'].get(key) is not None]
            if not valid:
                continue
            c = min(valid,key=lambda c:(c['metrics'][key],c['stone'],c['pose']))
            if not any(c['stone']==v['stone'] and np.linalg.norm(np.array(c['pose'])-v['pose'])<.01 for v in choices):
                choices.append(c)
        # Include a different piece, even if the extrema all chose one stone.
        other_pieces = [c for c in r['frontier'] if c['stone'] not in {v['stone'] for v in choices}]
        if other_pieces:
            choices.append(min(other_pieces,key=lambda c:(c['metrics']['edge_excess_q90_inches'],c['stone'])))
        near_options=r.get('nearfit_frontier',[])
        for key in ('movement','shape'):
            if near_options:
                c=min(near_options,key=lambda c:(c['space_plan']['max_neighbor_translation_inches'],
                      -c['shrink_probe']['linear_scale']) if key=='movement' else
                      (c['shrink_probe']['metrics']['edge_excess_q90_inches'],c['stone']))
                if not any(c['stone']==v['stone'] and np.linalg.norm(np.array(c['pose'])-v['pose'])<.01 for v in choices):
                    choices.append(c)
        aggressive=r.get('nearfit_probe_representatives',{}).get('0.8')
        movement_extreme=max(near_options,key=lambda c:c['space_plan']['max_neighbor_translation_inches']) if near_options else None
        for c in (aggressive,movement_extreme):
            if c and not any(c['stone']==v['stone'] and np.linalg.norm(np.array(c['pose'])-v['pose'])<.01 for v in choices):
                choices.append(c)
        original_poses = [c for c in r['original_options'] if c['origin']!='retain_original']
        if original_poses:
            original = min(original_poses,key=lambda c:c['metrics']['edge_excess_q90_inches'])
            original['benefit_gate']='original_reposition_comparator'
            choices.insert(0,original)
        t = targets[r['target']]
        fig,axes = plt.subplots(1,len(choices)+1,figsize=(4.5*(len(choices)+1),6.7),squeeze=False)
        draw(axes[0,0],dict(target=t,candidate=choices[0]),False)
        labels = []
        for i,c in enumerate(choices,1):
            ax = axes[0,i]
            draw(ax,dict(target=t,candidate=c),True)
            label = 'original repositioned' if c['stone']==t['old'] else 'reserve piece'
            is_near=c.get('nearfit_gate',{}).get('status')=='passes_nearfit_review_screen'
            if is_near:
                label=f"near-fit · {c['shrink_probe']['linear_scale']:.0%} probe"
                lines(ax,shrink_probe(b.world(c['stone'],c['pose']),c['shrink_probe']['linear_scale']).boundary,
                      color='#215c9c',linestyle='--',linewidth=1.1,zorder=7)
                for sid,(dx,dy) in c['space_plan']['translations_ft'].items():
                    lines(ax,translate(b.ORIGINAL[sid],dx,dy).boundary,color='#23835e',linestyle='--',linewidth=1.,zorder=6)
                    point=b.ORIGINAL[sid].centroid
                    ax.annotate('',xy=(point.x+dx,point.y+dy),xytext=(point.x,point.y),
                                arrowprops=dict(arrowstyle='->',color='#23835e'),zorder=7)
            ax.set_title(f"{chr(64+i)} · {c['stone']} · {label}",fontsize=10)
            m = c['metrics']
            benefit = {'original_reposition_comparator':'Compare with retaining this piece',
                       'robust_geometric_advantage':'Geometric advantage: needs review',
                       'conditional_value_requires_review':'Conditional on making room: review',
                       'insertion_value_requires_review':'Insertion value: needs review'}.get(c['benefit_gate'],'Tradeoff: retain by default')
            info=(f"probe edge tail {c['shrink_probe']['metrics']['edge_excess_q90_inches']:.2f} in | "
                  f"neighbor move ≤{c['space_plan']['max_neighbor_translation_inches']:.2f} in") if is_near else (
                  f"edge tail {m['edge_excess_q90_inches']:.2f} in | angle {m['edge_angle_q75_degrees']:.1f}°")
            ax.set_xlabel(f"{info}\nfull-size clearance depth {m['max_clearance_intrusion_inches']:.2f} in\n{benefit}",fontsize=9)
            labels.append(dict(option=chr(64+i),candidate=c,plausible_fit=None,
                               improvement_over_current=None,preferred_over_other_options=None,
                               neighbor_motion_acceptable=None,notes=''))
        fig.suptitle(t['id']+(' · synthetic removal' if t['removed'] else ' · original board'),fontsize=13)
        # Match framing across every option, including a repositioned original.
        xlim=(min(ax.get_xlim()[0] for ax in axes[0]),max(ax.get_xlim()[1] for ax in axes[0]))
        ylim=(min(ax.get_ylim()[0] for ax in axes[0]),max(ax.get_ylim()[1] for ax in axes[0]))
        for ax in axes[0]:
            ax.set_xlim(xlim);ax.set_ylim(ylim)
        fig.text(.5,.025,'Full-size proposals. Cyan: fixed pocket | purple: caps | red: clearance intrusion | dashed blue: shrink probe | dashed green: neighbor endpoints. No physics.',ha='center',fontsize=10)
        fig.subplots_adjust(left=.015,right=.985,top=.90,bottom=.23,wspace=.05)
        filename = f"{len(records)+1:03}_{t['id']}.png"
        fig.savefig(dest/filename,dpi=120)
        plt.close(fig)
        records.append(dict(target=t['id'],image=filename,options=labels,opening_mask_acceptable=None))
        if len(records)>=limit:
            break
    atomic_json(dest/'labels.json',records)
    content = ['<!doctype html><meta charset="utf-8"><title>Selection review</title>',
               '<style>body{font:16px system-ui;margin:28px}img{width:100%;height:auto}section{margin-bottom:40px}</style>',
               '<h1>Geometry selection review</h1><p>Compare plausible fit and improvement separately. '
               'Choices are deterministic frontier extremes. They are not certified placements. '
               'Labels and exact poses are in labels.json; no physics has run.</p>']
    content += [f'<section><h2>{html.escape(r["target"])}</h2><img src="{r["image"]}"></section>' for r in records]
    (dest/'index.html').write_text('\n'.join(content))
    print(f'Exported {len(records)} openings to {dest}',flush=True)


def diagnose(out):
    rows = []
    for r in json.loads((b.DEST/'placements.json').read_text()):
        t,c = r['target'],r['candidate']
        p,free,obstacles = environment(t)
        proposal = evaluate(c['stone'],c['pose'],p,free,obstacles,DEFAULT_CONFIG,'frozen_review_pose')
        reference = evaluate(t['old'],b.STONES[t['old']]['pose'],p,free,obstacles,DEFAULT_CONFIG,'retain_original') if t['old'] else None
        paired = comparison(proposal['metrics'],reference['metrics']) if reference else None
        rows.append(dict(number=r['number'],target=t['id'],proposal=proposal,reference=reference,
                         comparison=paired, review_only_replacement_recommendation=bool(paired and paired['verdict']=='candidate_dominates'),
                         evidence='Development diagnostic, not held-out validation'))
    atomic_json(out/'review_diagnostic.json',rows)
    atomic_json(out/'diagnostic_provenance.json',dict(dependencies=dependencies(),
        placements_sha256=digest(b.DEST/'placements.json'),config=DEFAULT_CONFIG,
        note='Post-hoc development: edge-only metrics initially favored 5/6; added angle and intrusion axes. Not validation.'))
    lines = ['# Six-case development diagnostic','',
             'Post-hoc hypotheses checked against already reviewed cases. No new poses or physics.', '',
             '| Case | Paired geometric result | Default disposition |', '|---|---|---|']
    lines += [f"| {r['number']:02} | {r['comparison']['verdict'] if r['comparison'] else 'insertion fit only'} | "
              f"{'review geometric advantage' if r['review_only_replacement_recommendation'] else 'retain / review'} |" for r in rows]
    lines += ['', 'Only #4 has no material disadvantage on these measured axes. #3/#5/#6 are tradeoffs, so retaining the original is the default. '
              'This reproduces the development comparison; it does not estimate generalization or an approval rate. '
              'The original-reorientation competition is evaluated by the full search, not by this frozen-pose diagnostic.']
    (out/'review_diagnostic.md').write_text('\n'.join(lines)+'\n')
    print([(r['number'],r['comparison']['verdict'] if r['comparison'] else 'fit_only') for r in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','run','report','review','diagnose'])
    ap.add_argument('--out',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--config',type=Path,help='Complete JSON config; see DEFAULT_CONFIG')
    ap.add_argument('--target',action='append',help='Prepare only this frozen opening (repeatable)')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--max-tasks',type=int,help='Bound a smoke run; omitted for full resume')
    ap.add_argument('--review-limit',type=int,default=12)
    args = ap.parse_args()
    if args.workers < 1 or (args.max_tasks is not None and args.max_tasks < 0):
        ap.error('workers must be positive; max-tasks must be nonnegative')
    out = args.out.resolve()
    # An output path must never alias any frozen campaign or review directory.
    protected = [b.SRC.resolve(),b.DEST.resolve()]
    if out == HERE or any(out==p or p in out.parents or out in p.parents for p in protected):
        ap.error('Choose an independent output directory outside frozen inputs')
    with lock(out):
        if args.command=='prepare':
            config = json.loads(args.config.read_text()) if args.config else DEFAULT_CONFIG
            if not 1 <= config['angle_step_degrees'] <= 90 or config['poses_per_piece'] < 1:
                raise ValueError('Invalid angle step or pose count')
            prepare(out,config,args.target)
        elif args.command=='run':
            run(out,args.workers,args.max_tasks)
        elif args.command=='report':
            report(out)
        elif args.command=='review':
            if args.review_limit<1:
                raise ValueError('review-limit must be positive')
            review(out,args.review_limit)
        else:
            diagnose(out)


if __name__=='__main__':
    main()
