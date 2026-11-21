"""Existing native tools used as proposal/diagnostic helpers, never acceptance gates."""
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import shapely
from shapely.geometry import Point, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


def optional_tool(value, kind):
    if value is None:
        return None
    if value != 'auto':
        path = Path(value).expanduser().absolute()
        if not path.is_file():
            raise ValueError(f'Missing {kind}: {path}')
        return str(path)
    env = os.environ.get('PATIO_PACKINGSOLVER_PYTHON' if kind == 'native_python' else 'PATIO_JTS_JAR')
    if env:
        return optional_tool(env, kind)
    paths = ([Path.home()/'miniconda3/envs/patio-packingsolver/bin/python'] if kind == 'native_python'
             else [Path.home()/'.cache/elliot-patio/jts-core-1.20.0.jar'])
    return next((str(p.absolute()) for p in paths if p.is_file()), None)


def coverage_diagnostics(layout, contract, data):
    """Same strict radius test as the old scan; index candidates once for all stages."""
    import corner_growth as c
    from flagstone.phase1 import transformed
    started = time.monotonic()
    fixed = [transformed(layout.shapes[r['stone_id']], r['pose']) for r in data['parent_history']]
    obstacles = unary_union([layout.patio.boundary, *fixed])
    polygons = np.array([transformed(layout.shapes[r['stone_id']], r['pose']) for r in data['candidates']], dtype=object)
    tree = shapely.STRtree(polygons)
    index_seconds = time.monotonic()-started
    zero = []
    for stage in range(len(data['parent_history'])+2, data['target']+1, 2):
        pts = c.coverage_samples(layout.patio, c.region(contract, stage).intersection(layout.patio), .16)
        pts = pts[shapely.distance(pts, obstacles) >= 1/12]
        covered = np.zeros(len(pts), dtype=bool)
        pairs, distances = tree.query_nearest(pts, max_distance=1/12, return_distance=True, all_matches=False)
        if pairs.shape[1]:
            # Only one nearest witness is needed, not every covering pose.
            # The query's radius includes equality; coverage remains strict.
            covered[pairs[0, distances < 1/12]] = True
        zero.extend(dict(stage=stage, center=[p.x, p.y]) for p in pts[~covered])
    return dict(candidates=len(polygons), zero_cover=zero,
                index_seconds=index_seconds, coverage_seconds=time.monotonic()-started-index_seconds)


def native_input(trial, parent, target, output):
    import corner_growth as c
    from flagstone.phase1 import transformed
    _, contract, layout = c.load_trial(trial)
    history = c.checked_history(layout, contract, json.loads((Path(parent)/'history.json').read_text()))
    used = {r['stone_id'] for r in history}
    def proxy(g):
        q = g.simplify(.002, preserve_topology=True).buffer(.003, quad_segs=4)
        return q if q.geom_type == 'Polygon' and q.covers(g) else g
    def coords(g):
        return [[float(x), float(y)] for x, y in orient(g).exterior.coords[:-1]]
    items = [dict(stone_id=sid, outline=coords(proxy(g))) for sid,g in layout.shapes.items() if sid not in used]
    # Keep the physical patio as the bin. Search-only exclusions discourage
    # full groups on the far side of the patio; they are never gap obstacles
    # or acceptance geometry. The original proposal family remains available.
    domain=c.region(contract,target).intersection(layout.patio)
    a,b,d,e=domain.bounds; padding=.75
    x0,y0,x1,y1=layout.patio.bounds
    rectangles=[box(x0,y0,a-padding,y1),box(d+padding,y0,x1,y1),
                box(x0,y0,x1,b-padding),box(x0,e+padding,x1,y1)]
    exclusions=[]
    for rectangle in rectangles:
        clipped=rectangle.intersection(layout.patio)
        parts=list(clipped.geoms) if hasattr(clipped,'geoms') else [clipped]
        for part in parts:
            if not part.is_empty and part.geom_type=='Polygon':
                exclusions.append(dict(outline=coords(part),holes=[list(r.coords)[:-1] for r in part.interiors]))
    payload = dict(patio=coords(layout.patio), parent_history=history, target=target,
                   fixed=[coords(transformed(proxy(layout.shapes[r['stone_id']]), r['pose'])) for r in history],
                   items=items, minimum_clearance_ft=.5/12+.0001,
                   proposal_exclusions=exclusions, proposal_focus_padding_ft=padding,
                   proposal_proxies='checked enclosing simplify(.002ft)/buffer(.003ft); original verification required')
    if not 0 < target-len(history) <= len(items):
        raise ValueError('Invalid native continuation size')
    c.write(output, payload)
    return payload


def check_native_patterns(trial, native_file, parent, target, output, groups):
    """Rank full native groups using independent all-prefix checks; never promote here."""
    import corner_growth as c
    from corner_controller import rank
    from flagstone.phase1 import transformed
    _, contract, layout = c.load_trial(trial)
    history = c.checked_history(layout, contract, json.loads((Path(parent)/'history.json').read_text()))
    payload = json.loads(Path(native_file).read_text())
    if payload['parent_history'] != history or payload['target'] != target:
        raise ValueError('Native patterns belong to a different parent/target')
    destination = Path(output); destination.mkdir()
    ranked = []; seen = set(); rejected_locality=0
    for pattern in payload['patterns']:
        records = pattern['records']
        if len(records) != target-len(history):
            raise ValueError('Partial native group')
        c.checked_history(layout,contract,history+[
            dict(step=len(history)+i+1,stone_id=r['stone_id'],pose=r['pose']) for i,r in enumerate(records)])
        domain=c.region(contract,target).intersection(layout.patio)
        if any(transformed(layout.shapes[r['stone_id']],r['pose']).distance(domain) >= 1/12 for r in records):
            # The full patio is the native bin. A group placed elsewhere is
            # retained in the raw evidence but cannot help this continuation.
            rejected_locality+=1
            continue
        # Native output order is not a prefix certificate. Try both directions,
        # including a corner-first order, under a small explicit group budget.
        orders = [records, sorted(records, key=lambda r: max(r['pose'][0], r['pose'][1])), list(reversed(records))]
        for order in orders:
            complete = history + [dict(step=len(history)+i+1, stone_id=r['stone_id'], pose=r['pose']) for i,r in enumerate(order)]
            complete = c.checked_history(layout, contract, complete)
            key = tuple((r['stone_id'], *r['pose']) for r in complete)
            if key in seen: continue
            seen.add(key)
            evaluation = c.evaluate_history(trial, complete)
            folder = destination/f'group_{len(ranked):03d}'; folder.mkdir()
            c.write(folder/'history.json', complete); c.write(folder/'evaluation.json', evaluation)
            ranked.append((rank(evaluation), str(folder/'history.json')))
            if len(ranked) >= groups*3: break
        if len(ranked) >= groups*3: break
    ranked.sort()
    return dict(histories=[p for _,p in ranked[:groups]], evaluated_groups=len(ranked),
                rejected_locality=rejected_locality, acceptance_authority=False)


def jts_crosscheck(layout, contract, evaluation, jar, output, timeout=20):
    """Recompute every returned witness on originals; an approximate pass grants nothing."""
    import corner_growth as c
    from flagstone.phase1 import transformed
    output = Path(output); runtime = output.parent/'jts_runtime'; runtime.mkdir(exist_ok=True)
    if not (runtime/'CornerGapProbe.class').exists():
        subprocess.run(['javac','-cp',jar,'-d',str(runtime),str(Path(__file__).with_name('CornerGapProbe.java'))],
                       check=True, capture_output=True, text=True, timeout=timeout)
    lines = []; cases = []; stones = []
    for n,r in enumerate(evaluation['history'],1):
        stones.append(transformed(layout.shapes[r['stone_id']], r['pose']))
        obstacles = unary_union([layout.patio.boundary, *stones])
        domain = c.region(contract,n).intersection(layout.patio)
        cases.append((obstacles, domain))
        lines.append(f'{n}\t{.001/12}\t{obstacles.wkt}\t{domain.wkt}')
    input_file = output/'jts_input.tsv'; input_file.write_text('\n'.join(lines)+'\n')
    run = subprocess.run(['java','-XX:ActiveProcessorCount=1','-cp',f'{runtime}{os.pathsep}{jar}',
                          'CornerGapProbe',str(input_file)], check=True,capture_output=True,text=True,timeout=timeout)
    rows = run.stdout.splitlines()
    if len(rows) != len(cases): raise ValueError('Incomplete JTS batch')
    checked = []; witnesses = []
    for n,(line,(obstacles,domain),result) in enumerate(zip(rows,cases,evaluation['results']),1):
        fields = line.split('\t')
        if len(fields) != 5 or fields[0] != str(n): raise ValueError('Malformed JTS row')
        values = [float(v) for v in fields[1:]]
        if not all(math.isfinite(v) for v in values): raise ValueError('Nonfinite JTS output')
        center = Point(values[:2])
        if not domain.covers(center): raise ValueError('JTS witness outside center domain')
        diameter = 24*center.distance(obstacles)
        if (result['passed'] and diameter >= 2) or diameter > result['upper_diameter_in']+1e-7:
            raise RuntimeError('JTS witness contradicts certified upper bound')
        if abs(diameter-24*values[2]) > 1e-7:
            raise ValueError('JTS radius disagrees with original geometry')
        checked.append(dict(step=n,center=list(center.coords[0]),candidate_diameter_in=diameter))
        if diameter >= 2: witnesses.append(dict(step=n,center=list(center.coords[0])))
    input_file.unlink()
    return dict(status='checked', acceptance_authority=False, candidates=checked, witnesses=witnesses)
