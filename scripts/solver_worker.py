"""Standalone worker: run with the solver's own Python; inputs/outputs are JSON."""
import json, math, os, sys, time
from pathlib import Path

solver, source, destination, seconds, seed = sys.argv[1:]
seconds = float(seconds); seed = int(seed)
data = json.loads(Path(source).read_text())
S = 1000.0
def points(vertices): return [(float(x)*S, float(y)*S) for x,y in vertices]
stones = data['stones']
items = [points(x['outline']) for x in stones]
patio = points(data['patio'])
started = time.monotonic()
result = {'solver': solver, 'seed': seed, 'poses': [], 'settings': {}}
if solver == 'packingsolver':
    import packingsolver.irregular as p
    builder = p.InstanceBuilder(); builder.set_objective(p.Objective.Knapsack)
    builder.add_bin_type(p.build_shape(patio), copies=1, item_bin_minimum_spacing=10.1)
    builder.set_item_item_minimum_spacing(10.1)
    for shape in items:
        builder.add_item_type(p.build_shape(shape), copies=1, allowed_rotations=[(0,360,False)])
    params = p.OptimizeParameters(); params.time_limit = seconds; params.verbosity_level = 0
    output = p.optimize(builder.build(), params)
    certificate = destination + '.certificate.json'; output.solution.write(certificate)
    raw = json.loads(Path(certificate).read_text())
    for bin in raw.get('bins', []):
        for item in bin['items']:
            if item.get('mirror'): raise ValueError('unexpected reflection')
            result['poses'].append([int(item['id']), item['x']/S, item['y']/S, math.radians(item['angle'])])
    result['settings'] = {'time_limit': seconds, 'rotation': 'continuous', 'spacing': .0101, 'edge': .0101}
elif solver == 'unesting':
    import u_nesting
    angles = list(range(0, 360, 30))
    geoms = [{'id': s['id'], 'polygon': item, 'rotations': angles} for s,item in zip(stones,items)]
    strategy = os.environ.get('PATIO_SOLVER_STRATEGY', 'nfp')
    gap = float(os.environ.get('PATIO_SOLVER_GAP', '.0101'))
    edge = float(os.environ.get('PATIO_SOLVER_EDGE', '.0101'))
    config = {'strategy':strategy, 'spacing':gap*S, 'margin':edge*S, 'time_limit_ms':int(seconds*1000)}
    if strategy in ('ga','brkga','sa','gdrr','alns'): config['seed'] = seed
    raw = u_nesting.solve_2d(geoms, {'polygon': patio}, config)
    result['raw'] = raw
    ids = {s['id']:i for i,s in enumerate(stones)}
    for p in raw.get('placements', []):
        result['poses'].append([ids[p['geometry_id']], p['position'][0]/S, p['position'][1]/S, p['rotation'][0]])
    result['settings'] = {'time_limit':seconds, 'strategy':strategy, 'rotation_step_degrees':30, 'spacing':gap, 'edge':edge}
elif solver == 'ironnest':
    import ironnest
    inset = points(data['inset'])
    gap = float(os.environ.get('PATIO_SOLVER_GAP', '.0101'))
    strategy = os.environ.get('PATIO_SOLVER_STRATEGY', 'sampling')
    budget = int(os.environ.get('PATIO_SOLVER_BUDGET', '50'))
    positions, unplaced = ironnest.nest(items, qty=[1]*len(items), container=inset, holes=[], min_sep=gap*S,
        rotations=list(range(0,360,30)), seed=seed, budget=budget, strategy=strategy, separation_effort='fast')
    result['poses'] = [[int(i),x/S,y/S,math.radians(a)] for i,x,y,a in positions]
    result['unplaced'] = unplaced
    result['settings'] = {'budget':budget, 'strategy':strategy, 'rotation_step_degrees':30, 'spacing':gap, 'edge':'pre-inset'}
else: raise ValueError(solver)
result['elapsed_seconds'] = time.monotonic()-started
Path(destination).write_text(json.dumps(result, indent=2))
