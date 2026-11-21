"""Run PackingSolver with validated, atomic incumbent saves; search itself restarts.

Run with patio-packingsolver's Python from the project root. A rerun preserves
the best previous valid layout but does not resume internal optimizer state.
"""
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import packingsolver.irregular as p
from flagstone.phase1 import load_json, save_json, validate
from flagstone.diagnostics import sampled_voids

ap = argparse.ArgumentParser()
ap.add_argument('--input', default='examples/flagstone_reference/surplus_inventory.json')
ap.add_argument('--output', default='examples/long_search')
ap.add_argument('--seconds', type=float, default=60)
ap.add_argument('--gap', type=float, default=.01, help='Pair clearance in layout units')
ap.add_argument('--edge', type=float, default=.01, help='Edge clearance in layout units')
ap.add_argument('--allowance', type=float, default=.0001, help='Extra solver clearance in layout units')
ap.add_argument('--void-diameter', type=float, help='Diameter of large empty circle to monitor, in layout units')
ap.add_argument('--void-spacing', type=float, help='Sampling grid spacing in layout units')
args = ap.parse_args()
if not math.isfinite(args.seconds) or args.seconds <= 0: ap.error('seconds must be positive')
if not all(math.isfinite(v) and v >= 0 for v in (args.gap, args.edge, args.allowance)): ap.error('clearances must be finite and nonnegative')
if (args.void_diameter is None) != (args.void_spacing is None): ap.error('specify both void options')
if args.void_diameter is not None and not all(math.isfinite(v) and v>0 for v in (args.void_diameter,args.void_spacing)): ap.error('void options must be finite and positive')
out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
src = Path(args.input); digest = hashlib.sha256(src.read_bytes()).hexdigest()
settings = {'input_sha256':digest, 'gap':args.gap, 'edge':args.edge, 'solver_gap':args.gap+args.allowance,
            'solver_edge':args.edge+args.allowance, 'coordinate_scale':1000, 'rotations':'continuous'}
if args.void_diameter is not None: settings.update(void_diameter=args.void_diameter,void_spacing=args.void_spacing)
meta = out / 'settings.json'
if meta.exists() and json.loads(meta.read_text()) != settings:
    raise ValueError('Output directory contains different input or settings')
if not meta.exists(): meta.write_text(json.dumps(settings, indent=2))
layout = load_json(src)
best_path = out / 'best.json'; best_metrics = out / 'best_metrics.json'
if best_path.exists():
    existing = load_json(best_path)
    if validate(existing, gap=args.gap, edge=args.edge): raise ValueError('Saved incumbent invalid')
    best = sum(existing.shapes[s].area for i,s in enumerate(existing.ids) if existing.active[i]) / existing.patio.area
else: best = 0.0
void_best_path=out/'fewest_voids.json'; void_metrics=out/'fewest_voids_metrics.json'
void_key=None
if void_best_path.exists() and args.void_diameter is not None:
    previous=load_json(void_best_path)
    if validate(previous,gap=args.gap,edge=args.edge): raise ValueError('Saved gap incumbent invalid')
    v=sampled_voids(previous,spacing=args.void_spacing,diameter=args.void_diameter)
    c=sum(previous.shapes[s].area for i,s in enumerate(previous.ids) if previous.active[i])/previous.patio.area
    void_key=(v['broad_core_area'],-c)
start = time.monotonic(); lock = Lock(); count = 0

def save_candidate(output):
    global best, count, void_key
    with lock:
        count += 1
        temp_cert = out / 'candidate.tmp.json'
        output.solution.write(str(temp_cert))
        cert = json.loads(temp_cert.read_text())
        if cert is None:
            return
        candidate = load_json(src)
        seen = set()
        for bin in cert.get('bins', []):
            for item in bin['items']:
                i = int(item['id'])
                if i in seen or not 0 <= i < len(candidate.ids) or item.get('mirror'):
                    raise ValueError('Invalid solver item ID or reflection')
                seen.add(i); candidate.active[i] = True
                candidate.poses[i] = [item['x']/1000, item['y']/1000, math.radians(item['angle'])]
        errors = validate(candidate, gap=args.gap, edge=args.edge)
        score = sum(candidate.shapes[s].area for i,s in enumerate(candidate.ids) if candidate.active[i]) / candidate.patio.area
        if not errors and args.void_diameter is not None:
            v=sampled_voids(candidate,spacing=args.void_spacing,diameter=args.void_diameter)
            key=(v['broad_core_area'],-score)
            if void_key is None or key<void_key:
                tmp=out/'fewest_voids.tmp.json';save_json(candidate,tmp);os.replace(tmp,void_best_path)
                info={'coverage':score,'placed':len(seen),'elapsed_seconds':time.monotonic()-start,'voids':v}
                tmp=out/'fewest_voids_metrics.tmp.json';tmp.write_text(json.dumps(info,indent=2));os.replace(tmp,void_metrics)
                void_key=key
        if not errors and score > best + 1e-12:
            tmp = out / 'best.tmp.json'; save_json(candidate, tmp); os.replace(tmp, best_path)
            best = score
            record = {'coverage':score, 'placed':len(seen), 'elapsed_seconds':time.monotonic()-start,
                      'candidate':count, 'input_sha256':digest, 'valid':True}
            tmp = out / 'best_metrics.tmp.json'; tmp.write_text(json.dumps(record, indent=2)); os.replace(tmp, best_metrics)
            with (out/'history.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n'); f.flush(); os.fsync(f.fileno())
            print(json.dumps(record), flush=True)

builder = p.InstanceBuilder(); builder.set_objective(p.Objective.Knapsack)
coords = lambda shape: [(float(x)*1000, float(y)*1000) for x,y in list(shape.exterior.coords)[:-1]]
builder.add_bin_type(p.build_shape(coords(layout.patio)), copies=1, item_bin_minimum_spacing=(args.edge+args.allowance)*1000)
builder.set_item_item_minimum_spacing((args.gap+args.allowance)*1000)
for sid in layout.ids:
    builder.add_item_type(p.build_shape(coords(layout.shapes[sid])), copies=1, allowed_rotations=[(0,360,False)])
params = p.OptimizeParameters(); params.time_limit = args.seconds; params.verbosity_level = 0
params.new_solution_callback = save_candidate
output = p.optimize(builder.build(), params)
save_candidate(output)
print(json.dumps({'completed':True, 'elapsed_seconds':time.monotonic()-start, 'best_coverage':best, 'callbacks':count}), flush=True)
