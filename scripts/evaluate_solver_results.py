"""Reconstruct and independently validate solver output in project coordinates."""
import json, sys
from pathlib import Path
from flagstone.phase1 import load_json, save_json, validate, plot
from flagstone.search import coverage

for name in sys.argv[1:]:
    path = Path('examples/solver_benchmark') / (name + '.json')
    result = json.loads(path.read_text())
    x = load_json('examples/flagstone_reference/patio_inventory.json')
    seen = set()
    for i, px, py, angle in result['poses']:
        if i in seen or not 0 <= i < len(x.ids): raise ValueError('duplicate or invalid ID')
        seen.add(i); x.active[i] = True; x.poses[i] = (px, py, angle)
    errors = validate(x, gap=.01, edge=.01)
    report = {'solver':name, 'placed':len(seen), 'physical_coverage':coverage(x),
              'valid':not errors, 'violations':errors, 'elapsed_seconds':result['elapsed_seconds'],
              'settings':result['settings']}
    path.with_suffix('.evaluation.json').write_text(json.dumps(report, indent=2))
    save_json(x, path.with_suffix('.layout.json'))
    plot(x, path.with_suffix('.png'))
    print(json.dumps({k:v for k,v in report.items() if k != 'violations'}, indent=2), 'violation_count=',len(errors))
