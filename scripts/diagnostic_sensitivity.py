"""Compute sampled proximity diagnostics at several spatial resolutions."""
import argparse
import json
from pathlib import Path
from flagstone.diagnostics import empty_space, sampled_neighbor_distances
from flagstone.phase1 import load_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', type=Path, default=Path('examples/search/layout.json'))
p.add_argument('--output', type=Path, default=Path('examples/diagnostic_sensitivity.json'))
p.add_argument('--edge-strip-width', type=float, default=.05)
p.add_argument('--broad-distance', type=float, default=.05)
p.add_argument('--neighbor-cutoff', type=float, default=.1)
p.add_argument('--grid', type=int, nargs='+', default=[25, 50, 100])
p.add_argument('--spacing', type=float, nargs='+', default=[.02, .01, .005])
a = p.parse_args()
x = load_json(a.input)
result = {'units': x.units, 'input': str(a.input),
          'note': 'Sampled distances are proximity proxies, not certified clearance or full joint width.',
          'grid': {str(n): empty_space(x, resolution=n, edge_strip_width=a.edge_strip_width,
                                      broad_distance=a.broad_distance) for n in a.grid},
          'boundary': {str(s): sampled_neighbor_distances(x, spacing=s, max_distance=a.neighbor_cutoff)
                       for s in a.spacing}}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2, allow_nan=False))
print(json.dumps({s: {'matched': r['matched_samples'], 'median': (r['distance_quantiles'] or {}).get('0.5')}
                  for s, r in result['boundary'].items()}, indent=2))
