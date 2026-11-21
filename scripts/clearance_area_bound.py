"""Optimistic area bound from half-gap buffers and a fractional knapsack.

Ignoring shape arrangement, each selected stone's half-gap buffer must fit in
the patio inset by edge minus half-gap. Allowing fractional stones makes the
result an upper bound rather than a feasible packing.
"""
import argparse, json
from flagstone.phase1 import load_json

ap = argparse.ArgumentParser()
ap.add_argument('--input', default='examples/flagstone_reference/surplus_inventory.json')
ap.add_argument('--gap', type=float, default=.01)
ap.add_argument('--edge', type=float, default=.01)
args = ap.parse_args()
x = load_json(args.input)
gap, edge = args.gap, args.edge
if gap < 0 or edge < gap/2: ap.error('need nonnegative gap and edge >= gap/2')
radius = gap/2
available = x.patio.buffer(-(edge-radius), quad_segs=64).area
items = sorted(((s.area, s.buffer(radius, quad_segs=64).area) for s in x.shapes.values()),
               key=lambda item:item[0]/item[1], reverse=True)
remaining = available; bound = 0
for area, expanded in items:
    fraction = min(1, max(0,remaining)/expanded)
    bound += fraction*area
    remaining -= fraction*expanded
    if remaining <= 1e-12: break
print(json.dumps({'gap':gap, 'edge':edge, 'available_buffer_area':available,
                  'fractional_area_upper_bound':bound/x.patio.area,
                  'note':'Polygonal buffer approximation; ignores placement constraints and allows fractional stones.'}, indent=2))
