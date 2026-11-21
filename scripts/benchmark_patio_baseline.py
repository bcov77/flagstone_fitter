"""Run the existing edge-proposal insertion baseline on the sketch-based patio."""
import json, time
from pathlib import Path
from flagstone.phase1 import load_json
from flagstone.search import greedy_insert

seconds = 15
x = load_json('examples/flagstone_reference/patio_inventory.json', seed=1)
t = time.monotonic(); passes = 0
while time.monotonic()-t < seconds and not x.active.all():
    greedy_insert(x, gap=.01, edge=.01, proposals=20, strategy='edge',
                  time_limit=max(0, seconds-(time.monotonic()-t)))
    passes += 1
result = {'solver':'baseline', 'seed':1,
          'poses':[[i,*p] for i,p in enumerate(x.poses.tolist()) if x.active[i]],
          'elapsed_seconds':time.monotonic()-t,
          'settings':{'strategy':'edge', 'time_limit':seconds, 'proposals_per_stone':20,
                      'passes':passes, 'spacing':.01, 'edge':.01}}
Path('examples/solver_benchmark/baseline.json').write_text(json.dumps(result, indent=2))
