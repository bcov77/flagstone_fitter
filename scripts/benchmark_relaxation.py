"""Measure a controlled containment plateau with and without restarts."""
import json
from pathlib import Path
import numpy as np
from flagstone.phase1 import Layout, polygon, local_shape
from flagstone.relaxation import relax, relax_with_restarts

big = local_shape([(-.2,-.2),(.2,-.2),(.2,.2),(-.2,.2)])
small = local_shape([(-.05,-.05),(.05,-.05),(.05,.05),(-.05,.05)])
patio = polygon([(0,0),(1,0),(1,1),(0,1)])
rows = []
for seed in range(10):
    def make():
        return Layout(patio, {'big': big, 'small': small}, [[.5,.5,0],[.5,.5,0]], [True,True], ('big','small'), np.random.default_rng(seed))
    plain, restarted = make(), make()
    a = relax(plain, mobility=[0,1], max_steps=20)
    b = relax_with_restarts(restarted, mobility=[0,1], max_steps=20, restarts=10, translation_noise=.35)
    rows.append({'seed': seed, 'plain_valid': not a.violations, 'restart_valid': not b.violations,
                 'plain_final_energy': a.final_energy, 'restart_final_energy': b.final_energy,
                 'restarts_used': b.restarts_used})
out = {'case': 'centered small square fully contained by frozen large square',
       'seeds': len(rows), 'plain_successes': sum(r['plain_valid'] for r in rows),
       'restart_successes': sum(r['restart_valid'] for r in rows), 'runs': rows}
Path('examples/relaxation_benchmark.json').write_text(json.dumps(out, indent=2))
print(json.dumps({k:v for k,v in out.items() if k != 'runs'}, indent=2))
