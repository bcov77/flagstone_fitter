"""Small fixed-work experiment for optional vibration and pressure pulses."""
import json
from pathlib import Path
import time
import numpy as np
from flagstone.generate import inventory
from flagstone.moves import pulse
from flagstone.search import greedy_insert, coverage
from flagstone.phase1 import validate

runs=[]
for seed in range(3):
    for mode in ('none', 'vibration', 'pressure'):
        x=inventory(seed=seed, counts=(3,6,12))
        greedy_insert(x, proposals=2, strategy='random')
        before=coverage(x); start=time.perf_counter()
        move=None
        if mode=='vibration':
            move=pulse(x, rounds=2, translation_noise=.004, angle_noise=.02, relaxation_steps=2)
        elif mode=='pressure':
            move=pulse(x, rounds=2, pressure=[.001, 0], relaxation_steps=2)
        result=greedy_insert(x, proposals=3, strategy='random')
        runs.append({'seed':seed,'mode':mode,'initial_coverage':before,'final_coverage':result.coverage,
                     'pulse_accepted':move.accepted if move else 0,'elapsed_seconds':time.perf_counter()-start,
                     'violations':validate(x)})
summary={mode:{'mean_final_coverage':float(np.mean([r['final_coverage'] for r in runs if r['mode']==mode])),
               'mean_elapsed_seconds':float(np.mean([r['elapsed_seconds'] for r in runs if r['mode']==mode]))}
         for mode in ('none','vibration','pressure')}
out={'config':{'seeds':3,'proposals_before':2,'proposals_after':3,
               'note':'Fixed-work comparison, not equal-time; pulse modes consume different RNG draws.'},'summary':summary,'runs':runs}
Path('examples/pulse_benchmark.json').write_text(json.dumps(out,indent=2))
print(json.dumps(summary,indent=2))
