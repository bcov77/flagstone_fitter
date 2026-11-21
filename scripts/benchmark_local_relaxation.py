"""Compare insertion search with no, full, or neighborhood relaxation."""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
import shapely
from flagstone.generate import inventory
from flagstone.search import greedy_insert

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seed-start', type=int, default=61)
parser.add_argument('--seeds', type=int, default=3)
parser.add_argument('--seconds', type=float, default=2)
parser.add_argument('--modes', nargs='+', choices=['none','local','selective_005','selective_020','targeted_020'], default=['none','local','selective_005','selective_020'])
parser.add_argument('--output', type=Path, default=Path('examples/benchmark_selective_relaxation.json'))
args=parser.parse_args()
if args.seeds < 1 or not np.isfinite(args.seconds) or args.seconds <= 0:
    parser.error('seeds and seconds must be positive')
runs=[]
for seed in range(args.seed_start, args.seed_start+args.seeds):
    for mode in args.modes:
        x=inventory(seed=seed, counts=(6,50,100), areas=(.025,.008,.003), vertex_profile='detailed')
        start=time.perf_counter(); attempts=accepted=passes=relaxed=skipped=successes=failures=visits=0
        while time.perf_counter()-start < args.seconds and not x.active.all():
            remaining=max(0,args.seconds-(time.perf_counter()-start))
            result=greedy_insert(x, strategy='edge', proposals=10, time_limit=remaining,
                                 relaxation_steps=0 if mode=='none' else 2,
                                 local_relaxation=(mode!='none'),
                                 max_relaxation_penalty={'selective_005':.05, 'selective_020':.2, 'targeted_020':.2}.get(mode),
                                 mobility_policy='clearance' if mode=='targeted_020' else 'circle')
            attempts+=result.attempts; accepted+=result.accepted; passes+=1
            relaxed+=result.relaxation_attempts; skipped+=result.relaxation_skipped
            successes+=result.relaxation_successes; failures+=result.relaxation_failures
            visits+=result.mobile_stone_visits
        runs.append({'seed':seed,'mode':mode,'coverage':result.coverage,'placed':int(x.active.sum()),
                     'attempts':attempts,'accepted':accepted,'passes':passes,'relaxation_attempts':relaxed,'relaxation_skipped':skipped,
                     'relaxation_successes':successes,'relaxation_failures':failures,'mobile_stone_visits':visits,
                     'elapsed_seconds':time.perf_counter()-start,'violations':result.violations})
summary={mode:{'mean_coverage':float(np.mean([r['coverage'] for r in runs if r['mode']==mode])),
               'mean_placed':float(np.mean([r['placed'] for r in runs if r['mode']==mode])),
               'mean_attempts':float(np.mean([r['attempts'] for r in runs if r['mode']==mode]))}
         for mode in args.modes}
out={'config':{'seconds_per_run':args.seconds,'seeds':list(range(args.seed_start,args.seed_start+args.seeds)), 'modes':args.modes,'counts':[6,50,100],
               'areas':[.025,.008,.003],'vertex_profile':'detailed','relaxation_steps':2,
               'note':'Single machine; time checked between proposals; whole-search results include final full validation.',
               'python':platform.python_version(),'numpy':np.__version__,'shapely':shapely.__version__},
     'summary':summary,'runs':runs}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(out,indent=2))
print(json.dumps(summary,indent=2))
