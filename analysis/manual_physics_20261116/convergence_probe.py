"""Short-time response probe; ongoing jitter is not stationary settling."""
import json,math,random
from pathlib import Path
import numpy as np
from patio_physics.engine import PhysicsSession
OUT=Path(__file__).resolve().parent
screen=json.loads((OUT/'screening.json').read_text());d=json.loads(Path(screen['source']).read_text())
# Same human-reference anchor treatment and drive sequence as slot 35, seed 17.
s=next(s for s in d['stones'] if str(s['id'])=='35')
d['physics']['pieces']['35']['anchor']=s['pose'][:]
session=PhysicsSession(d,seed=17)
rows=[]
for phase in ['jitter_on','jitter_off']:
 if phase=='jitter_off':session.edit('settings',values={'rotation_magnitude':0,'translation_magnitude':0})
 session.start()
 for sid in session.drives:
  rng=random.Random(f'17:{sid}:{0 if phase=="jitter_on" else 3}')
  session.drives[sid]=[(rng.uniform(.35,1.35),rng.uniform(0,2*math.pi)) for _ in range(3)]
 previous=0.;poses={sid:p.pose[:] for sid,p in session.pieces.items() if p.active}
 for t in [.125,.25,.5,1.,2.]:
  session.advance(t-previous)
  displacement=[math.hypot(p.pose[0]-poses[sid][0],p.pose[1]-poses[sid][1])*12 for sid,p in session.pieces.items() if p.active]
  speeds=[b.velocity.length*12 for b in session.bodies.values()]
  report=session.clearance_report()
  rows.append(dict(phase=phase,seconds=t,interval_seconds=t-previous,mean_speed_inches_per_second=float(np.mean(speeds)),max_speed_inches_per_second=max(speeds),mean_translation_since_previous_sample_inches=float(np.mean(displacement)),max_translation_since_previous_sample_inches=max(displacement),pair_clearance_violations=sum(e['kind']=='pair' for e in report['errors']),wall_clearance_violations=sum(e['kind']=='boundary' for e in report['errors'])))
  poses={sid:p.pose[:] for sid,p in session.pieces.items() if p.active};previous=t
 session.stop()
result={'method':'One human-reference board, slot 35 seed 17; 2 seconds jitter on, then jitter magnitudes zero for 2 seconds; saved tethers retained. Quiet phase starts from the 2-second jitter state. Descriptive probe, not a calibrated stopping criterion.','samples':rows}
(OUT/'convergence_probe.json').write_text(json.dumps(result,indent=2))
for r in rows:print(r['phase'],r['seconds'],'mean speed',round(r['mean_speed_inches_per_second'],4),'max',round(r['max_speed_inches_per_second'],4),'pair violations',r['pair_clearance_violations'],flush=True)
