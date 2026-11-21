"""Shared immutable inputs, eligibility, metrics, and full-size physics."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import copy,hashlib,importlib.metadata,json,math,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'analysis/aggressive_trajectories'),str(ROOT/'analysis/physics_trajectories'),str(ROOT/'physics_patio'),str(ROOT/'analysis/physics_placement_campaign'),str(ROOT/'src')]
import numpy as np
import shapely
from shapely.ops import unary_union
from common import read,write,sha,state,restore,shapes,world,measure,clearance,GAP
from selector import apply,fit,pockets,parts
from aggressive_geometry import eligible,canonical,joint_summary
from aggressive_physics import relax
from aggressive_archive import enrich
from physics import session_for,quiet,motion
# Restore our module namespace after loading the frozen campaign dependencies.
sys.path.insert(0,str(HERE))
BATCH=ROOT/'examples/aggressive_trajectories_20261118_r2/aggressive'
PILOT=ROOT/'analysis/late_board_branching_20261118'
DEFAULT_OUT=ROOT/'examples/frontier_search_20261118'

class SearchInterrupted(Exception):pass

def checked(stopped):
 if stopped():raise SearchInterrupted()

def config(path=HERE/'config.json'):
 cfg=read(path);base=read(ROOT/'analysis/aggressive_trajectories/config.json');base.update(cfg['physics']);cfg['physics']=base
 if base['minimum_replacement_area_ratio']<1 or base['minimum_reorientation_degrees']<45:raise ValueError('No downsizing and >=45 degree edits required')
 if base['max_speed_ips']>.02 or base['max_angular_speed_dps']>.1 or base['stable_samples']<2:raise ValueError('Settling requirements cannot be weakened')
 if not cfg['roots'] or len(cfg['roots'])!=len(set(cfg['roots'])) or set(cfg['roots'])-{62,105,70,76}:raise ValueError('Select the four researched roots')
 valid_ops={'insert','replace','reorient','pair_fill','neighbor_refill','pair_repack','triple_repack','local_pulse'}
 if not cfg['operator_cycle'] or set(cfg['operator_cycle'])-valid_ops:raise ValueError('Unknown/empty operator cycle')
 if cfg['frontier_limit']<8:raise ValueError('Frontier needs space for roots and axis champions')
 for key in ['grid_steps_ft','angle_steps_degrees','target_budgets','piece_budgets']:
  if not cfg[key] or any(v<=0 for v in cfg[key]):raise ValueError('Positive search grids/budgets required')
 for key,maximum in [('max_intrusion_fraction',.05),('max_intrusion_inches',.4),('probe_max_intrusion_fraction',.02),('probe_max_overhang_fraction',.12)]:
  if base[key]>maximum:raise ValueError('Fit prerequisites cannot be weakened')
 if base['min_pocket_coverage']<.55 or base['probe_min_coverage']<.55:raise ValueError('Coverage prerequisites cannot be weakened')
 if min(cfg['frontier_limit'],cfg['max_depth'],cfg['repeats_per_action'],cfg['geometry_beam'])<1:raise ValueError('Positive search limits required')
 return cfg

def dependencies():
 paths=[*HERE.glob('*.py'),HERE/'config.json',ROOT/'physics_patio/frontier_search.sh',*sorted((ROOT/'analysis/aggressive_trajectories').glob('*.py')),*sorted((ROOT/'analysis/physics_trajectories').glob('*.py')),ROOT/'analysis/aggressive_trajectories/config.json',ROOT/'physics_patio/patio_physics/engine.py',*sorted((ROOT/'analysis/physics_placement_campaign').glob('*geometry.py')),*[ROOT/f'src/flagstone/{p}.py' for p in ['quarter_gaps','quarter','phase1']]]
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}

def versions():return dict(python=sys.version,**{n:importlib.metadata.version(n) for n in ['numpy','scipy','shapely','pymunk','pyvoronoi','matplotlib']})
def evaluate(doc):return enrich(doc,measure(doc))
def seed_for(master,identity):return int(hashlib.sha256(f'{master}:{identity}'.encode()).hexdigest()[:14],16)

def geometry_key(frame):
 # Novelty only; this rounding never changes a physical pose or eligibility test.
 return canonical([(sid,round(p[0]*120),round(p[1]*120),round(math.degrees(p[2])%360/2)%180) for sid,p in sorted(frame['poses'].items())])

def action_key(p):
 if p.get('operator')=='local_pulse':return canonical(dict(operator='local_pulse',movable=sorted(p['movable_ids']),focus=p['focus']))
 return canonical(dict(remove=sorted(p['remove']),placements=sorted([(x['stone'],round(x['pose'][0]*240),round(x['pose'][1]*240),round(math.degrees(x['pose'][2])%360)%360) for x in p['placements']]),protocol=p['physics_protocol']))

def axes(m):
 g=m.get('gaps',{});q=m.get('quality',{})
 return [-m['area_percent'],g.get('largest_interior_inches',1e9),g.get('largest_edge_inches',1e9),q.get('broad_void_sqft',1e9),-q.get('wall_joint_inches',0)]

def dominates(a,b):
 aa,bb=axes(a['metrics']),axes(b['metrics'])
 return all(x<=y+1e-7 for x,y in zip(aa,bb)) and any(x<y-1e-7 for x,y in zip(aa,bb))

def frontier(nodes,limit):
 if len(nodes)<=limit:return list(nodes)
 # Keep original roots plus a champion on every measured axis. Then reserve
 # capacity for nondominated boards AND geometrically different setup moves.
 mandatory={n['id']:n for n in nodes if n.get('kind')=='original'}
 for axis in range(5):
  n=min(nodes,key=lambda n:axes(n['metrics'])[axis]);mandatory[n['id']]=n
 chosen=list(mandatory.values());seen={geometry_key(n['frame']) for n in chosen}
 pareto=[n for n in nodes if not any(dominates(o,n) for o in nodes)]
 # Deterministic recency-biased interleaving; visit-based selection happens later.
 explored=sorted(nodes,key=lambda n:n.get('sequence',0),reverse=True)
 candidates=[]
 for i in range(max(len(pareto),len(explored))):
  if i<len(pareto):candidates.append(pareto[-1-i])
  if i<len(explored):candidates.append(explored[i])
 for n in candidates:
  key=geometry_key(n['frame'])
  if key not in seen:chosen.append(n);seen.add(key)
  if len(chosen)>=limit:break
 return chosen

def local_pulse(doc,p,cfg,seed):
 candidate=copy.deepcopy(doc);old=copy.deepcopy(candidate['physics']);movable=set(p['movable_ids'])
 if not movable or any(old['pieces'].get(s,{}).get('locked') for s in movable):raise ValueError('Cannot release an original lock')
 for s in candidate['stones']:
  sid=str(s['id']);candidate['physics']['pieces'][sid]['anchor']=s['pose'][:]
  if s['active']:candidate['physics']['pieces'][sid]['locked']=sid not in movable
 candidate['physics']['settings'].update(gap_inches=.5,piece_hardness=60,wall_hardness=80,angular_stiffness=2,translation_stiffness=2,**{k:cfg[k] for k in ['rotation_magnitude','rotation_force','rotation_frequency','translation_magnitude','translation_force','translation_frequency']})
 session=session_for(candidate,seed);initial=state(session.export());pulses=[]
 for duration in cfg['pulse_seconds']:
  session.start();session.advance(duration);pulses.append(dict(seconds=duration,motion=motion(session)));session.stop()
 settled,trace=quiet(session,cfg);c=clearance(session.export());repair=not c['valid'];phases=[dict(name='local_free_pulses',pulses=pulses),dict(name='direct',settled=settled,trace=trace)]
 if repair:
  session.settings.update(gap_inches=cfg['repair_gap_inches'],rotation_magnitude=4,translation_magnitude=.25,rotation_force=12,translation_force=12,angular_stiffness=2,translation_stiffness=2)
  session._build();session._randomize_drives();session.running=True;session.advance(cfg['repair_inflate_seconds'])
  session.settings['gap_inches']=.5;session._build();session._randomize_drives();session.running=True;session.advance(cfg['repair_deflate_seconds']);settled,trace=quiet(session,cfg)
  phases.append(dict(name='gap_cycle',settled=settled,trace=trace))
 velocity=motion(session);end=session.export();c=clearance(end);session.stop();end['physics']=old
 for s in end['stones']:end['physics']['pieces'][str(s['id'])]['anchor']=s['pose'][:]
 return dict(accepted=bool(settled and c['valid']),reason='valid_and_observed_settled' if settled and c['valid'] else 'clearance_failure_rolled_back' if not c['valid'] else 'unsettled_timeout_rolled_back',settled=settled,clearance=c,phases=phases,final_motion=velocity,attempted_document=end,proposed_start=initial,repair_attempted=repair,physics_seed=seed)
