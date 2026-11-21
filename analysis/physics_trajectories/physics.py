"""Bounded relaxation, conditional gap-cycle repair and observed settling."""
import copy, math
from common import clearance, state
from patio_physics.engine import PhysicsSession
from selector import apply
PROXIES={}

def motion(session):
    return dict(max_speed_ips=max((b.velocity.length*12 for b in session.bodies.values()),default=0.),
        max_angular_speed_dps=max((abs(math.degrees(b.angular_velocity)) for b in session.bodies.values()),default=0.))

def session_for(document,seed):
    doc=copy.deepcopy(document);active={str(s['id']):s['active'] for s in doc['stones']}
    for s in doc['stones']:s['active']=False
    session=PhysicsSession(doc,seed=seed)
    for sid,p in session.pieces.items():p.active=active[sid]
    session.proxy_cache=PROXIES;session._build()
    return session

def quiet(session,cfg):
    # Direct force settings preserve existing velocities. No artificial stop before measuring.
    session.settings.update(piece_hardness=48,locked_hardness=60,wall_hardness=89,
        rotation_magnitude=0,translation_magnitude=0,rotation_force=0,translation_force=0,
        angular_stiffness=0,translation_stiffness=0)
    if not session.running:session.start()
    count=0;trace=[]
    for i in range(math.ceil(cfg['quiet_max_seconds']/cfg['quiet_sample_seconds'])):
        session.advance(cfg['quiet_sample_seconds']);v=motion(session);c=clearance(session.export())
        good=c['valid'] and v['max_speed_ips']<cfg['max_speed_ips'] and v['max_angular_speed_dps']<cfg['max_angular_speed_dps']
        count=count+1 if good else 0
        trace.append(dict(seconds=(i+1)*cfg['quiet_sample_seconds'],motion=v,clearance=c))
        if count>=cfg['stable_samples']:return True,trace
    return False,trace

def relax(document,proposal,cfg,seed):
    candidate=apply(document,proposal);meta=candidate.setdefault('physics',{});oldlocks={sid:v.get('locked',False) for sid,v in meta.get('pieces',{}).items()}
    # Each attempt tethers to its own start, not yesterday's original positions.
    for s in candidate['stones']:
        sid=str(s['id']);meta.setdefault('pieces',{}).setdefault(sid,{})['anchor']=s['pose'][:]
    original_settings=copy.deepcopy(meta.get('settings',{}))
    meta['settings']=dict(gap_inches=.5,piece_hardness=18,locked_hardness=60,wall_hardness=60,
        angular_stiffness=2,translation_stiffness=2,rotation_magnitude=4,rotation_force=12,
        rotation_frequency=1,translation_magnitude=.25,translation_force=12,translation_frequency=1,speed=1)
    temporary=[p['stone'] for p in proposal['placements'] if not oldlocks.get(p['stone'],False)]
    for sid in temporary:meta['pieces'][sid]['locked']=True
    session=session_for(candidate,seed);initial=state(session.export());session.start();session.advance(cfg['explore_seconds'])
    # Unlock proposed pieces for finishing. Rebuilding is followed by measured quiet integration.
    for sid in temporary:session.pieces[sid].locked=False
    session._build();session._randomize_drives();session.running=True
    settled,trace=quiet(session,cfg);after_direct=session.export();direct_clearance=clearance(after_direct)
    phases=[dict(name='direct',settled=settled,trace=trace)];repair=False
    if not direct_clearance['valid']:
        repair=True
        # A single short inflation/deflation cycle, then the identical settling check.
        session.settings.update(gap_inches=cfg['repair_gap_inches'],rotation_magnitude=4,translation_magnitude=.25,
            rotation_force=12,translation_force=12,angular_stiffness=2,translation_stiffness=2)
        session._build();session._randomize_drives();session.running=True;session._randomize_drives();session.advance(cfg['repair_inflate_seconds'])
        session.settings['gap_inches']=.5;session._build();session._randomize_drives();session.running=True
        session.advance(cfg['repair_deflate_seconds']);settled,trace=quiet(session,cfg)
        phases.append(dict(name='gap_cycle',inflate_gap_inches=cfg['repair_gap_inches'],inflate_seconds=cfg['repair_inflate_seconds'],
            deflate_seconds=cfg['repair_deflate_seconds'],settled=settled,trace=trace))
    # Read velocities BEFORE stop() zeroes them.
    final_motion=motion(session);attempted=session.export();c=clearance(attempted)
    session.stop()
    accepted=c['valid'] and settled
    # Keep ordinary editor settings for interchange; each next attempt sets its own force protocol.
    attempted['physics']['settings']=original_settings
    for s in attempted['stones']:
        sid=str(s['id']);attempted['physics']['pieces'][sid]['anchor']=s['pose'][:]
        attempted['physics']['pieces'][sid]['locked']=oldlocks.get(sid,False) if s['active'] else False
    return dict(accepted=accepted,reason='valid_and_observed_settled' if accepted else 'clearance_failure_rolled_back' if not c['valid'] else 'unsettled_timeout_rolled_back',
        repair_attempted=repair,proposed_start=initial,attempted_document=attempted,phases=phases,final_motion=final_motion,
        clearance=c,settled=settled,physics_seed=seed)
