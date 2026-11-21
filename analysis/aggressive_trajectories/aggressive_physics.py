"""Full-size free-body pulses; bounded quiet/repair and complete rollback."""
import copy
from common import clearance,state
from selector import apply
from physics import relax as ordinary_relax,session_for,quiet,motion
from aggressive_geometry import eligible


def relax(doc,p,cfg,seed):
    if not eligible(doc,p,cfg):raise ValueError('Ineligible edit')
    if p['physics_protocol'] not in ('held_release','free_pulses'):raise ValueError('Unknown physics protocol')
    if p['physics_protocol']=='held_release':
        r=ordinary_relax(doc,p,cfg,seed);r['physics_protocol']='held_release';return r
    candidate=apply(doc,p);meta=candidate.setdefault('physics',{});original=copy.deepcopy(meta)
    for s in candidate['stones']:meta.setdefault('pieces',{}).setdefault(str(s['id']),{})['anchor']=s['pose'][:]
    meta.setdefault('settings',{}).update(gap_inches=.5,piece_hardness=60,wall_hardness=80,angular_stiffness=2,translation_stiffness=2,
        **{k:cfg[k] for k in ['rotation_magnitude','rotation_force','rotation_frequency','translation_magnitude','translation_force','translation_frequency']})
    session=session_for(candidate,seed);initial=state(session.export());pulse_trace=[]
    for duration in cfg['pulse_seconds']:
        session.start();session.advance(duration);pulse_trace.append(dict(seconds=duration,motion=motion(session)));session.stop()
    # stop() mirrors human pulse restarts, but acceptance always measures motion
    # during positive quiet integration, before any final velocity reset.
    settled,trace=quiet(session,cfg);c=clearance(session.export());repair=not c['valid']
    phases=[dict(name='free_pulses',pulses=pulse_trace),dict(name='direct',settled=settled,trace=trace)]
    if repair:
        session.settings.update(gap_inches=cfg['repair_gap_inches'],rotation_magnitude=4,translation_magnitude=.25,
                                rotation_force=12,translation_force=12,angular_stiffness=2,translation_stiffness=2)
        session._build();session._randomize_drives();session.running=True;session.advance(cfg['repair_inflate_seconds'])
        session.settings['gap_inches']=.5;session._build();session._randomize_drives();session.running=True
        session.advance(cfg['repair_deflate_seconds']);settled,trace=quiet(session,cfg)
        phases.append(dict(name='gap_cycle',settled=settled,trace=trace))
    final_motion=motion(session);end=session.export();c=clearance(end);session.stop()
    # Restore editor settings/locks, retaining all updated full-size poses.
    end['physics']['settings']=copy.deepcopy(original.get('settings',{}))
    for s in end['stones']:
        sid=str(s['id']);end['physics']['pieces'][sid].update(anchor=s['pose'][:],locked=original.get('pieces',{}).get(sid,{}).get('locked',False) if s['active'] else False)
    accepted=bool(settled and c['valid'])
    return dict(accepted=accepted,reason='valid_and_observed_settled' if accepted else 'clearance_failure_rolled_back' if not c['valid'] else 'unsettled_timeout_rolled_back',
                repair_attempted=repair,physics_protocol='free_pulses',proposed_start=initial,attempted_document=end,
                phases=phases,final_motion=final_motion,clearance=c,settled=settled,physics_seed=seed)
