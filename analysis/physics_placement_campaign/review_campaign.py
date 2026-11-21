"""Audit completed campaign validation and build analysis tables/readiness record.
Run in physics_patio/.venv. Rendering is a separate plotting-environment script.
"""
import argparse,csv,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path
import campaign as c
import pilot
from patio_physics.engine import PhysicsSession


def review(out):
    m=c.load_manifest(out)
    done=c.completed_ids(out,m)
    validation=set(m['validation_task_ids'])
    if not validation<=done:
        raise ValueError(f'Validation incomplete: {len(validation & done)}/{len(validation)}')
    source=json.loads((out/'source.json').read_text())
    cases={case['id']:case for case in m['cases']}
    targets={target['id']:target for target in m['targets']}
    rows=[json.loads((out/'results'/f'{tid}.json').read_text()) for tid in sorted(done)]
    lookup={(r['task']['case'],r['task']['seed'],r['task']['protocol']):r for r in rows}
    # Geometry-only session: production checker reads original outlines and saved poses.
    for stone in source['stones']:stone['active']=False
    session=PhysicsSession(source)
    audit=[]
    for r in rows:
        for sid,p in session.pieces.items():
            p.active=sid in r['final_poses']
            if p.active:p.pose=r['final_poses'][sid]
        checked=session.clearance_report()
        agrees=(checked['passes_clearance']==r['after']['valid'] and
                sum(e['kind']=='pair' for e in checked['errors'])==r['after']['pair_violations'] and
                sum(e['kind']=='boundary' for e in checked['errors'])==r['after']['wall_violations'])
        if not agrees:raise AssertionError(f'Checker disagreement: {r["task"]["id"]}')
        audit.append(r['task']['id'])
    table=[]
    for r in rows:
        task=r['task'];case=cases[task['case']];target=targets[case['target']]
        control=lookup.get((case['control'],task['seed'],task['protocol']))
        aq=r['after_quality'];delta={}
        if control is not None:
            cq=control['after_quality']
            delta={'area_gain_sqft':aq['covered_area_sqft']-cq['covered_area_sqft'],
                   'void_delta_sqft':aq['broad_void_1']-cq['broad_void_1'],
                   'far_void_delta_sqft':aq['far_void_1']-cq['far_void_1'],
                   'local_void_delta_sqft':(aq['broad_void_1']-aq['far_void_1'])-(cq['broad_void_1']-cq['far_void_1'])}
        label='invalid_or_unfinished'
        if r['after']['valid'] and r['converged']:
            if delta and delta['local_void_delta_sqft']<-.02 and delta['far_void_delta_sqft']>.02:
                label='local_benefit_remote_cost'
            elif delta and delta['void_delta_sqft']<-.02 and delta['area_gain_sqft']>=-1e-8:
                label='provisional_quality_improvement'
            else:label='feasible_other'
        row={'id':task['id'],'case':case['id'],'target':case['target'],'board':case['context'],
             'family':case['family'],'kind':case['kind'],'lineage_group':target['lineage_group'],
             'region_group':target['region_group'],'seed':task['seed'],'protocol':task['protocol'],
             'valid':r['after']['valid'],'converged':r['converged'],'label':label,
             'pair_violations':r['after']['pair_violations'],'wall_violations':r['after']['wall_violations'],
             'pair_deficit_inches':r['after']['pair_deficit_inches'],
             'quiet_seconds':r['quiet_seconds'],'wall_seconds':r['wall_seconds'],
             'control_available':control is not None,**delta}
        if case.get('features'):
            f=case['features'];profile=f['scale_profiles']['1.0']
            row.update(area_sqft=f['area_sqft'],intrusion_sqft=profile['intrusion_sqft'],
                       mean_excess_inches=profile['mean_excess_inches'],near_fraction=profile['near_fraction'],
                       longest_near_run_ft=profile['longest_near_run_ft'],
                       near_tangent_alignment=f['near_boundary_tangent_abs_cosine'])
        table.append(row)
    fields=list(dict.fromkeys(k for row in table for k in row))
    with (out/'outcomes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(table)
    # JSONL feature table stays separate from outcomes to prevent accidental leakage.
    with (out/'features.jsonl').open('w') as f:
        for case in m['cases']:
            target=targets[case['target']]
            f.write(json.dumps({'case':case['id'],'target':case['target'],'family':case['family'],
                                'kind':case['kind'],'lineage_group':target['lineage_group'],
                                'region_group':target['region_group'],'features':case.get('features')})+'\n')
    vr=[r for r in table if r['id'] in validation]
    controls=[r for r in vr if r['kind']=='control']
    candidates=[r for r in vr if r['kind']!='control']
    checks={'all_validation_complete':len(vr)==len(validation),
            'engine_checker_agreement':len(audit)==len(rows),
            'controls_feasible_and_converged':all(r['valid'] and r['converged'] for r in controls),
            'positive_candidates_present':any(r['valid'] and r['converged'] for r in candidates),
            'invalid_or_unfinished_candidates_present':any(not (r['valid'] and r['converged']) for r in candidates),
            'matched_controls_available':all(r['control_available'] for r in vr),
            'source_original_unchanged':pilot.digest(pilot.SOURCE)==m['source_sha256']}
    groups=[]
    for family in ['insertion','replacement']:
        for protocol in ['direct','gap_cycle']:
            subset=[r for r in candidates if r['family']==family and r['protocol']==protocol]
            groups.append({'family':family,'protocol':protocol,'n':len(subset),
                           'valid':sum(r['valid'] for r in subset),'converged':sum(r['converged'] for r in subset)})
    runs=[json.loads(p.read_text()) for p in (out/'runs').glob('*.json')]
    runtime=sum(r['elapsed_seconds'] for r in runs)
    completed_new=sum(r['completed_new'] for r in runs)
    seconds_per_trial=runtime/completed_new if completed_new else None
    readiness={'ready':all(checks.values()),'manifest_hash':m['manifest_hash'],'checks':checks,
               'validation_trials':len(vr),'controls':len(controls),'candidates':len(candidates),
               'candidate_valid':sum(r['valid'] for r in candidates),
               'candidate_converged':sum(r['converged'] for r in candidates),
               'candidate_labels':dict(Counter(r['label'] for r in candidates)),
               'groups':groups,'completed_all':len(rows),'planned_all':len(m['tasks']),
               'median_individual_wall_seconds':statistics.median(r['wall_seconds'] for r in vr),
               'observed_parent_seconds_per_trial':seconds_per_trial,
               'estimated_remaining_wall_hours_at_observed_rate':(len(m['tasks'])-len(rows))*seconds_per_trial/3600 if seconds_per_trial else None,
               'audit_endpoint_count':len(audit),
               'independent_layout_lineages':len({t['lineage_group'] for t in m['targets']}),
               'local_maximum_gap_rule_checked':False}
    pilot.write(out/'readiness.json',readiness)
    text=f'''# Campaign readiness validation\n\n**{'READY for the frozen batch' if readiness['ready'] else 'NOT READY: see checks below'}**

Prepared {len(m['contexts'])} board contexts, {len(m['targets'])} targets,
{len(m['cases'])} proposal/control cases, {len(m['tasks'])} paired-seed/protocol
trials. Completed a coverage-balanced {len(vr)}-trial validation across every
board context. No remaining large-batch tasks have been launched by this report.

## Checks

'''
    for key,value in checks.items():text+=f'- {key}: {"PASS" if value else "FAIL"}\n'
    text+=f'''\n{len(controls)} matched control trials; {sum(r['valid'] and r['converged'] for r in controls)}
were both clearance-valid and converged. Of {len(candidates)} candidate trials,
{readiness['candidate_valid']} passed clearance and {readiness['candidate_converged']}
met the combined motion/clearance/void-stability criterion. Failed candidates
remain in the dataset. This mix is required for the large-batch readiness check.

| Family | Protocol | Valid / candidate trials | Converged / candidate trials |
|---|---|---:|---:|
'''
    for row in groups:text+=f"| {row['family']} | {row['protocol']} | {row['valid']}/{row['n']} | {row['converged']}/{row['n']} |\n"
    text+='''\n## Data and interpretation\n\nRandom proposals are sampled conditionally from a recorded area-stratified
inventory/pose pool. Geometry-ranked and high-intrusion proposals are retained
alongside them. Replacements include the original stone at ±15 degrees and
unused inventory choices; the control restores its original pose. Empty-removal
controls are also scheduled in the full plan. No candidate may reuse a stone
already active elsewhere in its context.

Each trial resets from frozen source/context geometry. Direct and gap-cycle
protocols use equal phase times and matched per-ID random drives. Quiet phases
release restoring gains and keep the physical gap at 0.5 inches. Convergence
checks original geometry, velocity, angular velocity, and stability of sampled
broad-void area (global and far-region) across a window. Failures/timeouts are
observations, not proof of infeasibility.

The validation deliberately includes controls and candidates across every board,
not an unbiased estimate of the final batch's success rate. Proposal-kind and
family breakdowns must be considered when interpreting aggregate rates.

Features contain original-geometry boundary profiles at scales 0.85/0.9/0.95/1,
contiguous matching lengths, local tangent alignment, directional candidate and
clearance-free opening widths, neighborhood/cavity descriptors, and intrusion.
Tangent directions use an inch-wide chord to suppress traced-pixel noise.
Replacement reference radius is an area-equivalent radius, not a measured gap
peak. All physics uses original full-size pieces. Sampled one-inch-grid gap
metrics and derived quality labels do not certify the local-maximum gap rule.
The derived labels use provisional 0.02-ft² differences; raw values are retained.

All eight contexts are the same original board or reproducibly loosened subsets.
They share one layout lineage. Region tags identify related areas but do not
create independent-layout train/test data. No learned-rule accuracy or broad
packing generalization is claimed. New independently arranged boards are needed
for a genuine layout holdout. Preserve paired variants in the same split.

`manifest.json` freezes config/code/source hashes, contexts, targets, proposals,
features and task IDs. `proposal_pools.json` preserves screening pools.
`outcomes.csv` records matched deltas and statuses; `features.jsonl` keeps inputs
separate from outcomes. Results store compact phase poses and quiet traces.
Code/config/source mismatches and corrupt checkpoint identities block resumption.
New trial outputs are atomic; completed tasks are skipped. Budget expiry stops
new dispatch and lets at most the current worker count finish.
'''
    if seconds_per_trial:
        text+=f'''\n## Measured execution budget\n\nRecorded parent execution time averages {seconds_per_trial:.2f} wall seconds per
completed trial at the worker settings in `runs/`. At that rate the remaining
{len(m['tasks'])-len(rows)} tasks would take about
{readiness['estimated_remaining_wall_hours_at_observed_rate']:.2f} hours. This
excludes preparation/audit/rendering and is an estimate, not a duration guarantee.
The validation mix is not the full-batch mix. The runner has an explicit wall
budget and can resume from per-trial checkpoints.
'''
    (out/'validation_report.md').write_text(text)
    print(json.dumps(readiness,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=c.DEFAULT)
    review(parser.parse_args().out)
