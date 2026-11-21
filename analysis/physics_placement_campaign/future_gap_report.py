"""Review/export planning assessments without changing the compute job hashes."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.affinity import translate

import future_gap_campaign as job
from future_gap_geometry import compare
from render_prefit_review import draw,lines
from nearfit_geometry import shrink_probe


def completed_connections(result,config,min_run_ft=.25):
    """Rank contiguous owner-specific runs, not scattered matching points.

    Three inches is an experimental review threshold, not a learned definition
    of a finished side. Provisional stability remains a soft, visible quantity.
    """
    a=result['assessment'];runs={owner:v['longest_run_ft'] for owner,v in a['joints']['by_owner'].items() if v['longest_run_ft']>=min_run_ft}
    hard=set(config['locked_stones'])|{owner for owner in runs if owner.startswith('wall_')}
    stable=result['provisional_stability']
    return dict(min_run_ft=min_run_ft,finished_owner_count=len(runs),finished_runs=runs,
        hard_finished_length_ft=sum(v for k,v in runs.items() if k in hard),
        provisional_finished_length_ft=sum(v*stable.get(k,{}).get('score',0.) for k,v in runs.items() if k not in hard),
        finished_length_ft=sum(runs.values()))


def build(out,limit=18,render=True):
    m=job.load(out);targets={t['id']:t for t in m['targets']};groups={}
    for task in m['tasks']:
        path=out/'results'/f"{task['id']}.json"
        if not path.exists():raise RuntimeError('Complete the run before exporting the review')
        groups.setdefault(task['target'],[]).append(json.loads(path.read_text()))
    decisions=[];audit=[]
    feedback_path=Path(m['source'])/'review_18_v1/human_feedback.jsonl'
    labels={r['review_number']:r for r in map(json.loads,feedback_path.read_text().splitlines())} if feedback_path.exists() else {}
    for target,records in groups.items():
        baseline=next(r for r in records if r['role']=='baseline')
        old=targets[target]['old']
        originals=[r for r in records if r['role']=='baseline' or (r['candidate'] and r['candidate']['stone']==old)]
        for r in records:
            if r['role']=='baseline':continue
            decision=compare(r['assessment'],baseline['assessment'],m['config'])
            other=[dict(reference=ref['id'],**compare(r['assessment'],ref['assessment'],m['config'])) for ref in originals]
            flags=sum(c['status'] not in ('geometric_planning_advantage_review','unclear_retain_or_review') for c in other)
            before=completed_connections(baseline,m['config']);after=completed_connections(r,m['config'])
            closure_delta={k:after[k]-before[k] for k in ('finished_owner_count','hard_finished_length_ft','provisional_finished_length_ft','finished_length_ft')}
            d=dict(id=r['id'],target=target,stone=r['candidate']['stone'],pose=r['candidate']['pose'],
                   baseline_comparison=decision,comparisons_to_original_options=other,
                   flagged_original_comparisons=flags,review_number=r['review_number'],
                   completed_connections=after,connection_changes=closure_delta,
                   requires_clearance_refinement=r['subject_min_clearance_inches']<.5-1e-6)
            decisions.append(d)
            if r['review_number']:
                label=labels.get(r['review_number'],{})
                audit.append(dict(number=r['review_number'],result_id=r['id'],
                    user_comment_verbatim=label.get('user_comment_verbatim'),
                    assistant_interpreted_verdict=label.get('interpreted_verdict'),comparison=decision,
                    note='Post-hoc development diagnostic; not held-out validation or an approval-rate estimate'))
    job.v4.atomic_json(out/'planning_comparisons.json',decisions)
    job.v4.atomic_json(out/'review_audit.json',sorted(audit,key=lambda r:r['number']))
    if audit:
        text=['# Reviewed-case planning diagnostic','',
            'Known development cases. Status labels expose feature tradeoffs, not human-approval predictions.', '',
            '| # | User judgment (assistant coding) | Planning status | Main flags |', '|---|---|---|---|']
        for r in sorted(audit,key=lambda r:r['number']):
            c=r['comparison'];flags=c['new_risk_axes']+[f"alignment: {x}" for x in c['lost_alignment_options']]
            text.append(f"| {r['number']:02} | {r['assistant_interpreted_verdict']} | {c['status']} | {', '.join(flags) or '—'} |")
        text+=['', 'The measurements currently flag the indent in #8, corner-reach loss in #9, '
            'fixed-corner/wall costs in #12, and original alignment options in #16/#17. '
            'They also caution on approved #4/#7/#14 and do not reproduce every concern in #10/#18. '
            'These disagreements remain visible; no classifier or weights were fitted.']
        (out/'review_audit.md').write_text('\n'.join(text)+'\n')
    if not render:return
    dest=out/'review'
    if (dest/'placements.json').exists():raise RuntimeError('Review numbering frozen; use a fresh output directory for another selection')
    dest.mkdir(exist_ok=True)
    status_order={'geometric_planning_advantage_review':0,'unclear_retain_or_review':1,
                  'future_alignment_tradeoff_review':2,'corner_reach_tradeoff_review':3,
                  'provisional_anchor_tradeoff_review':4,'new_residual_risk_review':5}
    picks=[]
    for target in sorted(groups):
        options=[d for d in decisions if d['target']==target and d['stone']!=targets[target]['old']]
        if not options:continue
        results={r['id']:r for r in groups[target]}
        def key(d):
            a=results[d['id']]['assessment']
            closure=d['connection_changes']
            return (bool(d['baseline_comparison']['new_risk_axes']),bool(d['flagged_original_comparisons']),
                    status_order[d['baseline_comparison']['status']],
                    -closure['hard_finished_length_ft'],-closure['provisional_finished_length_ft'],
                    -closure['finished_owner_count'],d['requires_clearance_refinement'],d['id'])
        d=min(options,key=key);picks.append((d,results[d['id']]))
    # Opening order is deterministic. The complete comparison JSON keeps every
    # alternative, including original reorientations and risk-bearing choices.
    records=[]
    for number,(decision,result) in enumerate(picks[:limit],1):
        target=targets[decision['target']];c=result['candidate']
        fig,axes=plt.subplots(1,2,figsize=(13,8))
        draw(axes[0],dict(target=target,candidate=c),False);draw(axes[1],dict(target=target,candidate=c),True)
        if c.get('nearfit_gate',{}).get('status')=='passes_nearfit_review_screen':
            lines(axes[1],shrink_probe(job.v4.b.world(c['stone'],c['pose']),c['shrink_probe']['linear_scale']).boundary,
                  color='#215c9c',linestyle='--',linewidth=1.,zorder=7)
            for sid,(dx,dy) in c['space_plan']['translations_ft'].items():
                lines(axes[1],translate(job.v4.b.ORIGINAL[sid],dx,dy).boundary,color='#23835e',linestyle='--',linewidth=1.,zorder=6)
        caption=decision['baseline_comparison']['status'].replace('_',' ')
        fig.suptitle(f"{number:02} · {target['id']} · piece {c['stone']}",fontsize=14)
        fig.text(.5,.07,caption+f" | original-option cautions: {decision['flagged_original_comparisons']}",ha='center',fontsize=10)
        fig.text(.5,.025,'Same full-size geometry and color meanings as v4. Planning hypotheses require review; no physics.',ha='center',fontsize=10)
        xlim=(min(ax.get_xlim()[0] for ax in axes),max(ax.get_xlim()[1] for ax in axes))
        ylim=(min(ax.get_ylim()[0] for ax in axes),max(ax.get_ylim()[1] for ax in axes))
        for ax in axes:ax.set_xlim(xlim);ax.set_ylim(ylim)
        fig.subplots_adjust(left=.025,right=.98,top=.88,bottom=.14,wspace=.06)
        filename=f'case_{number:02}.png';fig.savefig(dest/filename,dpi=140);plt.close(fig)
        records.append(dict(number=number,target=target,candidate=c,planning_decision=decision,
            diagnostics_file=f"../results/{decision['id']}.json",image=filename,
            plausible_fit=None,improvement=None,dead_end_risk=None,notes=''))
    job.v4.atomic_json(dest/'placements.json',records)
    content=['<!doctype html><meta charset="utf-8"><title>Future-gap review</title>',
        '<style>body{font:17px system-ui;margin:30px auto;max-width:1600px}img{width:100%;height:auto}section{margin-bottom:35px}</style>',
        '<h1>Future-gap planning review</h1><p>Deterministic opening-wise choices, not certified improvements. '
        'Review fit, replacement benefit and future repair risk separately. Full alternatives are in planning_comparisons.json.</p>']
    content+=[f'<section><h2>{r["number"]:02} · {html.escape(r["target"]["id"])}</h2><img loading="lazy" src="{r["image"]}"></section>' for r in records]
    (dest/'index.html').write_text('\n'.join(content))
    job.v4.atomic_json(dest/'provenance.json',dict(report_script_sha256=job.sha(__file__),
        min_finished_run_ft=.25,note='Contiguous-run review threshold and risk ordering are experimental; not trained quality weights.'))
    print('Saved',len(records),'paired review examples and',len(decisions),'planning comparisons.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=job.DEFAULT_OUT)
    ap.add_argument('--review-limit',type=int,default=18);ap.add_argument('--audit-only',action='store_true')
    args=ap.parse_args();build(args.out.resolve(),args.review_limit,not args.audit_only)
