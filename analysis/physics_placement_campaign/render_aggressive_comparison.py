"""Compact current / largest accepted movement / 80%-probe comparison."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.affinity import translate
import selection_campaign as s
from nearfit_geometry import shrink_probe
from render_prefit_review import draw,lines


def render(out):
    manifest=s.load_manifest(out)
    targets={t['id']:t for t in manifest['targets']}
    accepted=[];probes=[]
    for task in manifest['tasks']:
        path=out/'results'/f"{task['id']}.json"
        if not path.exists():
            raise RuntimeError('Complete the search first')
        for c in json.loads(path.read_text())['candidates']:
            c={**c,'target_id':task['target']}
            if c.get('nearfit_gate',{}).get('status')=='passes_nearfit_review_screen':
                accepted.append(c)
            if c.get('shrink_probe',{}).get('linear_scale')==.8 and c['shrink_probe']['fit_gate']['status']=='passes_provisional_screen':
                probes.append(c)
    if not accepted or not probes:
        raise RuntimeError('This comparison needs a passing near-fit and a shape-passing 80% probe')
    motion=max(accepted,key=lambda c:c['space_plan']['max_neighbor_translation_inches'])
    probes=[c for c in probes if c['target_id']==motion['target_id']]
    if not probes:
        raise RuntimeError('No 80% probe at the largest-movement opening')
    probe=min(probes,key=lambda c:c['shrink_probe']['metrics']['edge_excess_q90_inches'])
    target=targets[motion['target_id']]
    fig,axes=plt.subplots(1,3,figsize=(16,7))
    draw(axes[0],dict(target=target,candidate=motion),False)
    for ax,c in zip(axes[1:],[motion,probe]):
        draw(ax,dict(target=target,candidate=c),True)
        scale=c['shrink_probe']['linear_scale']
        lines(ax,shrink_probe(s.b.world(c['stone'],c['pose']),scale).boundary,color='#215c9c',linestyle='--',linewidth=1.2,zorder=7)
        valid=c['nearfit_gate']['status']=='passes_nearfit_review_screen'
        if valid:
            for sid,(dx,dy) in c['space_plan']['translations_ft'].items():
                lines(ax,translate(s.b.ORIGINAL[sid],dx,dy).boundary,color='#23835e',linestyle='--',linewidth=1.1,zorder=6)
                point=s.b.ORIGINAL[sid].centroid
                ax.annotate('',xy=(point.x+dx,point.y+dy),xytext=(point.x,point.y),arrowprops=dict(arrowstyle='->',color='#23835e'),zorder=7)
            caption=f"Full-size endpoint plan checked\nLargest neighbor move: {c['space_plan']['max_neighbor_translation_inches']:.2f} inches\nCONDITIONAL REVIEW — NO PHYSICS"
        else:
            reason=c.get('space_plan',{}).get('reason','full-size gate failed')
            if reason.startswith('no_sampled_slack_for_'):
                reason=f"No sampled room for neighbor {reason.split('_')[-1]}"
            elif reason=='too_many_affected_neighbors':
                reason=f"Affects more than {manifest['config']['nearfit']['max_neighbors']} neighbors"
            caption=f"Reduced shape passes; full-size plan not found\n{reason}\nDIAGNOSTIC ONLY — NOT SELECTED"
        ax.set_title(f"{c['stone']} · full size / dashed {scale:.0%} probe",fontsize=11)
        ax.set_xlabel(caption,fontsize=10)
    xlim=(min(ax.get_xlim()[0] for ax in axes),max(ax.get_xlim()[1] for ax in axes))
    ylim=(min(ax.get_ylim()[0] for ax in axes),max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:
        ax.set_xlim(xlim);ax.set_ylim(ylim)
    fig.suptitle('More aggressive search · 80% probes · neighbor movement limit 1.5 inches',fontsize=15)
    fig.text(.5,.025,'Orange: full-size piece | dashed blue: probe | dashed green: neighbor endpoints | cyan: fixed pocket | red: current clearance intrusion.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.025,right=.98,top=.90,bottom=.23,wspace=.08)
    dest=out/'review';dest.mkdir(exist_ok=True)
    fig.savefig(dest/'aggressive_comparison.png',dpi=140);plt.close(fig)
    s.atomic_json(dest/'aggressive_comparison.json',dict(target=target,conditional_candidate=s.compact(motion),
        eighty_percent_candidate=s.compact(probe),script_sha256=s.digest(__file__),physics_steps=0))
    print('Saved',dest/'aggressive_comparison.png')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=s.DEFAULT_OUT)
    render(ap.parse_args().out.resolve())
