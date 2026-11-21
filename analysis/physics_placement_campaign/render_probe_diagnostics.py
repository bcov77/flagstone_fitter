"""Show shape-matched probes rejected by movement checks, clearly unselected."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import selection_campaign as s
from nearfit_geometry import shrink_probe
from render_prefit_review import draw,lines


def render(out,factor):
    manifest=s.load_manifest(out)
    targets={t['id']:t for t in manifest['targets']}
    items=[]
    for task in manifest['tasks']:
        path=out/'results'/f"{task['id']}.json"
        if not path.exists():
            raise RuntimeError('Complete the opening search before diagnostic export')
        for c in json.loads(path.read_text())['candidates']:
            if (c.get('shrink_probe',{}).get('linear_scale')==factor and
                c['shrink_probe']['fit_gate']['status']=='passes_provisional_screen' and
                c['nearfit_gate']['status']=='reject'):
                items.append((targets[task['target']],c))
    if not items:
        print('No rejected shape-passing probes at this scale')
        return
    items.sort(key=lambda item:(item[1]['shrink_probe']['metrics']['edge_excess_q90_inches'],item[1]['stone']))
    target=items[0][0];chosen=[];reasons=set()
    for t,c in items:
        if t['id']!=target['id']:
            continue
        reason=c.get('space_plan',{}).get('reason','preliminary_full_size_gate')
        if reason in reasons:
            continue
        chosen.append(c);reasons.add(reason)
        if len(chosen)>=2:
            break
    fig,axes=plt.subplots(1,len(chosen)+1,figsize=(5*(len(chosen)+1),6.7))
    draw(axes[0],dict(target=target,candidate=chosen[0]),False)
    for i,c in enumerate(chosen,1):
        ax=axes[i];draw(ax,dict(target=target,candidate=c),True)
        lines(ax,shrink_probe(s.b.world(c['stone'],c['pose']),factor).boundary,
              color='#215c9c',linestyle='--',linewidth=1.2,zorder=7)
        reason=c.get('space_plan',{}).get('reason','preliminary_full_size_gate')
        if reason.startswith('no_sampled_slack_for_'):
            description=f"No sampled room for neighbor {reason.split('_')[-1]}"
        elif reason=='too_many_affected_neighbors':
            description=f"Affects {len(c['space_plan']['affected_neighbors'])} neighbors; search limit {manifest['config']['nearfit']['max_neighbors']}"
        else:
            description=reason.replace('_',' ')
        ax.set_title(f"{c['stone']} · full size / dashed {factor:.0%} probe",fontsize=11)
        ax.set_xlabel(f"Reduced shape passes; full-size plan not found\n{description}\nDIAGNOSTIC ONLY — NOT SELECTED",fontsize=10)
    xlim=(min(ax.get_xlim()[0] for ax in axes),max(ax.get_xlim()[1] for ax in axes))
    ylim=(min(ax.get_ylim()[0] for ax in axes),max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:
        ax.set_xlim(xlim);ax.set_ylim(ylim)
    fig.suptitle(f"{factor:.0%} probe diagnostics · {target['id']} · full-size motion budget {manifest['config']['nearfit']['max_translation_inches']:g} inches",fontsize=14)
    fig.text(.5,.025,'Orange: full-size piece | dashed blue: probe | cyan: fixed pocket | red: current clearance intrusion. No physics.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.025,right=.98,top=.90,bottom=.23,wspace=.08)
    dest=out/'review';dest.mkdir(exist_ok=True)
    prefix=f'probe_{round(factor*100):02}_diagnostics'
    fig.savefig(dest/f'{prefix}.png',dpi=140);plt.close(fig)
    s.atomic_json(dest/f'{prefix}.json',dict(target=target,scale=factor,
        candidates=[s.compact(c) for c in chosen],status='diagnostic_only_not_selected',
        qualifying_rejected_poses=len(items),script_sha256=s.digest(__file__)))
    print(f'Saved {dest/prefix}.png')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=s.DEFAULT_OUT)
    ap.add_argument('--scale',type=float,default=.8)
    args=ap.parse_args()
    render(args.out.resolve(),args.scale)
