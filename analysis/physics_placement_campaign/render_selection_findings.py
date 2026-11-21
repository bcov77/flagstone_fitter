"""Named examples from the completed search; selected for review, not approval."""
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
    manifest=s.load_manifest(out);targets={t['id']:t for t in manifest['targets']}
    reports={r['target']:r for r in json.loads((out/'report.json').read_text())}
    choices=[
        ('board_00_replace_62','91','frontier','Only swap with strict geometric advantage'),
        ('board_00_replace_62','67','nearfit_frontier','Alternative using a 95% probe and neighbor movement'),
        ('board_06_insert_pocket_04','141','nearfit_frontier','New insertion choice in a synthetic removal opening'),
    ]
    fig,axes=plt.subplots(3,2,figsize=(13,21));records=[]
    for row,(tid,sid,pool,title) in enumerate(choices):
        target=targets[tid]
        options=[c for c in reports[tid][pool] if c['stone']==sid]
        c=min(options,key=lambda c:c['metrics']['edge_excess_q90_inches'])
        draw(axes[row,0],dict(target=target,candidate=c),False)
        draw(axes[row,1],dict(target=target,candidate=c),True)
        near=c.get('nearfit_gate',{}).get('status')=='passes_nearfit_review_screen'
        movement=0.
        if near:
            factor=c['shrink_probe']['linear_scale']
            lines(axes[row,1],shrink_probe(s.b.world(sid,c['pose']),factor).boundary,color='#215c9c',linestyle='--',linewidth=1.1,zorder=7)
            for neighbor,(dx,dy) in c['space_plan']['translations_ft'].items():
                lines(axes[row,1],translate(s.b.ORIGINAL[neighbor],dx,dy).boundary,color='#23835e',linestyle='--',linewidth=1.1,zorder=6)
                point=s.b.ORIGINAL[neighbor].centroid
                axes[row,1].annotate('',xy=(point.x+dx,point.y+dy),xytext=(point.x,point.y),arrowprops=dict(arrowstyle='->',color='#23835e'),zorder=7)
            movement=c['space_plan']['max_neighbor_translation_inches']
        axes[row,0].set_title(f"{row+1:02} · CURRENT / OPENING\n{tid}",fontsize=11)
        axes[row,1].set_title(f"{sid} · {title}",fontsize=11)
        xlim=(min(ax.get_xlim()[0] for ax in axes[row]),max(ax.get_xlim()[1] for ax in axes[row]))
        ylim=(min(ax.get_ylim()[0] for ax in axes[row]),max(ax.get_ylim()[1] for ax in axes[row]))
        for ax in axes[row]:ax.set_xlim(xlim);ax.set_ylim(ylim)
        status=f"Endpoint plan checked; neighbors move ≤{movement:.2f} inches" if near else 'No movement plan needed; fixed-neighbor screen passes'
        axes[row,1].set_xlabel(f"{status}\nHuman fit / replacement value unreviewed; no physics",fontsize=10)
        records.append(dict(number=row+1,target=target,candidate=c,purpose=title,human_fit=None,human_value=None))
    fig.suptitle('Completed selection search · three comparisons to review',fontsize=18,y=.985)
    fig.text(.5,.012,'Orange: full size | blue: original | dashed blue: shrink probe | green dashes/arrows: neighbor endpoints | cyan: pocket | red: clearance intrusion',ha='center',fontsize=10)
    fig.subplots_adjust(left=.03,right=.98,top=.95,bottom=.055,hspace=.28,wspace=.07)
    dest=out/'completed_analysis';dest.mkdir(exist_ok=True)
    fig.savefig(dest/'new_examples.png',dpi=140);plt.close(fig)
    s.atomic_json(dest/'new_examples.json',records)
    print('Saved',dest/'new_examples.png')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=s.DEFAULT_OUT)
    render(ap.parse_args().out.resolve())
