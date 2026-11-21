"""An expanded numbered paired gallery from the completed v4 search.

Curated opening/piece coverage, with fixed deterministic pose selection. This
is a review sample, not a population quality estimate or an approval ranking.
"""
import argparse
import html
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.affinity import translate

import selection_campaign as s
from nearfit_geometry import shrink_probe
from render_prefit_review import draw,lines


PICKS=[
    ('board_00_replace_50','116','frontier'),
    ('board_00_replace_60','26','frontier'),
    ('board_00_replace_60','21','nearfit_frontier'),
    ('board_00_replace_61','105','frontier'),
    ('board_00_replace_63','100','frontier'),
    ('board_00_replace_63','11','nearfit_frontier'),
    ('board_00_replace_64','25','frontier'),
    ('board_00_replace_74','109','frontier'),
    ('board_00_replace_89','116','frontier'),
    ('board_00_replace_92','48','frontier'),
    ('board_00_replace_95','101','frontier'),
    ('board_00_replace_98','146','nearfit_frontier'),
    ('board_00_replace_106','11','frontier'),
    ('board_00_replace_147','28','frontier'),
    ('board_07_insert_pocket_01','117','nearfit_frontier'),
]


def prepare(out,dest):
    if (dest/'placements.json').exists():
        raise RuntimeError('Numbered review is frozen; use --render-only or a new output directory')
    m=s.load_manifest(out);targets={t['id']:t for t in m['targets']}
    reports={r['target']:r for r in json.loads((out/'report.json').read_text())}
    records=json.loads((out/'completed_analysis/new_examples.json').read_text())
    for r in records:
        r['overall_feedback']='Those 3 are fantastic.'
        r['review_status']='Favorable overall judgment on previous three-case panel; fit/value not separately scored'
    for tid,sid,pool in PICKS:
        options=[c for c in reports[tid][pool] if c['stone']==sid]
        if not options:
            raise ValueError(f'No saved passing candidate for {tid} / {sid}')
        def key(c):
            metrics=c['shrink_probe']['metrics'] if pool=='nearfit_frontier' else c['metrics']
            return (metrics['edge_excess_q90_inches'],metrics['edge_angle_q75_degrees'],-metrics['coverage'])
        c=min(options,key=key)
        records.append(dict(number=len(records)+1,target=targets[tid],candidate=c,
            purpose='Additional paired review',human_fit=None,human_value=None,
            neighbor_motion_acceptable=None,overall_feedback=None,review_status='unreviewed'))
    assert len(records)==18
    for r in records:
        r['number']=records.index(r)+1
        r['candidate']=s.compact(r['candidate'])
        r['notes']=''
    s.atomic_json(dest/'placements.json',records)
    s.atomic_json(dest/'provenance.json',dict(source_manifest_sha256=s.digest(out/'manifest.json'),
        source_examples_sha256=s.digest(out/'completed_analysis/new_examples.json'),
        script_sha256=s.digest(__file__),physics_steps=0,
        selection='First three exact previously reviewed poses; 15 additional opening/piece comparisons from saved frontiers. No new search.'))
    return records


def action(r):
    t,c=r['target'],r['candidate']
    return f"Replace {t['old']} → {c['stone']}" if t['old'] else f"Insert {c['stone']}"


def card(fig,slot,r,small=False):
    grid=slot.subgridspec(3,2,height_ratios=[.13,1,.18],hspace=.04,wspace=.045)
    title=fig.add_subplot(grid[0,:]);title.axis('off')
    t,c=r['target'],r['candidate']
    context='synthetic opening' if t['removed'] else 'original board'
    title.text(0,.65,f"{r['number']:02}   {action(r)}",fontsize=11 if small else 15,fontweight='bold')
    title.text(1,.65,context,ha='right',fontsize=8 if small else 10,color='#5f5f5f')
    axes=[fig.add_subplot(grid[1,i]) for i in range(2)]
    for ax,proposed in zip(axes,[False,True]):draw(ax,dict(target=t,candidate=c),proposed)
    near=c.get('nearfit_gate',{}).get('status')=='passes_nearfit_review_screen'
    if near:
        factor=c['shrink_probe']['linear_scale']
        lines(axes[1],shrink_probe(s.b.world(c['stone'],c['pose']),factor).boundary,
              color='#215c9c',linestyle='--',linewidth=1.1,zorder=7)
        for sid,(dx,dy) in c['space_plan']['translations_ft'].items():
            lines(axes[1],translate(s.b.ORIGINAL[sid],dx,dy).boundary,color='#23835e',linestyle='--',linewidth=1.,zorder=6)
            p=s.b.ORIGINAL[sid].centroid
            axes[1].annotate('',xy=(p.x+dx,p.y+dy),xytext=(p.x,p.y),
                             arrowprops=dict(arrowstyle='->',color='#23835e'),zorder=7)
    xlim=(min(ax.get_xlim()[0] for ax in axes),max(ax.get_xlim()[1] for ax in axes))
    ylim=(min(ax.get_ylim()[0] for ax in axes),max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:ax.set_xlim(xlim);ax.set_ylim(ylim)
    axes[0].set_title('CURRENT' if t['old'] else 'OPENING',fontsize=8 if small else 10)
    axes[1].set_title('PROPOSED · FULL SIZE',fontsize=8 if small else 10)
    caption=fig.add_subplot(grid[2,:]);caption.axis('off')
    mode=(f"{c['shrink_probe']['linear_scale']:.0%} probe · neighbors move ≤{c['space_plan']['max_neighbor_translation_inches']:.3g}″") if near else 'Fixed-neighbor screen passes'
    caption.text(0,.75,mode,fontsize=8 if small else 11)
    status='Previously reviewed: “fantastic”' if r['number']<=3 else 'New review · fit and improvement to judge'
    caption.text(0,.28,status+' · no physics',fontsize=7.5 if small else 9,color='#555555')


def sheet(records,path,rows,cols,figsize,title):
    fig=plt.figure(figsize=figsize,facecolor='white')
    grid=fig.add_gridspec(rows,cols,left=.025,right=.985,top=.935,bottom=.045,hspace=.15,wspace=.08)
    for i,r in enumerate(records):card(fig,grid[i//cols,i%cols],r,small=cols>1)
    fig.suptitle(title,fontsize=20,fontweight='bold',y=.985)
    fig.text(.5,.958,'01–03: same approved examples. 04–18: additional comparisons. Each pair uses the same scale.',ha='center',fontsize=10 if cols==1 else 12)
    fig.text(.5,.021,'Orange: full size | blue: original | blue dashes: shrink probe | green dashes/arrows: neighbor endpoints | cyan: pocket | red: clearance intrusion',ha='center',fontsize=9 if cols==1 else 11)
    fig.savefig(path,dpi=140,facecolor='white');plt.close(fig)


def render(out,dest,render_only=False):
    s.load_manifest(out);dest.mkdir(parents=True,exist_ok=True)
    records=json.loads((dest/'placements.json').read_text()) if render_only else prepare(out,dest)
    sheet(records,dest/'review_18.png',6,3,(24,39),'Expanded selection review · 18 comparisons')
    pages=[]
    for start in range(0,len(records),3):
        filename=f'page_{start//3+1:02}_{start+1:02}_{min(start+3,len(records)):02}.png'
        sheet(records[start:start+3],dest/filename,3,1,(13,22),f'Selection review · {start+1:02}–{min(start+3,len(records)):02}')
        pages.append(filename)
    for r in records:
        sheet([r],dest/f"case_{r['number']:02}.png",1,1,(13,9),f"Selection review · {r['number']:02}")
    content=['<!doctype html><meta charset="utf-8"><title>18 placement comparisons</title>',
        '<style>body{font:17px system-ui;max-width:1500px;margin:30px auto;padding:0 20px}img{width:100%;height:auto}nav a{margin-right:14px}section{margin:35px 0}</style>',
        '<h1>18 placement comparisons</h1><p>01–03 repeat the three proposals you approved. 04–18 are additional choices for separate fit and improvement judgments. Orange pieces are full size. No physics was run.</p>',
        '<p><a href="review_18.png">Full overview</a> · <a href="placements.json">Numbered poses and review fields</a></p>',
        '<nav>'+''.join(f'<a href="#case-{r["number"]}">{r["number"]:02}</a>' for r in records)+'</nav>']
    for r in records:
        number=r['number'];label=html.escape(action(r))
        content.append(f'<section id="case-{number}"><h2>{number:02} · {label}</h2><a href="case_{number:02}.png"><img loading="lazy" src="case_{number:02}.png"></a></section>')
    (dest/'index.html').write_text('\n'.join(content))
    readme=['# Expanded placement review — 18 comparisons','',
        'First three poses are unchanged from the approved three-case panel. Additional choices are unreviewed. '
        'This is a deliberately varied comparison sample, not an approval-rate estimate. No new search or physics.', '',
        '[Full overview](review_18.png) · [Browsable gallery](index.html) · [Numbered poses and labels](placements.json)', '',
        '| Page | Cases |','|---|---|']
    readme += [f'| [Page {i+1}]({filename}) | {i*3+1:02}–{i*3+3:02} |' for i,filename in enumerate(pages)]
    readme += ['', 'Same zoom within each pair; scale varies between comparisons. Synthetic removal openings are labeled. '
        'Proposals are independent alternatives and can reuse inventory pieces; this is not a simultaneous patio plan.', '',
        'Review plausible shape fit, improvement over the current arrangement, and acceptable neighbor movement separately. '
        'The first three have favorable overall feedback; those dimensions were not separately scored.']
    (dest/'README.md').write_text('\n'.join(readme)+'\n')
    s.atomic_json(dest/'render_provenance.json',dict(renderer_sha256=s.digest(__file__),
        placements_sha256=s.digest(dest/'placements.json'),overview='review_18.png',pages=pages,
        individual_images=[f"case_{r['number']:02}.png" for r in records]))
    print('Saved',len(records),'comparisons to',dest,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=s.DEFAULT_OUT)
    ap.add_argument('--dest',type=Path);ap.add_argument('--render-only',action='store_true')
    args=ap.parse_args();out=args.out.resolve()
    render(out,args.dest.resolve() if args.dest else out/'review_18_v1',args.render_only)
