"""Derived trajectory PNGs, pooled outcome table, and endpoint layout export."""
import argparse,csv,json,math
from pathlib import Path
from common import ROOT,restore,shapes,write,read,attempt_files,attempt_path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch


def draw(ax,document,title):
    patio=__import__('shapely').Polygon(document['patio']);board=shapes(document)
    for sid,g in board.items():
        ax.add_patch(Patch(list(g.exterior.coords),facecolor='#d9b786',edgecolor='#655747',lw=.5))
        z=g.representative_point();ax.text(z.x,z.y,sid,fontsize=5,ha='center',va='center',clip_on=True)
    x,y=patio.exterior.xy;ax.plot(x,y,color='#4b4841',lw=1)
    ax.set_aspect('equal');ax.set_xlim(patio.bounds[0]-.1,patio.bounds[2]+.1);ax.set_ylim(patio.bounds[1]-.1,patio.bounds[3]+.1)
    ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,fontsize=10)

def caption(metrics):
    g=metrics['gaps'];edge=g.get('largest_edge_inches');inside=g.get('largest_interior_inches')
    gaps=f'interior {inside:.2f}″ / edge {edge:.2f}″' if edge is not None and inside is not None else 'gap audit unavailable'
    return f'{metrics["area_percent"]:.2f}% area · {gaps}'

def render_run(out,number):
    out=Path(out);folder=out/'runs'/f'run_{number:06d}';source=json.loads((out/'source.json').read_text());m=json.loads((out/'manifest.json').read_text())
    records=[read(p) for p in attempt_files(folder)]
    if not records:return
    final=restore(source,records[-1]['end'])
    fig,axes=plt.subplots(1,2,figsize=(15,8))
    draw(axes[0],source,'Start · '+caption(m['start_metrics']));draw(axes[1],final,'End · '+caption(records[-1]['end_metrics']))
    fig.suptitle(f'Run {number} · {records[0]["mode"]} · {sum(r["accepted"] for r in records)}/{len(records)} accepted',fontsize=17)
    fig.tight_layout();fig.savefig(folder/'start_end.png',dpi=150);plt.close(fig)
    frames=[(source,'Start',m['start_metrics'])]+[(restore(source,r['end']),f'{r["number"]:02d} · '+('kept' if r['accepted'] else 'undo'),r['end_metrics']) for r in records]
    cols=5;rows=math.ceil(len(frames)/cols);fig,axes=plt.subplots(rows,cols,figsize=(20,rows*4.3),squeeze=False)
    for ax,(doc,title,metrics) in zip(axes.flat,frames):draw(ax,doc,title+'\n'+caption(metrics))
    for ax in list(axes.flat)[len(frames):]:ax.axis('off')
    fig.suptitle(f'Run {number} · {records[0]["mode"]} · retained board after each attempt',fontsize=20)
    fig.tight_layout(rect=(0,0,1,.98));fig.savefig(folder/'trajectory.png',dpi=120);plt.close(fig)
    metrics=[m['start_metrics']]+[r['end_metrics'] for r in records];x=list(range(len(metrics)))
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    axes[0].plot(x,[v['area_percent'] for v in metrics],'-o');axes[0].set_ylabel('Area %')
    for key,label in [('largest_interior_inches','Interior'),('largest_edge_inches','Edge')]:axes[1].plot(x,[v['gaps'].get(key,float('nan')) for v in metrics],'-o',label=label)
    axes[1].set_ylabel('Largest gap diameter (in)');axes[1].legend()
    axes[2].scatter([r['number'] for r in records],[1 if r['accepted'] else 0 for r in records],c=['#2c8664' if r['accepted'] else '#be5e4d' for r in records]);axes[2].set_yticks([0,1],['Undo','Keep']);axes[2].set_xlabel('Attempt')
    fig.tight_layout();fig.savefig(folder/'metrics.png',dpi=150);plt.close(fig)

def export_rows(out):
    out=Path(out);rows=[]
    for folder in sorted((out/'runs').glob('run_*')):
        for path in attempt_files(folder):
            r=read(path);p=r.get('chosen') or {};a=r.get('attempted_end_metrics',r['end_metrics']);c=p.get('comparison',{})
            rows.append(dict(run=r['run'],mode=r['mode'],attempt=r['number'],compute_revision=r.get('compute_revision','original'),accepted=r['accepted'],reason=r['reason'],
                removed=','.join(p.get('remove',[])),placed=','.join(x['stone'] for x in p.get('placements',[])),
                double=len(p.get('placements',[]))==2,selection_probability=r['selection'].get('probability'),tier=p.get('tier'),
                planning_status=c.get('status'),new_risk_axes=','.join(c.get('new_risk_axes',[])),lost_alignment=','.join(c.get('lost_alignment_options',[])),
                hard_joint_inches=p.get('finished_connections',{}).get('hard_length_inches'),
                repair_attempted=r['repair_attempted'],settled=r.get('settled'),
                before_area_percent=r['start_metrics']['area_percent'],attempted_area_percent=a['area_percent'],end_area_percent=r['end_metrics']['area_percent'],
                attempted_interior_gap_inches=a['gaps'].get('largest_interior_inches'),attempted_edge_gap_inches=a['gaps'].get('largest_edge_inches'),
                end_interior_gap_inches=r['end_metrics']['gaps'].get('largest_interior_inches'),end_edge_gap_inches=r['end_metrics']['gaps'].get('largest_edge_inches')))
    if rows:
        with (out/'attempts.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups={}
    for mode in ['normal','golden','silver']:
        rs=[r for r in rows if r['mode']==mode]
        groups[mode]=dict(attempts=len(rs),kept=sum(r['accepted'] for r in rs),repair_attempts=sum(r['repair_attempted'] for r in rs))
    write(out/'outcomes.json',dict(groups=groups,causal_warning='Adaptive, selected trajectories from one source board. Conditional choice probabilities are logged, but pooled associations are not causal treatment effects.'))
    print(f'Saved {len(rows)} attempt rows to {out}/attempts.csv')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--run',type=int);ap.add_argument('--export-attempt',type=int);ap.add_argument('--frame',choices=['start','end','attempted_end'],default='end');ap.add_argument('--output',type=Path);args=ap.parse_args()
    if args.export_attempt is not None:
        if args.run is None or args.output is None:ap.error('--export-attempt requires --run and --output')
        if args.output.exists():ap.error('Output exists')
        source=json.loads((args.out/'source.json').read_text());r=read(attempt_path(args.out/'runs'/f'run_{args.run:06d}',args.export_attempt))
        write(args.output,restore(source,r[args.frame]));return
    if args.run is not None:render_run(args.out,args.run)
    else:
        for p in sorted((args.out/'runs').glob('run_*/summary.json')):render_run(args.out,int(p.parent.name.split('_')[1]))
    export_rows(args.out)

if __name__=='__main__':main()
