"""Render every physical placement episode and the sequence's larger patterns."""
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch

from study import HERE,load,document,shapes,metrics


def main():
    source,final,records,_=load();cases=json.loads((HERE/'fixtures.json').read_text())
    final_board=shapes(final)
    for page in range(4):
        group=cases[page*4:(page+1)*4]
        fig,axes=plt.subplots(len(group),4,figsize=(17,5.3*len(group)),squeeze=False)
        fig.subplots_adjust(left=.025,right=.995,top=.9,bottom=.025,wspace=.07,hspace=.6)
        for ri,case in enumerate(group):
            boards=[shapes(document(source,case[k])) for k in ['pre','proposed','last_stop']]+[final_board]
            selected=set(case['selected']);removed=set(case['removed'])
            targets=[g for b in boards[:2] for sid,g in b.items() if sid in selected|removed]
            x0=min(g.bounds[0] for g in targets)-.65;x1=max(g.bounds[2] for g in targets)+.65
            y0=min(g.bounds[1] for g in targets)-.65;y1=max(g.bounds[3] for g in targets)+.65
            size=max(x1-x0,y1-y0,2.7);cx=(x0+x1)/2;cy=(y0+y1)/2
            for ci,(ax,board,title) in enumerate(zip(axes[ri],boards,['Before decision','Hand-placed, before physics','After recorded pulses','Final recorded board'])):
                ax.set_facecolor('#f4f3e9')
                for sid,g in board.items():
                    g=g.simplify(.003,preserve_topology=True);is_subject=sid in (removed if ci==0 else selected)
                    color=('#85afd1' if ci==0 else '#e9a24e' if ci==1 else '#8abe9e') if is_subject else '#dcd6c4'
                    ax.add_patch(Patch(list(g.exterior.coords),facecolor=color,edgecolor='#766e5b',lw=.5))
                    p=g.representative_point();ax.text(p.x,p.y,sid,ha='center',va='center',fontsize=5.5,clip_on=True)
                patio=source['patio']+[source['patio'][0]];ax.plot(*zip(*patio),color='#345741',lw=1)
                ax.set_xlim(cx-size/2,cx+size/2);ax.set_ylim(cy-size/2,cy+size/2);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,fontsize=9)
            r=case['removed'];s=case['selected'];m=case['last_metrics']
            caption=f"Run {case['run']} · {' + '.join(r) if r else 'opening'} → {' + '.join(s)} · {len(case['pulse_durations'])} pulse(s), {sum(case['pulse_durations']):.2f}s · coverage {case['before_metrics']['coverage']:.2f} → {m['coverage']:.2f}% · recorded clearance shortfalls {m['pair_failures']}/{m['wall_failures']}"
            fig.text(.03,axes[ri,0].get_position().y1+.035,caption,fontsize=10,fontweight='bold')
        fig.suptitle('Human placement sequence · every physical placement episode\nBlue = removed original(s); orange = proposal; green = selected piece(s). Fixed local crop; endpoints are not approval labels.',fontsize=13,y=.995)
        fig.savefig(HERE/f'placements_{page+1}.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    for ax,doc,label in zip(axes,[source,final],['Recording start','Recording finish']):
        board=shapes(doc)
        for sid,g in board.items():
            g=g.simplify(.003);ax.add_patch(Patch(list(g.exterior.coords),facecolor='#d7c5a6',edgecolor='#75634e',lw=.6));p=g.representative_point();ax.text(p.x,p.y,sid,ha='center',va='center',fontsize=7)
        patio=source['patio']+[source['patio'][0]];ax.plot(*zip(*patio),color='#345741');ax.set_aspect('equal');ax.set_xlim(-.1,7.91);ax.set_ylim(-.1,7.91);m=metrics(doc);ax.set_title(f"{label} · {m['active_count']} stones · {m['coverage']:.2f}%",fontsize=13)
    fig.tight_layout();fig.savefig(HERE/'start_finish.png',dpi=150);plt.close(fig)
    stops=[r for r in records if r.get('action')=='simulation_stop']
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    runs=[r['after']['run_number'] for r in stops]
    axes[0].plot(runs,[max(v['center_inches'] for v in r['motion_peaks'].values()) for r in stops],'-o',ms=3);axes[0].set_ylabel('Peak center travel (in)')
    axes[1].bar(runs,[sum(v['center_inches']>=.25 for v in r['motion_peaks'].values()) for r in stops],color='#649176');axes[1].set_ylabel('Stones moving ≥¼ inch');axes[1].set_xlabel('Recorded simulation pulse / run number')
    fig.suptitle('Recorded per-integration-step motion · relative to each pulse start');fig.tight_layout();fig.savefig(HERE/'pulse_motion.png',dpi=145);plt.close(fig)


if __name__=='__main__':main()
