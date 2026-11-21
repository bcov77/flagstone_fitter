"""Recorded-pose diagnostic: retrieval evidence, not autonomous reproduction."""
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.affinity import scale
from shapely.ops import unary_union

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'human_trajectory_fa791034'))
from study import load,document,shapes,metrics
from proposals import GAP,canonical,detect_seams


def measure(source,case):
    before=document(source,case['pre']);board=shapes(before)
    remaining={s:g for s,g in board.items() if s not in case['removed']}
    patio=shapely.Polygon(before['patio'])
    free=patio.buffer(-GAP).difference(unary_union(list(remaining.values())).buffer(GAP))
    chosen=shapes(document(source,case['proposed']))
    trials={}
    for factor in [1.,.9,.8,.7]:
        probe=unary_union([scale(chosen[s],factor,factor,origin='centroid') for s in case['selected']])
        trials[str(factor)]=dict(intrusion_fraction=probe.difference(free).area/probe.area,
                                wall_intrusion_sqin=144*probe.difference(patio.buffer(-GAP)).area)
    movement=[]
    after=case['last_stop']['pieces']
    for sid,g in remaining.items():
        a=case['pre']['pieces'][sid]['pose'];b=after[sid]['pose']
        movement.append(dict(stone=sid,translation_inches=12*math.dist(a[:2],b[:2]),
                             rotation_degrees=abs(math.degrees(math.remainder(b[2]-a[2],2*math.pi))),
                             initial_conflict=any(g.distance(chosen[s])<GAP for s in case['selected'])))
    return dict(case=case['id'],selected=case['selected'],removed=case['removed'],scales=trials,
                neighbors=sorted(movement,key=lambda x:-x['translation_inches']),
                pulse_seconds=case['pulse_durations'],
                recorded_stop_metrics=metrics(document(source,case['last_stop'])))


def draw(ax,geom,**kwargs):
    for g in getattr(geom,'geoms',[geom]):
        if g.geom_type=='Polygon':ax.fill(*g.exterior.xy,**kwargs)


def main():
    source,*_=load();cases=json.loads((HERE.parent/'human_trajectory_fa791034/fixtures.json').read_text())
    rows=[measure(source,c) for c in cases]
    result=dict(scope='All 15 recorded physical-placement episodes, at recorded poses after removals. '
                'The pair insertion scales each member separately. Clearance intrusion includes the half-inch joint and walls. '
                'This is a necessary geometry diagnostic, not an autonomous retrieval or physics-success rate.',
                fixtures_sha256=canonical(cases),rows=rows,
                counts={str(limit):sum(r['scales']['0.8']['intrusion_fraction']<=limit for r in rows)
                        for limit in [1e-8,.005,.02]},
                linear_scale=.8,area_scale=.64)
    (HERE/'probe_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    with (HERE/'probe_audit.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['case','pieces','removed','100% intrusion %','90% intrusion %','80% intrusion %','70% intrusion %'])
        for r in rows:w.writerow([r['case'],'+'.join(r['selected']),'+'.join(r['removed']),*[100*r['scales'][str(s)]['intrusion_fraction'] for s in [1.,.9,.8,.7]]])
    case=next(c for c in cases if c['id']=='run_21');before=document(source,case['pre']);board=shapes(before)
    selected=shapes(document(source,case['proposed']))['13'];end=shapes(document(source,case['last_stop']))
    fig,axes=plt.subplots(1,3,figsize=(15,5.5),constrained_layout=True)
    for ax in axes:
        for sid,g in board.items():
            draw(ax,g,facecolor='#ddd8ce',edgecolor='#77756f',linewidth=.6)
            if 1<g.centroid.x<4.7 and 2<g.centroid.y<4.5:ax.text(g.centroid.x,g.centroid.y,sid,ha='center',va='center',fontsize=8)
        ax.set(xlim=(.7,4.8),ylim=(1.9,4.5),aspect='equal',xlabel='feet');ax.grid(alpha=.15)
    draw(axes[0],selected,facecolor='#f5a344',edgecolor='#a75a00',alpha=.65)
    axes[0].set_title('Human full-size insertion\n45.3% clearance intrusion')
    probe=scale(selected,.8,.8,origin='centroid')
    draw(axes[1],probe,facecolor='#63b7dc',edgecolor='#18688a',alpha=.8)
    free=shapely.Polygon(before['patio']).buffer(-GAP).difference(unary_union(list(board.values())).buffer(GAP))
    draw(axes[1],probe.difference(free),facecolor='#d84a45',edgecolor='none',alpha=.85)
    axes[1].set_title('80% length AND width\n36.1% still in clearance exclusion')
    axes[2].cla()
    for sid,g in end.items():draw(axes[2],g,facecolor='#f5a344' if sid=='13' else '#ddd8ce',edgecolor='#77756f',linewidth=.6)
    for sid,g in board.items():
        if sid in end and g.centroid.distance(end[sid].centroid)>.25/12:
            a=g.centroid;b=end[sid].centroid
            axes[2].annotate('',xy=(b.x,b.y),xytext=(a.x,a.y),arrowprops=dict(arrowstyle='->',color='#2a77a0',lw=1))
    axes[2].set(xlim=(.7,4.8),ylim=(1.9,4.5),aspect='equal',xlabel='feet',title='Recorded result after pulses\nBlue arrows: neighbor centroid movement')
    axes[2].grid(alpha=.15)
    fig.suptitle('The horizontal bar needs a moving-neighbor placement operator',fontsize=15)
    fig.savefig(HERE/'horizontal_probe.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    values=[100*r['scales']['0.8']['intrusion_fraction'] for r in rows]
    labels=[r['case'].replace('run_','')+': '+'+'.join(r['selected']) for r in rows]
    ax.bar(labels,values,color=['#d84a45' if r['case']=='run_21' else '#438aab' for r in rows])
    ax.axhline(2,color='#333',linestyle='--',label='Exploratory 2% gate')
    ax.set(ylabel='% of 80% probe in clearance exclusion',xlabel='Episode: piece(s)',title='80% probes help many moves, but miss the horizontal insertion')
    ax.tick_params(axis='x',rotation=45);ax.legend();fig.savefig(HERE/'probe_comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(counts=result['counts'],long_bar=next(r for r in rows if r['case']=='run_21')),indent=2))


if __name__=='__main__':main()
