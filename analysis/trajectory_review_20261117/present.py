"""Produce an inspectable placement index and representative before/after sheets."""
import gzip
import html
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as Patch
from shapely.geometry import Polygon
from shapely.affinity import affine_transform

OUT=Path(__file__).parent
ROOT=OUT.parents[1]
BATCH=ROOT/'examples/physics_trajectories_20261117_r3'


def mean(items,key):
    return statistics.mean(key(r) for r in items) if items else None


def main():
    review=json.loads((OUT/'review.json').read_text());rows=review['attempts']
    actions=[r for r in rows if r['type']!='no_candidate']
    completed=[r for r in actions if r['complete']]
    groups={}
    for typ in sorted({r['type'] for r in actions}):
        z=[r for r in actions if r['type']==typ]
        groups[typ]=dict(n=len(z),verdicts=dict(Counter(r['verdict'] for r in z)),
                         mean_coverage=mean(z,lambda r:r['delta_coverage']),
                         mean_joints=mean(z,lambda r:r['delta_joints']),
                         mean_interior=mean(z,lambda r:r['delta_interior']),
                         mean_edge=mean(z,lambda r:r['delta_edge']))
    adj=[r for r in actions if r['type']=='reorientation']
    hole_stats={}
    for name,z in [('all',actions),('completed',completed)]:
        hs=[h for r in z for h in r['holes']]
        hole_stats[name]=dict(witnesses=len(hs),moves=sum(bool(r['holes']) for r in z),
                              repaired_at_end=sum(h['repaired_at_end'] for h in hs),
                              ever_repaired=sum(h['ever_repaired'] for h in hs),
                              persistent=sum(not h['repaired_at_end'] for h in hs),
                              moves_all_repaired=sum(bool(r['holes']) and all(h['repaired_at_end'] for h in r['holes']) for r in z))
    stats=dict(groups=groups,holes=hole_stats,
               adjustment=dict(count=len(adj),under45=sum(r['small_adjustment'] for r in adj),
                               under10=sum(r['rotation']<10 for r in adj),median_angle=statistics.median(r['rotation'] for r in adj),
                               median_translation=statistics.median(r['translation'] for r in adj if r['small_adjustment'])),
               lost_planned_joints=sum(r['joints_planned']['total']>=3 and r['joints_after']['total']<r['joints_planned']['total']/2 for r in actions),
               predicted_at_least3=sum(r['joints_planned']['total']>=3 for r in actions),
               losses_after_peak=sum(r['best_coverage']-r['final']['coverage']>=.5 for r in review['runs'] if r['complete']))
    (OUT/'statistics.json').write_text(json.dumps(stats,indent=2))
    lookup={(r['run'],r['attempt']):r for r in actions}
    cases=[('good_connections',[(44,13),(14,18),(42,7),(29,5)]),
           ('bad_tradeoffs',[(1,12),(20,24),(23,28),(29,30)]),
           ('neutral_and_small_adjustments',[(10,25),(69,21),(29,25),(29,27)]),
           ('later_repairs',[(9,12),(46,3),(43,23),(22,13)])]
    source=json.loads((BATCH/'source.json').read_text());locals={str(s['id']):Polygon(s['outline']).simplify(.002) for s in source['stones']}
    def board(state):
        result={}
        for sid,p in state['poses'].items():
            c,s=math.cos(p[2]),math.sin(p[2]);result[sid]=affine_transform(locals[sid],[c,-s,s,c,p[0],p[1]])
        return result
    case_notes=[]
    for sheet,keys in cases:
        fig,axes=plt.subplots(len(keys),4,figsize=(17,20))
        fig.subplots_adjust(left=.025,right=.995,top=.91,bottom=.025,wspace=.06,hspace=.55)
        for rowi,key in enumerate(keys):
            r=lookup[key];folder=BATCH/'runs'/f'run_{r["run"]:06d}'
            record=json.loads(gzip.decompress((folder/f'attempt_{r["attempt"]:03d}.json.gz').read_bytes()))
            last=max(folder.glob('attempt_*.json.gz'));final=json.loads(gzip.decompress(last.read_bytes()))
            boards=[board(record[k]) for k in ['start','proposed_start','attempted_end']]+[board(final['end'])]
            target=[g for b in boards[:3] for sid,g in b.items() if sid in r['placed']+r['removed']]
            x0=min(g.bounds[0] for g in target)-.55;y0=min(g.bounds[1] for g in target)-.55
            x1=max(g.bounds[2] for g in target)+.55;y1=max(g.bounds[3] for g in target)+.55
            size=max(x1-x0,y1-y0,2.8);cx=(x0+x1)/2;cy=(y0+y1)/2
            bounds=[cx-size/2,cx+size/2,cy-size/2,cy+size/2]
            for col,(ax,b,title) in enumerate(zip(axes[rowi],boards,['Before','Placed','After physics',f'Last retained · attempt {final["number"]}'])):
                ax.set_facecolor('#f4f3ea')
                for sid,g in b.items():
                    selected=sid in (r['removed'] if col==0 else r['placed'])
                    color=('#7aaed4' if col==0 else '#ed9b45' if col==1 else '#81bc9a') if selected else '#ded8c5'
                    ax.add_patch(Patch(list(g.exterior.coords),facecolor=color,edgecolor='#77725e',lw=.45))
                    p=g.representative_point();ax.text(p.x,p.y,sid,ha='center',va='center',fontsize=5.5,clip_on=True)
                patio=source['patio']+[source['patio'][0]];ax.plot([p[0] for p in patio],[p[1] for p in patio],color='#375b49',lw=1.5)
                if col==2:
                    for h in r['holes']:
                        if bounds[0]<h['center'][0]<bounds[1] and bounds[2]<h['center'][1]<bounds[3]:
                            ax.add_patch(Circle(h['center'],h['post']/24,fill=False,edgecolor='#bc4338',lw=1,ls='--'))
                ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
                metric=r['before'] if col==0 else r['after'] if col==2 else r['final'] if col==3 else None
                caption=f'\n{metric["coverage"]:.2f}% · gaps {metric["interior"]:.2f}″ / {metric["edge"]:.2f}″' if metric else '\nOrange = selected pieces'
                ax.set_title(title+caption,fontsize=9)
            repaired=sum(h['repaired_at_end'] for h in r['holes'])
            verdict='uncertain (user: short follow-up; original fit good)' if key==(23,28) else r['verdict']
            info=f'Run {r["run"]} / attempt {r["attempt"]} · {r["type"]} · {verdict} | Δarea {r["delta_coverage"]:+.3f} pp · Δjoint {r["delta_joints"]:+.1f}″ · Δwall {r["delta_wall"]:+.1f}″ | enlarged sites later recovered {repaired}/{len(r["holes"])}'
            fig.text(.03,axes[rowi,0].get_position().y1+.04,info,fontsize=9,fontweight='bold')
            case_notes.append(dict(sheet=sheet,run=r['run'],attempt=r['attempt'],verdict=r['verdict'],holes=r['holes']))
        fig.suptitle(sheet.replace('_',' ').title()+' · blue original / orange placed / green after physics\nLocal crop held fixed across the row; red circles mark enlarged saved gap witnesses. Last panel includes all later actions.',fontsize=14,y=.998)
        fig.savefig(OUT/f'{sheet}.png',dpi=135);plt.close(fig)
    (OUT/'illustrated_cases.json').write_text(json.dumps(case_notes,indent=2))

    # Keep the index lightweight; full provenance and histories remain in review.json.
    thin=[]
    for r in actions:
        thin.append({k:r.get(k) for k in ['run','mode','attempt','complete','type','verdict','small_adjustment','rotation','translation','placed','removed','delta_coverage','delta_joints','delta_wall','delta_interior','delta_edge','persistent_holes','remaining_attempts','positives','negatives']})
        thin[-1].update(new_holes=len(r['holes']),repaired_holes=sum(h['repaired_at_end'] for h in r['holes']))
        thin[-1].update(repair_attempts=sorted({h['sustained_repair_attempt'] for h in r['holes'] if h['sustained_repair_attempt']}),removed_later=r['removed_later'])
        if (r['run'],r['attempt'])==(23,28):
            thin[-1].update(screening_verdict=r['verdict'],verdict='uncertain',
                            review_note='User: only two later attempts; long-term value uncertain. Previous fit was already pretty good.')
    template=(OUT/'index_template.html').read_text()
    (OUT/'index.html').write_text(template.replace('__DATA__',json.dumps(thin,separators=(',',':')).replace('</','<\\/')))
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
