"""Compare saved human endpoints without inventing an unavailable edit sequence."""
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
import numpy as np
import shapely
from shapely.affinity import affine_transform

OUT=Path(__file__).parent;ROOT=OUT.parents[1]
FILES=dict(start=ROOT/'examples/physics_trajectories_20261117_r3/source.json',
           earlier_manual=ROOT/'physics_patio/sessions/recovery_before_gap_upgrade_20261115_102954.json',
           later_manual=Path('/Users/brian/Downloads/patio-physics-2026-11-16T15-45-13.444Z.json'))


def main():
    docs={key:json.loads(path.read_text()) for key,path in FILES.items()}
    all_stones={key:{str(s['id']):s for s in doc['stones']} for key,doc in docs.items()}
    active={key:{sid:s for sid,s in stones.items() if s.get('active',False)} for key,stones in all_stones.items()}
    def world(s):
        x,y,a=s['pose'];c,t=math.cos(a),math.sin(a)
        return affine_transform(shapely.Polygon(s['outline']),[c,-t,t,c,x,y])
    boards={key:{sid:world(s) for sid,s in stones.items()} for key,stones in active.items()}
    patio=shapely.Polygon(docs['start']['patio']);measurements={};changes={}
    for key,board in boards.items():
        pairs=[];overlaps=[];walls=[]
        for i,(sid,g) in enumerate(board.items()):
            wall=g.distance(patio.boundary)*12
            if wall<.5-1.2e-7 or not patio.covers(g):walls.append(dict(stone=sid,distance_inches=wall,outside_sqft=g.difference(patio).area))
            for tid,h in list(board.items())[i+1:]:
                distance=g.distance(h)*12
                if distance<.5-1.2e-7:pairs.append(dict(stones=[sid,tid],distance_inches=distance))
                overlap=g.intersection(h).area
                if overlap>1e-10:overlaps.append(dict(stones=[sid,tid],area_sqft=overlap))
        measurements[key]=dict(source=str(FILES[key]),sha256=hashlib.sha256(FILES[key].read_bytes()).hexdigest(),
                               active_count=len(board),nominal_coverage=100*sum(g.area for g in board.values())/patio.area,
                               union_coverage=100*shapely.union_all(list(board.values())).intersection(patio).area/patio.area,
                               pair_violations=len(pairs),wall_violations=len(walls),overlap_pairs=len(overlaps),
                               pairwise_overlap_sqin=sum(p['area_sqft'] for p in overlaps)*144,
                               settings=docs[key].get('physics',{}).get('settings',{}),
                               locked_active=[sid for sid in board if docs[key].get('physics',{}).get('pieces',{}).get(sid,{}).get('locked')])
    for start,end in [('start','earlier_manual'),('start','later_manual'),('earlier_manual','later_manual')]:
        rows=[]
        for sid in sorted(set(active[start])&set(active[end]),key=int):
            a,b=active[start][sid],active[end][sid]
            assert a['outline']==b['outline'],'Inventory outline changed'
            p,q=a['pose'],b['pose'];rot=abs(math.degrees(math.remainder(q[2]-p[2],2*math.pi)))
            center=math.dist(p[:2],q[:2])*12
            va=np.asarray(boards[start][sid].exterior.coords);vb=np.asarray(boards[end][sid].exterior.coords)
            vertex=float(np.max(np.linalg.norm(vb-va,axis=1)))*12
            rows.append(dict(stone=sid,center_inches=center,rotation_degrees=rot,max_corresponding_vertex_inches=vertex))
        moves=[r['center_inches'] for r in rows]
        changes[f'{start}_to_{end}']=dict(common_stones=len(rows),added=sorted(set(active[end])-set(active[start]),key=int),
                                           removed=sorted(set(active[start])-set(active[end]),key=int),
                                           center_median=float(np.median(moves)),center_p90=float(np.quantile(moves,.9)),center_max=max(moves),
                                           more_than_1_5_inches=sum(v>1.5 for v in moves),more_than_3_inches=sum(v>3 for v in moves),more_than_6_inches=sum(v>6 for v in moves),
                                           rotation_median=float(np.median([r['rotation_degrees'] for r in rows])),rows=rows)
    result=dict(measurements=measurements,changes=changes,
                limitation='These are three saved endpoints, not an ordered edit/physics sequence. Net changes include deliberate relocation and intervening physics; they do not establish per-insertion motion or peak excursion.')
    (OUT/'manual_reference.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(18,6.5))
    for ax,(key,board) in zip(axes,boards.items()):
        for sid,g in board.items():
            new=sid not in boards['start'];ax.add_patch(Patch(list(g.exterior.coords),facecolor='#e9a254' if new else '#d7cbb0',edgecolor='#655c4b',lw=.5))
            p=g.representative_point();ax.text(p.x,p.y,sid,fontsize=6,ha='center',va='center')
        xy=list(patio.exterior.coords);ax.plot(*zip(*xy),color='#355944',lw=1.3)
        ax.set_aspect('equal');ax.set_xlim(-.15,7.96);ax.set_ylim(-.15,7.96);ax.set_xticks([]);ax.set_yticks([])
        m=measurements[key];ax.set_title(f'{key.replace("_"," ").title()} · {m["active_count"]} stones\n{m["union_coverage"]:.2f}% union coverage · pair/wall shortfalls {m["pair_violations"]}/{m["wall_violations"]}',fontsize=11)
    fig.suptitle('Saved human packing references · orange = added since algorithm start\nSnapshots only; intermediate order and motion paths were not saved.',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(OUT/'manual_reference.png',dpi=145);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,9))
    change={r['stone']:r for r in changes['start_to_later_manual']['rows']}
    for sid,r in change.items():
        old=boards['start'][sid].simplify(.004);new=boards['later_manual'][sid].simplify(.004)
        ax.plot(*old.exterior.xy,color='#9b9e99',lw=.7,ls='--')
        color='#7eb593' if r['center_inches']<=3 else '#e3ad5e' if r['center_inches']<=6 else '#cf7c69'
        ax.add_patch(Patch(list(new.exterior.coords),facecolor=color,edgecolor='#655c4b',lw=.6,alpha=.75))
        p,q=active['start'][sid]['pose'],active['later_manual'][sid]['pose'];ax.annotate('',xy=q[:2],xytext=p[:2],arrowprops=dict(arrowstyle='->',lw=.8,color='#485e72'))
        pt=new.representative_point();ax.text(pt.x,pt.y,f'{sid}\n{r["center_inches"]:.1f}″',ha='center',va='center',fontsize=7)
    ax.plot(*patio.exterior.xy,color='#355944');ax.set_aspect('equal');ax.set_xlim(-.1,7.91);ax.set_ylim(-.1,7.91)
    ax.set_title('Common retained stones: algorithm start → later human endpoint\nDashed = starting outline · arrows = net center displacement, not motion paths\nGreen ≤3″ · amber 3–6″ · red >6″',fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'manual_retained_motion.png',dpi=145);plt.close(fig)
    print(json.dumps(dict(measurements={k:{a:b for a,b in v.items() if a not in ['settings','source','sha256','locked_active']} for k,v in measurements.items()},changes={k:{a:b for a,b in v.items() if a!='rows'} for k,v in changes.items()}),indent=2))


if __name__=='__main__':main()
