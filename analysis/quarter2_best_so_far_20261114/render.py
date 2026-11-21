import sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from flagstone.phase1 import load_json,transformed,validate
out=Path(__file__).resolve().parent
run=ROOT/'examples/quarter2_20h_ready_20261114'
choices=[]
for p in run.glob('tasks/*/verification.json'):
    v=json.loads(p.read_text());b=p.with_name('best.json')
    if not b.exists():continue
    data=b.read_bytes();g=v.get('gaps',{});groups=g.get('groups',{})
    if (v.get('rule')=='local-gap-circles-edge5-interior2-v1' and v.get('original_geometry') is True
        and v.get('reference_matched') is True and v.get('clearance_errors')==[] and g.get('stable') is True
        and set(groups)=={'edge','interior'} and v.get('layout_sha256')==hashlib.sha256(data).hexdigest()):
        score=max(groups['edge']['upper_diameter_in']/5,groups['interior']['upper_diameter_in']/2)
        choices.append((score,str(b),data,v))
score,path,data,v=min(choices,key=lambda x:(x[0],x[1]))
(out/'best.json').write_bytes(data)
(out/'verification.json').write_text(json.dumps(v,indent=2))
layout=load_json(out/'best.json')
errors=validate(layout,gap=.5/12,edge=.5/12)
assert not errors,errors
fig,ax=plt.subplots(figsize=(10,11),facecolor='#faf8f4')
ax.set_facecolor('#ebe5da')
colors=['#c8ad89','#a6aaa2','#b8a18b','#9ba6a2','#c2b69e','#ac9891','#b7b8aa']
area=0
for i,sid in enumerate(layout.ids):
    if not layout.active[i]:continue
    geom=transformed(layout.shapes[sid],layout.poses[i]);area+=geom.area
    ax.fill(*geom.exterior.xy,facecolor=colors[i%len(colors)],edgecolor='#5b574e',linewidth=1.1,zorder=2)
    pt=geom.representative_point()
    ax.text(pt.x,pt.y,str(sid),ha='center',va='center',fontsize=8,color='#302e29',zorder=3)
ax.plot(*layout.patio.exterior.xy,color='#373a36',linewidth=2,zorder=4)
for kind,color in [('edge','#b36c12'),('interior','#b52e40')]:
    g=v['gaps']['groups'][kind]
    diameter=(g['lower_diameter_in']+g['upper_diameter_in'])/2
    ax.add_patch(Circle(g['center'],diameter/24,facecolor=color+'18',edgecolor=color,linewidth=2.2,zorder=5,
                        label=f"Largest {kind} gap: {g['upper_diameter_in']:.2f} in (limit {g['limit_in']} in)"))
ax.set_aspect('equal');ax.set_xlim(-.2,8.01);ax.set_ylim(-.2,8.01)
ax.set_xlabel('Feet',fontsize=11);ax.set_ylabel('Feet',fontsize=11)
ax.spines[['top','right']].set_visible(False)
fig.suptitle('Best checked layout so far',fontsize=22,fontweight='bold',y=.97,color='#302e29')
ax.set_title(f"20-hour campaign · Task 83 · Adaptive grid\n{int(layout.active.sum())} stones · {area/layout.patio.area:.1%} coverage · 7.81 × 7.81 ft test patio",fontsize=12,pad=16,color='#514d46')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.09),frameon=False,fontsize=11)
fig.text(.5,.035,'Original stone outlines and sizes; labels are inventory IDs. Half-inch clearances pass.\nGap targets remain unmet. Dual-scale numerical checks include a 0.002-inch margin.',ha='center',fontsize=10,color='#605b52')
fig.subplots_adjust(top=.86,bottom=.17,left=.10,right=.96)
fig.savefig(out/'best_layout.png',dpi=200,facecolor=fig.get_facecolor())
plt.close(fig)
metadata=dict(source=path,snapshot_utc=datetime.now(timezone.utc).isoformat(),checked_layouts=len(choices),normalized_upper=score,placed=int(layout.active.sum()),coverage=area/layout.patio.area,groups=v['gaps']['groups'])
(out/'snapshot.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata,indent=2))
