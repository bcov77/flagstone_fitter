"""Show a validated gap-repair result and the largest sampled empty circles."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as Patch
from flagstone.phase1 import load_json, transformed, validate

root=Path('examples/gap_repair'); report=json.loads((root/'report.json').read_text())
before=load_json(report['input']); after=load_json(root/'best.json')
for x in (before,after):
    if validate(x,gap=1/24,edge=1/24):raise ValueError('Invalid layout')
added={sid for i,sid in enumerate(after.ids) if after.active[i] and not before.active[i]}
fig,axes=plt.subplots(2,1,figsize=(13,10),facecolor='#f8f6f2')
fig.subplots_adjust(top=.91,bottom=.085,left=.035,right=.965,hspace=.22)
for ax,x,key in zip(axes,(before,after),('before','after')):
    ax.add_patch(Patch(x.patio.exterior.coords,facecolor='#ded7cb',edgecolor='#453c37',lw=1))
    for i,sid in enumerate(x.ids):
        if x.active[i]:
            shape=transformed(x.shapes[sid],x.poses[i])
            ax.add_patch(Patch(shape.exterior.coords,facecolor='#bd7862' if sid in added else '#c2ada4',edgecolor='#6a5b55',lw=.4))
    v=report[key]; ax.add_patch(Circle(v['largest_center'],v['maximum_diameter']/2,fill=False,color='#a4342b',lw=2,linestyle='--'))
    cov=report['coverage_'+key];n=report['placed_'+key]
    ax.set_title(f"{key.capitalize()}: {cov:.1%} coverage · {n} stones · largest sampled circle {v['maximum_diameter']*12:.1f} in",loc='left',fontsize=12)
    ax.set_aspect('equal');ax.autoscale_view();ax.margins(.02);ax.axis('off')
fig.suptitle('Filling the largest gaps',fontsize=19,fontweight='bold')
fig.text(.5,.025,'Terracotta = added stone  •  dashed circle = sampled empty circle  •  synthetic patio, ½-inch minimum spacing',ha='center',fontsize=9)
fig.savefig(root/'before_after.png',dpi=180,bbox_inches='tight',facecolor=fig.get_facecolor());plt.close(fig)
