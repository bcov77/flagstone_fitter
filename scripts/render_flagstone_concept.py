"""Render an independently validated solver layout as a shareable concept."""
from pathlib import Path
import sys
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
from flagstone.phase1 import load_json, transformed, validate

root=Path(sys.argv[1]) if len(sys.argv)>1 else Path('examples/nonparallel_search')
layout=load_json(root/'run/best.json')
errors=validate(layout,gap=1/24,edge=1/24)
if errors: raise ValueError(f'Invalid layout: {len(errors)} clearance violations')
coverage=sum(layout.shapes[s].area for i,s in enumerate(layout.ids) if layout.active[i])/layout.patio.area
fig, ax=plt.subplots(figsize=(15,8),facecolor='#f8f6f2',layout='constrained')
ax.set_facecolor('#f8f6f2')
ax.add_patch(Patch(layout.patio.exterior.coords,facecolor='#dbd3c8',edgecolor='#4b4541',lw=1.4,zorder=0))
palette=['#c4aaa0','#cdb9a9','#bda7a2','#d2c3b7','#b5a7a3','#ceb0a5','#b9b1ac','#d2bfb3']
for i,sid in enumerate(layout.ids):
    if layout.active[i]:
        s=transformed(layout.shapes[sid],layout.poses[i]); color=palette[hashlib.sha256(sid.encode()).digest()[0]%len(palette)]
        ax.add_patch(Patch(s.exterior.coords,facecolor=color,edgecolor='#6a5b55',lw=.45,zorder=2))
ax.set_aspect('equal');ax.autoscale_view();ax.margins(.045);ax.axis('off')
fig.suptitle('Flagstone patio concept',fontsize=25,fontweight='semibold',color='#332d2a')
ax.set_title(f'Synthetic layout  ·  {int(layout.active.sum())} stones  ·  {coverage:.1%} stone coverage',fontsize=12,color='#645a54',pad=12)
fig.text(.5,.04,'25-foot-wide patio sketch   •   minimum ½-inch spacing   •   concept, not a measured plan',ha='center',fontsize=10,color='#645a54')
fig.savefig(root/'flagstone_concept.png',dpi=200,bbox_inches='tight',facecolor=fig.get_facecolor());plt.close(fig)
print('coverage',coverage,'stones',sum(layout.active),'violations',len(errors))
