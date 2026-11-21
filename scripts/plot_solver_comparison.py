"""Compare validated layouts at the same physical scale."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
from flagstone.phase1 import load_json, transformed, validate

root=Path('examples/revised_solver_comparison')
cases=[('PackingSolver',root/'packingsolver/best.json'),('U-nesting',root/'unesting.layout.json'),('Ironnest',root/'ironnest_best.layout.json')]
fig,axes=plt.subplots(3,1,figsize=(14,12),layout='constrained')
for ax,(name,path) in zip(axes,cases):
    x=load_json(path); errors=validate(x,gap=1/24,edge=1/24)
    if errors: raise ValueError(f'{name} has {len(errors)} violations')
    coverage=sum(x.shapes[s].area for i,s in enumerate(x.ids) if x.active[i])/x.patio.area
    ax.add_patch(Patch(x.patio.exterior.coords,facecolor='#e9e3d9',edgecolor='#222',lw=1))
    for i,sid in enumerate(x.ids):
        if x.active[i]: ax.add_patch(Patch(transformed(x.shapes[sid],x.poses[i]).exterior.coords,facecolor='#c3a99e',edgecolor='#393333',lw=.5))
    ax.set_aspect('equal');ax.autoscale_view();ax.margins(.02);ax.axis('off')
    ax.set_title(f'{name}: {coverage:.1%} coverage, {sum(x.active)} stones',loc='left',fontsize=13)
fig.suptitle('Three-minute solver comparison — same synthetic patio and stones',fontsize=16,weight='bold')
fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
