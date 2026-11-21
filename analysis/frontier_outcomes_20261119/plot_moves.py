"""Render original full-size outlines for representative saved moves."""
import json, gzip, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'examples/frontier_search_20261118_cached_r2'
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
source=read(OUT/'source.json');local={str(s['id']):s['outline'] for s in source['stones']}
fig,axs=plt.subplots(3,2,figsize=(12,16),layout='constrained')
for row,number in enumerate([5222,3104,3575]):
 d=OUT/'jobs'/f'job_{number:08d}';r=read(d/'result.json.gz');j=read(d/'job.json');parent=read(OUT/'nodes'/(j['parent']+'.json.gz'));p=r['chosen']
 for col,(frame,m,label) in enumerate([(r['start'],parent['metrics'],'Before'),(r['end'],r['end_metrics'],'After')]):
  ax=axs[row,col];ax.add_patch(Polygon(source['patio'],facecolor='#f8f5ec',edgecolor='#343a40',lw=1.5))
  selected=set(p['remove'] if col==0 else [x['stone'] for x in p['placements']])
  for sid,(x,y,a) in frame['poses'].items():
   c,s=math.cos(a),math.sin(a);xy=[(x+c*u-s*v,y+s*u+c*v) for u,v in local[sid]]
   ax.add_patch(Polygon(xy,facecolor='#e9a348' if sid in selected else '#d4d1c8',edgecolor='#796649' if sid in selected else '#88877f',lw=.6));ax.text(x,y,sid,fontsize=6,ha='center',va='center')
  g=m['gaps'];circle=g['groups']['interior'];ax.add_patch(Circle(circle['center'],g['largest_interior_inches']/24,fill=False,edgecolor='#c33643',lw=2));ax.plot(*circle['center'],'+',color='#c33643')
  ax.set_aspect('equal');ax.autoscale_view();ax.set_axis_off();ax.set_title(f"Job {number}: {j['operator']} — {label}\n{m['area_percent']:.3f}% coverage; worst interior gap {g['largest_interior_inches']:.2f} in",fontsize=11)
fig.suptitle('Useful gains and temporary gap tradeoffs\nOrange: removed / placed stones. Red: largest interior gap.',fontsize=16)
fig.savefig(HERE/'move_examples.png',dpi=150)
plt.close(fig)
