from experiment import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
folder=HERE/'run_000062';source=read(folder/'source.json');root=read(folder/'root_metrics.json');a=read(folder/'root_compound_replace_s0.json.gz');b=read(folder/'follow_root_compound_replace_s0_0_probe_replace.json.gz')
frames=[state(source),a['end'],b['end']];metrics=[root,a['end_metrics'],b['end_metrics']];docs=[restore(source,f) for f in frames]
ids=set(a['action']['remove']+b['action']['remove'])|{p['stone'] for r in [a,b] for p in r['action']['placements']}
core=unary_union([g for doc in docs for sid,g in shapes(doc).items() if sid in ids]);x0,y0,x1,y1=core.bounds
fig,axs=plt.subplots(1,3,figsize=(14,5.5),layout='constrained')
for i,(ax,frame,m,title) in enumerate(zip(axs,frames,metrics,['Saved board','Step 1: 100 → 121 + 53','Step 2: 135 → 79'])):
 ax.add_patch(Polygon(source['patio'],facecolor='white',edgecolor='#303b41',lw=1.5))
 chosen=set() if i==0 else {p['stone'] for p in [a,b][i-1]['action']['placements']};old=set(a['action']['remove']+b['action']['remove'])
 for s in source['stones']:
  sid=str(s['id']);pose=frame['poses'].get(sid)
  if pose is None:continue
  x,y,t=pose;c=math.cos(t);sn=math.sin(t);points=[(x+c*u-sn*v,y+sn*u+c*v) for u,v in s['outline']]
  color='#efa849' if sid in chosen or (i==0 and sid in old) else '#a9cdd9' if i==2 and sid in {'121','53'} else '#d9d4c8'
  ax.add_patch(Polygon(points,facecolor=color,edgecolor='#666d6a',lw=.8));ax.text(x,y,sid,ha='center',va='center',fontsize=9,clip_on=True)
 ax.set_xlim(x0-.25,x1+.25);ax.set_ylim(max(0,y0-.25),y1+.25);ax.set_aspect('equal');ax.axis('off');ax.set_title(f'{title}\nCoverage {m["area_percent"]:.3f}%\nInterior / edge max gaps: {m["gaps"]["largest_interior_inches"]:.2f} / {m["gaps"]["largest_edge_inches"]:.2f} in',fontsize=11)
fig.savefig(HERE/'setup_sequence.png',dpi=150)
print('Rendered setup sequence')
