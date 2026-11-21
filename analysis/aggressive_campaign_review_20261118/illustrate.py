"""Show representative retained endpoints; no simulation or geometry changes."""
import gzip,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
HERE=Path(__file__).resolve().parent
BATCH=HERE.parents[1]/'examples/aggressive_trajectories_20261118_r2/aggressive'
source=json.loads((BATCH/'source.json').read_text())
cases=[(76,15,'Useful late insertion: stone 55'),(82,16,'Two-piece fill: stones 135 + 128'),(57,25,'Coverage gain with larger gaps: 86 → 12')]
fig,axs=plt.subplots(3,2,figsize=(12,15),layout='constrained');details=[]
for axes,(run,attempt,title) in zip(axs,cases):
 with gzip.open(BATCH/f'runs/run_{run:06d}/attempt_{attempt:03d}.json.gz','rt') as f:r=json.load(f)
 p=r['chosen'];chosen=set(x['stone'] for x in p['placements']);removed=set(p['remove'])
 item=dict(run=run,attempt=attempt,title=title,chosen=p)
 for ax,key,mkey,label in zip(axes,['start','end'],['start_metrics','end_metrics'],['Before','Retained after physics']):
  frame=r[key];m=r[mkey];item[key+'_metrics']=m
  ax.add_patch(Polygon(source['patio'],facecolor='white',edgecolor='#303b41',lw=1.5))
  for s in source['stones']:
   sid=str(s['id']);pose=frame['poses'].get(sid)
   if pose is None:continue
   x,y,a=pose;c=math.cos(a);t=math.sin(a)
   points=[(x+c*u-t*v,y+t*u+c*v) for u,v in s['outline']]
   high=sid in (removed if key=='start' else chosen)
   ax.add_patch(Polygon(points,facecolor='#efa849' if high else '#d9d4c8',edgecolor='#985910' if high else '#666d6a',lw=.6))
   ax.text(x,y,sid,ha='center',va='center',fontsize=6)
  ax.autoscale_view();ax.set_aspect('equal');ax.axis('off')
  ax.set_title(f'{title} · run {run}, attempt {attempt}\n{label}: {m["area_percent"]:.3f}%\nLargest gap: interior {m["gaps"]["largest_interior_inches"]:.2f} in · edge {m["gaps"]["largest_edge_inches"]:.2f} in',fontsize=10)
 details.append(item)
fig.savefig(HERE/'move_examples.png',dpi=140);fig.savefig(HERE/'move_examples.svg')
(HERE/'illustrated_cases.json').write_text(json.dumps(details,indent=2))
print('Saved three before/after examples')
