"""Full-board illustrations of useful findings and their tradeoffs."""
from experiment import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def main():
 cases=[(62,'fine_insert_c0_s0','Finer-grid insertion of stone 53'),(62,'repack_s0','Two-for-two replacement'),(105,'root_reorientation_s0','Same-piece reorientation')]
 fig,axs=plt.subplots(3,2,figsize=(12,15),layout='constrained')
 for axes,(n,ident,title) in zip(axs,cases):
  folder=HERE/f'run_{n:06d}';source=read(folder/'source.json');root=read(folder/'root_metrics.json');r=read(folder/(ident+'.json.gz'));p=r['action'];chosen={x['stone'] for x in p['placements']};removed=set(p['remove'])
  for ax,frame,m,sel,label in zip(axes,[state(source),r['end']],[root,r['end_metrics']],[removed,chosen],['Saved board','Retained after physics']):
   ax.add_patch(Polygon(source['patio'],facecolor='white',edgecolor='#303b41',lw=1.5))
   for s in source['stones']:
    sid=str(s['id']);pose=frame['poses'].get(sid)
    if pose is None:continue
    x,y,a=pose;c=math.cos(a);t=math.sin(a);points=[(x+c*u-t*v,y+t*u+c*v) for u,v in s['outline']];high=sid in sel
    ax.add_patch(Polygon(points,facecolor='#efa849' if high else '#d9d4c8',edgecolor='#985910' if high else '#666d6a',lw=.6));ax.text(x,y,sid,ha='center',va='center',fontsize=6)
   ax.autoscale_view();ax.set_aspect('equal');ax.axis('off')
   ax.set_title(f'{title} · run {n}\n{label}: {m["area_percent"]:.3f}%\nLargest gap: interior {m["gaps"]["largest_interior_inches"]:.2f} in · edge {m["gaps"]["largest_edge_inches"]:.2f} in',fontsize=10)
 fig.savefig(HERE/'illustrated_findings.png',dpi=140);fig.savefig(HERE/'illustrated_findings.svg')
 print('Rendered three representative comparisons')
if __name__=='__main__':main()
