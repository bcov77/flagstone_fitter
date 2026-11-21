"""Check witnesses under the proposed edge/interior gap limits; no layout changes."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import shapely
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from flagstone.phase1 import load_json,validate,transformed
from flagstone.quarter import world,check_reference
from five_stone_edge_start import bounded_gap
OUT=Path(__file__).resolve().parent
P=ROOT/'examples/quarter_campaign_20261113_161051_26702/tasks/task_217_multi_rebuild/best.json'
l=load_json(P);check_reference(l,load_json(P.parents[2]/'trial/input.json'))
gs=list(world(l).values());top=l.patio.bounds[3]
def constraints(z):
 p=shapely.Point(z[:2])
 return np.array([p.distance(g)-z[2] for g in gs]+[p.distance(l.patio.boundary)-z[2]])
global_gap=bounded_gap(l.patio,gs,l.patio,error=.001/12,max_cells=200000,stop_on_failure=False)
witnesses={}
for kind,xy,limit in [('interior',(3.958333333333334,5.708333333333334),2),('edge',tuple(global_gap['center']),5)]:
 z=[*xy,min(shapely.Point(xy).distance(g) for g in gs)]
 cons=[dict(type='ineq',fun=constraints)]
 if kind=='edge':cons.append(dict(type='eq',fun=lambda z:top-z[1]-z[2]))
 result=minimize(lambda z:-z[2],z,method='SLSQP',constraints=cons,bounds=[(xy[0]-.5,xy[0]+.5),(xy[1]-.5,min(top,xy[1]+.5)),(0,1)],options=dict(maxiter=500,ftol=1e-12))
 assert result.success and min(constraints(result.x))>=-1e-9
 x,y,r=map(float,result.x)
 if kind=='interior':r-=1e-9 # Ensure strict obstacle separation for the witness.
 point=shapely.Point(x,y);boundary_slack=point.distance(l.patio.boundary)-r
 assert (abs(boundary_slack)<1e-9) if kind=='edge' else (boundary_slack>0)
 witnesses[kind]=dict(center_ft=[x,y],diameter_in=24*r,allowed_diameter_in=limit,excess_diameter_in=24*r-limit,boundary_separation_in=12*boundary_slack,minimum_stone_separation_in=12*(min(point.distance(g) for g in gs)-r),fails=24*r>limit)
report=dict(layout=str(P.relative_to(ROOT)),layout_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),rule=dict(edge_circle_diameter_limit_in=5,interior_circle_diameter_limit_in=2),original_reference_matched=True,clearance_errors=validate(l,gap=.5/12,edge=.5/12),global_gap=global_gap,witnesses=witnesses,passes=False,note='Interior witness is a local optimum, not a certified global maximum for a separately defined interior category. Both witnesses individually disprove compliance; global upper also bounds either category. Boundary tangency checked to 1e-9 ft numerical tolerance.')
assert not report['clearance_errors'] and all(w['fails'] for w in witnesses.values())
(OUT/'check.json').write_text(json.dumps(report,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(9,9.5),facecolor='#faf9f6');ax.set_facecolor('#eee9dd')
colors=['#c2ad8c','#9faca2','#b5a8a0','#a9b8bf','#c7b99b','#a7a1a7']
for i,sid in enumerate(l.ids):
 if l.active[i]:
  g=transformed(l.shapes[sid],l.poses[i]);ax.fill(*g.exterior.xy,color=colors[i%len(colors)],edgecolor='#4d4942',lw=.8);ax.text(g.centroid.x,g.centroid.y,sid,ha='center',va='center',fontsize=8)
ax.plot(*l.patio.exterior.xy,color='#292824',lw=2)
for kind,color,textpos in [('edge','#bd302e',(5.65,8.12)),('interior','#1b538a',(2.4,6.4))]:
 w=witnesses[kind];ax.add_patch(Circle(w['center_ft'],w['diameter_in']/24,fill=False,ec=color,lw=2.5))
 ax.annotate(f"{kind.capitalize()}: {w['diameter_in']:.2f} in\nLimit: {w['allowed_diameter_in']} in",xy=w['center_ft'],xytext=textpos,ha='center',fontsize=11,color=color,arrowprops=dict(arrowstyle='->',color=color))
ax.set_aspect('equal');ax.set_xlim(-.12,7.94);ax.set_ylim(-.12,8.45);ax.set_xlabel('Feet');ax.set_ylabel('Feet')
fig.suptitle('Best saved layout — proposed gap rule',fontsize=18,weight='bold',y=.965)
ax.set_title('27 stones · 72.27% coverage · Both gap limits fail',fontsize=12,pad=15)
fig.text(.5,.04,'Original outlines and sizes. Half-inch clearance checks pass.\nCircles show independently checked empty-space witnesses; layout is unchanged.',ha='center',fontsize=10)
fig.subplots_adjust(left=.10,right=.96,top=.88,bottom=.12);fig.savefig(OUT/'checked_layout.png',dpi=180);plt.close(fig)
print(json.dumps(report,indent=2))
