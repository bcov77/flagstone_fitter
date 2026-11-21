"""Real, matched before/probe/after examples of current edge-fit score limitations.
Uses saved trajectories only. The 90% panel is a geometric probe, not a run.
"""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
from shapely.geometry import Polygon
from shapely.affinity import affine_transform,scale
from shapely.ops import unary_union
OUT=Path(__file__).with_name('campaign_v1');DEST=OUT/'completed_analysis'
manifest=json.loads((OUT/'manifest.json').read_text());cases={c['id']:c for c in manifest['cases']}
source=json.loads((OUT/'source.json').read_text());stones={str(s['id']):s for s in source['stones']};patio=Polygon(source['patio'])
metrics=pd.read_csv(DEST/'trial_outcomes.csv').set_index('id')
PAIRS=[
 ('Same opening, stone 135: the higher initial matching score leaves more space',
  ['c994ee38c9085712b3d526fa','cde41c43cdb78b835fd17a3d']),
 ('Same replacement opening, stone 144: a better-looking shrink probe still loses',
  ['f438d829177cd8cb305cf116','9a5cabe10c2b5702a18bd2f9'])]

def world(sid,pose):
 x,y,t=pose;c,s=math.cos(t),math.sin(t)
 return affine_transform(Polygon(stones[sid]['outline']),(c,-s,s,c,x,y))
def patches(ax,geometry,**kwargs):
 if geometry.is_empty:return
 parts=list(geometry.geoms) if hasattr(geometry,'geoms') else [geometry]
 for g in parts:
  if g.geom_type=='Polygon':ax.add_patch(Patch(list(g.exterior.coords),**kwargs))

def draw(ax,r,case,stage,bounds):
 sid=case['candidate']['id'];initial=r['initial_poses'];final=r['final_poses']
 current=final if stage=='after' else initial
 if stage=='after':
  for other,pose in initial.items():
   if other!=sid:
    g=world(other,pose);x,y=g.exterior.xy;ax.plot(x,y,color='#738091',linestyle='--',linewidth=.55,alpha=.75,zorder=4)
 neighbors=[]
 for other,pose in current.items():
  if other==sid:continue
  g=world(other,pose);neighbors.append(g)
  patches(ax,g,facecolor='#ddd6c8',edgecolor='#70695f',linewidth=.7,zorder=1)
  p=g.representative_point();ax.text(p.x,p.y,other,fontsize=8,color='#504a42',ha='center',va='center',clip_on=True,zorder=5)
 candidate=world(sid,current[sid])
 if stage=='probe':
  full=candidate;x,y=full.exterior.xy;ax.plot(x,y,'--',color='#bc773f',linewidth=1,zorder=5)
  candidate=scale(candidate,xfact=.9,yfact=.9,origin='centroid')
 patches(ax,candidate,facecolor='#e5a567',edgecolor='#78461e',linewidth=1.2,zorder=3)
 if stage!='after':
  exclusion=unary_union([unary_union(neighbors).buffer(.5/12),patio.buffer(4).difference(patio.buffer(-.5/12))])
  intrusion=candidate.intersection(exclusion)
  patches(ax,intrusion,facecolor='#df7773',edgecolor='#a8413a',linewidth=.4,hatch='///',alpha=.8,zorder=4)
 else:
  x,y=candidate.exterior.xy;ax.plot(x,y,color='#357c53',linewidth=1.2,zorder=5)
 p=candidate.representative_point();ax.text(p.x,p.y,sid,fontsize=11,fontweight='bold',ha='center',va='center',zorder=6)
 x,y=patio.exterior.xy;ax.plot(x,y,color='#303030',linewidth=1.1,zorder=2)
 ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.set_aspect('equal');ax.tick_params(labelsize=8)
 ax.set_facecolor('#fcfbf8');ax.set_xlabel('feet',fontsize=8)
 for spine in ax.spines.values():spine.set_color('#aaa49a')

fig=plt.figure(figsize=(16,22),facecolor='white')
grid=fig.add_gridspec(4,3,left=.06,right=.97,bottom=.065,top=.89,hspace=.40,wspace=.12)
fig.text(.5,.975,'Why the current edge-fit score is not enough',ha='center',va='top',fontsize=22,fontweight='bold')
fig.text(.5,.948,'Rows A/B: stone 135. Rows C/D: stone 144. Same starting neighbors within each pair; direct protocol, seed 17.',ha='center',fontsize=12)
fig.text(.5,.928,'Orange = proposed stone   Red hatch = violates the 0.5″ clearance region   Dashed gray = neighbors’ starting outlines',ha='center',fontsize=11)
metadata=[]
for pair_index,(heading,ids) in enumerate(PAIRS):
 records=[json.loads((OUT/'results'/f'{tid}.json').read_text()) for tid in ids]
 pair_cases=[cases[r['task']['case']] for r in records]
 # Validate that the displayed differences are not changes in inventory or neighbors.
 assert pair_cases[0]['target']==pair_cases[1]['target']
 sid=pair_cases[0]['candidate']['id'];assert sid==pair_cases[1]['candidate']['id']
 assert records[0]['task']['seed']==records[1]['task']['seed']==17
 assert records[0]['task']['protocol']==records[1]['task']['protocol']=='direct'
 assert {k:v for k,v in records[0]['initial_poses'].items() if k!=sid}=={k:v for k,v in records[1]['initial_poses'].items() if k!=sid}
 all_shapes=[world(sid,r[stage][sid]) for r in records for stage in ['initial_poses','final_poses']]
 a,b,c,d=unary_union(all_shapes).bounds;span=max(c-a,d-b)+1.25;cx,cy=(a+c)/2,(b+d)/2
 x0,y0=cx-span/2,cy-span/2
 if x0<-.25:x0=-.25
 if y0<-.25:y0=-.25
 if x0+span>patio.bounds[2]+.25:x0=patio.bounds[2]+.25-span
 if y0+span>patio.bounds[3]+.25:y0=patio.bounds[3]+.25-span
 bounds=(x0,y0,x0+span,y0+span)
 row_top=.89-pair_index*(.825/2+.035)
 # Place pair labels using actual subplot geometry, not an approximate row-height.
 first_ax=None
 for within,(tid,r,case) in enumerate(zip(ids,records,pair_cases)):
  row=pair_index*2+within;label=chr(ord('A')+row);f=case['features'];full=f['scale_profiles']['1.0'];probe=f['scale_profiles']['0.9'];observed=metrics.loc[tid]
  for column,stage in enumerate(['before','probe','after']):
   ax=fig.add_subplot(grid[row,column]);draw(ax,r,case,stage,bounds)
   if column==0:
    ax.set_title(f"{label} · Stone {sid}, before settling\nClearance intrusion: {full['intrusion_sqft']/f['area_sqft']:.1%} | Near-edge perimeter: {full['near_fraction']:.0%}",fontsize=11,pad=10)
    if within==0:first_ax=ax
   elif column==1:
    ax.set_title(f"90% size probe — never simulated\nClearance intrusion: {probe['intrusion_sqft']/(f['area_sqft']*.81):.1%} | Near-edge perimeter: {probe['near_fraction']:.0%}",fontsize=11,pad=10)
   else:
    good=observed.delta_local_void<0
    ax.set_title(f"Full size, after settling — clearance passes\nLocal wide-gap change: {observed.delta_local_void:+.2f} ft² {'(less space)' if good else '(more space)'}",fontsize=11,pad=10,color='#266746' if good else '#a43d37')
  metadata.append({'row':label,'task':tid,'case':case['id'],'stone':sid,'pair_heading':heading,
                   'initial_intrusion_fraction':full['intrusion_sqft']/f['area_sqft'],
                   'initial_near_fraction':full['near_fraction'],
                   'probe_intrusion_fraction_of_probe':probe['intrusion_sqft']/(f['area_sqft']*.81),
                   'probe_near_fraction':probe['near_fraction'],
                   'delta_local_wide_gap_sqft':observed.delta_local_void,
                   'delta_far_wide_gap_sqft':observed.delta_far_void,
                   'delta_global_wide_gap_sqft':observed.delta_void,'same_neighbor_poses_verified':True,'bounds_ft':list(bounds)})

fig.text(.06,.041,'Near-edge perimeter = sampled boundary 0.4–1.0″ from a neighbor or wall. It is a score, not a guarantee of fit.',fontsize=11)
fig.text(.06,.025,'Wide-gap change is relative to the matched empty-slot/original-piece control. Negative is better. Views match within each pair.',fontsize=10)
fig.text(.06,.010,'Both poses in each pair settle legally. These illustrate score limitations; they do not establish that a particular bump caused the difference.',fontsize=10,color='#555555')
fig.savefig(DEST/'edge_fit_before_after.png',dpi=165,facecolor='white');plt.close(fig)
(DEST/'edge_fit_before_after_examples.json').write_text(json.dumps(metadata,indent=2))
print('Saved',DEST/'edge_fit_before_after.png')

# Smaller pair-specific figures for easier on-screen comparison.
for pair_index,(heading,ids) in enumerate(PAIRS):
 fig,axes=plt.subplots(2,3,figsize=(16,11),facecolor='white')
 pairmeta=metadata[pair_index*2:pair_index*2+2]
 for within,(tid,meta) in enumerate(zip(ids,pairmeta)):
  r=json.loads((OUT/'results'/f'{tid}.json').read_text());case=cases[r['task']['case']]
  for column,stage in enumerate(['before','probe','after']):
   ax=axes[within,column];draw(ax,r,case,stage,meta['bounds_ft'])
   if stage=='before':title=f"{meta['row']} · Full size before settling\nIntrusion {meta['initial_intrusion_fraction']:.1%}; near-edge perimeter {meta['initial_near_fraction']:.0%}"
   elif stage=='probe':title=f"90% probe — never simulated\nIntrusion {meta['probe_intrusion_fraction_of_probe']:.1%}; near-edge perimeter {meta['probe_near_fraction']:.0%}"
   else:title=f"Full size after settling — clearance passes\nLocal wide-gap change {meta['delta_local_wide_gap_sqft']:+.2f} ft²"
   ax.set_title(title,fontsize=11,pad=9,color=('#a43d37' if stage=='after' and meta['delta_local_wide_gap_sqft']>0 else '#333333'))
 fig.suptitle(heading,fontsize=17,fontweight='bold',y=.98)
 fig.text(.5,.94,'Orange: candidate | red hatch: clearance violation | dashed gray: original neighbor positions',ha='center',fontsize=11)
 fig.text(.5,.015,'Saved direct trials, seed 17. Negative wide-gap change is better; reference is the matched empty-slot/original-piece control.',ha='center',fontsize=10)
 fig.subplots_adjust(left=.055,right=.98,bottom=.07,top=.88,hspace=.32,wspace=.11)
 fig.savefig(DEST/f"edge_fit_stone_{pairmeta[0]['stone']}.png",dpi=165,facecolor='white');plt.close(fig)
