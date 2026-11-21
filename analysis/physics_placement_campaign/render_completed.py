"""Figures and counterexamples for the finished placement-learning campaign."""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
OUT=Path(__file__).with_name('campaign_v1');DEST=OUT/'completed_analysis'
trial=pd.read_csv(DEST/'trial_outcomes.csv');trial=trial[~trial.kind.isin(['control','empty_diagnostic'])]
cases=pd.read_csv(DEST/'case_features_outcomes.csv');bands=pd.read_csv(DEST/'intrusion_bands.csv');summary=json.loads((DEST/'summary.json').read_text())
method=pd.read_csv(DEST/'proposal_kind_comparison.csv')
fig,axes=plt.subplots(2,2,figsize=(13,10))
ax=axes[0,0]
for j,family in enumerate(['insertion','replacement']):
 rows=method[(method.family==family)&(method.kind.isin(['random','geometry','stress']))].set_index('kind').loc[['random','geometry','stress']]
 ax.bar(np.arange(3)+(j-.5)*.35,rows.useful_rate*100,width=.35,label=family)
ax.set_xticks(range(3),['Random','Geometry-ranked','High intrusion']);ax.set_ylim(0,100);ax.set_ylabel('Useful trial outcomes (%)');ax.legend();ax.set_title('The same screen behaves differently by action')
ax=axes[0,1];x=np.arange(len(bands))
ax.plot(x,bands.success_rate*100,'o-',label='Settled with clearance');ax.plot(x,bands.useful_rate*100,'s-',label='Useful packing improvement')
ax.set_xticks(x,bands.intrusion_bin);ax.set_xlabel('Initial fraction intruding into clearance exclusion');ax.set_ylim(0,105);ax.set_ylabel('Trial outcomes (%)');ax.legend();ax.set_title('Easy to settle does not mean useful')
ax=axes[1,0];p=summary['paired_protocols']
ax.bar([0,1],[p['only_cycle_feasible'],p['only_direct_feasible']],color=['#56877c','#bd8264'])
ax.set_xticks([0,1],['Only gap-cycle settles','Only direct settles']);ax.set_ylabel('Matched placement/seed pairs');ax.set_title('Gap cycle: selective rescue, with reversals')
for i,n in enumerate([p['only_cycle_feasible'],p['only_direct_feasible']]):ax.text(i,n+.25,str(n),ha='center')
ax=axes[1,1];ranking=pd.read_csv(DEST/'target_ranking_comparison.csv')
for family,marker in [('insertion','o'),('replacement','x')]:
 rows=ranking[(ranking.family==family)&(ranking.ranker.isin(['random','area_only','area_overlap','plus_boundary','plus_scale']))].groupby('ranker').top3_useful_rate.mean()
 names=['random','area_only','area_overlap','plus_boundary','plus_scale']
 ax.plot(range(5),[rows[n]*100 for n in names],marker+'-',label=family)
ax.set_xticks(range(5),['Random','Largest area','Area + overlap','+ Boundary','+ Scaling'],rotation=15);ax.set_ylim(0,105);ax.set_ylabel('Useful among top-three proposals (%)');ax.legend();ax.set_title('Within-pool ranking; grouped model predictions')
fig.suptitle('Completed campaign: 3,840 candidate trials / 960 proposed piece-poses\nUseful = feasible, preserves area, reduces broad voids, without flagged local-benefit/remote-cost tradeoff')
fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(DEST/'findings.png',dpi=170);plt.close(fig)
# Deliberately selected examples, not a representative success-rate sample.
selected=[]
def pick(label,frame,key,largest=True):
 if len(frame):
  r=frame.sort_values(key,ascending=not largest).iloc[0];selected.append((label,r))
pick('Useful insertion',trial[(trial.family=='insertion')&trial.useful], 'delta_void',False)
pick('Useful larger replacement',trial[(trial.family=='replacement')&trial.useful&(trial.delta_covered_area>.05)],'delta_covered_area')
pick('Smaller replacement opens space',trial[(trial.family=='replacement')&trial.success&(trial.delta_covered_area<-.2)],'delta_void')
pick('Local gain, remote damage',trial[trial.remote_cost],'delta_far_void')
pick('High-intrusion failure',trial[~trial.success],'pair_deficit')
pick('Large neighbor rearrangement',trial[trial.success],'max_neighbor_shift_inches')
m=json.loads((OUT/'manifest.json').read_text());lookup={c['id']:c for c in m['cases']};source=json.loads((OUT/'source.json').read_text());stones={str(s['id']):s for s in source['stones']}
fig,axes=plt.subplots(2,3,figsize=(15,11));examples=[]
for ax,(label,row) in zip(axes.ravel(),selected):
 r=json.loads((OUT/'results'/f'{row.id}.json').read_text());case=lookup[row['case']];cand=case['candidate']
 for sid,pose in r['final_poses'].items():
  x,y,t=pose;c,s=math.cos(t),math.sin(t);g=affine_transform(Polygon(stones[sid]['outline']),(c,-s,s,c,x,y))
  ax.add_patch(Patch(list(g.exterior.coords),facecolor='#cb8d60' if sid==cand['id'] else '#ddd3c0',edgecolor='#554a39',linewidth=.5))
  q=g.representative_point();ax.text(q.x,q.y,sid,fontsize=6,ha='center',va='center',clip_on=True)
 sid=cand['id'];x,y,t=cand['pose'];c,s=math.cos(t),math.sin(t);g=affine_transform(Polygon(stones[sid]['outline']),(c,-s,s,c,x,y));xx,yy=g.exterior.xy;ax.plot(xx,yy,'--',color='#377bab',linewidth=.8)
 if case['family']=='replacement':
  old=stones[case['old_id']];x,y,t=old['pose'];c,s=math.cos(t),math.sin(t)
  oldshape=affine_transform(Polygon(old['outline']),(c,-s,s,c,x,y));ox,oy=oldshape.exterior.xy
  ax.plot(ox,oy,':',color='#39864c',linewidth=1)
 patio=Polygon(source['patio']);xx,yy=patio.exterior.xy;ax.plot(xx,yy,'k-',linewidth=.8)
 cx,cy=case['center'];ax.set_xlim(cx-2,cx+2);ax.set_ylim(cy-2,cy+2);ax.set_aspect('equal')
 ax.set_title(f"{label}\nArea Δ {row.delta_covered_area:+.2f} ft²; void Δ {row.delta_void:+.2f} ft²\nFar void Δ {row.delta_far_void:+.2f} ft²; settled={row.success}",fontsize=9)
 examples.append({'label':label,'task_id':row.id,'case':row['case'],'seed':int(row.seed),'protocol':row.protocol,
                  'area_gain_sqft':row.delta_covered_area,'void_change_sqft':row.delta_void,'far_void_change_sqft':row.delta_far_void,
                  'max_neighbor_shift_inches':row.max_neighbor_shift_inches})
fig.suptitle('Selected counterexamples and successes — original outlines\nOrange: candidate; blue dashed: initial candidate; green dotted: removed piece; local views')
fig.subplots_adjust(left=.04,right=.99,bottom=.04,top=.89,wspace=.18,hspace=.4);fig.savefig(DEST/'examples.png',dpi=160);plt.close(fig)
(DEST/'selected_examples.json').write_text(json.dumps(examples,indent=2))
print('Rendered findings and selected examples.')
