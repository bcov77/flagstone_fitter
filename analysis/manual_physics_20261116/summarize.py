"""Aggregate completed trials, matched deltas and a labeled settled comparison."""
import csv,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
OUT=Path(__file__).resolve().parent
screen=json.loads((OUT/'screening.json').read_text())
split=[OUT/f'trials_{target}_{seed}.json' for target in ['29','119','35'] for seed in [17,41]]
paths=split if all(p.exists() for p in split) else [OUT/f'trials_{target}.json' for target in ['29','119','35']]
parts=[json.loads(p.read_text()) for p in paths]
trials=[r for p in parts for r in p['trials']]
expected=sum((len(s['selected'])+1)*4 for s in screen['slots'])
assert len(trials)==expected,(len(trials),expected)
assert len({(r['target'],r['candidate'],r['seed'],r['protocol']) for r in trials})==expected
method=parts[0]['method']
(OUT/'trials.json').write_text(json.dumps({'method':method,'trials':trials},indent=2))
lookup={(r['target'],r['candidate'],r['seed'],r['protocol']):r for r in trials}
fields=['target','candidate','selected_by','seed','protocol','passes_clearance','clearance_pair_count','clearance_wall_count','pair_deficit_sum_inches','pair_overlap_sqft','nominal_coverage','broad_void_sqft','far_broad_void_sqft','local_broad_void_sqft','local_void_delta_vs_human_sqft','far_void_delta_vs_human_sqft','far_void_delta_vs_empty_sqft','mean_translation_inches','max_translation_inches']
rows=[]
for r in trials:
 row={k:r[k] for k in ['target','candidate','seed','protocol']};row['selected_by']=' / '.join(r['selected_by']);row.update(r['after'])
 row['local_broad_void_sqft']=r['after']['broad_void_sqft']-r['after']['far_broad_void_sqft']
 human=lookup[(r['target'],r['target'],r['seed'],r['protocol'])]['after']
 row['local_void_delta_vs_human_sqft']=row['local_broad_void_sqft']-(human['broad_void_sqft']-human['far_broad_void_sqft'])
 for name,candidate in [('human',r['target']),('empty',None)]:
  baseline=lookup[(r['target'],candidate,r['seed'],r['protocol'])]
  row[f'far_void_delta_vs_{name}_sqft']=r['after']['far_broad_void_sqft']-baseline['after']['far_broad_void_sqft']
 rows.append(row)
with (OUT/'outcomes.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
groups=defaultdict(list)
for r in rows:groups[(r['target'],r['candidate'],r['protocol'])].append(r)
summary=[]
for (target,candidate,protocol),rs in groups.items():
 row=dict(target=target,candidate=candidate,protocol=protocol,selected_by=rs[0]['selected_by'],valid_trials=sum(r['passes_clearance'] for r in rs))
 for k in fields[6:]:row[k]=float(np.mean([r[k] for r in rs]))
 summary.append(row)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
source=json.loads(Path(screen['source']).read_text());stones={str(s['id']):s for s in source['stones']}
fig,axes=plt.subplots(3,4,figsize=(15,12))
for axs,slot in zip(axes,screen['slots']):
 target=slot['target'];s=stones[target];x,y,t=s['pose'];c,sn=math.cos(t),math.sin(t)
 ref=affine_transform(Polygon(s['outline']),(c,-sn,sn,c,x,y));cx,cy=ref.centroid.coords[0]
 for ax,case in zip(axs,list(slot['selected'])+[None]*(4-len(slot['selected']))):
  if case is None:ax.axis('off');continue
  r=lookup[(target,case['id'],17,'direct')]
  for sid,pose in r['poses'].items():
   x,y,t=pose;c,sn=math.cos(t),math.sin(t)
   g=affine_transform(Polygon(stones[sid]['outline']),(c,-sn,sn,c,x,y))
   ax.add_patch(Patch(list(g.exterior.coords),facecolor='#cf9468' if sid==case['id'] else '#ddd3c0',edgecolor='#554a39',linewidth=.6))
   pt=g.representative_point();ax.text(pt.x,pt.y,sid,fontsize=7,ha='center',va='center',clip_on=True)
  patio=Polygon(source['patio']);px,py=patio.exterior.xy;ax.plot(px,py,'k-',linewidth=1)
  ax.set_xlim(cx-1.6,cx+1.6);ax.set_ylim(cy-1.6,cy+1.6);ax.set_aspect('equal')
  kind='human' if case['id']==target else ', '.join(x.replace('_score','') for x in case['selected_by'])
  ax.set_title(f"Slot {target}: {case['id']} ({kind})\nBoard deficits: {r['after']['pair_deficit_sum_inches']:.1f} in; far void: {r['after']['far_broad_void_sqft']:.2f} ft²",fontsize=9)
fig.suptitle('Full-size replacements after 16 seconds: direct protocol, seed 17\nOriginal outlines; all neighbors movable; none certified clearance-valid')
fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/'settled_comparison.png',dpi=160);plt.close(fig)
print('Completed trials',len(trials),'clearance valid',sum(r['after']['passes_clearance'] for r in trials))
for r in summary:print(r['target'],r['candidate'],r['protocol'],'deficit',round(r['pair_deficit_sum_inches'],2),'far delta',round(r['far_void_delta_vs_human_sqft'],3),'void',round(r['broad_void_sqft'],3))
