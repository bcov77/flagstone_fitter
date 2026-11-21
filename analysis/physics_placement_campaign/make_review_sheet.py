"""Balanced 24-placement human-review sheet from original provisional useful set.
No rerun of simulation. Selects one trial per target and distinct candidate IDs.
"""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
from shapely.ops import unary_union

OUT=Path(__file__).with_name('campaign_v1');ANALYSIS=OUT/'completed_analysis';DEST=ANALYSIS/'review_24'
DEST.mkdir(exist_ok=True)
if (DEST/'human_feedback.jsonl').exists():
 raise RuntimeError('This numbered review set has human labels and is frozen. Use a new review directory for a new sample.')
m=json.loads((OUT/'manifest.json').read_text());source=json.loads((OUT/'source.json').read_text())
cases={c['id']:c for c in m['cases']};contexts={c['id']:c for c in m['contexts']};stones={str(s['id']):s for s in source['stones']};patio=Polygon(source['patio'])
d=pd.read_csv(ANALYSIS/'trial_outcomes.csv');rot=pd.read_csv(ANALYSIS/'rotation_review.csv')
d=d.merge(rot[['id','stone','max_saved_excursion_degrees','net_rotation_degrees']],on='id')
# Keep original useful definition. Exclude the four examples just reviewed.
shown_cases={x['case'] for x in json.loads((ANALYSIS/'edge_fit_before_after_examples.json').read_text())}
d=d[d.useful & ~d['case'].isin(shown_cases)].reset_index(drop=True)
rows=[];lower=[];upper=[]
def constrain(mask,lo,hi):
 rows.append(np.asarray(mask,dtype=float));lower.append(lo);upper.append(hi)
for board in sorted(d.board.unique()):
 constrain(d.board==board,3,3)
 for family in ['insertion','replacement']:constrain((d.board==board)&(d.family==family),1,2)
quotas={('insertion','geometry'):4,('insertion','random'):4,('insertion','stress'):4,
        ('replacement','geometry'):3,('replacement','random'):3,('replacement','stress'):3,
        ('replacement','same_piece_reoriented'):3}
for (family,kind),n in quotas.items():constrain((d.family==family)&(d.kind==kind),n,n)
for target in d.target.unique():constrain(d.target==target,0,1)
for stone in d.stone.unique():constrain(d.stone==stone,0,1)
turn=d.max_saved_excursion_degrees
constrain(turn<10,8,8);constrain((turn>=10)&(turn<=45),10,10);constrain(turn>45,6,6)
constrain(turn>90,1,2)
constrain(d.max_neighbor_shift_inches>6,6,12)
constrain(d.max_neighbor_shift_inches>12,2,8)
constrain(d.protocol=='direct',12,12)
constrain(d.seed==17,12,12)
rng=np.random.default_rng(20261116)
result=milp(c=rng.random(len(d)),integrality=np.ones(len(d)),bounds=Bounds(0,1),
            constraints=LinearConstraint(csr_matrix(np.asarray(rows)),lower,upper),
            options={'time_limit':60,'mip_rel_gap':0.0})
if result.x is None:raise RuntimeError(result.message)
chosen=d.loc[result.x>.5].sort_values(['board','family','kind']).reset_index(drop=True)
assert len(chosen)==24 and chosen.target.nunique()==24 and chosen.stone.nunique()==24
for mask,lo,hi in zip(rows,lower,upper):
 value=float(np.dot(mask,result.x>.5));assert lo-1e-6<=value<=hi+1e-6

def poly(sid,pose):
 x,y,t=pose;c,s=math.cos(t),math.sin(t)
 return affine_transform(Polygon(stones[sid]['outline']),(c,-s,s,c,x,y))

def paint(ax,g,**kwargs):
 if g.is_empty:return
 for part in (g.geoms if hasattr(g,'geoms') else [g]):
  if part.geom_type=='Polygon':ax.add_patch(Patch(list(part.exterior.coords),**kwargs))

entries=[]
for index,row in chosen.iterrows():
 r=json.loads((OUT/'results'/f'{row.id}.json').read_text());case=cases[row['case']];sid=case['candidate']['id']
 initial={key:poly(key,pose) for key,pose in r['initial_poses'].items()}
 final={key:poly(key,pose) for key,pose in r['final_poses'].items()}
 old=poly(case['old_id'],stones[case['old_id']]['pose']) if case['family']=='replacement' else None
 candidates=[initial[sid],final[sid]]+([old] if old is not None else [])
 x0,y0,x1,y1=unary_union(candidates).bounds;span=max(x1-x0,y1-y0)+1.15;cx,cy=(x0+x1)/2,(y0+y1)/2
 bounds=(cx-span/2,cy-span/2,cx+span/2,cy+span/2)
 # Retain at most a quarter foot of empty space outside the patio where possible.
 a,b,c,e=bounds
 left=min(-.25,x0-.15);bottom=min(-.25,y0-.15)
 right=max(patio.bounds[2]+.25,x1+.15);top=max(patio.bounds[3]+.25,y1+.15)
 if a<left:c+=left-a;a=left
 if b<bottom:e+=bottom-b;b=bottom
 if c>right:a-=c-right;c=right
 if e>top:b-=e-top;e=top
 assert a<=x0 and b<=y0 and c>=x1 and e>=y1
 neighbors=[g for key,g in initial.items() if key!=sid]
 exclusion=unary_union([unary_union(neighbors).buffer(.5/12),patio.buffer(8).difference(patio.buffer(-.5/12))])
 record={'number':index+1,'id':row.id,'case':row['case'],'target':row.target,'board':row.board,
         'family':row.family,'kind':row.kind,'stone':sid,'removed_stone':case['old_id'],
         'seed':int(row.seed),'protocol':row.protocol,'max_saved_turn_degrees':row.max_saved_excursion_degrees,
         'net_turn_degrees':row.net_rotation_degrees,'max_neighbor_translation_inches':row.max_neighbor_shift_inches,
         'area_gain_sqft':0. if abs(row.delta_covered_area)<1e-8 else row.delta_covered_area,'local_void_change_sqft':row.delta_local_void,
         'remote_void_change_sqft':row.delta_far_void,'global_void_change_sqft':row.delta_void,
         'source_board_stones':contexts[case['context']]['active_count'],
         'initial_candidate_pose':r['initial_poses'][sid],'final_candidate_pose':r['final_poses'][sid],
         'original_provisional_useful':True,'human_piece_choice':None,'human_starting_pose':None,'human_settled_fit':None,'human_notes':''}
 entries.append({'record':record,'initial':initial,'final':final,'old':old,'bounds':(a,b,c,e),
                 'intrusion':initial[sid].intersection(exclusion)})


def draw_panel(ax,entry,phase,small):
 rec=entry['record'];sid=rec['stone'];shapes=entry[phase]
 for other,g in shapes.items():
  if other==sid:continue
  paint(ax,g,facecolor='#ddd7ca',edgecolor='#7f786d',linewidth=.45)
  p=g.representative_point();ax.text(p.x,p.y,other,fontsize=5 if small else 7,color='#666054',ha='center',va='center',clip_on=True)
 if entry['old'] is not None:
  x,y=entry['old'].exterior.xy;ax.plot(x,y,':',color='#3f7b9c',linewidth=.7 if small else 1.1,zorder=4)
 paint(ax,shapes[sid],facecolor='#df9e60',edgecolor='#7d4b23',linewidth=.8,zorder=3)
 if phase=='initial':paint(ax,entry['intrusion'],facecolor='#dc7772',edgecolor='#a2453f',hatch='///',linewidth=.2,alpha=.8,zorder=4)
 p=shapes[sid].representative_point();ax.text(p.x,p.y,sid,fontsize=8 if small else 11,fontweight='bold',ha='center',va='center',zorder=5)
 x,y=patio.exterior.xy;ax.plot(x,y,color='#383838',linewidth=.8)
 a,b,c,e=entry['bounds'];ax.set_xlim(a,c);ax.set_ylim(b,e);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
 ax.set_facecolor('#fcfbf8')
 for spine in ax.spines.values():spine.set_color('#aaa59d');spine.set_linewidth(.5)
 ax.set_title('PROPOSED' if phase=='initial' else 'SETTLED',fontsize=7 if small else 10,pad=3,color='#67615a')
 # Six-inch ruler, consistent physical units even though zoom varies between cards.
 if phase=='final':
  length=max(v for v in [.5,.25,1/12] if v<=(c-a)*.24);start=a+.06*(c-a);y0=b+.06*(e-b)
  ax.plot([start,start+length],[y0,y0],color='#333333',linewidth=1.8)
  ax.text(start+length/2,y0+.025*(e-b),f'{length*12:g}″',ha='center',va='bottom',fontsize=6 if small else 8,color='#333333',bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.5))


def make_sheet(subset,nrows,ncols,figsize,path,small):
 fig=plt.figure(figsize=figsize,facecolor='white')
 outer=fig.add_gridspec(nrows,ncols,left=.025,right=.985,bottom=.055,top=.91,hspace=.23,wspace=.10)
 for position,entry in enumerate(subset):
  rec=entry['record'];grid=outer[position//ncols,position%ncols].subgridspec(3,2,height_ratios=[.17,1,.20],hspace=.13,wspace=.04)
  title=fig.add_subplot(grid[0,:]);title.axis('off')
  action=f"INSERT {rec['stone']}" if rec['family']=='insertion' else f"REPLACE {rec['removed_stone']} → {rec['stone']}"
  if rec['kind']=='same_piece_reoriented':action=f"REORIENT {rec['stone']}"
  title.text(0,.68,f"{rec['number']:02d}   {action}",fontsize=11 if small else 15,fontweight='bold',va='center')
  title.text(1,.68,f"{rec['source_board_stones']} stones before",fontsize=8 if small else 11,ha='right',va='center',color='#66605a')
  for column,phase in enumerate(['initial','final']):draw_panel(fig.add_subplot(grid[1,column]),entry,phase,small)
  caption=fig.add_subplot(grid[2,:]);caption.axis('off')
  caption.text(0,.75,f"Turn {rec['max_saved_turn_degrees']:.0f}°   |   Neighbor shift up to {rec['max_neighbor_translation_inches']:.1f}″",fontsize=8 if small else 11,va='center')
  caption.text(0,.22,f"Area gain {rec['area_gain_sqft']:+.2f} ft²   |   Wide-gap change {rec['global_void_change_sqft']:+.2f} ft²",fontsize=8 if small else 11,va='center',color='#57514b')
 fig.suptitle('24 placements the original metrics called “useful”' if len(subset)==24 else f"Placement review · {subset[0]['record']['number']:02d}–{subset[-1]['record']['number']:02d}",fontsize=24 if small else 21,fontweight='bold',y=.98)
 fig.text(.5,.948,'Orange: candidate   |   red hatch: initial clearance intrusion   |   blue dotted: removed stone   |   gray: neighbors',ha='center',fontsize=12 if small else 11)
 fig.text(.5,.929,'Same zoom before/after within each card; zoom varies across cards. Large rotation is included for review, not labeled as rejection.',ha='center',fontsize=11 if small else 10)
 fig.text(.03,.029,'Turn = largest deviation recorded at saved phases; intermediate peaks may be larger. Negative wide-gap change means less broad empty space.',fontsize=10)
 fig.text(.03,.013,'These are deliberately varied examples, not an unbiased sample or an endorsement. Please judge the piece choice, starting pose, and settled fit separately.',fontsize=10)
 fig.savefig(path,dpi=180 if small else 190,facecolor='white');plt.close(fig)

make_sheet(entries,6,4,(25,31),DEST/'placements_24.png',True)
for page in range(4):make_sheet(entries[page*6:page*6+6],3,2,(15,20),DEST/f'placements_{page*6+1:02d}_{page*6+6:02d}.png',False)
records=[e['record'] for e in entries]
(DEST/'placements.json').write_text(json.dumps(records,indent=2))
summary={'selected':24,'selection':'Deliberate balanced sample from original provisional useful outcomes; random objective subject to fixed diversity constraints; not a performance-rate sample.',
 'boards':8,'placements_per_board':3,'unique_targets':24,'unique_candidate_ids':24,
 'insertions':12,'replacements_including_reorientation':12,
 'turn_bins':{'under_10_degrees':8,'10_through_45_degrees':10,'over_45_degrees':6},
 'family_kind_quotas':{f'{family}/{kind}':n for (family,kind),n in quotas.items()},
 'seeds_and_protocols':'12 of each seed and 12 of each protocol',
 'original_source_and_results_modified':False,'new_physics_trials':0,
 'rotation_interpretation':'Large turn may mean a poor starting orientation, not a poor piece choice.'}
(DEST/'selection.json').write_text(json.dumps(summary,indent=2))
text='''# Review 24 placements\n\nThese were selected from the original provisional useful set, without applying\nthe later 45-degree rejection proposal. Large turns may indicate a poor starting\norientation rather than a poor piece choice. No new simulation was run.\n\nThere are 12 insertions and 12 replacements/reorientations, three placements\nfrom each of eight contexts, with distinct targets and candidate stone IDs.\nEight have turns below 10 degrees, ten between 10 and 45, and six above 45.\nOriginal board sizes range from 20 to 27 stones. The sample deliberately spans\nproposal methods, size and movement; it is not a representative success rate.\n\n[All 24](placements_24.png) · [01–06](placements_01_06.png) ·\n[07–12](placements_07_12.png) · [13–18](placements_13_18.png) ·\n[19–24](placements_19_24.png)\n\nEach card has the exact saved proposed and settled poses with matching zoom.\nRed hatch is intrusion into required half-inch clearance, not necessarily\nphysical overlap. Blue dotted outline is the removed original stone.\nAll selected endpoints passed clearance and the original settling criterion.\nThe scale ruler shows physical length; zoom differs between cards. Turn is\nmaximum observed phase-endpoint excursion, a lower bound on the continuous\npeak. Wide-gap change is measured against the matched no-insertion or restored-\noriginal control. It is a sampled proxy, not final gap-rule certification.\n\nYou can reply by number, for example: “03: good piece, bad starting angle;\n09: wrong piece; 17: good final fit.” Piece choice, starting pose and final\nresult are intentionally separate judgments.\n\n| # | Action | Candidate | Removed | Observed turn | Piece choice | Starting pose | Settled fit | Notes |\n|---|---|---|---|---:|---|---|---|---|\n'''
for r in records:text+=f"| {r['number']:02d} | {r['family']} | {r['stone']} | {r['removed_stone'] or '—'} | {r['max_saved_turn_degrees']:.1f}° | | | | |\n"
text+='\nExact source task IDs, poses, raw metrics and blank review fields are in `placements.json`. Reproduce with `.venv/bin/python analysis/physics_placement_campaign/make_review_sheet.py`.\n'
(DEST/'README.md').write_text(text)
print(json.dumps(summary,indent=2),flush=True)
