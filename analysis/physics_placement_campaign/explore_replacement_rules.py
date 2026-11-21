"""Exploratory follow-up: compare against the original piece and family-specific ranks.
These feature/policy choices are post-hoc hypotheses, not a new untouched test.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_completed import fit_predict

OUT=Path(__file__).with_name('campaign_v1')/'completed_analysis'
d=pd.read_csv(OUT/'case_features_outcomes.csv');old=json.loads((OUT/'old_piece_features.json').read_text())
extras=[]
for _,r in d.iterrows():
 f=old.get(r['target']);e={}
 if f:
  p=f['scale_profiles']['1.0']
  e={'old_mean_excess':p['mean_excess_inches'],'old_near_fraction':p['near_fraction'],
     'relative_near_fraction':r['near_fraction']-p['near_fraction'],
     'relative_excess':r['mean_excess_inches']-p['mean_excess_inches'],
     'old_near_run_fraction':p['longest_near_run_ft']/f['perimeter_ft'],
     'area_ratio_old':r['area_sqft']/f['area_sqft'],
     'relative_min_width':r['min_width']/min(f['widths_15deg_ft']),
     'relative_max_width':r['max_width']/max(f['widths_15deg_ft'])}
 extras.append(e)
extra=pd.DataFrame(extras,index=d.index);d=pd.concat([d,extra],axis=1)
base=['area_sqft','potential_area_gain','intrusion_fraction','board_stones','wall_distance_inches','cavity_radius']
edge=['mean_excess_inches','near_fraction','matching_run_fraction','tangent_alignment','neighbor_count',
      'width_to_peak','opening_free_area','min_width','max_width','convexity']
scale=['scaled_intrusion_fraction','intrusion_growth','scaled_excess_inches','scale_excess_change','scale_near_change']
metrics=[]
for family in ['insertion','replacement']:
 subset=d[d.family==family].copy()
 for name,columns in [('separate_basic',base),('separate_edges_old',base+edge+(list(extra.columns) if family=='replacement' else [])),
                      ('separate_scales_old',base+edge+scale+(list(extra.columns) if family=='replacement' else []))]:
  prediction=np.zeros(len(subset))
  for fold in range(5):
   test=subset.fold==fold
   if test.any():prediction[test]=fit_predict(subset[~test],subset[test],columns,'useful_rate')
  subset[name]=prediction
  for target,g in subset.groupby('target'):
   selected=g.nlargest(3,name)
   metrics.append({'family':family,'target':target,'ranker':name,'useful_rate':selected.useful_rate.mean(),'success_rate':selected.success_rate.mean()})
for target,g in d.groupby('target'):
 family=g.family.iloc[0]
 for name in ['hybrid','hybrid_intrusion50','hybrid_intrusion25','hybrid_feasibility90']:
  if family=='insertion':selected=g.assign(score=g.area_sqft-8*g.intrusion_sqft).nlargest(3,'score')
  elif name=='hybrid':selected=g.nlargest(3,'potential_area_gain')
  else:
   if name=='hybrid_feasibility90':eligible=g[g.success_rate_area_intrusion_prediction>=.9]
   else:eligible=g[g.intrusion_fraction<(.5 if name=='hybrid_intrusion50' else .25)]
   selected=eligible.nlargest(3,'potential_area_gain')
   if len(selected)<3:
    rest=g.loc[~g.index.isin(selected.index)].nlargest(3-len(selected),'success_rate_area_intrusion_prediction')
    selected=pd.concat([selected,rest])
  metrics.append({'family':family,'target':target,'ranker':name,'useful_rate':selected.useful_rate.mean(),'success_rate':selected.success_rate.mean()})
r=pd.DataFrame(metrics);r.to_csv(OUT/'exploratory_family_rankers.csv',index=False)
print(r.groupby(['family','ranker'])[['useful_rate','success_rate']].mean().to_string())
print('\nCombined',r.groupby('ranker')[['useful_rate','success_rate']].mean().to_string())
print('\nTargets with any consistently useful proposal',d.assign(consistent=d.useful_rate==1).groupby('family').apply(lambda a:a.groupby('target').consistent.any().sum(),include_groups=False))
