"""Export a useful one-step research result; never promote it as a final solution."""
import json
from pathlib import Path
import pandas as pd
OUT=Path(__file__).with_name('campaign_v1');DEST=OUT/'completed_analysis'
d=pd.read_csv(DEST/'trial_outcomes.csv');row=d[d.useful].sort_values(['covered_area','delta_void'],ascending=[False,True]).iloc[0]
r=json.loads((OUT/'results'/f'{row.id}.json').read_text());m=json.loads((OUT/'manifest.json').read_text());doc=json.loads((OUT/'source.json').read_text())
settings=m['config']['settings'].copy();settings.update(gap_inches=.5,piece_hardness=48,wall_hardness=89,rotation_magnitude=0,translation_magnitude=0,rotation_force=0,translation_force=0,angular_stiffness=0,translation_stiffness=0)
metadata={}
for stone in doc['stones']:
 sid=str(stone['id']);stone['active']=sid in r['final_poses']
 if stone['active']:stone['pose']=r['final_poses'][sid]
 metadata[sid]={'locked':False,'anchor':r['initial_poses'].get(sid,stone['pose'])}
doc['physics']={'version':1,'settings':settings,'pieces':metadata,'notice':'Research result: half-inch clearance verified; final local-maximum gap rule not checked. Saved settings are quiet finishing settings.'}
doc['research']={'task':row.id,'case':row['case'],'manifest_hash':m['manifest_hash'],'final_gap_rule_checked':False,'promoted_final_solution':False}
(DEST/'best_one_step_research.json').write_text(json.dumps(doc,separators=(',',':')))
patio=__import__('shapely').Polygon(doc['patio']);summary={'task':row.id,'case':row['case'],'coverage':row.covered_area/patio.area,'covered_area_sqft':row.covered_area,'area_gain_sqft':row.delta_covered_area,'broad_void_change_sqft':row.delta_void,'far_void_change_sqft':row.delta_far_void,'active_stones':len(r['final_poses']),'settings':'quiet finishing settings; restoring gains and jitter zero','final_gap_rule_checked':False}
(DEST/'best_one_step_research_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
