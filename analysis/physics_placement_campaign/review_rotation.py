"""Post-hoc user criterion: >45 degrees away from proposed orientation.
Saved phase endpoints only bound the maximum excursion from below. No new physics.
"""
import json,math,csv
from pathlib import Path
OUT=Path(__file__).with_name('campaign_v1');DEST=OUT/'completed_analysis'
m=json.loads((OUT/'manifest.json').read_text());cases={c['id']:c for c in m['cases']}
labels={r['id']:r for r in csv.DictReader((DEST/'trial_outcomes.csv').open())};rows=[]
for path in sorted((OUT/'results').glob('*.json')):
 r=json.loads(path.read_text());case=cases[r['task']['case']]
 if case['kind'] in ['control','empty_diagnostic']:continue
 sid=case['candidate']['id'];start=r['initial_poses'][sid][2]
 angles=[r['initial_poses'][sid][2]]+[s['poses'][sid][2] for s in r['stages']]+[r['final_poses'][sid][2]]
 net=abs(math.degrees(r['final_poses'][sid][2]-start))
 maximum=max(abs(math.degrees(a-start)) for a in angles)
 label=labels[r['task']['id']]
 rows.append({'id':r['task']['id'],'case':case['id'],'family':case['family'],'kind':case['kind'],
              'stone':sid,'net_rotation_degrees':net,'max_saved_excursion_degrees':maximum,
              'reject_starting_pose_over_45':maximum>45,'old_useful':label['useful']=='True',
              'success':label['success']=='True','trajectory_sampling':'phase endpoints; maximum is a lower bound'})
with (DEST/'rotation_review.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
summary={'criterion':'Reject starting pose if candidate moves >45 degrees away from proposed orientation; do not sum ordinary back-and-forth jitter.',
 'candidate_trials':len(rows),'flagged_at_saved_points':sum(r['reject_starting_pose_over_45'] for r in rows),
 'previously_useful_but_flagged':sum(r['old_useful'] and r['reject_starting_pose_over_45'] for r in rows),
 'limitation':'Intermediate angular maxima were not recorded. Unflagged trials are not certified to have stayed within 45 degrees.',
 'human_review':[{'stone':'135','cases':['board_00_insertion_2_proposal_09','board_00_insertion_2_proposal_08'],
                 'judgment':'User considers both starting poses unsuitable, regardless of aggregate void score.'}]}
(DEST/'rotation_review.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
