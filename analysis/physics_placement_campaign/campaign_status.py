"""Read-only progress monitor, requiring only Python's standard library."""
import argparse,json,statistics
from collections import Counter
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).with_name('campaign_v1'));args=p.parse_args()
if not (args.out/'manifest.json').exists():p.exit(1,'Campaign preparation has not finished.\n')
m=json.loads((args.out/'manifest.json').read_text());validation=set(m['validation_task_ids']);rows=[]
for path in (args.out/'results').glob('*.json'):
    r=json.loads(path.read_text())
    if r.get('manifest_hash')!=m['manifest_hash']:raise ValueError('Checkpoint manifest mismatch: '+str(path))
    rows.append(r)
ids={r['task']['id'] for r in rows};runs=[json.loads(x.read_text()) for x in (args.out/'runs').glob('*.json')]
print(json.dumps({'planned':len(m['tasks']),'completed':len(rows),'remaining':len(m['tasks'])-len(rows),
 'validation_completed':len(validation&ids),'validation_planned':len(validation),
 'valid_endpoints':sum(r['after']['valid'] for r in rows),'converged':sum(r['converged'] for r in rows),
 'completed_runs':len(runs),'recorded_execution_seconds':sum(r['elapsed_seconds'] for r in runs),
 'median_trial_wall_seconds':statistics.median(r['wall_seconds'] for r in rows) if rows else None},indent=2))
