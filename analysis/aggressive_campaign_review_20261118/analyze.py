"""Read-only analysis of the r2 campaign (which includes the original batch)."""
import collections,csv,gzip,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BATCH=ROOT/'examples/aggressive_trajectories_20261118_r2'
def avg(xs):return statistics.mean(xs) if xs else None
def describe(xs):
 return dict(n=len(xs),mean=avg(xs),median=statistics.median(xs) if xs else None,min=min(xs) if xs else None,max=max(xs) if xs else None)
def gap(m,k):return m.get('gaps',{}).get('largest_'+k+'_inches')
rows=[];runs=[];chains={}
for policy in ['aggressive','baseline']:
 for folder in sorted((BATCH/policy/'runs').glob('run_*')):
  records=[]
  for path in sorted(folder.glob('attempt_*.json.gz')):
   with gzip.open(path,'rt') as f:r=json.load(f)
   records.append({k:r[k] for k in ['run','number','mode','accepted','chosen','start_metrics','end_metrics']})
   p=r['chosen'];a=r['start_metrics'];b=r['end_metrics']
   row=dict(policy=policy,run=r['run'],attempt=r['number'],mode=r['mode'],accepted=r['accepted'],reason=r['reason'],family=p.get('family','baseline') if p else 'no_proposal',protocol=p.get('physics_protocol','held_release') if p else '',action=('insertion' if not p['remove'] else 'adjustment' if set(p['remove'])==set(x['stone'] for x in p['placements']) else 'replacement') if p else 'none',remove=','.join(map(str,p['remove'])) if p else '',placed=','.join(str(x['stone']) for x in p['placements']) if p else '',start=a['area_percent'],end=b['area_percent'],gain=b['area_percent']-a['area_percent'],interior_change=gap(b,'interior')-gap(a,'interior'),edge_change=gap(b,'edge')-gap(a,'edge'),band='>=80' if a['area_percent']>=80 else '78–80' if a['area_percent']>=78 else '<78',compute_revision=r.get('compute_revision','original'),pool_size=len(r.get('candidates',[])),probe_coverage=p.get('probe',{}).get('coverage') if p else None,probe_intrusion=p.get('probe',{}).get('intrusion_fraction') if p else None)
   rows.append(row)
  if not records:continue
  chains[(policy,records[0]['run'])]=records
  last=records[-1];sm=folder/'summary.json'
  best=max([records[0]['start_metrics']['area_percent']]+[r['end_metrics']['area_percent'] for r in records])
  runs.append(dict(policy=policy,run=last['run'],mode=last['mode'],complete=sm.exists(),attempts=len(records),final=last['end_metrics']['area_percent'],best=best,interior=gap(last['end_metrics'],'interior'),edge=gap(last['end_metrics'],'edge'),layout_pass=last['end_metrics'].get('layout_pass'),drawdown=best-last['end_metrics']['area_percent']))
def group_stats(key,subset):
 groups=collections.defaultdict(list)
 for r in subset:groups[tuple(r[k] for k in key)].append(r)
 return [dict(zip(key,k),n=len(v),accepted=sum(r['accepted'] for r in v),positive=sum(r['gain']>1e-7 for r in v),gain=describe([r['gain'] for r in v]),interior_change=describe([r['interior_change'] for r in v if r['accepted']]),edge_change=describe([r['edge_change'] for r in v if r['accepted']]),reasons=dict(collections.Counter(r['reason'] for r in v))) for k,v in sorted(groups.items())]
summary={}
for policy in ['aggressive','baseline']:
 rr=[r for r in runs if r['policy']==policy];done=[r for r in rr if r['complete']]
 summary[policy]=dict(runs=len(rr),complete=len(done),attempts=sum(r['attempts'] for r in rr),completed_final=describe([r['final'] for r in done]),completed_best=describe([r['best'] for r in done]),completed_ge80=sum(r['final']>=80 for r in done),completed_ge81=sum(r['final']>=81 for r in done),completed_ge82=sum(r['final']>=82 for r in done),layout_pass=sum(bool(r['layout_pass']) for r in done),top=sorted(rr,key=lambda r:-r['best'])[:10],largest_drawdown=sorted(done,key=lambda r:-r['drawdown'])[:5])
summary['by_family_action']=group_stats(['policy','family','action'],rows)
summary['by_band_family']=group_stats(['policy','band','family'],rows)
summary['by_band_action']=group_stats(['policy','band','action'],rows)
summary['by_mode']=[dict(policy=p,mode=m,final=describe([r['final'] for r in runs if r['policy']==p and r['mode']==m and r['complete']])) for p in ['aggressive','baseline'] for m in ['normal','golden','silver']]
summary['top_gains']=sorted([r for r in rows if r['policy']=='aggressive'],key=lambda r:-r['gain'])[:20]
summary['late_gains']=sorted([r for r in rows if r['policy']=='aggressive' and r['start']>=79],key=lambda r:-r['gain'])[:20]
# Later insertions following accepted replacements, limited to complete runs with >=5 attempts left.
follow=[]
for (policy,n),records in chains.items():
 if policy!='aggressive' or not any(r['policy']==policy and r['run']==n and r['complete'] for r in runs):continue
 for i,r in enumerate(records[:-5]):
  p=r['chosen']
  if not r['accepted'] or not p or not p['remove']:continue
  tail=records[i+1:i+6]
  follow.append(dict(run=n,attempt=r['number'],family=p['family'],immediate_gain=r['end_metrics']['area_percent']-r['start_metrics']['area_percent'],next5_gain=tail[-1]['end_metrics']['area_percent']-r['end_metrics']['area_percent'],next5_insertions=sum(bool(t['accepted'] and t['chosen'] and not t['chosen']['remove']) for t in tail),interior_change=gap(r['end_metrics'],'interior')-gap(r['start_metrics'],'interior')))
summary['replacement_followups']=dict(n=len(follow),with_next5_insertion=sum(r['next5_insertions']>0 for r in follow),next5_gain=describe([r['next5_gain'] for r in follow]),examples=sorted(follow,key=lambda r:-r['next5_gain'])[:10])
for name,data in [('summary',summary),('runs',runs),('replacement_followups',follow)]: (OUT/(name+'.json')).write_text(json.dumps(data,indent=2))
with (OUT/'moves.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:summary[k] for k in ['aggressive','baseline','by_mode']},indent=2))
