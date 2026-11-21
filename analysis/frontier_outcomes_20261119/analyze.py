"""Read-only outcome analysis; does not import or modify pinned search code."""
import csv, gzip, hashlib, json, statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'examples/frontier_search_20261118_cached_r2'
def read(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes())
def metric(m):
    return dict(area=m['area_percent'], interior=m['gaps'].get('largest_interior_inches'), edge=m['gaps'].get('largest_edge_inches'), void=m.get('quality',{}).get('broad_void_sqft'), wall=m.get('quality',{}).get('wall_joint_inches'))
nodes={p.name.removesuffix('.json.gz'):read(p) for p in (OUT/'nodes').glob('*.json.gz')}
manifest=read(OUT/'manifest.json')
summary=read(OUT/'summary.json')
inherited=read(OUT/'lineage/parent_manifest.json')
inherited_files=set(manifest['continuation']['inherited_file_hashes'])
parent_limit=max(int(p.split('/')[1].split('_')[1]) for p in manifest['continuation']['inherited_file_hashes'] if p.startswith('jobs/'))
rows=[];snapshot=[];chosen={};frames={}
for p in sorted((OUT/'jobs').glob('job_*/result.json.gz')):
    raw=p.read_bytes();snapshot.append(dict(path=str(p.relative_to(OUT)),sha256=hashlib.sha256(raw).hexdigest()))
    r=read(p);j=read(p.parent/'job.json');n=j['number'];parent=nodes[j['parent']];c=r.get('chosen',{});chosen[n]=c
    row=dict(job=n,root=j['root'],operator=j['operator'],parent=j['parent'],depth=j['depth'],revision='inherited' if str(p.relative_to(OUT)) in inherited_files else 'cache',seconds=r['seconds'],physics=bool(r.get('physics_evaluated')),accepted=bool(r.get('accepted')),usable=bool(r.get('metrics_usable')),reason=r.get('reason',r['status']),family=c.get('family',c.get('operator','none')),repair=bool(r.get('repair_attempted')),remove=','.join(c.get('remove',[])),place=','.join(x['stone'] for x in c.get('placements',[])))
    if row['physics']:
        a=metric(parent['metrics']);b=metric(r['attempted_metrics']);e=metric(r['end_metrics'])
        for k in a:
            row['before_'+k]=a[k];row['after_'+k]=e[k];row['delta_'+k]=e[k]-a[k] if a[k] is not None and e[k] is not None else None
        row['area_gain']=row['accepted'] and row['delta_area']>.00001
        row['gap_better']=row['accepted'] and row['delta_interior']<-.1
        row['gap_worse']=row['accepted'] and row['delta_interior']>.1
        row['balanced_gain']=row['area_gain'] and row['delta_interior']<=.1 and row['delta_edge']<=.1
        frames[n]=(r['start'],r['end'])
    rows.append(row)
byjob={r['job']:r for r in rows}
def aggregate(rs):
    phys=[r for r in rs if r['physics']];acc=[r for r in phys if r['accepted']]
    return dict(jobs=len(rs),physics=len(phys),accepted=len(acc),empty=len(rs)-len(phys),worker_hours=sum(r['seconds'] for r in rs)/3600,area_gains=sum(r.get('area_gain',False) for r in rs),balanced_gains=sum(r.get('balanced_gain',False) for r in rs),gap_improvements=sum(r.get('gap_better',False) for r in rs),gap_worsened=sum(r.get('gap_worse',False) for r in rs),sum_area_pp=sum(r.get('delta_area',0) for r in rs),median_gain_pp=statistics.median([r['delta_area'] for r in acc]) if acc else None,reasons=dict(Counter(r['reason'] for r in rs)))
def groups(keys):
    d=defaultdict(list)
    for r in rows:d[tuple(r[k] for k in keys)].append(r)
    return [dict(zip(keys,k),**aggregate(v)) for k,v in sorted(d.items())]
roots=[];chains={}
for root in manifest['config']['roots']:
    ns=[n for n in nodes.values() if n['root']==root]
    original=next(n for n in ns if n['kind']=='original');best=nodes[next(r['best_node'] for r in summary['roots'] if r['root']==root)];gap=min(ns,key=lambda n:metric(n['metrics'])['interior'])
    chain=[];n=best
    while n.get('parent'):
        chain.append(byjob[n['job']]);n=nodes[n['parent']]
    chain.reverse();chains[root]=chain
    baseline=max([metric(x['metrics'])['area'] for x in ns if x['kind']!='search' or 'nodes/'+x['id']+'.json.gz' in inherited_files])
    roots.append(dict(root=root,original=metric(original['metrics']),start_of_cache_best=baseline,best=metric(best['metrics']),best_node=best['id'],gap_best=metric(gap['metrics']),gap_node=gap['id'],chain_seed=n['id'],chain_seed_metrics=metric(n['metrics']),chain=[r['job'] for r in chain],chain_gain_by_operator={op:sum(r['delta_area'] for r in chain if r['operator']==op) for op in sorted({r['operator'] for r in chain})}))
# Observed descendants establish a successful continuation, not necessity of a setup.
children=defaultdict(list)
for n in nodes.values():
    if n.get('parent'):children[n['parent']].append(n)
setup=[]
for r in rows:
    if not(r['accepted'] and r['usable']) or r.get('delta_area',1)>.00001:continue
    nid=f'n{r["job"]:08d}';stack=list(children[nid]);desc=[]
    while stack:
        n=stack.pop();desc.append(n);stack.extend(children[n['id']])
    if desc:
        b=max(desc,key=lambda n:n['metrics']['area_percent']);gain=b['metrics']['area_percent']-r['after_area']
        setup.append(dict(job=r['job'],operator=r['operator'],root=r['root'],descendants=len(desc),best_descendant=b['id'],later_gain_pp=gain))
report=dict(output=str(OUT),snapshot_count=len(snapshot),parent_limit=parent_limit,totals=aggregate(rows),operators=groups(['operator']),cache_operators=[dict(operator=op,**aggregate([r for r in rows if r['operator']==op and r['revision']=='cache'])) for op in sorted({r['operator'] for r in rows})],root_operators=groups(['root','operator']),families=groups(['operator','family']),roots=roots,champion_chains=chains,setup_continuations=sorted(setup,key=lambda s:-s['later_gain_pp']),top_balanced=sorted([r for r in rows if r.get('balanced_gain')],key=lambda r:-r['delta_area'])[:20],top_gap=sorted([r for r in rows if r.get('gap_better')],key=lambda r:r['delta_interior'])[:20],gap_tradeoffs=sorted([r for r in rows if r.get('area_gain') and r.get('gap_worse')],key=lambda r:-r['delta_interior'])[:20])
(HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(HERE/'snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
fields=list(dict.fromkeys(k for r in rows for k in r))
with (HERE/'moves.csv').open('w') as f:
    w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
print(json.dumps({k:report[k] for k in ['totals','operators','cache_operators','roots','setup_continuations']},indent=2))
