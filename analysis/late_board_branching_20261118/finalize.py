"""Choose retained champions and preserve the nondominated alternatives."""
from experiment import *

def axes(m):return [-m['area_percent'],m['gaps']['largest_interior_inches'],m['gaps']['largest_edge_inches'],m['quality']['broad_void_sqft'],-m['quality']['wall_joint_inches']]
def dominates(a,b):
 aa,bb=axes(a['metrics']),axes(b['metrics'])
 return all(x<=y+1e-7 for x,y in zip(aa,bb)) and any(x<y-1e-7 for x,y in zip(aa,bb))

def main():
 results=[]
 for n in CASES:
  folder=HERE/f'run_{n:06d}';source=read(folder/'source.json');root=read(folder/'root_metrics.json');entries=[dict(id='saved_root',metrics=root,frame=state(source))]
  for path in sorted(folder.glob('*.json.gz')):
   r=read(path)
   if isinstance(r,dict) and r.get('accepted') and 'id' in r:entries.append(dict(id=r['id'],metrics=r['end_metrics'],frame=r['end']))
  best=max(entries,key=lambda e:e['metrics']['area_percent'])
  if best['metrics']['area_percent']<=root['area_percent']+1e-7:best=entries[0]
  gap=min(entries,key=lambda e:(e['metrics']['gaps']['largest_interior_inches'],e['metrics']['gaps']['largest_edge_inches']))
  pareto=[e for e in entries if not any(dominates(o,e) for o in entries)]
  unique={canonical(e['frame']):e for e in pareto};pareto=list(unique.values())
  write(folder/'best_coverage.json',restore(source,best['frame']));write(folder/'best_interior_gap.json',restore(source,gap['frame']));write(folder/'pareto_archive.json.gz',pareto)
  results.append(dict(run=n,root=root,best_coverage=best,best_interior_gap=gap,pareto_ids=[e['id'] for e in pareto]))
 write(HERE/'champions.json',results)
 print([(r['run'],r['best_coverage']['id'],r['best_coverage']['metrics']['area_percent']) for r in results])
if __name__=='__main__':main()
