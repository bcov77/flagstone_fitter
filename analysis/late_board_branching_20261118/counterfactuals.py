"""Diagnostic direct-action controls using poses discovered on second-step branches.
These are matched retrospective tests, not blind original-board searches.
"""
from experiment import HERE,CASES,SEEDS,config,read,write,restore,eligible,trial,shapely

def main():
 cfg=config();out=[]
 for n in CASES:
  folder=HERE/f'run_{n:06d}';doc=read(folder/'source.json')
  for path in sorted(folder.glob('follow_*.json.gz')):
   r=read(path);p=r.get('action')
   if not r.get('accepted') or not p:continue
   if not eligible(doc,p,cfg):continue
   roi=shapely.from_wkt(r['roi_wkt'])
   result=trial(folder,'direct_'+r['id'],doc,p,cfg,r['seed'],roi=roi)
   parent=read(folder/(r['parent']+'.json.gz'))
   out.append(dict(run=n,followup=r['id'],direct=result['id'],direct_accepted=result['accepted'],followup_accepted=r['accepted'],root_coverage=read(folder/'root_metrics.json')['area_percent'],parent_coverage=parent['end_metrics']['area_percent'],branch_coverage=r['end_metrics']['area_percent'],direct_coverage=result['end_metrics']['area_percent'],branch_local_void=r['local_end']['broad_void_sqft'],direct_local_void=result['local_end']['broad_void_sqft'],scope='Same action hypothesis/pose/seed on original root; discovered by branch, not blind; retained root move has its own area contribution'))
 write(HERE/'direct_action_comparisons.json',out)
 print(f'{len(out)} matched direct-action comparisons completed',flush=True)
if __name__=='__main__':main()
