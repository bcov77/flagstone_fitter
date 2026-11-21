"""Blind original-board insertion search with finer FFT grids, same fit gates.
No pose/piece ID from successful branches is supplied. Step sizes .02/.03 ft
replace the campaign's .04 ft proposal grid; full original outlines screen fits.
"""
from experiment import *
from local_mobility import insertion_search

def run(n):
 folder=HERE/f'run_{n:06d}';doc=read(folder/'source.json');cfg=config();all_candidates={};diagnostics=[]
 path=folder/'fine_insert_portfolio.json.gz'
 if path.exists():pool=read(path)['candidates']
 else:
  start=time.monotonic()
  for step in [.02,.03]:
   options,diag=insertion_search(doc,{**cfg,'raster_step_ft':step},20261118+n)
   for p in options:
    p['research_operator']='fine_grid_insertion';p['proposal_grid_ft']=step
    all_candidates.setdefault(canonical(dict(remove=p['remove'],placements=p['placements'])),p)
   diagnostics.append(dict(grid_step_ft=step,candidates=len(options),geometry=diag))
  pool=sorted(all_candidates.values(),key=lambda p:p['rank_key'])
  write(path,dict(candidates=pool,diagnostics=diagnostics,seconds=time.monotonic()-start))
 print(f'run {n}: finer grids found {len(pool)} insertion hypotheses',flush=True)
 for ci,p in enumerate(pool[:2]):
  for si,seed in enumerate(SEEDS):trial(folder,f'fine_insert_c{ci}_s{si}',doc,p,cfg,seed,roi=roi_for(doc,p))
 return n

if __name__=='__main__':
 import multiprocessing
 m=dict(files={Path(__file__).name:sha(Path(__file__))},seeds=SEEDS,scope=__doc__);dest=HERE/'fine_insert_manifest.json'
 if dest.exists():assert read(dest)==m
 else:write(dest,m)
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as ex:
  for f in as_completed([ex.submit(run,n) for n in CASES]):print(f'Finer insertion completed run {f.result()}',flush=True)
