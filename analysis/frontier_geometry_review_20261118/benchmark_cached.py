"""Cache-only ablation; requires completed paired benchmark."""
from benchmark import REVIEW, ROOT, DEFAULT_OUT, read, write, sha, restore, np, time, importlib
spec=importlib.util.spec_from_file_location('cached_planner',REVIEW/'planner_cached.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
results=read(REVIEW/'benchmark_results.json')
for row in results['cases']:
 j=row['job'];target=row['target'];node=read(DEFAULT_OUT/'nodes'/f'{j["parent"]}.json.gz');doc=restore(read(DEFAULT_OUT/'source.json'),node['frame']);cfg=read(DEFAULT_OUT/'manifest.json')['config']
 old_replace=module.replacement_targets;old_insert=module.insertion_targets
 module.replacement_targets=lambda *args:[t for t in old_replace(*args) if t['id']==target]
 module.insertion_targets=lambda *args:[t for t in old_insert(*args) if t['id']==target]
 start=time.monotonic()
 try:pool,diag=module.generate(doc,node['metrics'],cfg,j['operator'],j['phase'],np.random.default_rng(j['seed']),lambda:False)
 finally:module.replacement_targets=old_replace;module.insertion_targets=old_insert
 original=read(REVIEW/f'job_{j["number"]}_baseline_pool.json.gz')['candidates']
 row['cache_only']=dict(seconds=time.monotonic()-start,candidates=len(pool),exact_same_candidates=pool==original)
 assert pool==original
 print(j['number'],row['cache_only'],flush=True)
 write(REVIEW/'cache_results.json',results)
