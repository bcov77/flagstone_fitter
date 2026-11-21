"""Matched CPU/Metal batches including exact validation and search remeasurement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import shapely
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flagstone.phase1 import load_json
from flagstone.quarter_batch import BatchScorer
from flagstone import quarter_search2 as search


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--sizes',type=int,nargs='+',default=[1024,4096,16384]);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    if min(a.sizes)<1 or a.repeats<1:p.error('Positive batch sizes and repeats required')
    layout=load_json(a.input,seed=22);points=search.points_for(layout);coordinates=shapely.get_coordinates(points[::max(1,len(points)//512)])
    original=layout.snapshot();center=search.measure(layout,points)['largest_center'];results={}
    for size in a.sizes:
        row={}
        for backend in ('cpu','gpu'):
            scorer=BatchScorer(layout,coordinates,backend);times=[];outcomes=[]
            for repeat in range(a.repeats):
                layout.restore(original);layout.rng.bit_generator.state=np.random.default_rng(22+repeat).bit_generator.state
                started=time.perf_counter()
                detail=search.batch_move(layout,center,points,backend=backend,scorer=scorer,batch_size=size)
                times.append(time.perf_counter()-started)
                outcomes.append(dict(accepted=detail['accepted'],energy=detail['after']['sampled_energy'],valid_shortlist=detail['valid_shortlist']))
            row[backend]=dict(median_end_to_end_seconds=float(np.median(times)),cold_seconds=times[0],samples_seconds=times,outcomes=outcomes,last_action=detail)
        row['speedup']=row['cpu']['median_end_to_end_seconds']/row['gpu']['median_end_to_end_seconds']
        row['outcomes_match']=all(x['accepted']==y['accepted'] and abs(x['energy']-y['energy'])<1e-9
                                    for x,y in zip(row['cpu']['outcomes'],row['gpu']['outcomes']))
        results[size]=row;print(json.dumps(dict(batch_size=size,speedup=row['speedup'],outcomes_match=row['outcomes_match'])),flush=True)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(layout_sha256=hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
         benchmark='proposal generation, approximate scoring, exact validation, search remeasurement',
         thread_limits={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']},results=results),indent=2)+'\n')


if __name__=='__main__':main()
