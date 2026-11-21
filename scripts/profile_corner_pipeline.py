"""Bounded timing of existing patio pool diagnostics; no placement search."""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import time

import numpy as np
import shapely
from shapely.ops import unary_union
import corner_growth as c
from corner_acceleration import coverage_diagnostics
from flagstone.phase1 import transformed


def profile(pool, output, probes=128):
    data=json.loads((Path(pool)/'candidate_pool.json').read_text())
    _,contract,layout=c.load_trial(data['trial'])
    started=time.perf_counter()
    polygons=np.array([transformed(layout.shapes[r['stone_id']],r['pose']) for r in data['candidates']],dtype=object)
    fixed=[transformed(layout.shapes[r['stone_id']],r['pose']) for r in data['parent_history']]
    obstacles=unary_union([layout.patio.boundary,*fixed])
    pts=c.coverage_samples(layout.patio,c.region(contract,data['target']).intersection(layout.patio),.16)
    pts=pts[shapely.distance(pts,obstacles)>=1/12][:probes]
    preparation=time.perf_counter()-started
    before=time.perf_counter()
    reference=np.array([bool(np.any(shapely.distance(polygons,p)<1/12)) for p in pts])
    reference_seconds=time.perf_counter()-before
    before=time.perf_counter();tree=shapely.STRtree(polygons)
    pairs,distances=tree.query_nearest(pts,max_distance=1/12,return_distance=True,all_matches=False)
    covered=np.zeros(len(pts),bool)
    if pairs.shape[1]:
        covered[pairs[0,distances<1/12]]=True
    indexed_seconds=time.perf_counter()-before
    if not np.array_equal(reference,covered):raise AssertionError('Indexed coverage changed results')
    profiler=cProfile.Profile();start=time.perf_counter();profiler.enable()
    result=coverage_diagnostics(layout,contract,data)
    profiler.disable();total=time.perf_counter()-start
    report=io.StringIO();pstats.Stats(profiler,stream=report).strip_dirs().sort_stats('cumulative').print_stats(25)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'coverage_profile.txt').write_text(report.getvalue())
    profiler.dump_stats(str(output/'coverage.prof'))
    summary=dict(pool=str(Path(pool).resolve()),candidates=len(polygons),compared_probes=len(pts),
                 preparation_seconds=preparation,reference_scan_seconds=reference_seconds,
                 indexed_including_tree_seconds=indexed_seconds,identical_covered_flags=True,
                 full_indexed_diagnostics_seconds=total,zero_cover_count=len(result['zero_cover']),
                 scope='same saved-pool geometry and probes; timing ratio is not an end-to-end speedup')
    (output/'timing.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--probes',type=int,default=128)
    args=parser.parse_args();profile(args.pool,args.output,args.probes)
