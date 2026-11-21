"""PackingSolver proposal worker: standard library plus the solver's own environment."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import time


def run(input_file, output, seconds=45, groups=6):
    import packingsolver.irregular as ps
    data = json.loads(Path(input_file).read_text())
    out = Path(output); out.mkdir(exist_ok=False)
    scale = 1000
    def shape(points): return ps.build_shape([(x*scale,y*scale) for x,y in points])
    builder = ps.InstanceBuilder(); builder.set_objective(ps.Objective.Knapsack)
    spacing = data['minimum_clearance_ft']*scale
    bid = builder.add_bin_type(shape(data['patio']),copies=1,item_bin_minimum_spacing=spacing)
    builder.set_item_item_minimum_spacing(spacing)
    for polygon in data['fixed']:
        did = builder.add_defect(bid,0,ps.ShapeWithHoles(shape(polygon)))
        builder.set_item_defect_minimum_spacing(bid,did,spacing)
    for exclusion in data.get('proposal_exclusions',[]):
        did=builder.add_defect(bid,0,ps.ShapeWithHoles(shape(exclusion['outline']),
            [shape(hole) for hole in exclusion['holes']]))
        builder.set_item_defect_minimum_spacing(bid,did,0)
    count = data['target']-len(data['parent_history'])
    resource = builder.add_bin_type_resource(bid,count)
    for item in data['items']:
        iid = builder.add_item_type(shape(item['outline']),copies=1,profit=1,allowed_rotations=[(0,360,False)])
        builder.add_resource_consumption(bid,resource,iid,[1])
    payload = dict(parent_history=data['parent_history'], target=data['target'], patterns=[],
                   acceptance_authority=False, version=importlib.metadata.version('packingsolver'),
                   engine='direct_trigonometry', native_workers=1)
    payload['proposal_focus_padding_ft']=data.get('proposal_focus_padding_ft')
    def save():
        path = out/'patterns.json'; temp = out/'patterns.tmp'
        temp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n'); temp.replace(path)
    save(); seen = set()
    def retain(result):
        certificate=out/'certificate.json'
        result.solution.write(str(certificate))
        cert = json.loads(certificate.read_text())
        if cert is None: return
        bins = cert.get('bins',[])
        if not bins: return
        if len(bins) != 1: raise ValueError('Unexpected native bin count')
        items = bins[0]['items']
        if len(items) > count: raise ValueError('Native resource count exceeded')
        if len(items) != count: return
        records = []; ids = set()
        for item in items:
            index = int(item['id'])
            if not 0 <= index < len(data['items']) or item.get('mirror'):
                raise ValueError('Native identity/reflection violation')
            sid = data['items'][index]['stone_id']
            if sid in ids: raise ValueError('Duplicate native stone')
            ids.add(sid)
            pose = [item['x']/scale,item['y']/scale,math.radians(item['angle'])]
            if not all(math.isfinite(v) for v in pose): raise ValueError('Nonfinite native pose')
            records.append(dict(stone_id=sid,pose=pose))
        key = tuple(sorted((r['stone_id'],*r['pose']) for r in records))
        if key not in seen and len(payload['patterns']) < groups:
            seen.add(key); payload['patterns'].append(dict(records=records)); save()
    parameters = ps.OptimizeParameters(); parameters.time_limit=seconds; parameters.verbosity_level=1
    parameters.use_local_search_direct_trigonometry=True
    # This mode makes the native separator single-worker: outer jobs own the CPU budget.
    parameters.optimization_mode=ps.OptimizationMode.NotAnytimeSequential
    parameters.new_solution_callback=retain
    started=time.monotonic()
    result=ps.optimize(builder.build(),parameters); retain(result)
    payload['final_item_count']=result.solution.number_of_items()
    payload['elapsed_seconds']=time.monotonic()-started; save()
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--seconds',type=float,default=45); parser.add_argument('--groups',type=int,default=6)
    args=parser.parse_args(); run(args.input,args.output,args.seconds,args.groups)
