"""Isolated area-packing bootstrap; no maximum-gap acceptance authority."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path


def original_pose(item, transform, scale=1000):
    theta = transform['angle_rad']
    cosine, sine = math.cos(theta), math.sin(theta)
    x = item['x']/scale-transform['offset_ft'][0]
    y = item['y']/scale-transform['offset_ft'][1]
    return [cosine*x+sine*y, -sine*x+cosine*y, math.radians(item['angle'])-theta]


def run(input_file, output, seconds):
    import packingsolver.irregular as ps
    data = json.loads(Path(input_file).read_text())
    out = Path(output)
    out.mkdir(exist_ok=False)
    scale = 1000
    def shape(coords):
        return ps.build_shape([(x*scale, y*scale) for x, y in coords])
    builder = ps.InstanceBuilder()
    builder.set_objective(ps.Objective.Knapsack)
    builder.add_bin_type(shape(data['patio']), copies=1,
                         item_bin_minimum_spacing=data['clearance_ft']*scale)
    builder.set_item_item_minimum_spacing(data['clearance_ft']*scale)
    for item in data['items']:
        builder.add_item_type(shape(item['outline']), copies=1, profit=item['area_sqft'],
                              allowed_rotations=[(0, 360, False)])
    payload = dict(patterns=[], version=importlib.metadata.version('packingsolver'),
                   acceptance_authority=False, objective='original stone area', native_workers=1)
    seen = set()
    def retain(result):
        result.solution.write(str(out/'certificate.json'))
        cert = json.loads((out/'certificate.json').read_text())
        if cert is None or not cert.get('bins'):
            return
        if len(cert['bins']) != 1:
            raise ValueError('Unexpected bins')
        records = []
        transform = data.get('bin_transform', dict(angle_rad=0, offset_ft=[0, 0]))
        for item in cert['bins'][0]['items']:
            if item.get('mirror') or not 0 <= int(item['id']) < len(data['items']):
                raise ValueError('Invalid native item')
            records.append(dict(stone_id=data['items'][int(item['id'])]['stone_id'],
                                pose=original_pose(item, transform, scale)))
        key = tuple(sorted((r['stone_id'], *r['pose']) for r in records))
        if key in seen:
            return
        seen.add(key)
        payload['patterns'].append(dict(records=records))
        # Later native incumbents improve area; preserve early diversity too.
        if len(payload['patterns']) > 16:
            del payload['patterns'][1]
        tmp = out/'patterns.tmp'
        tmp.write_text(json.dumps(payload, allow_nan=False))
        tmp.replace(out/'patterns.json')
        print(json.dumps(dict(native_pattern=len(seen), placed=len(records))), flush=True)
    params = ps.OptimizeParameters()
    params.time_limit = seconds
    params.verbosity_level = 1
    params.use_local_search_direct_trigonometry = True
    params.optimization_mode = ps.OptimizationMode.NotAnytimeSequential
    params.new_solution_callback = retain
    result = ps.optimize(builder.build(), params)
    retain(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, required=True)
    args = parser.parse_args()
    run(args.input, args.output, args.seconds)
