"""Export a self-contained, read-only HTML replay from committed trajectory records."""
import argparse
import gzip
import json
from pathlib import Path

from shapely.geometry import Polygon


def load_run(batch, run):
    folder = batch / 'runs' / f'run_{run:06d}'
    records = [json.loads(gzip.decompress(p.read_bytes()))
               for p in sorted(folder.glob('attempt_*.json.gz'))]
    assert records and all(r['run'] == run for r in records)
    frames = []

    def add(r, stage, state, metrics, highlight=(), removed=()):
        if metrics:
            metrics = dict(area_percent=metrics['area_percent'],
                           clearance=dict(valid=metrics['clearance']['valid']),
                           gaps={k: metrics['gaps'].get(k) for k in
                                 ('largest_interior_inches', 'largest_edge_inches')})
        frames.append(dict(attempt=r['number'] if stage != 'Pre' else 0,
                           stage=stage, poses=state['poses'], metrics=metrics,
                           highlight=list(highlight), removed=list(removed),
                           accepted=r['accepted'], reason=r['reason']))

    add(records[0], 'Pre', records[0]['start'], records[0]['start_metrics'])
    previous = records[0]['start']
    for r in records:
        assert r['start'] == previous, 'Broken committed trajectory chain'
        p = r['chosen']
        if p:
            ids = [x['stone'] for x in p['placements']]
            # This is the recorded solver input, before integration.
            assert all(x['stone'] in r['proposed_start']['poses'] for x in p['placements'])
            add(r, 'Placed', r['proposed_start'], None, ids, p['remove'])
            add(r, "Physics’d", r['attempted_end'], r['attempted_end_metrics'], ids, p['remove'])
            if not r['accepted']:
                add(r, 'Rolled back', r['end'], r['end_metrics'])
        else:
            add(r, 'No placement', r['end'], r['end_metrics'])
        previous = r['end']

    best = max((f for f in frames if f['metrics'] and
                (f['accepted'] or f['stage'] in ['Pre', 'Rolled back', 'No placement'])),
               key=lambda f: f['metrics']['area_percent'])
    return dict(run=run, mode=records[0]['mode'], frames=frames,
                best=frames.index(best), attempts=len(records),
                complete=(folder / 'summary.json').exists(),
                accepted=sum(r['accepted'] for r in records))


def build(batch, run, output, special_runs=False, all_runs=False):
    source = json.loads((batch / 'source.json').read_text())
    numbers = {run}
    if all_runs:
        for folder in (batch / 'runs').glob('run_*'):
            if next(folder.glob('attempt_*.json.gz'), None) is not None:
                numbers.add(int(folder.name.split('_')[1]))
    if special_runs:
        for path in (batch / 'runs').glob('run_*/summary.json'):
            summary = json.loads(path.read_text())
            if summary['mode'] in ('golden', 'silver') and summary['status'] == 'complete':
                numbers.add(summary['number'])
    runs = [load_run(batch, number) for number in sorted(numbers)]
    used = {sid for r in runs for f in r['frames'] for sid in f['poses']}
    stones = {}
    for s in source['stones']:
        if str(s['id']) not in used:
            continue
        center = Polygon(s['outline']).representative_point()
        stones[str(s['id'])] = dict(outline=s['outline'], label=[center.x, center.y])
    data = dict(initialRun=run, runs=runs, patio=source['patio'], stones=stones,
                batch=str(batch))
    template = Path(__file__).with_name('trajectory_viewer.html').read_text()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace('__TRAJECTORY_DATA__', json.dumps(data, separators=(',', ':')).replace('</', '<\\/')))
    print(f'{output}: {len(runs)} runs, {sum(len(r["frames"]) for r in runs)} frames, {len(stones)} shared original outlines')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--special-runs', action='store_true',
                        help='Include all completed Golden and Silver runs in the batch')
    parser.add_argument('--all-runs', action='store_true',
                        help='Include every run with committed attempts, including partial runs')
    args = parser.parse_args()
    build(args.batch, args.run, args.output, args.special_runs, args.all_runs)
