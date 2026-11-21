"""Verify bounded pilot artifacts and render comparison boards."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'human_trajectory_fa791034'))
from study import load,shapes,metrics
from proposals import canonical,eligible


def draw(ax,doc,highlight=()):
    for sid,g in shapes(doc).items():
        ax.fill(*g.exterior.xy,facecolor='#eba34f' if sid in highlight else '#ddd8ce',edgecolor='#77756f',lw=.4)
        if sid in highlight:ax.text(g.centroid.x,g.centroid.y,sid,ha='center',va='center',fontsize=8)
    x,y=zip(*doc['patio']);ax.plot(x,y,color='#333');ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])


def main():
    source,*_=load();original={str(s['id']):s['outline'] for s in source['stones']};cases=[];checked=0
    for path in sorted(HERE.glob('*/result.json.gz')):
        with gzip.open(path,'rt') as f:r=json.load(f)
        manifest=json.loads((path.parent/'manifest.json').read_text());cfg=manifest['config']
        for name,digest in manifest['files'].items():assert canonical((HERE.parents[1]/name).read_text())==digest,name
        nodes=r['nodes'];trials=r['trials']
        for row in [*nodes,*trials]:
            doc=row['document']
            assert len(doc['stones'])==len(original)
            assert all(s['outline']==original[str(s['id'])] for s in doc['stones'])
            if row['accepted']:
                assert row['settled'] and metrics(doc)['clearance_valid']
                assert row['motion']['max_speed_ips']<cfg['max_speed_ips']
                assert row['motion']['max_angular_speed_dps']<cfg['max_angular_speed_dps']
            if row['action']:
                parent=nodes[row['parent']]
                assert eligible(parent['document'],row['action'],cfg)
                expected={str(s['id']) for s in parent['document']['stones'] if s['active']}
                expected-=set(row['action']['remove']);expected|={p['stone'] for p in row['action']['placements']}
                assert {str(s['id']) for s in doc['stones'] if s['active']}==expected
            checked+=1
        cases.append(dict(case=path.parent.name,attempts=len(trials),accepted=sum(t['accepted'] for t in trials),
                          baseline=nodes[0]['metrics'],best_quality=nodes[r['best_quality']]['metrics'] if r['best_quality'] is not None else None,
                          best_coverage=nodes[r['best_coverage']]['metrics'] if r['best_coverage'] is not None else None,
                          families={family:dict(attempts=sum(t['action']['family']==family for t in trials),
                                                accepted=sum(t['action']['family']==family and t['accepted'] for t in trials))
                                    for family in sorted({t['action']['family'] for t in trials})}))
        fig,axes=plt.subplots(1,3,figsize=(10,5.6),constrained_layout=True)
        for ax,label,node in zip(axes,['Quiet control','Best quality','Best coverage'],[nodes[0],nodes[r['best_quality']] if r['best_quality'] is not None else nodes[0],nodes[r['best_coverage']] if r['best_coverage'] is not None else nodes[0]]):
            chain=[];n=node
            while n['parent'] is not None:
                chain.extend(p['stone'] for p in n['action']['placements']);n=nodes[n['parent']]
            draw(ax,node['document'],chain)
            ax.set_title(f'{label}\n{node["metrics"]["coverage"]:.3f}% coverage\n{node["metrics"]["broad_void_sqft"]:.3f} ft² broad void',fontsize=10)
        fig.suptitle(path.parent.name+' · full original outlines; accepted boards clearance-valid')
        fig.savefig(path.parent/'comparison.png',dpi=170);plt.close(fig)
    assert len(cases)==4
    summary=dict(cases=cases,checked_documents=checked,verification='Source/engine hashes, exact inventory outlines, whole-board clearance, observed settling, valid parent edits, inventory transitions checked.',
                 caveats='Four contexts, one seed, depth two, small budget. Recorded neighborhood centers supplied in three cases. No autonomous full-trajectory or held-out success claim. Void/joint scores are approximate.')
    (HERE/'pilot_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
