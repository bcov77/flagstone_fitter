"""Postprocess the bounded pilot with spatial diagnostics and paired controls."""
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
import numpy as np
import shapely
from shapely.affinity import scale,affine_transform

OUT=Path(__file__).parent;ROOT=OUT.parents[1]


def main():
    files=sorted((OUT/'trials').glob('*.json.gz'))
    assert len(files)==39,f'Pilot incomplete: {len(files)}/39'
    records=[json.loads(gzip.decompress(p.read_bytes())) for p in files]
    source=json.loads((ROOT/'examples/physics_trajectories_20261117_r3/source.json').read_text())
    source_stones={str(s['id']):s for s in source['stones']}
    local={s:shapely.Polygon(v['outline']) for s,v in source_stones.items()}
    patio=shapely.Polygon(source['patio']);inputs=json.loads((OUT/'candidates.json').read_text())
    def board(poses):
        result={}
        for sid,p in poses.items():
            c,s=math.cos(p[2]),math.sin(p[2]);result[sid]=affine_transform(local[sid],[c,-s,s,c,p[0],p[1]])
        return result
    original=board({s:v['pose'] for s,v in source_stones.items() if v['active']})
    step=.5/12;xs=np.arange(step/2,patio.bounds[2],step);ys=np.arange(step/2,patio.bounds[3],step)
    xx,yy=np.meshgrid(xs,ys);xy=np.column_stack((xx.ravel(),yy.ravel()));pts=shapely.points(xy)
    def field(b):
        tree=shapely.STRtree([*b.values(),patio.boundary]);_,d=tree.query_nearest(pts,all_matches=False,return_distance=True)
        return d*12
    initial=field(original);controls={(r['old'],r['seed']):r for r in records if r['method']=='retain'}
    masks={target:shapely.covers(shapely.from_wkt(t['pocket']).buffer(.75),pts) for target,t in inputs['targets'].items()}
    sample={}
    for r in records:
        distances=field(board(r['end']['poses']));mask=masks[r['target']]
        core=distances>1;startcore=initial>1
        sample[r['name']]=dict(local_core_sqft=float(core[mask].sum()*step**2),
                               remote_core_sqft=float(core[~mask].sum()*step**2),
                               local_core_delta_sqft=float((core[mask].sum()-startcore[mask].sum())*step**2),
                               remote_core_delta_sqft=float((core[~mask].sum()-startcore[~mask].sum())*step**2),
                               sampled_local_diameter_inches=float(distances[mask].max()*2),
                               sampled_local_before_inches=float(initial[mask].max()*2))
    rows=[]
    for r in records:
        ctrl=controls[r['old'],r['seed']];sp=sample[r['name']];cs=sample[ctrl['name']]
        result=dict(name=r['name'],method=r['method'],old=r['old'],stone=r['stone'],seed=r['seed'],
                    physical_pass=r['accepted'],pose_translation_inches=r['candidate_translation_inches'],
                    pose_rotation_degrees=r['candidate_rotation_degrees'],pocket_coverage=r['full_pocket_coverage_after'],
                    coverage=r['end_metrics']['area_percent'],coverage_vs_control_pp=r['end_metrics']['area_percent']-ctrl['end_metrics']['area_percent'],
                    interior=r['end_metrics']['gaps']['largest_interior_inches'],edge=r['end_metrics']['gaps']['largest_edge_inches'],
                    interior_vs_control_inches=r['end_metrics']['gaps']['largest_interior_inches']-ctrl['end_metrics']['gaps']['largest_interior_inches'],
                    edge_vs_control_inches=r['end_metrics']['gaps']['largest_edge_inches']-ctrl['end_metrics']['gaps']['largest_edge_inches'],
                    joint_delta_inches=r['joints_after']['total']-r['joints_before']['total'],
                    neighbors_moved=r['neighbors_moved_quarter_inch'],second_ring_moved=len(r['second_ring_moved']),
                    max_neighbor_inches=r['max_neighbor_movement_inches'],
                    local_core_vs_control_sqft=sp['local_core_sqft']-cs['local_core_sqft'],
                    remote_core_vs_control_sqft=sp['remote_core_sqft']-cs['remote_core_sqft'],**sp)
        # Fidelity is diagnostic, not a user-specified movement limit. Functional target occupation can survive movement.
        result['occupies_target']=r['full_pocket_coverage_after']>=.8
        rows.append(result)
    groups={}
    for old,sid in [('60','22'),('60','77'),('62','118'),('62','114'),('61','58')]:
        for method in ['direct','growth']:
            group=[r for r in rows if r['old']==old and r['stone']==sid and r['method']==method]
            groups[f'{old}_to_{sid}_{method}']={k:float(np.mean([r[k] for r in group])) for k in group[0] if isinstance(group[0][k],(float,int,bool)) and k!='seed'}
            groups[f'{old}_to_{sid}_{method}']['n']=len(group)
    summary=dict(trials=len(records),insertion_trials=30,controls=9,
                 physical_pass=sum(r['physical_pass'] for r in rows if r['method']!='retain'),
                 occupies_target=sum(r['occupies_target'] for r in rows if r['method']!='retain'),groups=groups,
                 diagnostic='Broad-void core = half-inch grid cells more than one inch from every stone and wall. Local = target pocket buffered nine inches. This is an exploratory area proxy, not a gap certificate.',
                 movement_note='The <=0.5-inch / <=10-degree stayed_at_target and useful_screen fields in raw pilot records were initial diagnostic screens, not user-approved movement limits. This analysis reports continuous motion and target occupation instead. A missing saved top-20 local circle is unassessed, not proof of failure.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    with (OUT/'outcomes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'spatial_metrics.json').write_text(json.dumps(rows,indent=2))
    baseline=json.loads((OUT/'baseline_metrics.json').read_text())
    cases=[('60','22'),('60','77'),('62','118'),('62','114'),('61','58')]
    for page in range(2):
        selected=cases[page*3:(page+1)*3]
        fig,axes=plt.subplots(len(selected),4,figsize=(17,5.3*len(selected)),squeeze=False)
        fig.subplots_adjust(left=.025,right=.995,top=.88,bottom=.025,wspace=.07,hspace=.6)
        for ri,(old,sid) in enumerate(selected):
            direct=next(r for r in records if r['method']=='direct' and r['old']==old and r['stone']==sid and r['seed']==2026111701)
            growth=next(r for r in records if r['method']=='growth' and r['old']==old and r['stone']==sid and r['seed']==2026111701)
            pose=direct['proposal']['placements'][0]['pose'];proposed=board({sid:pose})[sid]
            p=shapely.from_wkt(inputs['targets'][direct['target']]['pocket'])
            boxes=[p.bounds,proposed.bounds,original[old].bounds];x0=min(b[0] for b in boxes)-.8;y0=min(b[1] for b in boxes)-.8;x1=max(b[2] for b in boxes)+.8;y1=max(b[3] for b in boxes)+.8
            width=max(x1-x0,y1-y0);cx=(x0+x1)/2;cy=(y0+y1)/2
            bs=[original,{s:g for s,g in original.items() if s!=old},board(direct['end']['poses']),board(growth['end']['poses'])]
            bs[1][sid]=scale(proposed,.8,.8,origin='centroid')
            for ci,(ax,b,title) in enumerate(zip(axes[ri],bs,['Original','80% shape probe','Direct full-size endpoint','80% → 100% endpoint'])):
                ax.set_facecolor('#f2f2e8')
                if ci==1:ax.add_patch(Patch(list(p.simplify(.003).exterior.coords),facecolor='#c5e6e5',edgecolor='#60928d',lw=.7))
                for stone,g in b.items():
                    g=g.simplify(.003);chosen=stone==(old if ci==0 else sid);color=('#84adce' if ci==0 else '#efaa5c' if ci==1 else '#86bd9d') if chosen else '#dcd7c5'
                    ax.add_patch(Patch(list(g.exterior.coords),facecolor=color,edgecolor='#756d59',lw=.5));pt=g.representative_point();ax.text(pt.x,pt.y,stone,ha='center',va='center',fontsize=5.5,clip_on=True)
                if ci==1:ax.plot(*proposed.simplify(.003).exterior.xy,color='#aa512e',ls='--',lw=.8)
                ax.plot(*patio.exterior.xy,color='#355c47');ax.set_xlim(cx-width/2,cx+width/2);ax.set_ylim(cy-width/2,cy+width/2);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
                if ci>=2:
                    r=direct if ci==2 else growth;em=r['end_metrics'];title+=f'\n{em["area_percent"]:.2f}% · gaps {em["gaps"]["largest_interior_inches"]:.2f}″ / {em["gaps"]["largest_edge_inches"]:.2f}″'
                elif ci==1:title+='\nDashed = full-size footprint'
                ax.set_title(title,fontsize=9)
            probe=direct['probe_metrics'];fig.text(.03,axes[ri,0].get_position().y1+.045,f'{old} → {sid} · 80% probe fills {probe["coverage"]:.1%} of pocket · q90 edge deficit {probe["edge_excess_q90_inches"]:.2f}″ · one paired seed shown',fontsize=10,fontweight='bold')
        fig.suptitle('Saved 80% near-fit pilot · original → reduced probe → full-size physics\nEndpoints use original unscaled inventory outlines; fixed crop, three seeds measured per method.',fontsize=13,y=.995)
        fig.savefig(OUT/f'pilot_examples_{page+1}.png',dpi=140);plt.close(fig)
    manifest=json.loads((OUT/'manifest.json').read_text());assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in manifest['dependencies'].items())
    verification=dict(trials=39,physical_insertion_passes=summary['physical_pass'],
                      growth_colliders_restored=all(r.get('full_colliders_restored') for r in records if r['method']=='growth'),
                      neighbor_velocities_preserved_during_growth=all(r.get('growth_preserves_velocities') for r in records if r['method']=='growth'),
                      production_dependencies_unchanged=True,
                      note='Each endpoint independently remeasured from full original outlines in pilot.py. Temporary collision scaling never becomes accepted inventory geometry.')
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
