"""Render validation galleries in the repository plotting environment."""
import argparse,csv,json,math,random
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
from shapely.geometry import Polygon
from shapely.affinity import affine_transform


def main(out):
    m=json.loads((out/'manifest.json').read_text());source=json.loads((out/'source.json').read_text())
    cases={case['id']:case for case in m['cases']};stones={str(s['id']):s for s in source['stones']}
    rows=list(csv.DictReader((out/'outcomes.csv').open()))
    candidates=[r for r in rows if r['kind'] not in ['control','empty_diagnostic']]
    # Reproducible mixed review sample: both actions/kinds and observed outcomes.
    rng=random.Random(20261116);rng.shuffle(candidates)
    buckets={}
    for r in candidates:
        buckets.setdefault((r['family'],r['kind'],r['label']),[]).append(r)
    selected=[]
    for label in ['invalid_or_unfinished','local_benefit_remote_cost','provisional_quality_improvement']:
        example=next((r for r in candidates if r['label']==label),None)
        if example is not None:selected.append(example)
    for key,items in sorted(buckets.items()):
        if items[0]['id'] not in {r['id'] for r in selected}:selected.append(items[0])
    seen={r['id'] for r in selected}
    for r in candidates:
        if len(selected)>=12:break
        if r['id'] not in seen:selected.append(r);seen.add(r['id'])
    selected=selected[:12]
    fig,axes=plt.subplots(4,3,figsize=(14,17))
    for ax,row in zip(axes.ravel(),selected):
        case=cases[row['case']];r=json.loads((out/'results'/f"{row['id']}.json").read_text());candidate=case['candidate']
        for sid,pose in r['final_poses'].items():
            x,y,t=pose;c,s=math.cos(t),math.sin(t)
            g=affine_transform(Polygon(stones[sid]['outline']),(c,-s,s,c,x,y))
            ax.add_patch(Patch(list(g.exterior.coords),facecolor='#cd9264' if candidate and sid==candidate['id'] else '#ddd3c0',edgecolor='#554a39',linewidth=.5))
            q=g.representative_point();ax.text(q.x,q.y,sid,fontsize=6,ha='center',va='center',clip_on=True)
        if candidate:
            sid=candidate['id'];x,y,t=candidate['pose'];c,s=math.cos(t),math.sin(t)
            g=affine_transform(Polygon(stones[sid]['outline']),(c,-s,s,c,x,y));xx,yy=g.exterior.xy
            ax.plot(xx,yy,'--',color='#2671a5',linewidth=.8)
        patio=Polygon(source['patio']);x,y=patio.exterior.xy;ax.plot(x,y,'k-',linewidth=.7)
        cx,cy=case['center'];ax.set_xlim(cx-1.8,cx+1.8);ax.set_ylim(cy-1.8,cy+1.8);ax.set_aspect('equal')
        ax.set_title(f"{case['context']} {case['family']} / {case['kind']}\n{row['protocol']}, seed {row['seed']}; valid={row['valid']}, settled={row['converged']}",fontsize=9)
    for ax in axes.ravel()[len(selected):]:ax.axis('off')
    fig.suptitle('Campaign validation: mixed cases and outcomes\nOrange: candidate; blue dashed: initial candidate pose; original outlines',fontsize=13)
    fig.subplots_adjust(left=.05,right=.99,bottom=.04,top=.92,wspace=.15,hspace=.35)
    fig.savefig(out/'validation_gallery.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for family,color in [('insertion','#477a9f'),('replacement','#c68b58')]:
        rs=[r for r in candidates if r['family']==family]
        axes[0].scatter([float(r.get('intrusion_sqft') or 0) for r in rs],[float(r['pair_deficit_inches']) for r in rs],label=family,color=color,alpha=.7,s=24)
    axes[0].set_xlabel('Initial clearance intrusion (ft²)');axes[0].set_ylabel('Final summed pair deficit (in)');axes[0].legend()
    for protocol,marker in [('direct','o'),('gap_cycle','x')]:
        rs=[r for r in candidates if r['protocol']==protocol and r['valid']=='True' and r['control_available']=='True']
        axes[1].scatter([float(r['local_void_delta_sqft']) for r in rs],[float(r['far_void_delta_sqft']) for r in rs],label=protocol,marker=marker,alpha=.7,s=30)
    axes[1].axhline(0,color='gray',linewidth=.6);axes[1].axvline(0,color='gray',linewidth=.6)
    axes[1].set_xlabel('Local broad-void change vs control (ft²)');axes[1].set_ylabel('Far broad-void change (ft²)');axes[1].legend()
    axes[2].hist([float(r['wall_seconds']) for r in rows],bins=16,color='#5f857b');axes[2].set_xlabel('Per-trial wall seconds under concurrency');axes[2].set_ylabel('Trials')
    fig.suptitle('Validation diagnostics — selected validation mix, not a success-rate estimate')
    fig.tight_layout();fig.savefig(out/'validation_diagnostics.png',dpi=160);plt.close(fig)
    (out/'gallery_sample.json').write_text(json.dumps([r['id'] for r in selected],indent=2))
    print('Rendered validation gallery and diagnostics')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).with_name('campaign_v1'));main(p.parse_args().out)
