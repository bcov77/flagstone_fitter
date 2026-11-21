"""Retained-board champions, with approximate quality explicitly separated."""
import copy
from pathlib import Path
import numpy as np
import shapely
from common import read,write,state,restore,shapes
from aggressive_geometry import canonical,joint_summary


def enrich(doc,metrics):
    m=copy.deepcopy(metrics);board=shapes(doc);patio=shapely.Polygon(doc['patio']);step=.5/12
    x0,y0,x1,y1=patio.bounds;xx,yy=np.meshgrid(np.arange(x0+step/2,x1,step),np.arange(y0+step/2,y1,step))
    xy=np.column_stack([xx.ravel(),yy.ravel()]);points=shapely.points(xy);mask=shapely.contains(patio,points);points=points[mask]
    tree=shapely.STRtree([*[g.simplify(.002,preserve_topology=True) for g in board.values()],patio.boundary])
    _,dist=tree.query_nearest(points,all_matches=False,return_distance=True)
    j=joint_summary(board,patio)
    m['quality']=dict(broad_void_sqft=float(np.sum(dist>1/12)*step**2),wall_joint_inches=j['wall'],joint_inches=j['total'],
                      scope='Approximate half-inch-grid void area and sampled matched joints; not clearance/gap certification')
    return m


def axes(m):
    q=m['quality'];g=m['gaps']
    return [-m['area_percent'],g.get('largest_edge_inches',1e6),g.get('largest_interior_inches',1e6),q['broad_void_sqft'],-q['wall_joint_inches'],-q['joint_inches']]


def quality_key(m):
    a=axes(m)
    return (a[1],a[3],a[4],a[2],a[5],a[0])


def dominates(a,b):
    a,b=axes(a),axes(b)
    return all(x<=y+1e-9 for x,y in zip(a,b)) and any(x<y-1e-9 for x,y in zip(a,b))


def archive(folder,doc,attempt,metrics,limit=12):
    if not metrics['clearance']['valid']:return
    folder=Path(folder);path=folder/'archive.json';old=read(path) if path.exists() else dict(entries=[],best_coverage=None,best_quality=None)
    item=dict(attempt=attempt,metrics=metrics,frame=state(doc),key=canonical(state(doc)))
    entries=old['entries'][:]
    for k in ['best_coverage','best_quality']:
        if old.get(k) and not any(v['key']==old[k]['key'] for v in entries):entries.append(old[k])
    if not any(v['key']==item['key'] for v in entries):entries.append(item)
    coverage=min(entries,key=lambda v:(-v['metrics']['area_percent'],quality_key(v['metrics']),v['attempt']))
    quality=min(entries,key=lambda v:(quality_key(v['metrics']),v['attempt']))
    frontier=[v for v in entries if not any(dominates(o['metrics'],v['metrics']) for o in entries)]
    frontier=sorted(frontier,key=lambda v:quality_key(v['metrics']))[:limit]
    write(path,dict(entries=frontier,best_coverage=coverage,best_quality=quality,
                    policy='Retained valid endpoints only; bounded Pareto archive plus persistent coverage and gap-quality champions. No backtracking.'))
    for label,item in [('best_coverage',coverage),('best_quality',quality)]:
        write(folder/f'{label}.json',restore(doc,item['frame']))
