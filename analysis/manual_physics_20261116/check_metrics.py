"""Small geometric sanity checks for the proposed measurement logic."""
import json
import numpy as np
import shapely
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
from shapely.affinity import translate
from pathlib import Path
# Dimensions inches. A narrow bump overlaps little area but forces a whole
# twelve-inch edge away from its matching flat neighbor.
gap=.5; neighbor=box(-12,0,0,12)
flat=box(gap,0,12+gap,12)
bump=unary_union([flat,box(gap-.75,5.95,gap+.1,6.05)])
shifted=translate(bump,xoff=.75)
ys=np.linspace(.1,11.9,120)
flat_ds=shapely.distance(shapely.points(np.full(120,gap),ys),neighbor)
# Profile along the main straight edge after separating its bump.
shifted_ds=shapely.distance(shapely.points(np.full(120,gap+.75),ys),neighbor)
assert flat.distance(neighbor)==gap
assert abs(shifted.distance(neighbor)-gap)<1e-9
assert bump.intersection(neighbor.buffer(gap,join_style='mitre')).area<.1
assert np.mean(shifted_ds-gap)>.7
assert np.mean(flat_ds-gap)==0
# Stable grid mapping: nearest-tree result indices may be unordered.
pts=shapely.points([3,.2,7],[0,0,0]);polys=[box(0,-1,1,1),box(5,-1,6,1)]
idx,ds=shapely.STRtree(polys).query_nearest(pts,return_distance=True,all_matches=False)
mapped=np.zeros(3);mapped[idx[0]]=ds
assert np.allclose(mapped,[2,0,1])
result={'units':'inches','bump_intrusion_area_sq_inches':bump.intersection(neighbor.buffer(gap,join_style='mitre')).area,
 'flat_minimum_gap':flat.distance(neighbor),'bump_after_translation_minimum_gap':shifted.distance(neighbor),
 'flat_main_edge_mean_excess_gap':float(np.mean(flat_ds-gap)),
 'bump_main_edge_mean_excess_gap':float(np.mean(shifted_ds-gap)),
 'checks_passed':['small bump overlap does not imply good full-edge fit','equal minimum gaps can hide different edge profiles','nearest-tree grid values map to point indices']}
Path(__file__).with_name('metric_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
fig,axes=plt.subplots(1,2,figsize=(10,6))
for ax,g,title in zip(axes,[flat,shifted],['Matching flat edge','Small bump limits approach']):
 for poly,color in [(neighbor,'#dad0bf'),(g,'#ce9569')]:
  ax.add_patch(Patch(list(poly.exterior.coords),facecolor=color,edgecolor='#554a39'))
 ax.axvline(.5,color='#53769a',linestyle='--',linewidth=.8)
 ax.set_xlim(-2,4);ax.set_ylim(-.5,12.5);ax.set_aspect('equal');ax.set_title(title)
 ax.set_xlabel('Inches');ax.set_ylabel('Inches')
axes[0].annotate('0.5-inch joint',xy=(.25,9),xytext=(1.3,10),arrowprops=dict(arrowstyle='->'))
axes[1].annotate('0.5 inch at bump',xy=(.5,6),xytext=(1.6,6.7),arrowprops=dict(arrowstyle='->'))
axes[1].annotate('1.25 inches\nalong main edge',xy=(.6,9),xytext=(1.6,10),arrowprops=dict(arrowstyle='->'))
fig.suptitle('Same minimum clearance, different edge compatibility')
fig.tight_layout();fig.savefig(Path(__file__).with_name('bump_example.png'),dpi=160);plt.close(fig)
