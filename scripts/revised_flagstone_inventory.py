"""Create a varied, synthetic 2.5x inventory informed by the flagstone photo."""
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch, Rectangle
from shapely.affinity import rotate, scale
from flagstone.phase1 import Layout, load_json, local_shape, polygon, save_json

# Hand-designed silhouettes inspired by the photo: angular triangular,
# quadrilateral, elongated and irregular outlines; not traced real stones.
templates = [
    [(0,.04),(.95,0),(1,.25),(.77,.95),(.14,1),(.02,.73)],
    [(.02,.12),(.28,0),(.9,.06),(1,.48),(.77,1),(.08,.87)],
    [(.03,.1),(.89,0),(1,.24),(.93,.89),(.44,1),(0,.81)],
    [(.02,.02),(.91,0),(1,.15),(.86,.91),(.14,1),(0,.76)],
    [(.05,0),(.87,.04),(1,.37),(.84,.92),(.28,1),(0,.72)],
    [(.02,.12),(.38,0),(.98,.07),(.87,.31),(1,.52),(.72,.94),(.21,1),(0,.73)],
    [(0,.04),(.92,0),(1,.13),(.83,.31),(.96,.56),(.69,.94),(.14,1),(.02,.73)],
    [(0,.05),(.27,0),(1,.16),(.86,.96),(.18,1),(.04,.8)],
    [(0,.04),(.93,0),(1,.86),(.19,1),(.07,.75)],
    [(0,.08),(.89,0),(1,.24),(.82,1),(.03,.91)],
    [(.02,.05),(.93,.02),(1,.16),(.88,.88),(.14,1),(0,.73)],
    [(0,.12),(.2,0),(1,.08),(.83,.94),(.33,1),(.04,.75)],
    [(0,.08),(.26,.03),(.9,0),(1,.16),(.89,.85),(.71,.98),(.12,1)],
    [(0,.03),(.91,0),(1,.13),(.63,.92),(.12,1)],
    [(.04,.03),(.91,0),(1,.29),(.83,.95),(.13,1),(0,.78)],
    [(0,0),(1,.08),(.83,.87),(.64,1),(.07,.94)],
    [(0,.06),(.88,0),(1,.1),(.63,.91),(.44,1),(.05,.87)],
    [(0,.04),(.94,0),(.92,.09),(.55,.94),(.37,1),(.03,.8)],
    [(0,0),(1,.12),(.24,1)],
    [(0,.07),(.88,0),(1,.12),(.2,1),(.06,.91)],
    [(0,0),(1,.13),(.54,.97),(.42,1),(.07,.78)],
    [(.02,.09),(.87,0),(1,.15),(.38,1),(.14,.83)],
]
rng = np.random.default_rng(41)
reference = load_json('examples/flagstone_reference/surplus_25ft.json')
patio_area = reference.patio.area
# Five stones have S016's area. The rest have continuously varying sizes.
large = np.full(5, .025)
medium = rng.uniform(.0055, .0115, 195); medium *= 1.565 / medium.sum()
small = rng.uniform(.0018, .0044, 270); small *= .810 / small.sum()
areas = np.concatenate((large, medium, small))
assert abs(areas.sum()-2.5) < 1e-10 and max(areas[5:]) < min(large)
shapes = {}; ids = []
for area in areas:
    base = np.array(templates[int(rng.integers(len(templates)))], dtype=float)
    # Small independent corner perturbations give related pieces distinct edges.
    for _ in range(20):
        try:
            shape = polygon(base + rng.uniform(-.035, .035, base.shape)); break
        except ValueError:
            continue
    else: shape = polygon(base)
    shape = scale(shape, xfact=rng.uniform(1.05, 2.25), yfact=1, origin='centroid')
    shape = rotate(shape, rng.uniform(0,360), origin='centroid')
    size = math.sqrt(area * patio_area / shape.area)
    shape = scale(shape, xfact=size, yfact=size, origin='centroid')
    sid = f'S{len(ids):03d}'; ids.append(sid); shapes[sid] = local_shape(shape.exterior.coords)
layout = Layout(reference.patio, shapes, np.zeros((len(ids),3)), np.zeros(len(ids),bool),
                tuple(ids), rng, units='ft')
out = Path('examples/flagstone_reference'); out.mkdir(parents=True, exist_ok=True)
save_json(layout, out/'revised_25ft_inventory.json')

# Every displayed stone and the patio square share one physical scale.
fig = plt.figure(figsize=(15,10), facecolor='#f5f2ed')
gs = fig.add_gridspec(2,1,height_ratios=[1,1.35],hspace=.20)
ax=fig.add_subplot(gs[0]); ax.imshow(plt.imread('example_flagstones.png'))
ax.set_title('Reference photograph',loc='left',fontsize=14,weight='bold'); ax.axis('off')
ax=fig.add_subplot(gs[1]); ax.set_title('Synthetic sample, shown at one common scale',loc='left',fontsize=14,weight='bold')
indices = list(range(5)) + list(range(5,20)) + list(range(200,216))
for k,i in enumerate(indices):
    col,row=k%9,k//9
    pts=np.array(shapes[ids[i]].exterior.coords) + [col*4.5+2.25, (3-row)*4.5+2.25]
    ax.add_patch(Patch(pts,facecolor=['#c9a69b','#b9adb0','#d3bbb0'][k%3],edgecolor='#413e3a',lw=.8))
    ax.text(col*4.5+2.25,(3-row)*4.5+.1,ids[i],ha='center',fontsize=6)
ax.add_patch(Rectangle((42,.1),25,25,fill=False,lw=1.3,edgecolor='#413e3a'))
ax.text(54.5,12.6,'25 ft × 25 ft\nscale reference',ha='center',va='center')
ax.set_xlim(0,68); ax.set_ylim(-.2,25.5); ax.set_aspect('equal'); ax.axis('off')
fig.suptitle('Revised angular flagstones — visual review',fontsize=19,weight='bold')
fig.text(.05,.02,'Synthetic illustration, not a trace or stone survey. Full inventory: 470 stones, total area 2.5× patio; exactly five largest.\nReference square shows scale, not the patio outline. Individual areas vary continuously below the largest class.',fontsize=10)
fig.savefig(out/'revised_stones_review.png',dpi=160,bbox_inches='tight');plt.close(fig)
print('stones',len(ids),'largest',len(large),'total patio areas',sum(s.area for s in shapes.values())/patio_area)
print('sizes sqft',round(min(s.area for s in shapes.values()),2),round(max(s.area for s in shapes.values()),2))
