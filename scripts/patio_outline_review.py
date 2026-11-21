"""Make a reproducible polygon approximation of the user's sketch (not a survey)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.affinity import scale, translate
from shapely.geometry.polygon import orient
from flagstone.phase1 import load_json, polygon, save_json

# Approximate points read from rough_outline.png, in image pixels. Long curved
# segments are represented by straight chords; y is flipped to Cartesian axes.
pixels = [(56,78),(150,80),(268,80),(268,257),(360,267),(445,272),
          (455,72),(575,65),(730,61),(902,58),(903,139),(900,194),
          (892,243),(878,284),(858,316),(833,339),(802,360),(765,381),
          (719,402),(669,421),(615,439),(559,454),(507,465),(465,472),
          (428,473),(390,469),(352,460),(312,447),(271,430),(232,411),
          (192,389),(155,365),(122,341),(94,318),(74,298),(60,276),
          (50,252),(45,228),(43,199),(44,163),(49,120)]
shape = polygon([(x, -y) for x, y in pixels])
shape = scale(shape, xfact=1/np.sqrt(shape.area), yfact=1/np.sqrt(shape.area), origin=(0,0))
shape = orient(translate(shape, xoff=-shape.bounds[0], yoff=-shape.bounds[1]), sign=1)
layout = load_json('examples/flagstone_reference/inventory.json')
layout.patio = shape
out = Path('examples/flagstone_reference')
save_json(layout, out / 'patio_inventory.json')
inset = shape.buffer(-.01)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
axes[0].imshow(plt.imread('rough_outline.png')); axes[0].set_title('User sketch'); axes[0].axis('off')
x,y = shape.exterior.xy; axes[1].fill(x,y,color='#21617d',alpha=.9)
for part in ([inset] if inset.geom_type == 'Polygon' else list(inset.geoms)):
    x,y = part.exterior.xy; axes[1].plot(x,y,'--',color='white',lw=1)
axes[1].set_title('Benchmark polygon and 0.01 edge inset'); axes[1].set_aspect('equal'); axes[1].set_xlabel('Normalized units'); axes[1].set_ylabel('Normalized units')
fig.suptitle('Sketch-based patio: area normalized to 1; no real dimensions implied')
fig.savefig(out / 'patio_review.png', dpi=160); plt.close(fig)
print(f'area={shape.area:.6f}; vertices={len(shape.exterior.coords)-1}; inset={inset.geom_type}; inset area={inset.area:.6f}')
