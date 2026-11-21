"""Create an inspectable, image-inspired synthetic inventory, not measured stones."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch, Rectangle
import numpy as np
from flagstone.generate import flagstone_inventory
from flagstone.phase1 import save_json

out = Path('examples/flagstone_reference')
out.mkdir(parents=True, exist_ok=True)
layout = flagstone_inventory(seed=17)
save_json(layout, out / 'inventory.json')

fig = plt.figure(figsize=(14, 10), facecolor='#f5f2ed')
grid = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=.28)
ax = fig.add_subplot(grid[0])
ax.imshow(plt.imread('example_flagstones.png'))
ax.set_title('Reference photograph', loc='left', fontsize=15, weight='bold')
ax.axis('off')
ax = fig.add_subplot(grid[1])
ax.set_title('Synthetic sample: actual relative sizes; square = entire patio', loc='left', fontsize=14, weight='bold')
# One shared scale for all stone outlines and the patio square.
indices = list(range(6)) + list(range(6, 18)) + list(range(56, 74))
for k, i in enumerate(indices):
    col, row = k % 9, k // 9
    points = np.asarray(layout.shapes[layout.ids[i]].exterior.coords) + [col * .24 + .12, (3-row)*.23 + .12]
    ax.add_patch(Patch(points, facecolor=['#c9a69b','#b9adb0','#d3bbb0'][k%3], edgecolor='#413e3a', lw=.8))
ax.add_patch(Rectangle((2.3, .03), 1, 1, fill=False, lw=1.5, edgecolor='#413e3a'))
ax.text(2.8, .53, 'unit patio', ha='center', va='center')
ax.set_xlim(0, 3.35); ax.set_ylim(0, 1.07)
ax.set_aspect('equal'); ax.axis('off')
fig.suptitle('Angular flagstone inventory: visual review', fontsize=20, weight='bold')
fig.text(.06, .035, 'Synthetic, image-inspired silhouettes — not traced measurements. Sample: 6 large, 12 medium, 18 small.\nFull inventory: 156 stones; total stone area = 0.85 of patio area. The patio and gaps are provisional.', fontsize=10)
fig.savefig(out / 'review.png', dpi=160, bbox_inches='tight')
plt.close(fig)
print(out / 'inventory.json')
print(out / 'review.png')
