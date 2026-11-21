"""Render saved synthetic search results and unused-stone inventories."""
from pathlib import Path
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
import numpy as np

from flagstone.phase1 import load_json, transformed, validate
from flagstone.search import coverage

OUT = Path('examples/search_review')
BACKGROUND = '#f4f2ec'
PATIO = '#e8e3d8'
PLACED = '#a7bdb1'
UNUSED = '#d6ad79'
DARK = '#25332e'


def load_checked(path, edge):
    layout = load_json(path)
    errors = validate(layout, gap=.01, edge=edge)
    if errors:
        raise ValueError(f'{path}: {errors}')
    return layout


def draw_patio(ax, layout, title):
    ax.set_facecolor(BACKGROUND)
    ax.add_patch(PolygonPatch(np.asarray(layout.patio.exterior.coords), facecolor=PATIO,
                              edgecolor=DARK, lw=1.8, zorder=1))
    for i, sid in enumerate(layout.ids):
        if not layout.active[i]:
            continue
        shape = transformed(layout.shapes[sid], layout.poses[i])
        ax.add_patch(PolygonPatch(np.asarray(shape.exterior.coords), facecolor=PLACED,
                                  edgecolor=DARK, linewidth=.55, zorder=2))
    ax.set_title(title, loc='left', fontsize=14, fontweight='bold', pad=11)
    ax.set_aspect('equal', adjustable='box')
    minx, miny, maxx, maxy = layout.patio.bounds
    ax.set_xlim(minx-.03*(maxx-minx), maxx+.03*(maxx-minx))
    ax.set_ylim(miny-.03*(maxy-miny), maxy+.03*(maxy-miny))
    ax.set_xlabel('Patio width (normalized units)', fontsize=9)
    ax.set_ylabel('Patio depth (normalized units)', fontsize=9)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_unused(ax, layout):
    unused = [sid for i, sid in enumerate(layout.ids) if not layout.active[i]]
    cols = 8
    rows = max(1, math.ceil(len(unused)/cols))
    ax.set_title(f'Unused stones  •  {len(unused)}', loc='left', fontsize=14, fontweight='bold', pad=11)
    if unused:
        # A single common scale preserves relative sizes within this inventory.
        max_dimension = max(max(s.bounds[2]-s.bounds[0], s.bounds[3]-s.bounds[1])
                            for s in (layout.shapes[sid] for sid in unused))
        factor = .70 / max_dimension
        for index, sid in enumerate(unused):
            col, row = index % cols, rows-1-index // cols
            shape = np.asarray(layout.shapes[sid].exterior.coords)
            points = shape * factor + [col+.5, row+.57]
            ax.add_patch(PolygonPatch(points, facecolor=UNUSED, edgecolor=DARK, linewidth=.6))
            ax.text(col+.5, row+.10, sid, ha='center', va='center', fontsize=7, color=DARK)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    return len(unused)


def known_fit():
    witness = load_checked('examples/known_fit_large/witness.json', .005)
    result = load_checked('examples/known_fit_large_rebuild/result_90s.json', .005)
    if witness.ids != result.ids or any(not witness.shapes[s].equals_exact(result.shapes[s], 0) for s in witness.ids):
        raise ValueError('known-fit witness and result inventories differ')
    fig = plt.figure(figsize=(21, 8), facecolor=BACKGROUND)
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.15), wspace=.18, left=.045, right=.97, top=.80, bottom=.16)
    ax1, ax2, ax3 = (fig.add_subplot(grid[0, i]) for i in range(3))
    draw_patio(ax1, witness, f'Known feasible arrangement\n144 of 144  •  {coverage(witness):.1%} coverage')
    draw_patio(ax2, result, f'Search result\n{int(result.active.sum())} of 144  •  {coverage(result):.1%} coverage')
    draw_unused(ax3, result)
    fig.suptitle('Known-fit synthetic case: what the search missed', x=.045, y=.955, ha='left', fontsize=20, fontweight='bold', color=DARK)
    fig.text(.045, .892, 'The same 144 stones appear in both layouts. The left arrangement proves they all fit under the modeled clearances.', fontsize=11, color=DARK)
    fig.text(.045, .065, 'Green: placed stones   •   Tan: unused inventory, drawn at a common scale relative to each other (not patio scale)\nSynthetic unit-square patio; pair gap 0.01, edge gap 0.005. Both layouts pass exact geometric validation.', fontsize=10, color=DARK)
    fig.savefig(OUT/'01_known_fit_comparison.png', dpi=180, facecolor=BACKGROUND)
    plt.close(fig)


def mixed(path, filename, title, subtitle):
    layout = load_checked(path, .01)
    n = len(layout.ids); placed = int(layout.active.sum())
    fig = plt.figure(figsize=(16, 9), facecolor=BACKGROUND)
    grid = fig.add_gridspec(1, 2, width_ratios=(1, 1.2), wspace=.16, left=.055, right=.97, top=.80, bottom=.17)
    ax1, ax2 = (fig.add_subplot(grid[0, i]) for i in range(2))
    draw_patio(ax1, layout, f'Search result\n{placed} of {n}  •  {coverage(layout):.1%} coverage')
    draw_unused(ax2, layout)
    fig.suptitle(title, x=.055, y=.95, ha='left', fontsize=20, fontweight='bold', color=DARK)
    fig.text(.055, .89, subtitle, fontsize=11, color=DARK)
    fig.text(.055, .065, 'Green: placed stones   •   Tan: unused inventory, drawn at a common scale relative to each other (not patio scale)\nSynthetic unit-square patio; pair and edge gaps 0.01. Layout passes exact validation. No complete-fit witness is known for this inventory.', fontsize=10, color=DARK)
    fig.savefig(OUT/filename, dpi=180, facecolor=BACKGROUND)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    known_fit()
    mixed('examples/detailed_156_optimized/rebuild/result.json', '02_detailed_inventory.png',
          'Detailed mixed-size inventory', 'A search result with mostly rounded, approximately 20-vertex synthetic outlines.')
    mixed('examples/varied_156/rebuild/result.json', '03_varied_inventory.png',
          'Varied synthetic shapes', 'Squares, rectangles, rounded stones, and broad-notched outlines in one inventory.')
    print('\n'.join(str(p) for p in sorted(OUT.glob('*.png'))))


if __name__ == '__main__':
    main()
