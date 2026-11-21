"""Render original-outline layouts with independently witnessed gap circles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from flagstone.phase1 import load_json, transformed


def render(paths, labels, output):
    fig, axes = plt.subplots(1, len(paths), figsize=(6*len(paths), 7.1), squeeze=False)
    for axis, path, label in zip(axes[0], paths, labels):
        path = Path(path)
        layout = load_json(path)
        axis.plot(*layout.patio.exterior.xy, color='black', linewidth=1.5)
        area = 0
        for i, sid in enumerate(layout.ids):
            if not layout.active[i]:
                continue
            g = transformed(layout.shapes[sid], layout.poses[i])
            area += g.area
            axis.fill(*g.exterior.xy, color=plt.get_cmap('tab20')(i%20), alpha=.6)
            axis.text(g.centroid.x, g.centroid.y, sid, ha='center', va='center', fontsize=5.5)
        verification = path.with_name('initial_verification.json' if path.stem == 'initial' else 'verification.json')
        v = json.loads(verification.read_text())
        if v['layout_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Verification belongs to different layout bytes')
        gap = v['gap']
        axis.add_patch(Circle(gap['center'], gap['lower_diameter_in']/24, fill=False,
                              edgecolor='#b22222', linewidth=2))
        axis.set_title(f"{label}\n{int(layout.active.sum())} stones · {100*area/layout.patio.area:.1f}% coverage\n"
                       f"Global gap: [{gap['lower_diameter_in']:.2f}, {gap['upper_diameter_in']:.2f}] in",
                       fontsize=11)
        axis.set_aspect('equal')
        axis.set_xlabel('Feet')
        axis.set_ylabel('Feet')
        axis.margins(.03)
    fig.suptitle('Quarter-patio pilot — all layouts still fail the two-inch goal', fontsize=15)
    fig.text(.5, .025, 'Actual stone sizes and original outlines. Red circles are independently witnessed empty circles.\n'
             'Four square edges are physical boundaries of this synthetic test patio.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .94))
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layouts', nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if len(args.layouts) != len(args.labels):
        parser.error('Each layout requires a label')
    render(args.layouts, args.labels, args.output)
