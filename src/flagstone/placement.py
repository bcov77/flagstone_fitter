"""Inspectable placement manifests for validated layouts."""
import json
from pathlib import Path
import numpy as np
from .phase1 import polygon, transformed, validate
from .diagnostics import layout_metrics
from .uncertainty import verify_uncertainty_model


def placement_manifest(layout, *, gap, edge, tolerance=1e-9, stone_margins=None,
                       patio_margin=0, order=None, frozen_ids=(), uncertainty_model=None):
    """Export poses and local/world orientation marks after full validation.

    Coordinates share the patio's units and origin. Angles are counterclockwise
    radians around each local centroid. Default inventory order is not a build
    sequence. An explicit order records user intent, not a construction guarantee.
    """
    verify_uncertainty_model(layout, uncertainty_model, stone_margins, patio_margin)
    violations = validate(layout, gap, edge, tolerance, stone_margins=stone_margins,
                          patio_margin=patio_margin)
    if violations:
        raise ValueError(f'Cannot export invalid layout: {violations}')
    active = [sid for i, sid in enumerate(layout.ids) if layout.active[i]]
    if order is None:
        sequence = active
        sequence_kind = 'inventory_order_not_build_sequence'
    else:
        sequence = list(order)
        sequence_kind = 'user_supplied_order'
        if len(sequence) != len(active) or set(sequence) != set(active):
            raise ValueError('order must contain every active stone exactly once')
    frozen = set(frozen_ids)
    if not frozen.issubset(active):
        raise ValueError('frozen_ids must refer to active stones')
    entries = []
    for sid in sequence:
        i = layout.ids.index(sid)
        x, y, angle = layout.poses[i]
        lx, ly = layout.shapes[sid].exterior.coords[0]
        wx = x + lx*np.cos(angle) - ly*np.sin(angle)
        wy = y + lx*np.sin(angle) + ly*np.cos(angle)
        item = {'id': sid, 'centroid': [float(x), float(y)], 'angle_radians_ccw': float(angle),
                'orientation_mark_local': [float(lx), float(ly)],
                'orientation_mark_world': [float(wx), float(wy)],
                'orientation_reference_length': float(np.hypot(lx, ly)),
                'area': float(layout.shapes[sid].area),
                'perimeter': float(layout.shapes[sid].length), 'frozen': sid in frozen}
        local = np.asarray(layout.shapes[sid].exterior.coords[:-1])
        item['local_outline'] = local.tolist()
        item['world_outline'] = [list(p) for p in transformed(layout.shapes[sid], layout.poses[i]).exterior.coords[:-1]]
        item['edge_lengths'] = np.linalg.norm(np.roll(local, -1, axis=0) - local, axis=1).tolist()
        if sid in layout.source_outlines:
            c = polygon(layout.source_outlines[sid]).centroid
            item['source_centroid'] = [float(c.x), float(c.y)]
            item['orientation_mark_source'] = [float(lx+c.x), float(ly+c.y)]
        entries.append(item)
    return {'manifest_version': 1, 'units': layout.units,
            'coordinate_convention': 'patio coordinates; +x right, +y up; angles counterclockwise in radians',
            'patio_outline': [list(point) for point in layout.patio.exterior.coords[:-1]],
            'clearance': {'pair': gap, 'edge': edge, 'tolerance': tolerance,
                          'stone_margins': dict(stone_margins or {}), 'patio_margin': patio_margin},
            'sequence_kind': sequence_kind, 'stones': entries, 'metrics': layout_metrics(layout),
            'uncertainty_model': uncertainty_model}


def save_placement_manifest(layout, path, **kwargs):
    """Write a manifest after validation; invalid inputs do not overwrite a file."""
    data = placement_manifest(layout, **kwargs)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False))


def plot_placement(layout, path, *, gap, edge, tolerance=1e-9, stone_margins=None,
                   patio_margin=0, frozen_ids=(), uncertainty_model=None):
    """Render a validated coordinate map with IDs and orientation references.

    The first local outline vertex is the reference mark; marks in this figure
    describe the model and are not automatically marked on physical stones.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch
    from matplotlib.lines import Line2D
    data = placement_manifest(layout, gap=gap, edge=edge, tolerance=tolerance,
                              stone_margins=stone_margins, patio_margin=patio_margin,
                              frozen_ids=frozen_ids, uncertainty_model=uncertainty_model)
    fig, ax = plt.subplots(figsize=(9, 8))
    try:
        ax.set_facecolor('#f5f2eb')
        ax.add_patch(PolygonPatch(data['patio_outline'], closed=True, facecolor='#ebe6da', edgecolor='#202623', linewidth=2, zorder=1))
        for stone in data['stones']:
            color = '#a8bcb1' if stone['frozen'] else '#d2c7b0'
            ax.add_patch(PolygonPatch(stone['world_outline'], closed=True, facecolor=color, edgecolor='#303733', linewidth=.7, zorder=2))
            cx, cy = stone['centroid']; mx, my = stone['orientation_mark_world']
            ax.plot([cx + .2*(mx-cx), mx], [cy + .2*(my-cy), my], color='#574a38', linewidth=.55, zorder=3)
            ax.plot(mx, my, marker='o', markersize=2.5, color='#b55625', zorder=4)
            ax.text(cx, cy, str(stone['id']), ha='center', va='center', fontsize=6, color='#111713', zorder=5)
        minx, miny, maxx, maxy = layout.patio.bounds
        width, height = maxx-minx, maxy-miny
        target = min(width, height)/5
        power = 10 ** np.floor(np.log10(target))
        bar = max(f*power for f in (1, 2, 5) if f*power <= target)
        left = minx + width*.03
        bottom = miny - height*.1
        ax.plot([left, left+bar], [bottom, bottom], color='#1b211e', linewidth=2, solid_capstyle='butt')
        ax.plot([left, left], [bottom-height*.008, bottom+height*.008], color='#1b211e', linewidth=1)
        ax.plot([left+bar, left+bar], [bottom-height*.008, bottom+height*.008], color='#1b211e', linewidth=1)
        ax.text(left+bar/2, bottom-height*.02, f'{bar:g} {layout.units}', ha='center', va='top', fontsize=8)
        ax.set_xlim(minx-width*.05, maxx+width*.05)
        ax.set_ylim(miny-height*.17, maxy+height*.05)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f'x ({layout.units})'); ax.set_ylabel(f'y ({layout.units})')
        ax.set_title('Placement planning map', loc='left', fontweight='bold', pad=27)
        margin_note = '  ·  extra margins applied' if patio_margin or any((stone_margins or {}).values()) else ''
        ax.text(0, 1.012, f"{len(data['stones'])} stones  ·  gap {gap:g}  ·  edge {edge:g} {layout.units}{margin_note}", transform=ax.transAxes, ha='left', va='bottom', fontsize=8)
        ax.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor='#b55625',markeredgecolor='#b55625',markersize=4,label='Model orientation mark'),
                           Line2D([0],[0],marker='s',color='none',markerfacecolor='#a8bcb1',markeredgecolor='#303733',markersize=6,label='Frozen stone')],
                  loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2, frameon=False, fontsize=8)
        fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    finally:
        plt.close(fig)
