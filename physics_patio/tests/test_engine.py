import copy
import json
import math
from pathlib import Path

import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from patio_physics.engine import PhysicsSession, convex_parts, CONTROLS


def document():
    return {'schema_version': 1, 'units': 'ft',
            'patio': [[0, 0], [8, 0], [8, 8], [0, 8]],
            'stones': [{'id': str(i), 'outline': [[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]],
                        'pose': [2 + i * 2, 3, 0], 'active': i == 0}
                       for i in range(3)]}


def quiet(session):
    session.edit('settings', values={k: 0 for k in ['translation_force', 'rotation_force',
                                                  'translation_stiffness', 'angular_stiffness']})


def test_edit_transactions_and_inventory_uniqueness():
    s = PhysicsSession(document())
    before = s.export()
    s.edit('insert', sid='1', pose=[3, 3, .5])
    assert len(s.bodies) == 2
    with pytest.raises(ValueError):
        s.edit('insert', sid='1', pose=[4, 4, 0])
    assert len(s.history) == 1
    s.edit('lock', sid='1', locked=True)
    s.edit('pose', sid='1', pose=[2.7, 3, 1])
    assert s.pieces['1'].locked and s.pieces['1'].anchor == [2.7, 3, 1]
    s.undo()
    assert s.pieces['1'].pose == [3, 3, .5]
    s.undo()
    assert not s.pieces['1'].locked
    s.undo()
    assert s.export()['stones'] == before['stones']


def test_delete_returns_stone_and_undo_restores_lock_and_pose():
    s = PhysicsSession(document())
    s.edit('lock', sid='0', locked=True)
    s.edit('delete', sid='0')
    assert not s.pieces['0'].active and '0' not in s.bodies
    s.undo()
    assert s.pieces['0'].active and s.pieces['0'].locked


def test_pause_and_single_transaction_for_entire_run():
    s = PhysicsSession(document())
    before = s.snapshot()['pieces']
    s.start()
    drives = copy.deepcopy(s.drives)
    s.start()
    assert len(s.history) == 1
    for _ in range(10):
        s.advance(.025)
    assert s.snapshot()['pieces'] != before
    s.stop()
    paused = s.snapshot()['pieces']
    s.advance(.5)
    assert s.snapshot()['pieces'] == paused
    s.start()
    assert s.drives != drives
    s.stop()
    s.undo()
    assert s.snapshot()['pieces'] == paused
    s.undo()
    assert s.snapshot()['pieces'] == before


def test_edit_during_run_is_independently_undoable():
    s = PhysicsSession(document())
    s.start()
    s.advance(.1)
    old_pose = s.pieces['0'].pose[:]
    s.edit('pose', sid='0', pose=[4, 4, .25])
    assert s.running
    s.advance(.1)
    s.undo()  # following simulation
    assert s.pieces['0'].pose == [4, 4, .25]
    s.undo()  # manual manipulation
    assert s.pieces['0'].pose == old_pose
    s.undo()  # preceding simulation
    assert s.pieces['0'].pose == [2, 3, 0]


def test_locked_insert_pushes_neighbor_and_never_moves_itself():
    s = PhysicsSession(document())
    quiet(s)
    s.edit('insert', sid='1', pose=[2.8, 3, 0])
    s.edit('lock', sid='1', locked=True)
    s.start()
    s.advance(.4)
    assert s.pieces['1'].pose == [2.8, 3, 0]
    assert s.pieces['0'].pose[0] < 2
    assert s.clearance_report()['minimum_pair_in'] > .3


def test_hardness_actually_changes_overlap_response():
    def response(hz):
        s = PhysicsSession(document())
        quiet(s)
        s.edit('insert', sid='1', pose=[2.8, 3, 0])
        s.edit('lock', sid='1', locked=True)
        s.edit('settings', values={'locked_hardness': hz})
        s.start()
        s.advance(.05)
        return s.pieces['0'].pose[0]
    assert response(0) == pytest.approx(2)
    assert response(50) < response(1) < 2


def test_tether_pulls_to_manual_anchor():
    s = PhysicsSession(document())
    quiet(s)
    s.edit('settings', values={'translation_stiffness': 20})
    s.start()
    s.bodies['0'].position = (3, 3)
    s.advance(.5)
    assert abs(s.pieces['0'].pose[0]-2) < 1


def test_concave_proxy_encloses_original_and_keeps_large_notch():
    outline = [(0,0),(3,0),(3,1),(1,1),(1,3),(0,3)]
    original = Polygon(outline)
    parts = [Polygon(p) for p in convex_parts(outline)]
    proxy = unary_union(parts)
    assert proxy.covers(original)
    assert proxy.area < original.convex_hull.area
    assert all(p.area == pytest.approx(p.convex_hull.area) for p in parts)


def test_export_roundtrip_original_geometry_and_metadata():
    d = document()
    original = copy.deepcopy(d)
    s = PhysicsSession(d)
    s.edit('pose', sid='0', pose=[4, 5, 1])
    s.edit('lock', sid='0', locked=True)
    s.edit('settings', values={'rotation_force': 30})
    exported = s.export()
    restored = PhysicsSession(json.loads(json.dumps(exported)))
    assert restored.export() == exported
    assert d == original
    for a, b in zip(exported['stones'], original['stones']):
        assert a['outline'] == b['outline']


@pytest.mark.parametrize('values', [{'speed': float('nan')}, {'speed': 0}, {'bad': 3}, {'piece_hardness': 1000}])
def test_invalid_settings_are_atomic(values):
    s = PhysicsSession(document())
    before = s.snapshot()
    with pytest.raises(ValueError):
        s.edit('settings', values=values)
    assert s.snapshot() == before
    assert not s.history


def test_independent_clearance_check_finds_clashes():
    s = PhysicsSession(document())
    assert s.clearance_report()['passes_clearance']
    s.edit('insert', sid='1', pose=[2.8, 3, 0])
    report = s.clearance_report()
    assert not report['passes_clearance']
    assert report['errors'][0]['ids'] == ['0', '1']
    s.edit('pose', sid='0', pose=[0, 0, 0])
    assert any(e['kind'] == 'boundary' for e in s.clearance_report()['errors'])


def test_gap_inflation_updates_all_colliders_and_is_undoable():
    s = PhysicsSession(document())
    s.edit('insert', sid='1', pose=[4, 3, 0])
    s.edit('lock', sid='1', locked=True)
    before = s.snapshot()['pieces']
    s.edit('settings', values={'gap_inches': 1.25})
    assert s.snapshot()['pieces'] == before
    for shape in s.space.shapes:
        if shape.collision_type in (1, 2):
            assert shape.radius == pytest.approx(1.25 / 24)
    left_wall = next(sh for sh in s.space.shapes if sh.collision_type == 3 and sh.bb.left < -1)
    assert left_wall.bb.right == pytest.approx(1.25 / 24)
    restored = PhysicsSession(s.export())
    assert restored.settings['gap_inches'] == 1.25
    s.undo()
    assert s.settings['gap_inches'] == .5
    assert s.snapshot()['pieces'] == before


def test_larger_gap_pushes_apart_stones_that_already_meet_required_gap():
    def run(gap):
        s = PhysicsSession(document())
        quiet(s)
        s.edit('insert', sid='1', pose=[3.07, 3, 0])
        s.edit('lock', sid='1', locked=True)
        s.edit('settings', values={'gap_inches': gap})
        # Acceptance remains based on the real half-inch requirement.
        assert s.clearance_report()['passes_clearance']
        s.start()
        s.advance(.1)
        return s.pieces['0'].pose[0]
    assert run(.5) == pytest.approx(2)
    assert run(1) < 2


def test_old_saves_default_to_half_inch_and_gap_cannot_shrink_below_contract():
    s = PhysicsSession(document())
    assert s.settings['gap_inches'] == .5
    with pytest.raises(ValueError):
        s.edit('settings', values={'gap_inches': .4})


def test_real_winner_loads_and_exports_exact_originals():
    root = Path(__file__).resolve().parents[2]
    original = json.loads((root / 'examples/quarter2_20h_ready_20261114/tasks/task_576_escape/best.json').read_text())
    s = PhysicsSession(original)
    assert len(s.pieces) == 137 and len(s.bodies) == 27
    assert s.export()['stones'] == original['stones']
    assert s.clearance_report()['passes_clearance']
    s.start()
    s.advance(.2)
    assert all(math.isfinite(v) for p in s.pieces.values() for v in p.pose)
    s.undo()
    assert s.export()['stones'] == original['stones']
