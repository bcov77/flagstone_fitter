# Patio physics sandbox

A separate Pymunk backend with a local browser editor. No existing packing
module, campaign capsule, environment, or result file is changed. The initial
layout is task 576, the November 15 overnight winner, with all 137 original
stones available (27 on the field, 110 in reserve).

## Run

From the project root:

```bash
./physics_patio/run.sh
```

Open **http://127.0.0.1:8765**. Leave the terminal running; Ctrl-C stops the server.
The server listens only on loopback. One browser tab controls a session at a time.

The sandbox already has its own installed environment. To recreate it:

```bash
uv venv physics_patio/.venv --python python3.12
uv pip install --python physics_patio/.venv/bin/python -r physics_patio/requirements.txt
```

## Controls

- **Hold space** to simulate, **release** to freeze. The on-screen hold button
  works with a mouse or touch. Space also works when a range slider has focus;
  text/number inputs retain their normal editing keys.
- Click a field stone to select it. **Drag** to translate. Overlap is allowed,
  including moving locked stones into clashes. The next simulation attempts
  to separate them.
- **Q / E** or rotation buttons rotate by 5°. Shift changes keyboard rotation
  to 1°. The angle input accepts an absolute angle in degrees.
- Arrow keys nudge by ¼ inch; Shift nudges by one inch.
- **Lock position**, or **L**, freezes both translation and rotation during
  simulation. A locked stone remains manually movable/rotatable.
- **Delete / Backspace**, or **Return to inventory**, removes a stone from the
  field without deleting its original inventory record.
- Inventory is on the right at the **same pixels per foot** as the field.
  Scroll to browse, filter by ID, and drag onto the field to place a stone.
- **Back**, or ⌘/Ctrl-Z, undoes a manipulation or an entire hold-space run.
  Lock changes, insertion, deletion, rotation, translation, and settings changes
  are undoable. Each drag and each committed slider change is one transaction.
  History keeps the last 200 entries using compact state, not copied outlines.
- Editing while running splits the preceding run, manipulation, and following
  run into separate undo entries. Undo stops simulation. Mouse dragging pauses
  simulation until the drop; if space is still held, it resumes afterward.
- Switching tabs, losing focus, disconnecting, or losing the held-key heartbeat
  stops the backend. The heartbeat expires after 0.65 seconds.

## Sliders and their meaning

All eleven requested controls are present, plus patio edge hardness and a
**Simulation gap** slider (0.50–2.00 inches, in 0.05-inch steps). Increase it to
temporarily inflate collision shapes, jiggle, then return to **0.50 inches**
and settle again. The gap change is undoable and saves with the session. It
affects movable and locked stones and the patio-edge clearance. It changes
collision geometry without resizing original stone outlines or moving tether
anchors. The independent clearance check always uses the required half inch.

| Control | Meaning |
|---|---|
| Simulation gap | Temporary target separation in inches; 0.50 is the actual required minimum |
| Piece edge hardness | Contact frequency in Hz for two movable pieces; higher separates overlap more strongly |
| Locked piece edge hardness | Separate contact frequency when a locked piece touches a movable one |
| Angular constraint stiffness | Restoring angular acceleration per radian away from the piece's anchor angle |
| Translational constraint stiffness | Restoring acceleration per foot away from its anchor position |
| Rotational magnitude | Amplitude of the sinusoidal angular target, in degrees |
| Rotational force | Gain driving the piece toward the angular target |
| Rotational frequency | Global multiplier on independently randomized per-piece angular rhythms |
| Translational magnitude | Amplitude of each x/y target, in inches |
| Translational force | Gain driving the piece toward its positional target |
| Translational frequency | Global multiplier on independent x/y rhythms |
| Simulation speed | Simulated time per wall-clock time in the browser adapter |
| Patio edge hardness | Contact frequency against the patio walls |

Each run samples independent frequencies (0.35–1.35 Hz before the global
multiplier) and phases for x, y, and angle, separately for every piece. A zero
frequency multiplier freezes that target at its initial phase; zero magnitude
removes its oscillatory amplitude. Force is a mass/inertia-normalized drive
gain, rather than a dimensional Newton value. Magnitude sets a target, not a
hard displacement limit. Nonzero drive gain also pulls toward the anchor when
magnitude is zero. To remove all translational/rotational restoring force,
set both the corresponding constraint stiffness and drive force to zero.

Manual translation/rotation and lock changes update that piece's tether anchor.
Starting another simulation does not silently reanchor the whole layout.
Locked pieces receive no drive or tether forces. Damping is always present.

## Saving and interchange

### Record a complete manual trajectory

Use **Record trajectory** in the bar above the field before your first edit.
Then work normally: insert and move stones, hold space for physics, change
settings, and undo anything you reject. The recording captures these actions
and the intermediate physics poses, rather than only your final board.

Click **Finish & save trajectory ↓** when done. This pauses physics and
downloads a ZIP containing:

- `replay.html`: open locally; left/right arrows step through the actions.
  Enable **Include physics motion** to see sampled intermediate motion.
- `initial.json` and `final.json`: complete original outlines and editor layouts.
- `timeline.jsonl`: ordered before/after edits, settings, locks, undo, failed
  commands, start/stop events, physics poses and velocities. Simulation events
  preserve RNG state and drive rhythms. Maximum center and angular excursions
  are observed every integration step relative to the last start/edit baseline.
- `manifest.json`: version hashes, counts, timing and completion status.

The server writes recordings incrementally under
`physics_patio/sessions/recordings/`; downloading is not required to retain the
capture. Refresh/disconnect finishes the recording locally, and **Download last
trajectory** retrieves it after reconnecting. Opening another layout finishes
the current recording; start a new one for the new board. A server restart can
recover an interrupted saved prefix, explicitly marked `recovered` rather than
normally complete. An incomplete final append is ignored by the replay while
the original log is preserved.

The replay samples physics at about 0.2 simulated seconds; the raw timeline
retains the more frequent captured frames (at most about 0.1-second stepping).
Event boundaries are retained regardless of replay sampling. Recording does
not change the solver, enforce new placement rules, or certify clearance/gaps.

### Save just the current layout

**Save layout** downloads a JSON containing original outlines and active rigid
poses in the existing layout format, with an extra `physics` block for locks,
anchors, and slider settings. No collision proxy replaces an original outline.
**Open layout** starts a new editor session from such a JSON or another existing
quarter layout. Opening a layout resets undo history.

The server also atomically autosaves stopped/edited sessions and disconnects to
`physics_patio/sessions/latest.json`. It does not overwrite the original winner
and does not automatically replace your startup layout. Resume explicitly:

```bash
./physics_patio/run.sh --layout physics_patio/sessions/latest.json
```

The browser and server must remain connected for live control. Edits only sent
after a completed drop are committed. The local app does not provide a durable
undo journal across process restarts; exported poses/settings/locks persist.

## Headless use and clean separation

The backend has no HTTP or browser dependency in its interface:

```python
from patio_physics import PhysicsSession

session = PhysicsSession(document, seed=42)
session.edit('insert', sid='9', pose=[4.0, 4.0, 0.2])
session.edit('lock', sid='9', locked=True)
session.start()              # fresh per-piece rhythms; one undo boundary
session.advance(1.0)         # simulated seconds; no sleeping
session.stop()
coordinates = session.state()['pieces']
result = session.export()
session.undo()
```

Import with `physics_patio` on `PYTHONPATH`, using the sandbox environment.
There is also a bounded CLI step that refuses to overwrite an existing output:

```bash
./physics_patio/run.sh \
  --layout physics_patio/sessions/latest.json \
  --headless-seconds 10 --seed 42 \
  --output physics_patio/sessions/step_001.json
```

Headless duration is simulated time; the browser's speed slider only controls
wall-time playback. `engine.py` owns geometry, physics, edits, history and
interchange. `server.py` owns transport, held-key lifetime, and autosave.
`static/` owns drawing and pointer/keyboard controls. The old `src/flagstone`
package is not imported or modified. No Sites hosting or external service is
needed for a native local Python backend.

## Physics details and limitations

The current patio must be rectangular and layouts must use feet. Original
concave shapes become checked enclosing collision proxies, then convex parts
on a single Pymunk body. Simplification tolerance is 0.024 inch; a checked
outward buffer of 0.024 inch (or 0.048 inch when needed) makes it conservative.
If containment cannot be established, the original polygon is decomposed.
All parts carry a further radius of half the selected simulation gap.
The walls are inset by the same amount, supplying the selected edge clearance
in total. At the default 0.50-inch gap, both amounts are ¼ inch.
Proxies may therefore be slightly more restrictive than the original contract.

Pymunk performs collision detection and body integration. Contact callbacks
apply an implicit critically damped normal spring impulse, using contact depth,
relative velocity, and effective mass/inertia. Native hard-contact response is
disabled for these contacts so it cannot override the independent hardness
sliders. There is no contact bounce or tangential friction; general damping
helps the field settle. Correction impulses and external accelerations are
bounded to limit violent deep-insertion responses. The fixed timestep is 1/240 s.
This is a deliberately compliant search simulation, not a guarantee that every
overlap can resolve or that Pymunk's usual hard-contact solver is in use.

**Check clearances** independently measures original polygon pair distances
and boundary clearance. This first app does not run the local-maximum gap-circle
audit; a clearance pass does not establish the two-inch/five-inch maximum-gap
goal. Exported candidates should use the existing independent verifier before
promotion into the old campaign workflow.

## Validation

```bash
uv pip install --python physics_patio/.venv/bin/python 'pytest>=9,<10'
cd physics_patio
.venv/bin/python -m pytest -q tests
```

Tests exercise transaction undo, insertion/deletion/locks, pause semantics,
fresh drive rhythms, editing during a run, physical overlap recovery, hardness
response, tethers, concave geometry, exact-outline exports, the real winner,
and live HTTP/WebSocket start/stop/undo/export/disconnect/heartbeat behavior.

## Overnight adaptive placement trajectories

A separate foreground runner now generates and evaluates choices on the current
board, relaxes single/double placements, conditionally cycles the simulation gap,
and records 30-attempt trajectories using ten worker processes:

```bash
caffeinate -i ./physics_patio/overnight.sh
```

Every ten trajectories contain eight normal, one Golden and one Silver run.
Ctrl-C checkpoints; the same command resumes. See
[the strategy, saved artifacts and analysis commands](../analysis/physics_trajectories/README.md).
This runner uses the root `.venv`; it does not control the live editor or autosave.
