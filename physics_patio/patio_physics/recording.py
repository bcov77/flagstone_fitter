"""Durable editor telemetry. Observes session instances; engine code is unchanged."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import time
import uuid
import zipfile

import pymunk


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump(value, f, separators=(',', ':'), allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def capture(session):
    snapshot = session.snapshot()
    return dict(pieces=snapshot['pieces'], settings=snapshot['settings'], rng=snapshot['rng'],
                running=session.running, elapsed=session.elapsed, run_number=session.run_number,
                accumulator=session.accumulator, undo_count=len(session.history),
                drives=session.drives,
                velocities={sid:[b.velocity.x,b.velocity.y,b.angular_velocity]
                            for sid,b in session.bodies.items()})


def read_timeline(path):
    """An interrupted final append is recoverable; malformed complete rows are not."""
    with path.open() as f:
        for line in f:
            try:
                yield json.loads(line)
            except ValueError:
                if not line.endswith('\n'):
                    break
                raise


class Recorder:
    def __init__(self, root):
        self.root = Path(root)
        self.active = False
        self.handle = None
        self.latest = None
        self.frames = self.events = self.sequence = 0
        self.simulated = 0.
        self.reference = {}
        self.peaks = {}
        self.error = None
        manifests = sorted(self.root.glob('*/manifest.json'), key=lambda p:p.stat().st_mtime)
        for p in reversed(manifests):
            try:
                manifest = json.loads(p.read_text())
                if not manifest.get('complete') and not manifest.get('recovered'):
                    self.recover(p.parent,manifest)
                    manifest = json.loads(p.read_text())
                if manifest.get('complete') or manifest.get('recovered'):
                    if self.latest is None:
                        self.latest = p.parent.name
            except (ValueError, OSError):
                continue

    def recover(self, folder, manifest):
        initial = json.loads((folder/'initial.json').read_text())
        stones = {str(s['id']):s for s in initial['stones']}
        count = 0
        for record in read_timeline(folder/'timeline.jsonl'):
            count += 1
            if record['kind']=='event' and record.get('after'):
                snap = record['after']
                initial['physics']['settings'] = snap['settings']
                for sid,p in snap['pieces'].items():
                    stones[sid].update(pose=p['pose'],active=p['active'])
                    initial['physics']['pieces'][sid].update(anchor=p['anchor'],locked=p['locked'])
            elif record['kind']=='frame':
                for sid,stone in stones.items():
                    stone['active'] = sid in record['poses']
                    if stone['active']:
                        stone['pose'] = record['poses'][sid]
        atomic_json(folder/'final.json',initial)
        manifest.update(recovered=True,reason='Interrupted recording; recovered saved prefix, not a normal finish.',
                        recovered_records=count,final_sha256=hashlib.sha256((folder/'final.json').read_bytes()).hexdigest())
        atomic_json(folder/'manifest.json',manifest)

    def attach(self, session):
        """Observe every integration step without changing forces, RNG or stepping."""
        original = session._forces

        def observed_forces():
            if self.active:
                self.watch(session)
            original()

        session._forces = observed_forces

    def status(self):
        return dict(active=self.active, id=getattr(self,'id',None), frames=self.frames,
                    events=self.events, simulated_seconds=self.simulated, error=self.error,
                    latest_id=self.latest,
                    download_url=f'/recordings/{self.latest}/download' if self.latest else None)

    def start(self, session):
        if self.active:
            raise ValueError('A trajectory is already recording.')
        self.id = uuid.uuid4().hex
        self.folder = self.root / self.id
        self.folder.mkdir(parents=True)
        self.frames = self.events = self.sequence = 0
        self.simulated = 0.
        self.error = None
        self.last_sync = 0.
        self.manifest = dict(schema_version=1,id=self.id,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                             complete=False,python=platform.python_version(),pymunk=pymunk.version,
                             timestep_seconds=1/240,maximum_frame_step_seconds=.1,
                             peak_motion='Observed every physics step; relative to latest simulation-start or edit baseline.',
                             dependencies={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in [Path(__file__),Path(__file__).with_name('engine.py'),Path(__file__).with_name('server.py')]})
        atomic_json(self.folder/'initial.json',session.export())
        atomic_json(self.folder/'manifest.json',self.manifest)
        self.handle = (self.folder/'timeline.jsonl').open('a')
        self.active = True
        self.reset_motion(session)
        self.event('recording_started',None,capture(session))

    def _append(self, record):
        self.sequence += 1
        record.update(sequence=self.sequence,wall_time_unix=time.time(),simulated_seconds=self.simulated)
        try:
            self.handle.write(json.dumps(record,separators=(',', ':'),allow_nan=False)+'\n')
            self.handle.flush()
            if record['kind']=='event' or time.monotonic()-self.last_sync>=.5:
                os.fsync(self.handle.fileno())
                self.last_sync = time.monotonic()
        except (OSError, ValueError) as exc:
            self.error = str(exc)
            self.active = False
            self.handle.close()
            raise ValueError(f'Recording stopped because it could not be saved: {exc}') from exc

    def event(self, action, before, after, payload=None, error=None):
        if not self.active:
            return
        self.events += 1
        self._append(dict(kind='event',action=action,before=before,after=after,
                          payload=payload or {},error=error,motion_peaks=self.peaks.copy()))

    def reset_motion(self, session):
        self.reference = {sid:[b.position.x,b.position.y,b.angle] for sid,b in session.bodies.items()}
        self.peaks = {sid:dict(center_inches=0.,angle_degrees=0.) for sid in self.reference}

    def watch(self, session):
        for sid,b in session.bodies.items():
            ref = self.reference.setdefault(sid,[b.position.x,b.position.y,b.angle])
            p = self.peaks.setdefault(sid,dict(center_inches=0.,angle_degrees=0.))
            p['center_inches'] = max(p['center_inches'],12*math.hypot(b.position.x-ref[0],b.position.y-ref[1]))
            p['angle_degrees'] = max(p['angle_degrees'],abs(math.degrees(b.angle-ref[2])))

    def advance(self, session, seconds):
        if not self.active:
            session.advance(seconds)
            return
        remaining = seconds
        while remaining > 1e-10:
            duration = min(.1,remaining)
            before = session.elapsed
            session.advance(duration)
            self.simulated += max(0.,session.elapsed-before)
            self.watch(session)
            self.frames += 1
            self._append(dict(kind='frame',running=session.running,elapsed=session.elapsed,
                              run_number=session.run_number,
                              poses={sid:[b.position.x,b.position.y,b.angle] for sid,b in session.bodies.items()},
                              velocities={sid:[b.velocity.x,b.velocity.y,b.angular_velocity] for sid,b in session.bodies.items()}))
            remaining -= duration

    def finish(self, session, reason='finished'):
        if not self.active:
            return
        before = capture(session)
        self.watch(session)
        session.stop()
        self.event('recording_finished',before,capture(session),dict(reason=reason))
        atomic_json(self.folder/'final.json',session.export())
        self.handle.close()
        self.active = False
        self.manifest.update(complete=True,reason=reason,frames=self.frames,events=self.events,
                             simulated_seconds=self.simulated,
                             initial_sha256=hashlib.sha256((self.folder/'initial.json').read_bytes()).hexdigest(),
                             timeline_sha256=hashlib.sha256((self.folder/'timeline.jsonl').read_bytes()).hexdigest(),
                             final_sha256=hashlib.sha256((self.folder/'final.json').read_bytes()).hexdigest())
        atomic_json(self.folder/'manifest.json',self.manifest)
        self.latest = self.id

    def bundle(self, identifier):
        if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):
            raise ValueError('Invalid recording ID.')
        folder = self.root / identifier
        manifest = json.loads((folder/'manifest.json').read_text())
        if not (manifest.get('complete') or manifest.get('recovered')):
            raise ValueError('This recording is unfinished. Its durable timeline is still saved locally.')
        initial = json.loads((folder/'initial.json').read_text())
        frames = []
        last_time = -1.
        for record in read_timeline(folder/'timeline.jsonl'):
            if record['kind']=='frame':
                if record['simulated_seconds']-last_time<.2:
                    continue
                last_time = record['simulated_seconds']
                frames.append(dict(label='Physics',time=last_time,poses=record['poses'],highlight=[],event=False))
            else:
                for key in ['before','after']:
                    snap = record.get(key)
                    if snap is None:
                        continue
                    highlight = record['payload'].get('edit',{}).get('sid')
                    frames.append(dict(label=record['action'].replace('_',' ')+(f' · {key}' if record.get('before') else ''),
                                       time=record['simulated_seconds'],
                                       poses={sid:p['pose'] for sid,p in snap['pieces'].items() if p['active']},
                                       highlight=[str(highlight)] if highlight is not None else [],event=True))
        used = {sid for f in frames for sid in f['poses']}
        if not frames:
            frames.append(dict(label='Recovered initial state',time=0.,event=True,highlight=[],
                               poses={str(s['id']):s['pose'] for s in initial['stones'] if s['active']}))
            used = set(frames[0]['poses'])
        data = dict(patio=initial['patio'],stones={str(s['id']):s['outline'] for s in initial['stones'] if str(s['id']) in used},frames=frames)
        viewer = Path(__file__).with_name('recording_viewer.html').read_text().replace('__DATA__',json.dumps(data,separators=(',',':')).replace('</','<\\/'))
        output = io.BytesIO()
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
            for name in ['manifest.json','initial.json','timeline.jsonl','final.json']:
                archive.write(folder/name,name)
            archive.writestr('replay.html',viewer)
            archive.writestr('README.txt','Open replay.html in a browser. Left/right arrows step through actions; enable physics frames for sampled motion. The full timeline is retained in timeline.jsonl. Original outlines are in initial.json; final.json reopens in the editor. Recording does not certify clearance or gap quality.\n')
        return output.getvalue()
