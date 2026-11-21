import asyncio
import hashlib
import io
import json
import zipfile

from aiohttp import ClientSession, web

from patio_physics.engine import PhysicsSession
from patio_physics.recording import Recorder, capture
from patio_physics.server import create_app
from test_engine import document


def test_recording_preserves_dynamics_and_exports_durable_replay(tmp_path):
    a = PhysicsSession(document(),seed=12)
    b = PhysicsSession(document(),seed=12)
    recorder = Recorder(tmp_path)
    recorder.attach(b)
    initial = b.export()
    recorder.start(b)
    a.start(); b.start()
    recorder.reset_motion(b)
    recorder.event('simulation_start',None,capture(b))
    a.advance(.37)
    recorder.advance(b,.37)
    assert a.export() == b.export()
    assert a.drives == b.drives
    timeline = recorder.folder/'timeline.jsonl'
    entries = [json.loads(line) for line in timeline.read_text().splitlines()]
    assert len([r for r in entries if r['kind']=='frame']) == 4
    assert entries[-1]['simulated_seconds'] > .36
    before = capture(b)
    b.edit('insert',sid='1',pose=[4,4,.2])
    recorder.event('edit',before,capture(b),{'edit':{'action':'insert','sid':'1','pose':[4,4,.2]}})
    recorder.reset_motion(b)
    recorder.advance(b,.1)
    before = capture(b);b.undo()
    recorder.event('undo',before,capture(b))
    recorder.finish(b)
    assert not recorder.active and not b.running
    assert json.loads((recorder.folder/'initial.json').read_text()) == initial
    with zipfile.ZipFile(io.BytesIO(recorder.bundle(recorder.id))) as z:
        assert set(z.namelist()) == {'initial.json','final.json','manifest.json','timeline.jsonl','replay.html','README.txt'}
        manifest = json.loads(z.read('manifest.json'))
        assert manifest['timeline_sha256'] == hashlib.sha256(z.read('timeline.jsonl')).hexdigest()
        records = [json.loads(line) for line in z.read('timeline.jsonl').splitlines()]
        assert [r['sequence'] for r in records] == list(range(1,len(records)+1))
        assert {'edit','undo','recording_finished'} <= {r.get('action') for r in records}
        assert 'ArrowRight' in z.read('replay.html').decode()
        assert '__DATA__' not in z.read('replay.html').decode()
        final = json.loads(z.read('final.json'))
        assert PhysicsSession(final).export() == final
    reopened = Recorder(tmp_path)
    assert reopened.latest == recorder.id
    assert reopened.bundle(recorder.id)


def test_peak_motion_is_observed_between_frame_samples(tmp_path):
    s = PhysicsSession(document())
    recorder = Recorder(tmp_path);recorder.attach(s);recorder.start(s)
    s.start();recorder.reset_motion(s)
    # Integration steps observe an excursion even if the caller later restores the endpoint.
    s.bodies['0'].position = (3,3)
    s.advance(1/240)
    s.bodies['0'].position = (2,3)
    recorder.advance(s,.01)
    recorder.finish(s)
    events = [json.loads(line) for line in (recorder.folder/'timeline.jsonl').read_text().splitlines()]
    assert events[-1]['motion_peaks']['0']['center_inches'] >= 12


def test_restart_recovers_saved_prefix_without_rewriting_truncated_log(tmp_path):
    s = PhysicsSession(document())
    recorder = Recorder(tmp_path);recorder.attach(s);recorder.start(s)
    before = capture(s);s.edit('pose',sid='0',pose=[3,4,.5])
    recorder.event('edit',before,capture(s),{'edit':{'sid':'0','action':'pose','pose':[3,4,.5]}})
    s.start();recorder.advance(s,.12)
    expected = s.export()
    recorder.handle.write('{"kind":"frame"');recorder.handle.flush();recorder.handle.close()
    path = recorder.folder/'timeline.jsonl';original = path.read_bytes()
    recovered = Recorder(tmp_path)
    assert recovered.latest == recorder.id
    assert path.read_bytes() == original
    with zipfile.ZipFile(io.BytesIO(recovered.bundle(recorder.id))) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['recovered'] and not manifest['complete']
        assert json.loads(archive.read('final.json')) == expected


def test_websocket_capture_failed_edit_undo_download_and_disconnect(tmp_path):
    async def run():
        save = tmp_path/'latest.json'
        runner = web.AppRunner(create_app(PhysicsSession(document()),save))
        await runner.setup()
        site = web.TCPSite(runner,'127.0.0.1',0);await site.start()
        base = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
        async def until(ws,predicate):
            async with asyncio.timeout(8):
                while True:
                    d = await ws.receive_json()
                    if predicate(d):return d
        try:
            async with ClientSession() as client:
                async with client.ws_connect(base+'/ws') as ws:
                    await ws.receive_json()
                    await ws.send_json({'type':'record_start'})
                    start = await until(ws,lambda d:d.get('recording',{}).get('active'))
                    first_id = start['recording']['id']
                    await ws.send_json({'type':'edit','edit':{'action':'insert','sid':'1','pose':[4,4,0]}})
                    await until(ws,lambda d:any(p['id']=='1' and p['active'] for p in d.get('pieces',[])))
                    await ws.send_json({'type':'edit','edit':{'action':'insert','sid':'1','pose':[4,4,0]}})
                    await until(ws,lambda d:d['type']=='error')
                    await ws.send_str('{broken json')
                    await until(ws,lambda d:d['type']=='error')
                    await ws.send_json({'type':'start'})
                    await until(ws,lambda d:d.get('elapsed',0)>.12)
                    await ws.send_json({'type':'stop'})
                    await until(ws,lambda d:d['type']=='state' and not d['running'])
                    await ws.send_json({'type':'undo'})
                    await until(ws,lambda d:d['type']=='state' and d['undo_count']==1)
                    await ws.send_json({'type':'record_finish'})
                    saved = await until(ws,lambda d:d['type']=='recording_saved')
                    async with client.get(base+saved['download_url']) as response:
                        assert response.status==200
                        with zipfile.ZipFile(io.BytesIO(await response.read())) as z:
                            events = [json.loads(line) for line in z.read('timeline.jsonl').splitlines()]
                            actions = [r['action'] for r in events if r['kind']=='event']
                            assert actions.count('command_error')==2
                            assert 'undo' in actions and any(r['kind']=='frame' for r in events)
                    await ws.send_json({'type':'record_start'})
                    await until(ws,lambda d:d.get('recording',{}).get('active'))
                await asyncio.sleep(.1)
                async with client.ws_connect(base+'/ws') as ws:
                    init = await ws.receive_json()
                    assert not init['recording']['active']
                    assert init['recording']['latest_id']!=first_id
                    assert len(list((tmp_path/'recordings').glob('*/final.json')))==2
                    assert json.loads(save.read_text())['stones'][1]['active']
        finally:
            await runner.cleanup()
    asyncio.run(run())
