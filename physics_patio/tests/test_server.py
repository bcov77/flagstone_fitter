import asyncio
import json

from aiohttp import ClientSession, web

from patio_physics import PhysicsSession
from patio_physics.server import create_app
from test_engine import document


def test_websocket_edit_run_release_undo_and_disconnect(tmp_path):
    async def run():
        session = PhysicsSession(document())
        save_path = tmp_path / 'latest.json'
        runner = web.AppRunner(create_app(session, save_path))
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        async def until(ws, predicate):
            async with asyncio.timeout(5):
                while True:
                    data = await ws.receive_json()
                    if predicate(data):
                        return data

        try:
            async with ClientSession() as client:
                async with client.get(f'http://127.0.0.1:{port}/') as response:
                    assert response.status == 200
                    assert 'Hold' in await response.text()
                async with client.ws_connect(f'http://127.0.0.1:{port}/ws') as ws:
                    init = await ws.receive_json()
                    assert init['type'] == 'init' and len(init['pieces']) == 3
                    await ws.send_json({'type':'start'})
                    await until(ws, lambda d: d.get('running') and d.get('elapsed', 0) > .04)
                    await ws.send_json({'type':'stop'})
                    await until(ws, lambda d: d.get('type') == 'state' and not d['running'])
                    poses = session.snapshot()['pieces']
                    await asyncio.sleep(.1)
                    assert session.snapshot()['pieces'] == poses
                    await ws.send_json({'type':'undo'})
                    await until(ws, lambda d: d.get('undo_count') == 0)
                    assert session.pieces['0'].pose == [2,3,0]
                    await ws.send_json({'type':'edit','edit':{'action':'insert','sid':'1','pose':[4,4,0]}})
                    await until(ws, lambda d: any(p['id']=='1' and p['active'] for p in d.get('pieces',[])))
                    await ws.send_json({'type':'export'})
                    result = await until(ws, lambda d: d.get('type') == 'export')
                    assert result['document']['stones'][1]['active']
                    await ws.send_json({'type':'start'})
                    await until(ws, lambda d: d.get('running') is True)
                    # Missing held-key heartbeat must stop automatically.
                    await until(ws, lambda d: d.get('running') is False)
                    await ws.send_json({'type':'start'})
                    await until(ws, lambda d: d.get('running') is True)
                await asyncio.sleep(.1)
                assert not session.running
                assert json.loads(save_path.read_text())['stones'][1]['active']
        finally:
            await runner.cleanup()
    asyncio.run(run())
