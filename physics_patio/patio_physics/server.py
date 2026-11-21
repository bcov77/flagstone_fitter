"""Loopback-only HTTP/WebSocket adapter around the headless backend."""
import argparse
import asyncio
import contextlib
import json
from pathlib import Path
import time

from aiohttp import web

from .engine import PhysicsSession
from .recording import Recorder, capture

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAYOUT = ROOT / 'examples/quarter2_20h_ready_20261114/tasks/task_576_escape/best.json'
STATIC = Path(__file__).with_name('static')


def create_app(session, save_path):
    app = web.Application(client_max_size=32 * 1024**2)
    runtime = {'session': session, 'controller': None}
    recorder = Recorder(save_path.parent / 'recordings')
    recorder.attach(session)

    def save():
        save_path.parent.mkdir(parents=True, exist_ok=True)
        temp = save_path.with_suffix('.tmp')
        temp.write_text(json.dumps(runtime['session'].export(), separators=(',', ':')))
        temp.replace(save_path)

    async def index(request):
        return web.FileResponse(STATIC / 'index.html')

    def state_message(s, geometry=False):
        return {'type':'init' if geometry else 'state', **s.state(geometry=geometry),
                'recording':recorder.status()}

    async def download(request):
        try:
            payload = await asyncio.to_thread(recorder.bundle,request.match_info['identifier'])
        except (OSError, ValueError) as exc:
            raise web.HTTPNotFound(text=str(exc))
        return web.Response(body=payload,content_type='application/zip',
                            headers={'Content-Disposition':f'attachment; filename="patio-trajectory-{request.match_info["identifier"]}.zip"'})

    async def socket(request):
        # Only same-origin local pages may control the local solver.
        origin = request.headers.get('Origin')
        if origin and origin != f'{request.scheme}://{request.host}':
            raise web.HTTPForbidden(text='Use the local application page.')
        if runtime['controller'] is not None and not runtime['controller'].closed:
            raise web.HTTPConflict(text='Another tab controls this sandbox. Close it first.')
        ws = web.WebSocketResponse(max_msg_size=32 * 1024**2, heartbeat=10)
        await ws.prepare(request)
        runtime['controller'] = ws
        s = runtime['session']
        last_hold = time.monotonic()
        last_tick = last_hold
        dirty = False
        await ws.send_json(state_message(s,geometry=True))

        async def tick():
            nonlocal last_tick, dirty
            while not ws.closed:
                await asyncio.sleep(.05)
                now = time.monotonic()
                dt = min(.1, now - last_tick)
                last_tick = now
                if s.running and now - last_hold > .65:
                    before = capture(s) if recorder.active else None
                    s.stop()
                    recorder.event('heartbeat_stop',before,capture(s) if recorder.active else None)
                    dirty = True
                if s.running:
                    try:
                        recorder.advance(s,dt * s.settings['speed'])
                    except ValueError as exc:
                        s.stop()
                        await ws.send_json({'type':'error','message':str(exc)})
                    dirty = True
                elif dirty:
                    save()
                    dirty = False
                await ws.send_json(state_message(s))

        ticker = asyncio.create_task(tick())
        try:
            async for msg in ws:
                if msg.type != web.WSMsgType.TEXT:
                    continue
                data = None
                before = None
                try:
                    data = json.loads(msg.data)
                    if not isinstance(data, dict):
                        raise ValueError('Expected a JSON object.')
                    command = data.get('type')
                    before = capture(s) if recorder.active and command not in ('hold','check','export') else None
                    if command == 'start':
                        last_hold = time.monotonic()
                        was_running = s.running
                        s.start()
                        if not was_running:
                            recorder.reset_motion(s)
                            recorder.event('simulation_start',before,capture(s) if recorder.active else None)
                    elif command == 'hold':
                        last_hold = time.monotonic()
                    elif command == 'stop':
                        was_running = s.running
                        s.stop()
                        if was_running:
                            recorder.event('simulation_stop',before,capture(s) if recorder.active else None)
                        dirty = True
                    elif command == 'undo':
                        s.undo()
                        recorder.event('undo',before,capture(s) if recorder.active else None)
                        recorder.reset_motion(s)
                        dirty = True
                    elif command == 'edit':
                        s.edit(**data['edit'])
                        recorder.event('edit',before,capture(s) if recorder.active else None,{'edit':data['edit']})
                        recorder.reset_motion(s)
                        dirty = True
                    elif command == 'check':
                        await ws.send_json({'type': 'check', **s.clearance_report()})
                    elif command == 'export':
                        await ws.send_json({'type': 'export', 'document': s.export()})
                    elif command == 'load':
                        # Build first: malformed input cannot destroy a session.
                        replacement = PhysicsSession(data['document'])
                        recorder.finish(s,reason='opened_another_layout')
                        s.stop()
                        s = runtime['session'] = replacement
                        recorder.attach(s)
                        dirty = True
                        await ws.send_json(state_message(s,geometry=True))
                    elif command == 'record_start':
                        recorder.start(s)
                    elif command == 'record_finish':
                        recorder.finish(s)
                        dirty = True
                        await ws.send_json({'type':'recording_saved',**recorder.status()})
                    else:
                        raise ValueError('Unknown command.')
                    await ws.send_json(state_message(s))
                except (KeyError, ValueError, TypeError, OSError) as exc:
                    if recorder.active:
                        recorder.event('command_error',before,capture(s),
                                       {'command':data.get('type') if isinstance(data,dict) else None},str(exc))
                    await ws.send_json({'type': 'error', 'message': str(exc)})
        finally:
            ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError, ConnectionResetError):
                await ticker
            recorder.finish(s,reason='disconnected')
            s.stop()
            save()
            runtime['controller'] = None
        return ws

    app.router.add_get('/', index)
    app.router.add_get('/ws', socket)
    app.router.add_get('/recordings/{identifier}/download',download)
    app.router.add_static('/static/', STATIC)
    async def shutdown(app):
        ws = runtime['controller']
        if ws is not None:
            await ws.close()
        recorder.finish(runtime['session'],reason='server_shutdown')
        runtime['session'].stop()
        save()
    app.on_shutdown.append(shutdown)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--headless-seconds', type=float)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save', type=Path, default=ROOT / 'physics_patio/sessions/latest.json')
    args = parser.parse_args()
    session = PhysicsSession(json.loads(args.layout.read_text()), seed=args.seed)
    if args.headless_seconds is not None:
        if args.headless_seconds < 0 or args.headless_seconds > 86400 or not args.output:
            parser.error('Headless mode needs 0–86400 seconds and a new --output path.')
        if args.output.exists():
            parser.error('Output already exists; choose a new file.')
        session.start()
        remaining = args.headless_seconds
        while remaining > 1e-9:
            duration = min(1, remaining)
            session.advance(duration)
            remaining -= duration
        session.stop()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(session.export(), separators=(',', ':')))
        print(json.dumps(session.clearance_report(), indent=2))
        return
    print(f'Patio physics: http://127.0.0.1:{args.port}', flush=True)
    web.run_app(create_app(session, args.save), host='127.0.0.1', port=args.port, print=None)


if __name__ == '__main__':
    main()
