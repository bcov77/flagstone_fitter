"""Foreground supervisor for eight new-policy and two historical workers."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'physics_trajectories'))
from common import read,write
from aggressive_review import render_run,render_index


def relay(proc,label):
    for line in proc.stdout:print(f'[{label}] {line.rstrip()}',flush=True)


def index(out):
    options=[]
    for track in ['aggressive','baseline']:
        for p in (out/track/'runs').glob('run_*/viewer.html'):
            n=int(p.parent.name.split('_')[1]);mode='Golden' if n%10==9 else 'Silver' if n%10==0 else 'Normal'
            label=f'{track.title()} · {mode} {n}'+('' if (p.parent/'summary.json').exists() else ' · partial')
            options.append(({'Golden':0,'Silver':1,'Normal':2}[mode],track,n,f'<option value="{p.relative_to(out)}">{label}</option>'))
    html='''<!doctype html><meta charset="utf-8"><title>Overnight patio runs</title><style>body{font:16px system-ui;margin:12px;background:#f5f3ee}select{font:inherit;padding:6px}iframe{width:100%;height:91vh;border:0}</style><label>Run <select id="runs">__OPTIONS__</select></label><iframe id="view" title="Trajectory"></iframe><script>const s=document.getElementById('runs'),f=document.getElementById('view');s.onchange=()=>f.src=s.value;if(s.options.length)s.onchange();window.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();f.contentWindow.postMessage(e.key==='ArrowRight'?'patio-next':'patio-prev','*')}})</script>'''.replace('__OPTIONS__',''.join(v[3] for v in sorted(options)))
    tmp=out/'index.tmp';tmp.write_text(html);tmp.replace(out/'index.html')


def status(out):
    result={}
    for track in ['aggressive','baseline']:
        completed=list((out/track/'runs').glob('run_*/summary.json'));progress=[]
        for p in (out/track/'runs').glob('run_*/progress.json'):
            if not (p.parent/'summary.json').exists():progress.append(read(p))
        rows=[read(p) for p in completed]
        result[track]=dict(completed=len(completed),attempts=sum(r['attempts'] for r in rows),accepted=sum(r['accepted'] for r in rows),partial=progress)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=ROOT/'examples/aggressive_trajectories_20261118_r2')
    ap.add_argument('--workers',type=int,default=8,help='New-policy workers');ap.add_argument('--baseline-workers',type=int,default=2)
    ap.add_argument('--runs',type=int,help='Finite new-policy trajectory count');ap.add_argument('--baseline-runs',type=int)
    ap.add_argument('--config',type=Path,default=HERE/'config.json');ap.add_argument('--source',type=Path,default=ROOT/'examples/quarter2_20h_ready_20261114/tasks/task_576_escape/best.json')
    ap.add_argument('--status',action='store_true');ap.add_argument('--stop',action='store_true');args=ap.parse_args();out=args.out.resolve()
    if args.status:
        import json
        print(json.dumps(status(out),indent=2));return
    if args.stop:
        if not out.exists():ap.error('Output folder does not exist')
        (out/'STOP').touch()
        for track in ['aggressive','baseline']:
            if (out/track).exists():(out/track/'STOP').touch()
        print('Stop requested; active attempts will checkpoint.');return
    if args.workers<1 or args.baseline_workers<0 or any(v is not None and v<1 for v in [args.runs,args.baseline_runs]):ap.error('Positive counts required; baseline workers may be zero')
    if args.baseline_runs and not args.baseline_workers:ap.error('--baseline-runs requires baseline workers')
    if out==ROOT or out in args.source.resolve().parents:ap.error('Use a separate output directory')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.supervisor.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This overnight batch already has a foreground supervisor')
        (out/'STOP').unlink(missing_ok=True);stopping=False;children=[];threads=[];seen=set();failed=False
        def stop(*_):
            nonlocal stopping
            if not stopping:print('\nStopping both policies after their current attempts; please wait for checkpoints.',flush=True)
            stopping=True;(out/'STOP').touch()
        previous={s:signal.signal(s,stop) for s in [signal.SIGINT,signal.SIGTERM]}
        try:
            tracks=[('aggressive',args.workers,HERE/'run.py',args.config,args.runs)]
            if args.baseline_workers:tracks.append(('baseline',args.baseline_workers,HERE.parent/'physics_trajectories/run.py',HERE.parent/'physics_trajectories/config.json',args.baseline_runs or (max(1,math.ceil(args.runs/4)) if args.runs else None)))
            for label,workers,script,config,runs in tracks:
                dest=out/label;dest.mkdir(exist_ok=True)
                command=[sys.executable,str(script),'--out',str(dest),'--workers',str(workers),'--config',str(config),'--source',str(args.source)]
                if runs is not None:command+=['--runs',str(runs)]
                proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
                children.append((label,proc));t=threading.Thread(target=relay,args=(proc,label),daemon=True);t.start();threads.append(t)
            print(f'Foreground batch: {out}\nNew policy: {args.workers} workers; baseline: {args.baseline_workers}. Ctrl-C checkpoints both.\nViewer: {out}/index.html',flush=True)
            while any(p.poll() is None for _,p in children):
                if (out/'STOP').exists():stop()
                if any(p.poll() not in (None,0) for _,p in children):failed=True;stop()
                if stopping:
                    for label,_ in children:(out/label/'STOP').touch()
                # Historical runner writes PNGs; add the same offline replay.
                for label,_ in children:
                    for p in (out/label/'runs').glob('run_*/summary.json'):
                        if str(p) in seen:continue
                        if label=='baseline':render_run(out/label,int(p.parent.name.split('_')[1]),images=False)
                        seen.add(str(p))
                index(out);time.sleep(1)
            for t in threads:t.join(timeout=2)
            # Include the final completions/checkpointed prefixes in the viewer.
            for label,_ in children:
                for folder in (out/label/'runs').glob('run_*'):
                    if list(folder.glob('attempt_*.json.gz')):render_run(out/label,int(folder.name.split('_')[1]),images=False)
                render_index(out/label)
            index(out);write(out/'status.json',status(out))
            if failed or any(p.returncode for _,p in children):raise RuntimeError('A policy process failed. Both were stopped; inspect the foreground traceback before resuming.')
            print('All workers exited. Run the same command to resume checkpointed trajectories.',flush=True)
        finally:
            for label,proc in children:
                if proc.poll() is None:
                    (out/label/'STOP').touch()
            # A rendering/error exit must not orphan an overnight job after the
            # foreground supervisor disappears. Keep signal handlers installed
            # while children finish their transaction and release output locks.
            while any(proc.poll() is None for _,proc in children):
                for label,proc in children:
                    if proc.poll() is None:(out/label/'STOP').touch()
                time.sleep(1)
            for s,handler in previous.items():signal.signal(s,handler)


if __name__=='__main__':main()
