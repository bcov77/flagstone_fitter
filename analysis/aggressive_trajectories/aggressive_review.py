"""Offline arrow-key replay and a run selector, usable directly from Finder."""
import json
from pathlib import Path
from common import read,write,attempt_files,restore
from selector import apply


def render_run(out,number,images=True):
    out=Path(out);folder=out/'runs'/f'run_{number:06d}';source=read(out/'source.json');records=[read(p) for p in attempt_files(folder)]
    frames=[dict(label='Original layout',frame=__import__('common').state(source),selected=[])]
    for r in records:
        p=r['chosen'];selected=[x['stone'] for x in p['placements']] if p else []
        base=f'Attempt {r["number"]} · {p.get("family","baseline") if p else "no proposal"}'
        frames.append(dict(label=base+' · pre',frame=r['start'],selected=[]))
        if p:
            placed=__import__('common').state(apply(restore(source,r['start']),p))
            frames.append(dict(label=base+' · full-size placed',frame=placed,selected=selected,probe=p.get('probe')))
            frames.append(dict(label=base+' · physics · '+('accepted' if r['accepted'] else 'rejected: '+r['reason']),frame=r['attempted_end'],selected=selected))
        if not r['accepted']:frames.append(dict(label=base+' · rollback',frame=r['end'],selected=[]))
    payload=json.dumps(dict(source=source,frames=frames,mode=records[0]['mode'] if records else '',run=number),separators=(',',':')).replace('</',r'<\/')
    html='''<!doctype html><meta charset="utf-8"><title>Patio trajectory</title>
<style>body{font:16px system-ui;margin:18px;background:#f5f3ee;color:#292c2d}header{display:flex;gap:12px;align-items:center;flex-wrap:wrap}button,select{font:inherit;padding:6px}svg{display:block;height:77vh;max-width:100%;margin:12px auto}#info{min-height:2em}small{color:#59636a}</style>
<header><button id="prev">←</button><button id="next">→</button><select id="frames"></select></header>
<p id="info"></p><svg id="board" xmlns="http://www.w3.org/2000/svg"></svg><small>←/→ change state · orange highlights placed pieces · dashed blue shows an 80% probe. Physics endpoints use full-size stones.</small>
<script>const DATA=__DATA__;let index=0;const picker=document.getElementById('frames'),svg=document.getElementById('board');
DATA.frames.forEach((f,i)=>picker.add(new Option(f.label,i)));
function polygon(points,fill,stroke,dash=''){const p=document.createElementNS(svg.namespaceURI,'polygon');p.setAttribute('points',points.map(p=>p.join(',')).join(' '));p.setAttribute('fill',fill);p.setAttribute('stroke',stroke);p.setAttribute('stroke-width','.008');if(dash)p.setAttribute('stroke-dasharray',dash);svg.append(p)}
function draw(){const f=DATA.frames[index];picker.value=index;svg.replaceChildren();const xs=DATA.source.patio.map(p=>p[0]),ys=DATA.source.patio.map(p=>-p[1]);let x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x,h=Math.max(...ys)-y;svg.setAttribute('viewBox',`${x-.1} ${y-.1} ${w+.2} ${h+.2}`);polygon(DATA.source.patio.map(([x,y])=>[x,-y]),'#fff','#38434a');
for(const s of DATA.source.stones){let pose=f.frame.poses[String(s.id)];if(!pose)continue;let [x,y,a]=pose,c=Math.cos(a),t=Math.sin(a),selected=f.selected.includes(String(s.id));const points=s.outline.map(([u,v])=>[x+c*u-t*v,-y-t*u-c*v]);polygon(points,selected?'#eea547':'#d9d4c8',selected?'#a45700':'#666d6a');if(selected&&f.probe){let factor=f.probe.scale||.8;polygon(s.outline.map(([u,v])=>[x+factor*(c*u-t*v),-y-factor*(t*u+c*v)]),'none','#1679a6','.04 .025')}
const text=document.createElementNS(svg.namespaceURI,'text');text.setAttribute('x',x);text.setAttribute('y',-y);text.setAttribute('font-size','.075');text.setAttribute('text-anchor','middle');text.textContent=s.id;svg.append(text)}
document.getElementById('info').textContent=`Run ${DATA.run} · ${DATA.mode} · ${index+1}/${DATA.frames.length} · ${f.label}`;document.getElementById('prev').disabled=index===0;document.getElementById('next').disabled=index===DATA.frames.length-1;}
function move(delta){index=Math.max(0,Math.min(DATA.frames.length-1,index+delta));draw()}document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);picker.onchange=()=>{index=+picker.value;draw()};window.addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();move(e.key==='ArrowRight'?1:-1)}});window.addEventListener('message',e=>{if(e.data==='patio-next')move(1);if(e.data==='patio-prev')move(-1)});draw();</script>'''.replace('__DATA__',payload)
    (folder/'viewer.html').write_text(html)
    if images:
        from report import render_run as old_render
        old_render(out,number)


def render_index(out):
    out=Path(out);entries=[]
    for folder in (out/'runs').glob('run_*'):
        if not (folder/'viewer.html').exists():continue
        n=int(folder.name.split('_')[1]);mode='golden' if n%10==9 else 'silver' if n%10==0 else 'normal'
        complete=(folder/'summary.json').exists();entries.append(({'golden':0,'silver':1,'normal':2}[mode],n,mode,folder.name,complete))
    options=''.join(f'<option value="runs/{name}/viewer.html">{mode.title()} {n}'+('' if complete else ' · partial')+'</option>' for _,n,mode,name,complete in sorted(entries))
    html='''<!doctype html><meta charset="utf-8"><title>Aggressive patio runs</title><style>body{font:16px system-ui;margin:12px;background:#f5f3ee}select{font:inherit;padding:5px}iframe{width:100%;height:92vh;border:0}</style><label>Run <select id="runs">__OPTIONS__</select></label><iframe id="viewer" title="Trajectory viewer"></iframe><script>const s=document.getElementById('runs'),f=document.getElementById('viewer');s.onchange=()=>{f.src=s.value};if(s.options.length)s.onchange();window.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();f.contentWindow.postMessage(e.key==='ArrowRight'?'patio-next':'patio-prev','*')}})</script>'''.replace('__OPTIONS__',options)
    path=out/'index.html';tmp=out/'index.tmp';tmp.write_text(html);tmp.replace(path)
