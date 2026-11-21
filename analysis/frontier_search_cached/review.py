"""Offline branch replay; shared full-outline assets and persistent endpoint %s."""
from core import *
from html import escape

VIEWER='''<!doctype html><meta charset="utf-8"><title>Frontier branch</title><style>body{font:15px system-ui;margin:15px;background:#f5f3ee;color:#292c2d}header{display:flex;gap:9px;align-items:center;flex-wrap:wrap}button,select{font:inherit;padding:6px}svg{display:block;height:72vh;max-width:100%;margin:auto}#endpoint{font-weight:650}small{color:#59636a}</style><p id="endpoint"></p><header><button id="prev">←</button><button id="next">→</button><select id="frames"></select></header><p id="info"></p><svg id="board" xmlns="http://www.w3.org/2000/svg"></svg><small>Orange: selected pieces (or mobile neighbors during a physics-only pulse). Proposed and rejected states have not passed acceptance. ←/→ change state.</small><script src="../../assets.js"></script><script>const DATA=__DATA__,SOURCE=window.PATIO_SOURCE;let index=0;const picker=document.getElementById('frames'),svg=document.getElementById('board');DATA.frames.forEach((f,i)=>picker.add(new Option(f.label,i)));function polygon(points,fill,stroke){let p=document.createElementNS(svg.namespaceURI,'polygon');p.setAttribute('points',points.map(p=>p.join(',')).join(' '));p.setAttribute('fill',fill);p.setAttribute('stroke',stroke);p.setAttribute('stroke-width','.008');svg.append(p)}function draw(){const f=DATA.frames[index];picker.value=index;svg.replaceChildren();let xs=SOURCE.patio.map(p=>p[0]),ys=SOURCE.patio.map(p=>-p[1]),x=Math.min(...xs),y=Math.min(...ys);svg.setAttribute('viewBox',`${x-.1} ${y-.1} ${Math.max(...xs)-x+.2} ${Math.max(...ys)-y+.2}`);polygon(SOURCE.patio.map(([x,y])=>[x,-y]),'white','#38434a');for(let s of SOURCE.stones){let pose=f.frame.poses[String(s.id)];if(!pose)continue;let[x,y,a]=pose,c=Math.cos(a),t=Math.sin(a),selected=f.selected.includes(String(s.id));polygon(s.outline.map(([u,v])=>[x+c*u-t*v,-y-t*u-c*v]),selected?'#eea547':'#d9d4c8',selected?'#a45700':'#666d6a');let tx=document.createElementNS(svg.namespaceURI,'text');tx.setAttribute('x',x);tx.setAttribute('y',-y);tx.setAttribute('font-size','.075');tx.setAttribute('text-anchor','middle');tx.textContent=s.id;svg.append(tx)}document.getElementById('endpoint').textContent=`Root ${DATA.root} · Branch endpoint completion: ${DATA.completion.toFixed(3)}% · ${DATA.status}`;document.getElementById('info').textContent=`${index+1}/${DATA.frames.length} · ${f.label}`+(f.metrics?` · ${f.metrics.area_percent.toFixed(3)}% · interior ${f.metrics.gaps.largest_interior_inches?.toFixed(2)??'audit unavailable'} in · edge ${f.metrics.gaps.largest_edge_inches?.toFixed(2)??'audit unavailable'} in`:'');document.getElementById('prev').disabled=index===0;document.getElementById('next').disabled=index===DATA.frames.length-1}function move(n){index=Math.max(0,Math.min(DATA.frames.length-1,index+n));draw()}document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);picker.onchange=()=>{index=+picker.value;draw()};window.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();move(e.key==='ArrowRight'?1:-1)}});window.addEventListener('message',e=>{if(e.data==='patio-next')move(1);if(e.data==='patio-prev')move(-1)});draw();</script>'''


def node_url(node):return f'jobs/job_{node["job"]:08d}/viewer.html' if node.get('job') else f'seeds/{node["id"]}/viewer.html'


def frames_for(node,nodes,jobs,results):
 original=next(n for n in nodes.values() if n['root']==node['root'] and n['kind']=='original');frames=[dict(label=f'Saved root {node["root"]}',frame=original['frame'],metrics=original['metrics'],selected=[])]
 chain=[];current=node
 while current:
  chain.append(current);current=nodes.get(current.get('parent'))
 for n in reversed(chain):
  if n['kind']=='original':continue
  if n['kind']=='pilot_seed':frames.append(dict(label='Verified pilot starting endpoint',frame=n['frame'],metrics=n['metrics'],selected=[]));continue
  r=results[n['job']];append_trial(frames,jobs[n['job']],r,nodes)
 return frames


def append_trial(frames,meta,r,nodes):
 parent=nodes[meta['parent']];p=r['chosen'];selected=p.get('movable_ids') if p.get('operator')=='local_pulse' else [x['stone'] for x in p['placements']];label=f'Job {meta["number"]} · {meta["operator"]}'
 frames.append(dict(label=label+' · pre',frame=r['start'],metrics=parent['metrics'],selected=[]))
 frames.append(dict(label=label+' · proposed (unvalidated)',frame=r['proposed_start'],metrics=None,selected=selected))
 frames.append(dict(label=label+' · physics · '+('accepted' if r['accepted'] else 'rejected: '+r['reason']),frame=r['attempted_end'],metrics=r['attempted_metrics'],selected=selected))
 if not r['accepted']:frames.append(dict(label=label+' · rollback',frame=r['end'],metrics=parent['metrics'],selected=[]))


def write_view(path,root,frames,completion,status):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);payload=json.dumps(dict(root=root,frames=frames,completion=completion,status=status),separators=(',',':')).replace('</','<\\/');tmp=path.with_suffix('.tmp');tmp.write_text(VIEWER.replace('__DATA__',payload));tmp.replace(path)


def render(out,nodes,jobs,results,value,export=False):
 out=Path(out)
 for n in nodes.values():
  if n.get('job'):continue
  path=out/node_url(n)
  if not path.exists():write_view(path,n['root'],frames_for(n,nodes,jobs,results),n['metrics']['area_percent'],'original' if n['kind']=='original' else 'pilot seed')
 for number,r in results.items():
  if not r.get('physics_evaluated'):continue
  path=out/'jobs'/f'job_{number:08d}'/'viewer.html'
  if path.exists():continue
  meta=jobs[number];frames=frames_for(nodes[meta['parent']],nodes,jobs,results);append_trial(frames,meta,r,nodes)
  write_view(path,meta['root'],frames,r['end_metrics']['area_percent'],'accepted' if r['accepted'] else 'rolled back')
 data=[]
 for row in value['roots']:
  root=row['root'];root_nodes=[n for n in nodes.values() if n['root']==root];best=nodes[row['best_node']];gap=nodes[row['gap_node']];wall=min(root_nodes,key=lambda n:axes(n['metrics'])[4]);options=[];seen=set()
  for title,n in [('Best coverage',best),('Best interior gap',gap),('Best wall fit',wall)]+[('Retained branch',n) for n in reversed(root_nodes) if n['id'] in row['frontier']]:
   url=node_url(n)
   if url in seen:continue
   seen.add(url);options.append(dict(url=url,label=f'{title} · {n["id"]} · {n["metrics"]["area_percent"]:.3f}%'))
  recent=[(number,r) for number,r in results.items() if jobs[number]['root']==root and r.get('physics_evaluated')][-80:]
  for number,r in reversed(recent):
   url=f'jobs/job_{number:08d}/viewer.html'
   if url not in seen:options.append(dict(url=url,label=f'Job {number} · {jobs[number]["operator"]} · {"accepted" if r["accepted"] else "rollback"} · {r["end_metrics"]["area_percent"]:.3f}%'));seen.add(url)
  data.append(dict(root=root,label=f'Root {root} · saved {row["original"]:.3f}% · best {row["best"]:.3f}%',options=options,stats=f'{row["physics_attempts"]} physics attempts · {row["accepted"]} accepted · {row["empty_searches"]} empty searches'))
  if export:
   # Expensive complete layouts are rewritten only when the champion changes.
   for label,n in [('best_coverage',best),('best_interior_gap',gap),('best_wall_fit',wall)]:
    marker=out/'roots'/f'run_{root:06d}'/(label+'_node.json')
    if marker.exists() and read(marker)['id']==n['id']:continue
    source=read(out/'source.json');write(marker.parent/(label+'.json'),restore(source,n['frame']));write(marker,dict(id=n['id'],metrics=n['metrics']))
 payload=json.dumps(data,separators=(',',':')).replace('</','<\\/')
 html='''<!doctype html><meta charset="utf-8"><title>Patio frontier search</title><style>body{font:15px system-ui;background:#f5f3ee;margin:12px;color:#292c2d}h1{font-size:21px}header{display:flex;gap:12px;flex-wrap:wrap;align-items:center}select{font:inherit;padding:6px;max-width:90vw}iframe{width:100%;height:84vh;border:0}small{color:#59636a}</style><h1>Patio frontier search</h1><header><label>Starting board <select id="roots"></select></label><label>Branch <select id="branches"></select></label></header><p id="stats"></p><small>Refresh this page for new results. Every branch preserves the original full-size inventory. Empty searches are separate from physical attempts. Recent trials and retained frontier branches are listed; every physical job has its own saved viewer.</small><iframe id="view" title="Frontier trajectory"></iframe><script>const DATA=__DATA__,roots=document.getElementById('roots'),branches=document.getElementById('branches'),view=document.getElementById('view');DATA.forEach((d,i)=>roots.add(new Option(d.label,i)));function load(){const d=DATA[+roots.value];branches.replaceChildren();d.options.forEach(o=>branches.add(new Option(o.label,o.url)));document.getElementById('stats').textContent=d.stats;select()}function select(){if(branches.options.length)view.src=branches.value}roots.onchange=load;branches.onchange=select;window.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();view.contentWindow.postMessage(e.key==='ArrowRight'?'patio-next':'patio-prev','*')}});if(DATA.length)load();</script>'''.replace('__DATA__',payload)
 tmp=out/'index.tmp';tmp.write_text(html);tmp.replace(out/'index.html')
