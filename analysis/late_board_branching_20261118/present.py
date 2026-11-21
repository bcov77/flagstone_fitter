"""Offline replay and aggregate summaries for saved branching trials."""
import collections,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from experiment import HERE,CASES,read,restore,apply,arm,sha,manifest

def main():
 data=[];rows=[]
 for number in CASES:
  folder=HERE/f'run_{number:06d}'
  if not (folder/'source.json').exists():continue
  source=read(folder/'source.json');root=read(folder/'root_metrics.json');items={}
  for path in sorted(folder.glob('*.json.gz')):
   if 'pool' in path.name or 'portfolio' in path.name or path.name=='pareto_archive.json.gz':continue
   r=read(path)
   if 'accepted' not in r or 'id' not in r:continue
   items[r['id']]=r
  trials=[]
  for r in items.values():
   p=r['action'];family=r.get('operator') or (arm(p) if p else 'quiet_control');start_metrics=root if r['parent'] is None else items[r['parent']]['end_metrics']
   path_chain=[];current=r
   while current:
    path_chain.append(current);current=items.get(current['parent'])
   path_chain.reverse();frames=[dict(label='Saved root',frame=__import__('experiment').state(source),metrics=root,selected=[])]
   for item in path_chain:
    act=item['action'];selected=[x['stone'] for x in act['placements']] if act else item.get('movable_ids',[])
    frames.append(dict(label=item['id']+' · pre',frame=item['start'],selected=[],metrics=root if item['parent'] is None else items[item['parent']]['end_metrics']))
    if act:frames.append(dict(label=item['id']+' · proposed (not yet physics-validated)',frame=__import__('experiment').state(apply(restore(source,item['start']),act)),selected=selected,metrics=None))
    frames.append(dict(label=item['id']+' · physics · '+('accepted' if item['accepted'] else 'rejected: '+item['reason']),frame=item['attempted_end'],selected=selected,metrics=item['attempted_metrics']))
    if not item['accepted']:frames.append(dict(label=item['id']+' · rollback',frame=item['end'],selected=[],metrics=item['end_metrics']))
   m=r['end_metrics'];row=dict(stage='diagnostic' if r['id'].startswith('direct_') else 'followup' if r['parent'] else 'fine_grid' if r['id'].startswith('fine_insert_') else 'two_for_two' if r['id'].startswith('repack_') else 'local_mobility' if r.get('operator')=='local_agitation' else 'root',run=number,id=r['id'],parent=r['parent'],seed=r['seed'],arm=family,accepted=r['accepted'],reason=r['reason'],coverage=m['area_percent'],gain_from_root=m['area_percent']-root['area_percent'],gain_from_parent=m['area_percent']-start_metrics['area_percent'],interior=m['gaps']['largest_interior_inches'],edge=m['gaps']['largest_edge_inches'],broad_void=m['quality']['broad_void_sqft'],wall_joint=m['quality']['wall_joint_inches'],seconds=r['seconds'])
   if r.get('local_before'):row.update(local_void_change=r['local_end']['broad_void_sqft']-r['local_before']['broad_void_sqft'],local_stone_area_gain=r['local_end']['stone_area_sqft']-r['local_before']['stone_area_sqft'])
   rows.append(row);trials.append(dict(**row,frames=frames))
  data.append(dict(run=number,source=source,root=root,trials=trials))
 groups=collections.defaultdict(list)
 for r in rows:groups[(r['stage'],r['arm'])].append(r)
 summary=dict(cases=[dict(run=d['run'],root=d['root']['area_percent'],best=max([d['root']['area_percent']]+[r['coverage'] for r in d['trials'] if r['accepted']]),trials=len(d['trials'])) for d in data],groups=[dict(depth=k[0],arm=k[1],trials=len(v),accepted=sum(r['accepted'] for r in v)) for k,v in sorted(groups.items())],rows=rows)
 (HERE/'results.json').write_text(json.dumps(summary,indent=2))
 assets=lambda d:__import__('experiment').canonical(dict(patio=d['source']['patio'],stones=[dict(id=s['id'],outline=s['outline']) for s in d['source']['stones']]))
 assert len({assets(d) for d in data})<=1,'Sources differ in geometry; cannot share viewer assets'
 payload=json.dumps(dict(source=data[0]['source'] if data else None,cases=[{k:v for k,v in d.items() if k!='source'} for d in data]),separators=(',',':')).replace('</','<\\/')
 html='''<!doctype html><html><meta charset="utf-8"><title>Late patio board experiments</title><style>body{font:15px system-ui;background:#f5f3ee;color:#292c2d;margin:18px}header{display:flex;align-items:center;gap:10px;flex-wrap:wrap}button,select{font:inherit;padding:6px}h1{font-size:22px}svg{display:block;height:70vh;max-width:100%;margin:auto}#metrics{font-weight:600}#note{max-width:1000px;font-size:13px;color:#59636a}#state{max-width:70vw}#trial{max-width:80vw}</style><h1>Late patio board experiments</h1><p id="note">Saved 79–82% boards · full-size stone outlines · quiet-only controls · two physics seeds · selective two-move branches. Rejected endpoints are shown for diagnosis and rolled back. Proposed states have not passed physics checks.</p><header><label>Board <select id="case"></select></label><label>Trial <select id="trial"></select></label></header><p id="metrics"></p><header><button id="prev">←</button><button id="next">→</button><select id="state"></select></header><p id="info"></p><svg id="board" xmlns="http://www.w3.org/2000/svg"></svg><p>Orange: deliberately placed pieces. ←/→ changes replay state.</p><script>const PAYLOAD=__DATA__;const DATA=PAYLOAD.cases;DATA.forEach(d=>d.source=PAYLOAD.source);const cases=document.getElementById('case'),trials=document.getElementById('trial'),states=document.getElementById('state'),svg=document.getElementById('board');let ci=0,ti=0,fi=0;DATA.forEach((d,i)=>cases.add(new Option(`Run ${d.run} · saved ${d.root.area_percent.toFixed(3)}%`,i)));function polygon(points,fill,stroke){let p=document.createElementNS(svg.namespaceURI,'polygon');p.setAttribute('points',points.map(p=>p.join(',')).join(' '));p.setAttribute('fill',fill);p.setAttribute('stroke',stroke);p.setAttribute('stroke-width','.008');svg.append(p)}function loadCase(){ci=+cases.value;trials.replaceChildren();DATA[ci].trials.forEach((r,i)=>trials.add(new Option(`${r.id} · ${r.accepted?'accepted':'rollback'} · ${r.coverage.toFixed(3)}%`,i)));loadTrial()}function loadTrial(){ti=+trials.value;fi=0;states.replaceChildren();let r=DATA[ci].trials[ti];if(!r)return;r.frames.forEach((f,i)=>states.add(new Option(f.label,i)));draw()}function draw(){let d=DATA[ci],r=d.trials[ti],f=r.frames[fi];states.value=fi;svg.replaceChildren();let xs=d.source.patio.map(p=>p[0]),ys=d.source.patio.map(p=>-p[1]),x=Math.min(...xs),y=Math.min(...ys);svg.setAttribute('viewBox',`${x-.1} ${y-.1} ${Math.max(...xs)-x+.2} ${Math.max(...ys)-y+.2}`);polygon(d.source.patio.map(([x,y])=>[x,-y]),'white','#38434a');for(let s of d.source.stones){let pose=f.frame.poses[String(s.id)];if(!pose)continue;let[x,y,a]=pose,c=Math.cos(a),t=Math.sin(a),sel=f.selected.includes(String(s.id));polygon(s.outline.map(([u,v])=>[x+c*u-t*v,-y-t*u-c*v]),sel?'#eea547':'#d9d4c8',sel?'#a45700':'#666d6a');let tx=document.createElementNS(svg.namespaceURI,'text');tx.setAttribute('x',x);tx.setAttribute('y',-y);tx.setAttribute('font-size','.075');tx.setAttribute('text-anchor','middle');tx.textContent=s.id;svg.append(tx)}document.getElementById('metrics').textContent=`Saved root: ${d.root.area_percent.toFixed(3)}% · Branch endpoint: ${r.coverage.toFixed(3)}% · Gain: ${r.gain_from_root.toFixed(3)} points`;document.getElementById('info').textContent=`${fi+1}/${r.frames.length} · ${f.label}`+(f.metrics?` · Coverage ${f.metrics.area_percent.toFixed(3)}% · Interior gap ${f.metrics.gaps.largest_interior_inches.toFixed(2)} in · Edge gap ${f.metrics.gaps.largest_edge_inches.toFixed(2)} in`:'');document.getElementById('prev').disabled=fi===0;document.getElementById('next').disabled=fi===r.frames.length-1}function move(n){fi=Math.max(0,Math.min(DATA[ci].trials[ti].frames.length-1,fi+n));draw()}cases.onchange=loadCase;trials.onchange=loadTrial;states.onchange=()=>{fi=+states.value;draw()};document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);window.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();move(e.key==='ArrowRight'?1:-1)}});if(DATA.length)loadCase();</script></html>'''.replace('__DATA__',payload)
 (HERE/'index.html').write_text(html)
 print(json.dumps({k:summary[k] for k in ['cases','groups']},indent=2))
if __name__=='__main__':main()
