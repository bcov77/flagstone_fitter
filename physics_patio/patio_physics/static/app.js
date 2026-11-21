'use strict';
const $ = id => document.getElementById(id);
let ws, state, selected = null, scale = 50, fieldOrigin = [0, 0], inventoryItems = [];
let geometry = new Map(), dragging = null, holding = false, paintPending = false;
const colors = ['#adc3ad','#d4bca1','#a8bdc8','#d0b6a7','#c5bd91','#b6b6cc','#a4c5be','#ccb4bf'];
const names = {
  gap_inches:'Simulation gap',
  piece_hardness:'Piece edge hardness', locked_hardness:'Locked piece edge hardness',
  angular_stiffness:'Angular constraint stiffness', translation_stiffness:'Translational constraint stiffness',
  rotation_magnitude:'Rotational magnitude', rotation_force:'Rotational force', rotation_frequency:'Rotational frequency',
  translation_magnitude:'Translational magnitude', translation_force:'Translational force', translation_frequency:'Translational frequency',
  speed:'Simulation speed', wall_hardness:'Patio edge hardness'
};
function send(data) { if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(data)); }
function message(text, error=false) { $('message').textContent=text; $('message').classList.toggle('error',error); }
function updateRecording(recording) {
  if(!recording)return;
  $('recordStart').disabled=recording.active;
  $('recordFinish').disabled=!recording.active;
  $('recordStatus').textContent=recording.error?'Recording error':recording.active?`● Recording · ${recording.events} events · ${recording.simulated_seconds.toFixed(1)} s`:'Not recording';
  $('recordStatus').classList.toggle('recording',recording.active);
  $('recordDownload').hidden=!recording.download_url;
  if(recording.download_url)$('recordDownload').href=recording.download_url;
}
function connect() {
  ws = new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`);
  ws.onmessage = event => {
    const data=JSON.parse(event.data);
    if(data.type==='recording_saved') {
      updateRecording(data);holding=false;
      if(data.download_url){const a=document.createElement('a');a.href=data.download_url;a.download='';a.click();}
      message('Trajectory saved. The ZIP includes replay.html, every edit, physics frames and your final layout.');return;
    }
    if (data.type==='error') { message(data.message,true); return; }
    if (data.type==='export') {
      const url=URL.createObjectURL(new Blob([JSON.stringify(data.document)],{type:'application/json'}));
      const a=document.createElement('a'); a.href=url; a.download=`patio-physics-${new Date().toISOString().replaceAll(':','-')}.json`; a.click();
      setTimeout(()=>URL.revokeObjectURL(url),1000); message('Saved original outlines, poses, locks, anchors, and settings.'); return;
    }
    if (data.type==='check') {
      message(data.passes_clearance ? 'Original outlines pass the ½″ pair and boundary clearance checks. Maximum-gap goal not checked.' : `${data.errors.length} clearance violations on original outlines. Soft contacts may need more settling. Maximum-gap goal not checked.`, !data.passes_clearance);
      return;
    }
    if (data.type==='init') {
      geometry.clear(); selected=null;
      for (const p of data.pieces) {
        const path=new Path2D(); p.outline.forEach(([x,y],i)=> i ? path.lineTo(x,y) : path.moveTo(x,y)); path.closePath();
        const xs=p.outline.map(v=>v[0]), ys=p.outline.map(v=>v[1]);
        geometry.set(p.id,{path,outline:p.outline,bounds:[Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)]});
      }
      buildSliders(data.controls); message('Ready. Drag a reserve stone into place, lock it, then hold space to jiggle.');
      $('connection').textContent='Pymunk connected';
    }
    const prior=state; state=data;
    updateRecording(data.recording);
    // Compact state updates omit immutable geometry/patio and controls.
    if (data.type!=='init' && prior) state.patio=prior.patio;
    if (!state.patio) return;
    updateUI(); schedulePaint();
  };
  ws.onclose=()=>{holding=false; $('connection').textContent='Disconnected'; $('mode').textContent='Offline';document.body.classList.remove('running');message('Backend disconnected. Reload after restarting the local server. Another open tab may already control it.',true);};
  ws.onerror=()=>message('Cannot connect to the local backend.',true);
}
function buildSliders(controls) {
  $('sliders').replaceChildren();
  const groups=[['Contacts',['gap_inches','piece_hardness','locked_hardness','wall_hardness']],['Tethers',['angular_stiffness','translation_stiffness']],['Rotation',['rotation_magnitude','rotation_force','rotation_frequency']],['Translation',['translation_magnitude','translation_force','translation_frequency']],['Playback',['speed']]];
  for (const [title,keys] of groups) {
    const group=document.createElement('section');group.className='control-group'; const heading=document.createElement('h3');heading.textContent=title;group.append(heading);
    for (const key of keys) {
      if (!controls[key]) continue; // Allow a page refresh before backend restart.
      const [value,min,max,step,unit]=controls[key],label=document.createElement('label');label.className='slider';
      const row=document.createElement('div'),text=document.createElement('span'),output=document.createElement('output'),input=document.createElement('input');
      text.textContent=names[key];output.id=`value-${key}`;input.type='range';input.min=min;input.max=max;input.step=step;input.value=value;input.id=key;input.setAttribute('aria-label',names[key]);
      input.dataset.unit=unit;row.append(text,output);label.append(row,input);group.append(label);
      input.addEventListener('input',()=>{output.textContent=`${Number(input.value).toFixed(step<1?2:0)} ${unit}`;});
      input.addEventListener('change',()=>send({type:'edit',edit:{action:'settings',values:{[key]:Number(input.value)}}}));
    }
    $('sliders').append(group);
  }
}
function updateUI() {
  const count=state.pieces.filter(p=>p.active).length;
  $('placed').textContent=count; $('available').textContent=state.pieces.length-count;
  $('gapLabel').textContent=`Simulation gap ${(state.settings.gap_inches ?? .5).toFixed(2)}″ · required ½″`;
  document.body.classList.toggle('running',state.running);$('mode').textContent=state.running?'Jiggling':'Paused';$('simTime').textContent=`${state.elapsed.toFixed(2)} s`;
  $('undo').disabled=!state.undo_count;$('undoCount').textContent=state.undo_count;$('undo').title=state.undo_label?`Undo: ${state.undo_label}`:'Nothing to undo';
  for (const [key,value] of Object.entries(state.settings)) {
    const input=$(key); if (!input) continue;
    if (document.activeElement!==input) input.value=value;
    $(`value-${key}`).textContent=`${Number(input.value).toFixed(Number(input.step)<1?2:0)} ${input.dataset.unit}`;
  }
  const p=state.pieces.find(p=>p.id===selected),onField=p?.active;
  $('selectedLabel').textContent=p?`Stone ${p.id}${p.locked?' · locked':''}`:'Choose a stone';
  for(const id of ['ccw','cw','angle','lock','delete']) $(id).disabled=!onField;
  if(document.activeElement!==$('angle')) $('angle').value=onField?(p.pose[2]*180/Math.PI).toFixed(1):'';
  $('lock').textContent=p?.locked?'Unlock position':'Lock position';
}
function prepare(canvas) {
  const width=canvas.clientWidth,height=canvas.clientHeight,dpr=window.devicePixelRatio||1;
  if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);}
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);return ctx;
}
function pieceColor(id) { return colors[[...id].reduce((n,c)=>n+c.charCodeAt(0),0)%colors.length]; }
function drawPiece(ctx,p,pose,origin,selectedHere=false) {
  const geo=geometry.get(p.id);if(!geo)return;
  ctx.save();ctx.translate(origin[0]+pose[0]*scale,origin[1]-pose[1]*scale);ctx.scale(scale,-scale);ctx.rotate(pose[2]);
  ctx.fillStyle=pieceColor(p.id);ctx.strokeStyle=selectedHere?'#274f3d':p.locked?'#936d35':'#6d786b';ctx.lineWidth=(selectedHere?2.8:p.locked?2:1)/scale;
  ctx.fill(geo.path);ctx.stroke(geo.path);ctx.restore();
  const x=origin[0]+pose[0]*scale,y=origin[1]-pose[1]*scale;
  ctx.fillStyle='#293e36';ctx.font=`${Math.max(9,Math.min(13,scale*.20))}px -apple-system,sans-serif`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(p.id,x,y);
  if(p.locked){ctx.font='10px sans-serif';ctx.fillStyle='#805d2c';ctx.fillText('◆',x,y+15);}
}
function layoutBoards() {
  const canvas=$('field'),w=canvas.clientWidth,h=canvas.clientHeight,invW=$('inventoryScroll').clientWidth;
  const xs=state.patio.map(p=>p[0]),ys=state.patio.map(p=>p[1]);const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
  $('fieldDimensions').textContent=`${(maxX-minX).toFixed(2)} × ${(maxY-minY).toFixed(2)} ft`;
  const widest=Math.max(...[...geometry.values()].map(g=>g.bounds[2]-g.bounds[0]));
  scale=Math.min((w-60)/(maxX-minX),(h-60)/(maxY-minY),(invW-36)/widest);
  fieldOrigin=[(w-(maxX-minX)*scale)/2-minX*scale,(h+(maxY-minY)*scale)/2+minY*scale];
  $('scaleLabel').textContent=`1 ft = ${scale.toFixed(0)} screen px`;
  const filter=$('filter').value.trim().toLowerCase();let x=18,y=15,rowHeight=0;inventoryItems=[];
  for(const p of state.pieces){
    if(p.active||!p.id.toLowerCase().includes(filter))continue;
    const [a,b,c,d]=geometry.get(p.id).bounds,pw=(c-a)*scale,ph=(d-b)*scale;
    if(x+pw>invW-18&&x>18){x=18;y+=rowHeight+36;rowHeight=0;}
    inventoryItems.push({p,x:x-a*scale,y:y+d*scale,bounds:[x,y,x+pw,y+ph]});
    x+=pw+24;rowHeight=Math.max(rowHeight,ph);
  }
  const height=y+rowHeight+40,viewH=$('inventoryScroll').clientHeight;
  $('inventorySpacer').style.height=`${Math.max(0,height-viewH)}px`;
  $('inventory').style.width=`${invW}px`;$('inventory').style.height=`${viewH}px`;
}
function schedulePaint(){if(!paintPending){paintPending=true;requestAnimationFrame(()=>{paintPending=false;draw();});}}
function draw() {
  if(!state)return;layoutBoards();
  const ctx=prepare($('field')),w=$('field').clientWidth,h=$('field').clientHeight;
  ctx.fillStyle='#fbfaf5';ctx.fillRect(0,0,w,h);
  // One-foot grid, drawn in the physical patio's coordinate frame.
  ctx.strokeStyle='#e7e8de';ctx.lineWidth=.7;
  const xs=state.patio.map(p=>p[0]),ys=state.patio.map(p=>p[1]),maxX=Math.max(...xs),maxY=Math.max(...ys);
  for(let x=0;x<=maxX;x++){ctx.beginPath();ctx.moveTo(fieldOrigin[0]+x*scale,fieldOrigin[1]);ctx.lineTo(fieldOrigin[0]+x*scale,fieldOrigin[1]-maxY*scale);ctx.stroke();}
  for(let y=0;y<=maxY;y++){ctx.beginPath();ctx.moveTo(fieldOrigin[0],fieldOrigin[1]-y*scale);ctx.lineTo(fieldOrigin[0]+maxX*scale,fieldOrigin[1]-y*scale);ctx.stroke();}
  ctx.beginPath();state.patio.forEach(([x,y],i)=>i?ctx.lineTo(fieldOrigin[0]+x*scale,fieldOrigin[1]-y*scale):ctx.moveTo(fieldOrigin[0]+x*scale,fieldOrigin[1]-y*scale));ctx.closePath();ctx.strokeStyle='#657965';ctx.lineWidth=2;ctx.stroke();
  for(const p of state.pieces){if(!p.active)continue;drawPiece(ctx,p,dragging?.id===p.id?dragging.preview:p.pose,fieldOrigin,p.id===selected);}
  if(dragging?.from==='inventory'&&dragging.overField){const p=state.pieces.find(p=>p.id===dragging.id);ctx.globalAlpha=.65;drawPiece(ctx,p,dragging.preview,fieldOrigin,true);ctx.globalAlpha=1;}
  ctx.font='10px sans-serif';ctx.fillStyle='#83917f';ctx.textAlign='center';
  for(let x=0;x<=maxX;x+=2)ctx.fillText(`${x} ft`,fieldOrigin[0]+x*scale,fieldOrigin[1]+19);
  const inv=prepare($('inventory')),scroll=$('inventoryScroll').scrollTop;
  for(const item of inventoryItems){if(item.bounds[3]<scroll||item.bounds[1]>scroll+$('inventory').clientHeight)continue;drawPiece(inv,item.p,[0,0,0],[item.x,item.y-scroll],item.p.id===selected);}
}
function worldPoint(event) {const r=$('field').getBoundingClientRect();return[(event.clientX-r.left-fieldOrigin[0])/scale,(fieldOrigin[1]-(event.clientY-r.top))/scale];}
function isInside(geo,x,y,angle=0) {const c=Math.cos(angle),s=Math.sin(angle);const ctx=$('field').getContext('2d');ctx.save();ctx.setTransform(1,0,0,1,0,0);const yes=ctx.isPointInPath(geo.path,c*x+s*y,-s*x+c*y);ctx.restore();return yes;}
function fieldHit(event) {
  const [x,y]=worldPoint(event);
  return [...state.pieces].reverse().find(p=>p.active&&isInside(geometry.get(p.id),x-p.pose[0],y-p.pose[1],p.pose[2]));
}
function inventoryHit(event){const r=$('inventory').getBoundingClientRect(),x=event.clientX-r.left,y=event.clientY-r.top+$('inventoryScroll').scrollTop;return inventoryItems.find(i=>x>=i.bounds[0]&&x<=i.bounds[2]&&y>=i.bounds[1]&&y<=i.bounds[3]&&isInside(geometry.get(i.p.id),(x-i.x)/scale,(i.y-y)/scale));}
function stopHold(){holding=false;send({type:'stop'});}
$('recordStart').onclick=()=>send({type:'record_start'});
$('recordFinish').onclick=()=>{holding=false;send({type:'record_finish'});};
function beginHold(){if(holding||!state)return;holding=true;send({type:'start'});}
function edit(edit){send({type:'edit',edit});}
function beginDrag(event,from){
  if(!state||event.button!==0)return;
  $('field').focus({preventScroll:true});
  const hit=from==='field'?fieldHit(event):inventoryHit(event)?.p;
  if(!hit){selected=null;updateUI();schedulePaint();return;}
  selected=hit.id;updateUI();
  const point=worldPoint(event),wasHolding=holding;
  // Freeze while positioning; overlaps remain allowed. Continue on release
  // only if the user is still holding space/the on-screen hold button.
  send({type:'stop'});
  dragging={id:hit.id,from,start:[event.clientX,event.clientY],preview:hit.pose.slice(),offset:from==='field'?[point[0]-hit.pose[0],point[1]-hit.pose[1]]:[0,0],moved:false,overField:false,wasHolding};
  event.currentTarget.setPointerCapture(event.pointerId);event.preventDefault();schedulePaint();
}
function moveDrag(event){
  if(!dragging)return;
  if(Math.hypot(event.clientX-dragging.start[0],event.clientY-dragging.start[1])>3)dragging.moved=true;
  const [x,y]=worldPoint(event),r=$('field').getBoundingClientRect();dragging.overField=event.clientX>=r.left&&event.clientX<=r.right&&event.clientY>=r.top&&event.clientY<=r.bottom;
  dragging.preview=[x-dragging.offset[0],y-dragging.offset[1],dragging.from==='inventory'?0:dragging.preview[2]];
  if(dragging.from==='inventory'){
    const ghost=$('dragGhost'),canvas=ghost.querySelector('canvas'),p=state.pieces.find(p=>p.id===dragging.id),b=geometry.get(p.id).bounds;
    ghost.hidden=dragging.overField;canvas.style.width=`${(b[2]-b[0])*scale+10}px`;canvas.style.height=`${(b[3]-b[1])*scale+10}px`;
    const ctx=prepare(canvas);drawPiece(ctx,p,[0,0,0],[5-b[0]*scale,5+b[3]*scale],true);
    ghost.style.left=`${event.clientX+b[0]*scale-5}px`;ghost.style.top=`${event.clientY-b[3]*scale-5}px`;
  }
  schedulePaint();
}
function finishDrag(event,cancel=false){
  if(!dragging)return;const d=dragging;dragging=null;$('dragGhost').hidden=true;
  if(!cancel&&d.moved&&(d.from==='field'||d.overField))edit({action:d.from==='inventory'?'insert':'pose',sid:d.id,pose:d.preview});
  if(holding&&d.wasHolding)send({type:'start'});
  schedulePaint();
}
for(const id of ['field','inventory']){
  $(id).addEventListener('pointerdown',e=>beginDrag(e,id==='field'?'field':'inventory'));
  $(id).addEventListener('pointermove',moveDrag);
  $(id).addEventListener('pointerup',e=>finishDrag(e));
  $(id).addEventListener('pointercancel',e=>finishDrag(e,true));
}
function activeSelection(){return state?.pieces.find(p=>p.id===selected&&p.active);}
function rotate(degrees){const p=activeSelection();if(p)edit({action:'pose',sid:p.id,pose:[p.pose[0],p.pose[1],p.pose[2]+degrees*Math.PI/180]});}
$('ccw').onclick=()=>rotate(5);$('cw').onclick=()=>rotate(-5);
$('angle').onchange=()=>{const p=activeSelection(),angle=Number($('angle').value);if(p&&Number.isFinite(angle))edit({action:'pose',sid:p.id,pose:[p.pose[0],p.pose[1],angle*Math.PI/180]});};
$('lock').onclick=()=>{const p=activeSelection();if(p)edit({action:'lock',sid:p.id,locked:!p.locked});};
$('delete').onclick=()=>{const p=activeSelection();if(p)edit({action:'delete',sid:p.id});};
$('undo').onclick=()=>{stopHold();send({type:'undo'});};
$('check').onclick=()=>send({type:'check'});$('export').onclick=()=>send({type:'export'});
$('load').onclick=()=>{stopHold();$('file').click();};
$('file').onchange=async()=>{try{const file=$('file').files[0];if(file)send({type:'load',document:JSON.parse(await file.text())});}catch(e){message(`Could not open layout: ${e.message}`,true);}$('file').value='';};
$('filter').oninput=schedulePaint;$('inventoryScroll').onscroll=schedulePaint;
$('hold').onpointerdown=e=>{e.preventDefault();$('hold').setPointerCapture(e.pointerId);beginHold();};
$('hold').onpointerup=stopHold;$('hold').onpointercancel=stopHold;
window.addEventListener('keydown',e=>{
  if((e.target.tagName==='INPUT'&&e.target.type!=='range')||['TEXTAREA','SELECT'].includes(e.target.tagName)||e.target.isContentEditable)return;
  if(e.code==='Space'){e.preventDefault();if(!e.repeat)beginHold();return;}
  if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='z'){e.preventDefault();stopHold();send({type:'undo'});return;}
  const p=activeSelection();if(!p)return;
  if(e.key.toLowerCase()==='q'){e.preventDefault();rotate(e.shiftKey?1:5);}
  if(e.key.toLowerCase()==='e'){e.preventDefault();rotate(e.shiftKey?-1:-5);}
  if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();edit({action:'delete',sid:p.id});}
  if(e.key.toLowerCase()==='l')edit({action:'lock',sid:p.id,locked:!p.locked});
  if(e.key.startsWith('Arrow')){e.preventDefault();const pose=p.pose.slice(),step=(e.shiftKey?1:.25)/12;const changes={ArrowLeft:[0,-step],ArrowRight:[0,step],ArrowUp:[1,step],ArrowDown:[1,-step]};const [i,d]=changes[e.key];pose[i]+=d;edit({action:'pose',sid:p.id,pose});}
});
window.addEventListener('keyup',e=>{if(e.code==='Space'){e.preventDefault();stopHold();}});
window.addEventListener('blur',()=>{stopHold();if(dragging)finishDrag(null,true);});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stopHold();});
setInterval(()=>{if(holding&&!dragging)send({type:'hold'});},150);
new ResizeObserver(schedulePaint).observe($('fieldWrap'));
connect();
