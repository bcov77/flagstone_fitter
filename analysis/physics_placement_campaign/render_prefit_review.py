"""Fixed numbered pre-minimization review, selected using pre-fit geometry only."""
import json,math,hashlib
import numpy as np
import shapely
from shapely.geometry import box
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
import build_prefit_review as b

D=b.DEST
# One proposal per distinct physical opening. Insertion cases do not simply
# restore any of the stones removed to create their synthetic context.
PICKS=[('board_03_insert_pocket_02','11'),('board_07_insert_pocket_01','48'),
       ('board_00_replace_138','122'),('board_00_replace_50','116'),
       ('board_00_replace_60','67'),('board_00_replace_61','94')]

def eligible(c):
 m=c['metrics']
 return (m['coverage']>=.68 and m['max_clearance_intrusion_inches']<=.4
         and m['clearance_intrusion_fraction']<=.005 and m['physical_overlap_sqft']<1e-6)

def paint(ax,g,**kw):
 for part in b.parts(g):ax.add_patch(Patch(list(part.exterior.coords),**kw))

def lines(ax,g,**kw):
 if g.is_empty:return
 if hasattr(g,'geoms'):
  for part in g.geoms:lines(ax,part,**kw)
 elif hasattr(g,'xy'):
  x,y=g.xy;ax.plot(x,y,**kw)

def build():
 if (D/'human_feedback.jsonl').exists() or (D/'review_frozen.json').exists():
  raise RuntimeError('Numbered panel is frozen; use a new review directory.')
 records=[]
 for num,(target_id,sid) in enumerate(PICKS,1):
  initial=json.loads((D/'search_results'/f'{target_id}.json').read_text())
  refined=json.loads((D/'refined_results'/f'{target_id}.json').read_text())
  choices=[c for c in initial['candidates']+refined['candidates'] if c['stone']==sid and eligible(c)]
  if not choices:raise ValueError((target_id,sid,'no eligible full-size pose'))
  c=max(choices,key=lambda c:c['score']);t=initial['target']
  alternatives=[a for a in initial['candidates']+refined['candidates'] if a['stone']!=sid and a['stone']!=t['old'] and eligible(a)]
  alternative=max(alternatives,key=lambda a:a['score']) if alternatives else None
  r=dict(number=num,id=f'prefit_v1_{num:02}',target=t,candidate=c,baseline=initial['baseline'],alternate=alternative,
         review_status='unreviewed; geometry-screened, not human-approved',physics_steps=0,
         piece_choice=None,starting_pose=None,opening_mask=None,notes='')
  if t['old']:assert c['area_gain_sqft']>.01
  records.append(r)
 (D/'placements.json').write_text(json.dumps(records,indent=2))
 return records

def draw(ax,r,proposed):
 t=r['target'];c=r['candidate'];p=shapely.from_wkt(t['pocket']);free=shapely.from_wkt(t['free'])
 active=set(b.ACTIVE)-set(t['removed'])-({t['old']} if t['old'] else set())
 for sid in active:
  g=b.ORIGINAL[sid];paint(ax,g,facecolor='#dedad1',edgecolor='#827b6d',linewidth=.65)
  z=g.representative_point();ax.text(z.x,z.y,sid,color='#666052',fontsize=7,ha='center',va='center',clip_on=True)
 paint(ax,p,facecolor='#bfe9ed',edgecolor='#16839a',linewidth=.65,alpha=.8)
 # Artificial caps at trimmed joint mouths, distinguished from true obstacles.
 caps=p.boundary.difference(free.boundary.buffer(.001))
 lines(ax,caps,color='#9c60af',linestyle='--',linewidth=1.25,zorder=5)
 g=b.world(c['stone'],c['pose'])
 old=b.ORIGINAL[t['old']] if t['old'] else None
 shown=g if proposed else old
 if shown is not None:
  paint(ax,shown,facecolor='#e7a156' if proposed else '#8eabc5',edgecolor='#89501f' if proposed else '#426484',linewidth=1.1,zorder=3)
  if proposed:
   intr=g.difference(free);paint(ax,intr,facecolor='#d13e4a',edgecolor='#b11f32',linewidth=.3,hatch='////',zorder=5)
  sid=c['stone'] if proposed else t['old'];z=shown.representative_point();ax.text(z.x,z.y,sid,fontsize=12,fontweight='bold',ha='center',va='center',zorder=6)
  if old is not None and proposed:lines(ax,old.exterior,color='#426484',linestyle=':',linewidth=.7,zorder=4)
 x0,y0,x1,y1=p.bounds
 if old is not None:ox0,oy0,ox1,oy1=old.bounds;x0=min(x0,ox0);y0=min(y0,oy0);x1=max(x1,ox1);y1=max(y1,oy1)
 gx0,gy0,gx1,gy1=g.bounds;x0=min(x0,gx0);y0=min(y0,gy0);x1=max(x1,gx1);y1=max(y1,gy1)
 span=max(x1-x0,y1-y0)+.75;cx=(x0+x1)/2;cy=(y0+y1)/2
 a,bb,cc,dd=cx-span/2,cy-span/2,cx+span/2,cy+span/2
 ax.set_xlim(a,cc);ax.set_ylim(bb,dd);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([]);ax.set_facecolor('#fcfbf8')
 lines(ax,b.PATIO.exterior,color='#444',linewidth=1.1)
 for sp in ax.spines.values():sp.set_color('#c7c0b5')
 title='PROPOSED · full size' if proposed else ('CURRENT · original stone' if old is not None else 'OPENING · empty')
 ax.set_title(title,fontsize=10,pad=4,color='#5d574e')
 length=.5 if span>2.5 else .25;start=a+.06*span;y=bb+.065*span
 ax.plot([start,start+length],[y,y],color='#333',linewidth=2,zorder=7)
 ax.text(start+length/2,y+.025*span,f'{length*12:g}″',ha='center',fontsize=8,zorder=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.5))

def sheet(records,path,rows,cols,figsize):
 fig=plt.figure(figsize=figsize,facecolor='white');outer=fig.add_gridspec(rows,cols,left=.025,right=.985,bottom=.065,top=.89,hspace=.25,wspace=.11)
 for i,r in enumerate(records):
  grid=outer[i//cols,i%cols].subgridspec(3,2,height_ratios=[.16,1,.23],hspace=.12,wspace=.04)
  title=fig.add_subplot(grid[0,:]);title.axis('off');t=r['target'];c=r['candidate']
  action=f"INSERT {c['stone']}" if t['old'] is None else f"REPLACE {t['old']} → {c['stone']}"
  title.text(0,.6,f"{r['number']:02d}   {action}",fontsize=16,fontweight='bold')
  title.text(1,.6,'synthetic opening' if t['removed'] else 'original board',fontsize=10,ha='right',color='#656057')
  for j,proposed in enumerate([False,True]):draw(fig.add_subplot(grid[1,j]),r,proposed)
  cap=fig.add_subplot(grid[2,:]);cap.axis('off');m=c['metrics']
  baseline=r['baseline'];area=f"   |   area {c.get('area_gain_sqft',m['area_sqft']):+.2f} ft²" if baseline else f"   |   stone {m['area_sqft']:.2f} ft²"
  cap.text(0,.78,f"Pocket filled: {m['coverage']:.0%}"+(f"  (original {baseline['coverage']:.0%})" if baseline else '')+area,fontsize=10)
  cap.text(0,.34,f"Physical overlap: 0   |   clearance intrusion: {m['clearance_intrusion_fraction']:.2%}, sampled depth {m['max_clearance_intrusion_inches']:.2f}″",fontsize=9,color='#635c52')
 fig.suptitle('Before minimization · shape-fit review',fontsize=27,fontweight='bold',y=.98)
 fig.text(.5,.949,'Orange: proposed stone  |  blue: original  |  cyan: remaining opening after ½″ clearance  |  gray: fixed neighbors',ha='center',fontsize=12)
 fig.text(.5,.927,'Purple dashes: inferred cuts at narrow joint mouths. Red: clearance intrusion. Every pose is full size; no physics was run.',ha='center',fontsize=11)
 fig.text(.025,.039,'Same zoom within each pair; scale varies between cases. “Pocket filled” depends on the cyan mask and is a review aid, not a quality verdict.',fontsize=10)
 fig.text(.025,.018,'Review both the opening definition and the stone: does the shape match, is the rotation sensible, and are the remaining wedges acceptable?',fontsize=11)
 fig.savefig(path,dpi=180,facecolor='white');plt.close(fig)

if __name__=='__main__':
 records=build();sheet(records,D/'prefit_06.png',3,2,(21,22))
 for page in range(2):sheet(records[page*3:(page+1)*3],D/f'prefit_{page*3+1:02}_{page*3+3:02}.png',3,1,(12,24))
 for record in records:sheet([record],D/f"case_{record['number']:02}.png",1,1,(12,10))
 print('Rendered 6 new pre-fit cases, 2 insertion / 4 replacement; no physics')
