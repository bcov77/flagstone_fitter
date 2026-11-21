"""Meaningful failure cases for the isolated geometry prototype."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier_search'))
from core import *
import planner as baseline
from planner_v2 import diverse_poses,piece_round_robin,coarse_poses,proposal


def test_close_poses_cannot_erase_distant_orientation_or_position():
 rows=[(i*.001,'a',[0,0,i*.01]) for i in range(8)]
 rows.extend([(1,'a',[0,0,math.pi/2]),(2,'a',[.3,0,0])])
 selected=diverse_poses(rows,3)
 assert [x[2] for x in selected]==[[0,0,0],[0,0,math.pi/2],[.3,0,0]]


def test_piece_shortlist_cannot_be_monopolized_by_best_scoring_piece():
 rows=[(i*.01,'a',[i,0,0]) for i in range(8)]+[(1,'b',[0,0,0]),(2,'c',[0,0,0])]
 assert [x[1] for x in piece_round_robin(rows,3)]==['a','b','c']


def test_invalid_small_turns_do_not_consume_same_piece_shortlist():
 spec=dict(raster_step_ft=.05,angle_step_degrees=15,piece_budget=1,poses_per_angle=3,poses_per_piece=8,coarse_limit=24,old_angles={'a':0},minimum_reorientation_degrees=45)
 pocket=shapely.box(0,0,1,1);local={'a':shapely.box(-.15,-.15,.15,.15)}
 rows=coarse_poses(local,['a'],pocket,pocket,spec,np.random.default_rng(4),lambda:False,min_fraction=.01)
 assert rows
 assert all(abs(math.degrees(math.remainder(row[2][2],2*math.pi)))+1e-8>=45 for row in rows)


def test_cached_free_space_matches_original_scaled_proposal():
 cfg=config()['physics'];patio=shapely.box(0,0,3,3)
 local={'1':shapely.box(-.3,-.3,.3,.3),'2':shapely.box(-.4,-.4,.4,.4)}
 doc=dict(patio=list(patio.exterior.coords),stones=[dict(id=s,outline=list(g.exterior.coords),active=s=='1',pose=[1,1,0]) for s,g in local.items()],physics=dict(pieces={}))
 board=shapes(doc);original_free=patio.buffer(-GAP).difference(unary_union(list(board.values())).buffer(GAP))
 t=dict(remove=['1'],id='replace_1',pocket=shapely.box(.575,.575,1.425,1.425),free=patio.buffer(-GAP))
 t['existing_free_sqft']=t['pocket'].intersection(original_free).area
 placements=[dict(stone='2',pose=[1,1,0])]
 old=baseline.proposal(doc,t,placements,local,{},patio,cfg,.8)
 assert old is not None
 assert proposal(doc,t,placements,local,{},patio,cfg,.8)==old


def test_early_eligibility_still_rejects_downsizing_and_locked_removal():
 cfg=config()['physics'];patio=shapely.box(0,0,3,3)
 local={'1':shapely.box(-.4,-.4,.4,.4),'2':shapely.box(-.3,-.3,.3,.3)}
 doc=dict(stones=[dict(id=s,outline=list(g.exterior.coords),active=s=='1',pose=[1,1,0]) for s,g in local.items()],physics=dict(pieces={}))
 t=dict(remove=['1'])
 assert proposal(doc,t,[dict(stone='2',pose=[1,1,0])],local,{},patio,cfg) is None
 doc['physics']['pieces']['1']={'locked':True}
 assert proposal(doc,t,[dict(stone='1',pose=[1,1,math.pi/2])],local,{},patio,cfg) is None
