"""Ensure research grouping doesn't conceal compound replacements or invalid edits."""
import copy,math
from experiment import arm,eligible,config,read,BATCH

def action(remove,ids):
 return dict(remove=remove,placements=[dict(stone=s,pose=[1,1,0]) for s in ids],physics_protocol='held_release')

def test_operator_groups_distinguish_fills_from_substitutions():
 assert arm(action([],['a','b']))=='compound_fill'
 assert arm(action(['a'],['b','c']))=='compound_replace'
 assert arm(action(['a'],['a','b']))=='neighbor_refill'
 assert arm(action(['a','b'],['c']))=='compound_replace'
 assert arm(action(['a'],['a']))=='reorientation'

def test_real_root_constraints_and_original_inventory():
 doc=read(BATCH/'runs/run_000076/final.json');original=copy.deepcopy(doc);cfg=config()
 s=next(s for s in doc['stones'] if s['active'] and not doc['physics']['pieces'][str(s['id'])].get('locked'))
 sid=str(s['id']);p=action([sid],[sid]);p['placements'][0]['pose']=[*s['pose'][:2],s['pose'][2]+math.radians(44)]
 assert not eligible(doc,p,cfg)
 p['placements'][0]['pose'][2]=s['pose'][2]+math.radians(45)
 assert eligible(doc,p,cfg)
 doc['physics']['pieces'][sid]['locked']=True
 assert not eligible(doc,p,cfg)
 doc=original
 unused=next(s for s in doc['stones'] if not s['active']);p=action([sid],[str(unused['id'])])
 unused['outline']=[[0,0],[.001,0],[0,.001]]
 assert not eligible(doc,p,cfg)
